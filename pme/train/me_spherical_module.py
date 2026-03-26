import warnings
import copy
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from astropy import units as u
from astropy.visualization import ImageNormalize
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from pme.data.differential_rotation import carrington_rotation_velocity
from pme.data.util import cartesian_to_spherical
from pme.evaluation.loader import to_spherical, to_cartesian
from pme.model import MESphericalModel, VelocityCorrectionModel, LimbCorrectionModel, \
    NormalizationModule
from pme.train.artifact_correction import ArtifactCorrectionModule
from pme.train.me_atmosphere import HMIMEAtmosphere, PHIMEAtmosphere
from pme.train.physics import compute_physics_losses
from pme.train.soap import SOAP
from pme.train.util import acos_safe, atan2_safe, log_wandb_image, get_random_coordinates, random_time_shift_coords


class MESphericalModule(LightningModule):

    def __init__(self, image_shape, wavelength_config,
                 gauss_per_dB, Rs_per_ds, seconds_per_dt,
                 instrument_config,
                 lr_params=None, model_config=None, normalization_config=None,
                 time_shift_config=None,
                 lambda_config=None, **kwargs):
        super().__init__()
        lr_params = lr_params if lr_params is not None else {"start": 1e-3, "end": 1e-4, "iterations": 1e5}

        self.image_shape = image_shape

        # init model
        model_config = copy.deepcopy(model_config) if model_config is not None else {}
        self.parameter_model = MESphericalModel(**model_config)

        # init instrument models
        forward_models = {}
        velocity_correction_models = {}
        limb_correction_models = {}
        artifact_correction_models = {}
        for instrument in instrument_config:
            instrument_id = instrument.pop('instrument_id')
            instrument_type = instrument.pop('type')
            wavelength_center = wavelength_config[instrument_id]['wavelength_center']
            velocity_correction = instrument.pop('velocity_correction', False)
            limb_correction = instrument.pop('limb_correction', False)
            artifact_correction = instrument.pop('artifact_correction', False)

            if instrument_type == 'hmi':
                forward_models[instrument_id] = HMIMEAtmosphere(wavelength_center=wavelength_center, **instrument)
            elif instrument_type == 'phi':
                forward_models[instrument_id] = PHIMEAtmosphere(wavelength_center=wavelength_center, **instrument)
            else:
                raise ValueError(f"Unknown instrument type: {instrument_type}")

            if velocity_correction:
                velocity_correction_models[instrument_id] = VelocityCorrectionModel()
            if limb_correction:
                limb_correction_models[instrument_id] = LimbCorrectionModel()
            if artifact_correction:
                artifact_correction_models[instrument_id] = ArtifactCorrectionModule()

        self.forward_models = nn.ModuleDict(forward_models)
        self.velocity_correction_models = nn.ModuleDict(velocity_correction_models)
        self.limb_correction_models = nn.ModuleDict(limb_correction_models)
        self.artifact_correction_models = nn.ModuleDict(artifact_correction_models)

        self.lr_params = lr_params

        self.validation_outputs = {}
        normalization_config = normalization_config if normalization_config is not None else {}
        self.normalization = NormalizationModule()
        self.stokes_loss_function = nn.MSELoss(reduction='none')
        #
        scheduled_lambda_config = {}
        lambdas = {}
        available_lambdas = ['I', 'Q', 'U', 'V',
                             'induction', 'divergence', 'force_free', 'potential', 'dB_dt']
        lambda_config = lambda_config if lambda_config is not None else {}
        # add default lambdas for stokes vectors if not specified
        lambda_config['I'] = lambda_config.get('I', 1.0)
        lambda_config['Q'] = lambda_config.get('Q', 1.0)
        lambda_config['U'] = lambda_config.get('U', 1.0)
        lambda_config['V'] = lambda_config.get('V', 1.0)
        for k, v in lambda_config.items():
            assert k in available_lambdas, f"Unknown lambda key: {k}, must be one of {available_lambdas}"
            if isinstance(v, dict):
                lambda_type = v.get('type', 'exponential')
                if lambda_type == 'exponential':
                    gamma = (v['end'] / v['start']) ** (1 / v['iterations'])
                elif lambda_type == 'linear':
                    gamma = (v['end'] - v['start']) / v['iterations']
                elif lambda_type == 'step':
                    gamma = 0
                else:
                    raise ValueError(f"Unknown lambda type: {lambda_type}")
                scheduled_lambda_config[k] = {'end': v['end'], 'gamma': gamma, 'type': lambda_type,
                                              'iterations': v['iterations']}
                lambdas[k] = nn.Parameter(torch.tensor(v['start'], dtype=torch.float32), requires_grad=False)
            else:
                lambdas[k] = nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
        self.scheduled_lambda_config = scheduled_lambda_config
        self.lambdas = nn.ParameterDict(lambdas)

        # time shift config
        time_shift_config = {'scale': 0.0, 'iterations': 0.0} if time_shift_config is None else time_shift_config
        scale_value = time_shift_config['scale'] / seconds_per_dt
        self.time_shift_scale = nn.Parameter(torch.tensor(scale_value, dtype=torch.float32), requires_grad=False)
        self.time_shift_scale_gamma = scale_value / time_shift_config['iterations'] if time_shift_config['iterations'] > 0 else 0.0
        #
        self.gauss_per_dB = gauss_per_dB
        self.Rs_per_ds = Rs_per_ds
        self.meters_per_ds = Rs_per_ds * (1 * u.Rsun).to_value(u.m)
        self.seconds_per_dt = seconds_per_dt
        #
        self.val_outputs = []

    def configure_optimizers(self):
        parameters = (list(self.parameter_model.parameters()) +
                      list(self.forward_models.parameters()) +
                      list(self.velocity_correction_models.parameters()) +
                      list(self.limb_correction_models.parameters()) +
                      list(self.artifact_correction_models.parameters()))
        if isinstance(self.lr_params, dict):
            lr_start = self.lr_params['start']
            lr_end = self.lr_params['end']
            iterations = self.lr_params['iterations']
        elif isinstance(self.lr_params, (float, int)):
            lr_start = self.lr_params
            lr_end = self.lr_params
            iterations = 1
            self.lr_params = {'start': lr_start, 'end': lr_end, 'iterations': iterations}
        else:
            raise ValueError(f"Invalid lr_params: {self.lr_params}, must be dict or float/int")
        optimizer = SOAP(parameters, lr=lr_start)
        scheduler = ExponentialLR(optimizer, gamma=(lr_end / lr_start) ** (1 / iterations))

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        instrument_ids = list(batch.keys())

        coords = torch.cat([batch[k]['coords'] for k in instrument_ids])
        cartesian_to_spherical_transform = torch.cat(
            [batch[k]['cartesian_to_spherical_transform'] for k in instrument_ids])
        rtp_to_img_transform = torch.cat([batch[k]['rtp_to_img_transform'] for k in instrument_ids])
        v_obs_los = torch.cat([batch[k]['v_obs_los'] for k in instrument_ids])
        ds_lengths = {k: batch[k]['coords'].shape[0] for k in instrument_ids}

        # apply random shift to time coordinate
        if self.time_shift_scale.item() > 0:
            coords = random_time_shift_coords(coords, self.seconds_per_dt, scale=self.time_shift_scale.item())

        # forward step
        coords.requires_grad = True
        output = self.parameter_model(coords)

        transformed_output = self.transform_parameters(output, coords, cartesian_to_spherical_transform,
                                                       rtp_to_img_transform)

        #################################################
        # stokes profile synthesis
        forward_params = self.scale_parameters(output, transformed_output, v_obs_los)
        current_idx = 0
        stokes_pred_normalized = []
        stokes_true_normalized = []
        for instrument_id, n_samples in ds_lengths.items():
            ds_mu = batch[instrument_id]['mu']
            ds_wavelength_grid = batch[instrument_id]['wavelength_grid']
            ds_stokes_true = batch[instrument_id]['stokes']

            ds_forward_params = {k: v[current_idx:current_idx + n_samples] for k, v in forward_params.items()}
            # apply velocity correction if available
            if instrument_id in self.velocity_correction_models:
                time_coords = batch[instrument_id]['coords'][..., 0:1]
                v_obs_correction = self.velocity_correction_models[instrument_id](time_coords)
                ds_forward_params['vdop'] += v_obs_correction

            if instrument_id in self.limb_correction_models:
                limb_correction = self.limb_correction_models[instrument_id](ds_mu)
                ds_forward_params = self.correct_limb_effects(ds_forward_params, limb_correction)

            # compute stokes profiles for parameters
            I, Q, U, V = self.forward_models[instrument_id](**ds_forward_params, wavelength_grid=ds_wavelength_grid,
                                                            mu=ds_mu)
            ds_stokes_pred = torch.stack([I, Q, U, V], dim=-2)

            if instrument_id in self.artifact_correction_models:
                assert 'pix' in batch[
                    instrument_id], f"Artifact correction requires 'pix' in batch for instrument {instrument_id}"
                pix = batch[instrument_id]['pix']
                correction = self.artifact_correction_models[instrument_id](pix, ds_stokes_pred)
                ds_stokes_pred = correction['stokes_corr']

            # stokes profiles
            Ic = torch.quantile(ds_stokes_true[..., 0:1, :], 0.9, dim=-1, keepdim=True)
            stokes_true_normalized.append(self.normalization(ds_stokes_true, Ic=Ic))
            stokes_pred_normalized.append(self.normalization(ds_stokes_pred, Ic=Ic))
            current_idx += n_samples

        #################################################
        # compute stokes loss
        stokes_true_normalized = torch.cat(stokes_true_normalized, dim=0)
        stokes_pred_normalized = torch.cat(stokes_pred_normalized, dim=0)
        stokes_loss = self.stokes_loss_function(stokes_pred_normalized, stokes_true_normalized)

        # sum over wavelength axis
        stokes_loss = stokes_loss.sum(-1)

        # logging losses
        I_loss, Q_loss, U_loss, V_loss = stokes_loss.mean(dim=0)

        # weighted loss - apply lambda weights for each stokes parameter
        lambda_stokes = torch.stack([self.lambdas['I'], self.lambdas['Q'], self.lambdas['U'], self.lambdas['V']])
        stokes_loss = torch.einsum('...i,i->...', stokes_loss, lambda_stokes)

        #################################################
        # compute physics losses
        if not any(k in ['induction', 'divergence', 'force_free', 'potential', 'dB_dt'] and self.lambdas[k].item() > 0
                   for k, v in self.lambdas.items()):
            # skip physics losses if not required
            physics_losses = {}
        else:
            random_coords = get_random_coordinates(coords)

            # forward pass of random coordinates
            physics_out = self.parameter_model(random_coords)
            b = torch.cat([physics_out['b_x'], physics_out['b_y'], physics_out['b_z']], dim=-1)
            v = torch.cat([physics_out['v_x'], physics_out['v_y'], physics_out['v_z']], dim=-1)
            a_jac_matrix = physics_out['a_jac_matrix']

            # compute physics losses
            physics_losses = compute_physics_losses(b, v, a_jac_matrix, random_coords)

        #################################################
        # compute total loss
        total_loss = stokes_loss.mean()
        logging_loss = {"I": I_loss, "Q": Q_loss, "U": U_loss, "V": V_loss,
                        "stokes_loss": stokes_loss.mean()}

        #################################################
        # add physics losses to total loss with lambda weights if specified
        if 'induction' in physics_losses and 'induction' in self.lambdas:
            induction_loss = physics_losses['induction'].mean()
            assert not torch.isnan(induction_loss), f"Encountered invalid value. Induction loss is NaN"
            total_loss = total_loss + self.lambdas['induction'] * induction_loss
            logging_loss['induction'] = induction_loss
        if 'divergence' in physics_losses and 'divergence' in self.lambdas:
            divergence_loss = physics_losses['divergence'].mean()
            assert not torch.isnan(divergence_loss), f"Encountered invalid value. Divergence loss is NaN"
            total_loss = total_loss + self.lambdas['divergence'] * divergence_loss
            logging_loss['divergence'] = divergence_loss
        if 'force_free' in physics_losses and 'force_free' in self.lambdas:
            force_free_loss = physics_losses['force_free'].mean()
            assert not torch.isnan(force_free_loss), f"Encountered invalid value. Force-free loss is NaN"
            total_loss = total_loss + self.lambdas['force_free'] * force_free_loss
            logging_loss['force_free'] = force_free_loss
        if 'potential' in physics_losses and 'potential' in self.lambdas:
            potential_loss = physics_losses['potential'].mean()
            assert not torch.isnan(potential_loss), f"Encountered invalid value. Potential loss is NaN"
            total_loss = total_loss + self.lambdas['potential'] * potential_loss
            logging_loss['potential'] = potential_loss
        if 'dB_dt' in physics_losses and 'dB_dt' in self.lambdas:
            dB_dt_loss = physics_losses['dB_dt'].mean()
            assert not torch.isnan(dB_dt_loss), f"Encountered invalid value. dB/dt loss is NaN"
            total_loss = total_loss + self.lambdas['dB_dt'] * dB_dt_loss
            logging_loss['dB_dt'] = dB_dt_loss

        assert not torch.isnan(total_loss), f"Encountered invalid value. Loss is NaN"
        logging_loss['loss'] = total_loss
        return logging_loss

    def scale_parameters(self, output, transformed_output, v_obs_los):
        vdop = transformed_output['vdop'] * self.meters_per_ds / self.seconds_per_dt
        vdop = vdop + v_obs_los  # add doppler correction - spacecraft velocity

        forward_params = {'b_field': transformed_output['b_field'] * self.gauss_per_dB,
                          'sin_inc2': transformed_output['sin_inc2'],
                          'cos_inc': transformed_output['cos_inc'],
                          'inc': transformed_output['inc'],
                          'sin2azi': transformed_output['sin2azi'],
                          'cos2azi': transformed_output['cos2azi'],
                          'azi': transformed_output['azi'],
                          'vdop': vdop,
                          'vmac': output['vmac'], 'damping': output['damping'],
                          'b0': output['b0'], 'b1': output['b1'], 'kl': output['kl']}
        return forward_params

    def transform_parameters(self, output, coords, cartesian_to_spherical_transform, rtp_to_img_transform):
        # transform B
        b_xyz = torch.cat([output['b_x'], output['b_y'], output['b_z']], dim=-1)
        b_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, b_xyz)
        b_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)

        # b_img = (xi, eta, zeta)
        # (field, inclination, azimuth) = field, gamma, psi = b_field, inc, azi
        # b_xi = - field * sin(gamma) * sin(psi)
        # b_eta = field * sin(gamma) * cos(psi)
        # b_zeta = field * cos(gamma)
        b_field = torch.norm(b_img, dim=-1, keepdim=True)

        sin_inc2 = (b_img[..., 0:1] ** 2 + b_img[..., 1:2] ** 2) / (b_field ** 2 + 1e-8)
        cos_inc = b_img[..., 2:3] / (b_field + 1e-8)

        # 2 * sin(x) * cos(x) = sin(2*x)
        # sin(x)**2 - cos(x)**2 = -cos(2*x)
        sin2azi = -2 * b_img[..., 0:1] * b_img[..., 1:2] / (b_img[..., 0:1] ** 2 + b_img[..., 1:2] ** 2 + 1e-8)
        cos2azi = -(b_img[..., 0:1] ** 2 - b_img[..., 1:2] ** 2) / (b_img[..., 0:1] ** 2 + b_img[..., 1:2] ** 2 + 1e-8)

        # Shift polarizer position for HMI
        inc = acos_safe(b_img[..., 2:3] / (b_field + 1e-8))
        azi = atan2_safe(-b_img[..., 0:1], b_img[..., 1:2])

        # transform to carrington frame --> add rotation velocity
        spherical_coords = cartesian_to_spherical(coords[..., 1:], torch)
        colatitude = spherical_coords[..., 1]  # theta in spherical coordinates
        latitude = torch.pi / 2 - colatitude  # convert to latitude
        radius = spherical_coords[..., 0] * self.meters_per_ds  # r in spherical coordinates
        v_rot = carrington_rotation_velocity(latitude, radius)  # in m/s
        v_rot = v_rot / self.meters_per_ds * self.seconds_per_dt  # convert to ds/dt (model units)

        # transform V
        v_xyz = torch.cat([output['v_x'], output['v_y'], output['v_z']], dim=-1)
        v_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, v_xyz)
        v_rtp_alt = torch.stack([v_rtp[..., 0], v_rtp[..., 1], v_rtp[..., 2] + v_rot], dim=-1)
        v_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, v_rtp_alt)

        vdop = -v_img[..., 2:3]  # negative because doppler shift is defined in the observer frame

        return {'b_field': b_field,
                'sin2azi': sin2azi, 'cos2azi': cos2azi, 'azi': azi,
                'sin_inc2': sin_inc2, 'cos_inc': cos_inc, 'inc': inc,
                'vdop': vdop,
                'v_rtp': v_rtp, 'b_rtp': b_rtp, 'v_img': v_img, 'b_img': b_img,
                'b_xyz': b_xyz, 'v_xyz': v_xyz}

    def correct_limb_effects(self, parameters, limb_correction):
        # apply limb correction
        c_b0 = limb_correction['c_b0']
        c_b1 = limb_correction['c_b1']
        # c_vmac = limb_correction['c_vmac']
        # c_damping = limb_correction['c_damping']
        # c_kl = limb_correction['c_kl']
        c_vdop = limb_correction['c_vdop']

        # apply limb correction to parameters
        b0 = parameters['b0'] * c_b0
        b1 = parameters['b1'] * c_b1
        # vmac = parameters['vmac'] * c_vmac
        # damping = parameters['damping'] * c_damping
        # kl = parameters['kl'] * c_kl
        vdop = parameters['vdop'] + c_vdop

        corrected_output = {k: v for k, v in parameters.items() if
                            k not in ['b0', 'b1', 'vdop']}
        corrected_output['b0'] = b0
        corrected_output['b1'] = b1
        # corrected_output['vmac'] = vmac
        # corrected_output['damping'] = damping
        # corrected_output['kl'] = kl
        corrected_output['vdop'] = vdop

        return corrected_output

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        scheduler = self.lr_schedulers()
        if scheduler.get_last_lr()[0] > self.lr_params['end']:
            scheduler.step()
        self.log('Learning Rate', scheduler.get_last_lr()[0])

        if hasattr(self.parameter_model, 'step'):
            self.parameter_model.step(self.global_step)
        if hasattr(self.parameter_model, 'fine_weight'):
            self.log('fine_weight', self.parameter_model.fine_weight.item())

        for k in self.scheduled_lambda_config.keys():
            value = self.lambdas[k]
            gamma = self.scheduled_lambda_config[k]['gamma']
            lambda_type = self.scheduled_lambda_config[k]['type']

            if lambda_type == 'linear':
                if value > self.scheduled_lambda_config[k]['end'] and gamma < 0:
                    new_value = value + gamma
                elif value < self.scheduled_lambda_config[k]['end'] and gamma > 0:
                    new_value = value + gamma
                else:
                    new_value = self.scheduled_lambda_config[k]['end']
            elif lambda_type == 'exponential':
                if value > self.scheduled_lambda_config[k]['end']:
                    # update lambda value
                    new_value = value * gamma
                else:
                    new_value = value
            elif lambda_type == 'step':
                if self.global_step >= self.scheduled_lambda_config[k]['iterations']:
                    new_value = self.scheduled_lambda_config[k]['end']
                else:
                    new_value = value
            else:
                raise ValueError(f"Unknown lambda type: {lambda_type}")
            self.lambdas[k].copy_(new_value)
            self.log(f"lambda.{k}", self.lambdas[k].item())

        # update time shift scale
        new_scale = self.time_shift_scale - self.time_shift_scale_gamma
        if new_scale <= 0.0:
            new_scale = 0.0
        self.time_shift_scale.copy_(new_scale)
        self.log('time_shift_scale', self.time_shift_scale.item() * self.seconds_per_dt)

        # log results to WANDB
        self.log_dict({f'train.{k}': v.mean() for k, v in outputs.items()})

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb):
        coords = batch['coords']
        mu = batch['mu']
        stokes_true = batch['stokes']
        cartesian_to_spherical_transform = batch['cartesian_to_spherical_transform']
        rtp_to_img_transform = batch['rtp_to_img_transform']
        v_obs_los = batch['v_obs_los']
        wavelength_grid = batch['wavelength_grid']
        instrument_id = batch['instrument_id']

        # forward step
        coords.requires_grad = True
        output = self.parameter_model(coords)

        transformed_output = self.transform_parameters(output, coords,
                                                       cartesian_to_spherical_transform, rtp_to_img_transform)
        forward_params = self.scale_parameters(output, transformed_output, v_obs_los)

        if instrument_id in self.limb_correction_models:
            limb_correction = self.limb_correction_models[instrument_id](mu)
            forward_params = self.correct_limb_effects(forward_params, limb_correction)
        else:
            limb_correction = None

        v_obs_correction = torch.zeros_like(v_obs_los)
        if instrument_id in self.velocity_correction_models:
            time_coords = coords[..., 0:1]
            v_obs_correction = self.velocity_correction_models[instrument_id](time_coords)
            forward_params['vdop'] += v_obs_correction

        I, Q, U, V = self.forward_models[instrument_id](**forward_params, mu=mu, wavelength_grid=wavelength_grid)
        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        if instrument_id in self.artifact_correction_models:
            assert 'pix' in batch, f"Artifact correction requires 'pix' in batch for instrument {instrument_id}"
            pix = batch['pix']
            artifact_correction = self.artifact_correction_models[instrument_id](pix, stokes_pred)
            stokes_pred = artifact_correction['stokes_corr']
        else:
            artifact_correction = None

        Ic = torch.quantile(stokes_true[..., 0:1, :], 0.9, dim=-1, keepdim=True)
        stokes_true_normalized = self.normalization(stokes_true, Ic=Ic)
        stokes_pred_normalized = self.normalization(stokes_pred, Ic=Ic)

        diff = torch.abs(stokes_true_normalized - stokes_pred_normalized)

        b = transformed_output['b_xyz']
        v = transformed_output['v_xyz']
        a_jac_matrix = output['a_jac_matrix']
        # physics_losses = compute_physics_losses(b, v, a_jac_matrix, coords)

        res = {'diff': diff,
               'stokes_true': stokes_true_normalized, 'stokes_pred': stokes_pred_normalized,
               **forward_params,
               'b_rtp': transformed_output['b_rtp'] * self.gauss_per_dB,
               'v_rtp': transformed_output['v_rtp'] * self.meters_per_ds / self.seconds_per_dt,
               'b_img': transformed_output['b_img'] * self.gauss_per_dB,
               'v_img': transformed_output['v_img'] * self.meters_per_ds / self.seconds_per_dt,
               # 'induction': physics_losses['induction'],
               # 'dB_dt': physics_losses['dB_dt'],
               # 'curl_VxB': physics_losses['curl_VxB'],
               # 'divergence': physics_losses['divergence'],
               # 'force_free': physics_losses['force_free'],
               # 'dB_dr': physics_losses['dB_dr'],
               'mu': mu, 'v_obs_los': v_obs_los,
               'v_obs_correction': v_obs_correction,
               }
        if limb_correction is not None:
            res['c_b0'] = limb_correction['c_b0']
            res['c_b1'] = limb_correction['c_b1']
            # res['c_vmac'] = limb_correction['c_vmac']
            # res['c_damping'] = limb_correction['c_damping']
            # res['c_kl'] = limb_correction['c_kl']
            res['c_vdop'] = limb_correction['c_vdop']
        if artifact_correction is not None:
            res['artifact_correction_params'] = artifact_correction['correction_params']

        res["lin_idx"] = batch["lin_idx"].long()  # for assembling full images later

        return res

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        # Only rank-0 needs to keep outputs if you're running val on rank-0 only.
        if outputs is not None:
            # ensure CPU to keep GPU mem low
            cpu_out = {k: v.detach().cpu() for k, v in outputs.items()}
            self.val_outputs.append(cpu_out)

    def on_validation_epoch_end(self):
        outputs_list = self.val_outputs
        if not outputs_list or any(len(o) == 0 for o in outputs_list):
            return

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            if rank == 0:
                obj_gather_list = [None] * world
                dist.gather_object(outputs_list, obj_gather_list, dst=0)
                outputs_list = [item for sub in obj_gather_list for item in sub]
            else:  # only log for rank 0
                dist.gather_object(outputs_list, None, dst=0)
                return

        # ---- reorder the list itself ----
        # get a single scalar lin_idx for each batch element
        # (use mean or first value if it's a vector)
        idxs = []
        for i, out in enumerate(outputs_list):
            lin_idx = out["lin_idx"]
            if any([lin_idx == li for li, _ in idxs]):
                # already have this index (DDP added extra batches)
                continue
            if lin_idx.ndim > 0:
                lin_idx = lin_idx.view(-1)[0]  # take first sample in that batch
            idxs.append((int(lin_idx), i))
        # drop additional batches added by DDP
        idxs = [idx for idx in idxs if idx[0] < self.image_shape[0] * self.image_shape[1]]
        # sort by the scalar lin_idx
        outputs_list = [outputs_list[i] for _, i in sorted(idxs, key=lambda x: x[0])]

        outputs = {}
        for k in outputs_list[0].keys():
            outputs[k] = torch.cat([o[k] for o in outputs_list], dim=0)

        I_diff, Q_diff, U_diff, V_diff = torch.nanmean(outputs['diff'], dim=(0, 2))

        parameters = {}
        for k in ['b_field', 'inc', 'azi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl',
                  'v_rtp', 'b_rtp', 'v_img', 'b_img', 'v_obs_correction',
                  'induction', 'dB_dt', 'curl_VxB', 'divergence', 'force_free', 'dB_dr', 'mu', 'v_obs_los',
                  'c_b0', 'c_b1', 'c_vmac', 'c_damping', 'c_kl', 'c_vdop']:
            if k not in outputs:
                continue
            v = outputs[k].reshape(*self.image_shape[:2], -1).numpy().squeeze()
            parameters[k] = v
        self.plot_parameter_overview(parameters)
        self.plot_B_rtp_log(parameters)
        self.plot_B_rtp(parameters)
        self.plot_v_rtp(parameters)
        # self.plot_physics_losses(parameters)
        self.plot_limb_correction(parameters)
        plt.close('all')

        if 'artifact_correction_params' in outputs:
            artifact_params = outputs['artifact_correction_params'].reshape(*self.image_shape[:2], -1).cpu().numpy()
            self.plot_artifact_corrections(artifact_params)

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)

        self.plot_stokes(stokes_pred, stokes_true)

        warnings.simplefilter("ignore")  # ignore warnings for distributed logging
        self.log_dict({
            # log total stokes loss
            "valid.diff": torch.nanmean(outputs['diff']),
            # log stokes differences
            'valid.I': I_diff, 'valid.Q': Q_diff, 'valid.U': U_diff, 'valid.V': V_diff,
            # log physics losses
            # 'valid.induction': torch.nanmean(outputs['induction']),
            # 'valid.divergence': torch.nanmean(outputs['divergence']),
            # 'valid.force_free': torch.nanmean(outputs['force_free']),
        })

    def plot_physics_losses(self, parameters):
        induction = parameters['induction']
        dB_dt = parameters['dB_dt']
        curl_VxB = parameters['curl_VxB']
        divergence = parameters['divergence']
        dB_dr = parameters['dB_dr']

        fig, axs = plt.subplots(3, 3, figsize=(10, 6), dpi=100)
        ax = axs[0, 0]
        im = ax.imshow(induction, origin='lower', norm='log')
        ax.set_title("Induction")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(dB_dt, origin='lower', norm='log')
        ax.set_title("dB/dt")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(curl_VxB, origin='lower', norm='log')
        ax.set_title("Curl VxB")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(divergence, origin='lower', norm='log')
        ax.set_title("Divergence")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(dB_dr, origin='lower', norm='log')
        ax.set_title("dB/dr")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        ax.set_axis_off()

        ax = axs[2, 0]
        im = ax.imshow(parameters['force_free'], origin='lower', norm='log')
        ax.set_title("force_free")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 1]
        ax.set_axis_off()

        ax = axs[2, 2]
        ax.set_axis_off()

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, 'Physics Losses')
        plt.close(fig)

    def plot_limb_correction(self, outputs):
        if 'c_b0' not in outputs or 'c_b1' not in outputs or 'c_vmac' not in outputs:
            return
        c_b0 = outputs['c_b0']
        c_b1 = outputs['c_b1']
        # c_vmac = outputs['c_vmac']
        mu = outputs['mu']
        # c_damping = outputs['c_damping']
        # c_kl = outputs['c_kl']
        c_vdop = outputs['c_vdop']

        fig, axs = plt.subplots(2, 4, figsize=(10, 5), dpi=100)

        ax = axs[0, 0]
        im = ax.imshow(mu, origin='lower', cmap='cividis', vmin=0, vmax=1)
        ax.set_title("mu")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        ax.set_axis_off()
        # im = ax.imshow(c_damping, origin='lower', cmap='magma')
        # ax.set_title(r"$c_\text{damping}$")
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        ax.set_axis_off()
        # im = ax.imshow(c_kl, origin='lower', cmap='magma')
        # ax.set_title(r"$c_\text{kl}$")
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(im, cax=cax)

        axs[0, 3].set_axis_off()

        ax = axs[1, 0]
        im = ax.imshow(c_b0, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{b0}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(c_b1, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{b1}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        ax.set_axis_off()
        # im = ax.imshow(c_vmac, origin='lower', cmap='magma')
        # ax.set_title(r"$c_\text{vmac}$")
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(im, cax=cax)

        ax = axs[1, 3]
        v_min_max = np.nanmax(np.abs(c_vdop))
        im = ax.imshow(c_vdop, origin='lower', cmap='RdBu_r', vmin=-v_min_max, vmax=v_min_max)
        ax.set_title(r"$c_\text{vdop}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        fig.suptitle("Limb Correction Factors")
        fig.tight_layout()
        log_wandb_image(fig, "Limb Correction Factors")
        plt.close(fig)

    def plot_profile(self, stokes_pred, stokes_true):
        y_range = stokes_true.shape[0]
        x_range = stokes_true.shape[1]
        pos = np.stack(np.meshgrid(np.linspace(x_range * 0.1, x_range * 0.9, 3, dtype=int),
                                   np.linspace(y_range * 0.1, y_range * 0.9, 3, dtype=int)), -1)
        pos = pos.reshape(-1, 2)
        for x, y in pos:
            fig, axs = plt.subplots(4, 1, figsize=(8, 8))
            for i, label in enumerate(['I', 'Q', 'U', 'V']):
                axs[i].plot(stokes_true[y, x, i], label=f'true - {label}')
                axs[i].plot(stokes_pred[y, x, i], label=f'pred - {label}')
                if i == 0:
                    continue
                # v_min_max = np.abs(stokes_true[20, 20, i]).max()
                # v_min_max = max(1e-4, v_min_max)
                # axs[i].set_ylim([-v_min_max, v_min_max])

            [ax.legend(loc='upper right') for ax in axs]
            # log figure
            # fig.tight_layout()
            fig.subplots_adjust(wspace=0.25, hspace=0.25)
            log_wandb_image(fig, f"Profile x:{x:02d} y:{y:02d}")
            plt.close(fig)

    def plot_stokes(self, stokes_pred, stokes_true):
        # plot comparison of integrated stokes vectors
        integerated_stokes_true = np.abs(stokes_true).sum(-1)
        integerated_stokes_pred = np.abs(stokes_pred).sum(-1)
        fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
        for i, label in enumerate(['I', 'Q', 'U', 'V']):
            v_min = np.nanmin(integerated_stokes_true[:, :, i])
            v_max = np.nanmax(integerated_stokes_true[:, :, i])
            norm = ImageNormalize(vmin=v_min, vmax=v_max)
            im = ax[0, i].imshow(integerated_stokes_true[:, :, i], norm=norm, origin='lower')
            ax[0, i].set_title(f"true - {label}")
            divider = make_axes_locatable(ax[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
            im = ax[1, i].imshow(integerated_stokes_pred[:, :, i], norm=norm, origin='lower')
            ax[1, i].set_title(f"pred - {label}")
            divider = make_axes_locatable(ax[1, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        # fig.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, "Integrated Stokes vector - Comparison")
        plt.close(fig)

    def plot_parameter_overview(self, parameters):
        b = parameters['b_field']
        inc = parameters['inc']
        azi = parameters['azi']
        # reproject vectors (inc flip with negative B)
        b_xyz = to_cartesian(b, inc, azi)
        b, inc, azi = to_spherical(b_xyz)
        inc = inc % np.pi
        azi = azi % (2 * np.pi)

        fig, axs = plt.subplots(2, 5, figsize=(16, 4), dpi=100)
        ax = axs[0, 0]
        im = ax.imshow(b, cmap='viridis', vmin=.1, origin='lower', norm='log')
        ax.set_title("B")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 1]
        im = ax.imshow(inc % np.pi, cmap='seismic', vmin=0, vmax=np.pi, origin='lower')
        ax.set_title("Inclination")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 2]
        im = ax.imshow(azi % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower')
        ax.set_title("Azimuth")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 3]
        im = ax.imshow(parameters['b0'], origin='lower')
        ax.set_title("B0")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 4]
        im = ax.imshow(parameters['b1'], origin='lower')
        ax.set_title("B1")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 0]
        im = ax.imshow(parameters['vmac'], origin='lower')
        ax.set_title("Vmac")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 1]
        im = ax.imshow(parameters['damping'], origin='lower')
        ax.set_title("Damping")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 2]
        ax.set_axis_off()
        ax = axs[1, 3]
        vdop_max = np.nanmax(np.abs(parameters['vdop']))
        im = ax.imshow(parameters['vdop'], cmap='seismic_r', vmin=-vdop_max, vmax=vdop_max, origin='lower')
        ax.set_title("Vdop")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 4]
        im = ax.imshow(parameters['kl'], origin='lower')
        ax.set_title("Kl")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, "Parameter Overview")
        plt.close(fig)

    def plot_B_rtp_log(self, parameters):
        b_rtp = parameters['b_rtp']
        b_img = parameters['b_img']

        norm = SymLogNorm(linthresh=100, vmin=-5000, vmax=5000)

        fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=100)

        ax = axs[0, 0]
        im = ax.imshow(b_rtp[..., 0], norm=norm, origin='lower', cmap='seismic')
        ax.set_title("$B_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(b_rtp[..., 1], cmap='seismic', norm=norm, origin='lower')
        ax.set_title("$B_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(b_rtp[..., 2], cmap='seismic', norm=norm, origin='lower')
        ax.set_title("$B_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(b_img[..., 0], norm=norm, origin='lower', cmap='seismic')
        ax.set_title(r"$B_\text{xi}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(b_img[..., 1], norm=norm, origin='lower', cmap='seismic')
        ax.set_title(r"$B_\text{eta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        im = ax.imshow(b_img[..., 2], norm=norm, origin='lower', cmap='seismic')
        ax.set_title(r"$B_\text{zeta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, 'B')
        plt.close(fig)

    def plot_B_rtp(self, parameters):
        b_rtp = parameters['b_rtp']

        norm = Normalize(vmin=-500, vmax=500)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=100)

        ax = axs[0]
        im = ax.imshow(b_rtp[..., 0], norm=norm, origin='lower', cmap='gray')
        ax.set_title("$B_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1]
        im = ax.imshow(b_rtp[..., 1], cmap='gray', norm=norm, origin='lower')
        ax.set_title("$B_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2]
        im = ax.imshow(b_rtp[..., 2], cmap='gray', norm=norm, origin='lower')
        ax.set_title("$B_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, 'B_rtp')
        plt.close(fig)

    def plot_v_rtp(self, parameters):
        v_rtp = parameters['v_rtp']
        v_img = parameters['v_img']
        v_obs_correction = parameters['v_obs_correction']

        v_min_max = 5000  # m/s
        norm = Normalize(vmin=-v_min_max, vmax=v_min_max)

        fig, axs = plt.subplots(3, 3, figsize=(10, 7), dpi=100)
        ax = axs[0, 0]
        im = ax.imshow(v_rtp[..., 0], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title("$v_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(v_rtp[..., 1], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title("$v_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(v_rtp[..., 2], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title("$v_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(v_img[..., 0], cmap='RdBu', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{xi}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(v_img[..., 1], cmap='RdBu', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{eta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        im = ax.imshow(v_img[..., 2], cmap='RdBu', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{zeta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 0]
        im = ax.imshow(parameters['v_obs_los'], cmap='RdBu_r', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{OBS LOS}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 1]
        im = ax.imshow(v_obs_correction, cmap='RdBu_r', origin='lower', norm=norm)
        ax.set_title(r'$v_\text{OBS correction}$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 2]
        ax.set_axis_off()

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, "v")
        plt.close(fig)

    def plot_artifact_corrections(self, artifact_params):
        fig, axs = plt.subplots(2, 4, figsize=(16, 4), dpi=100)

        factor_idx = [0, 2, 4, 6]  # I, Q, U, V gains (Q/U/V share but repeated)
        leak_idx = [1, 3, 5, 7]  # I bias, and I->Q/U/V leakage coeffs

        for i, label in enumerate(['I', 'Q', 'U', 'V']):
            ax = axs[0, i]
            im = ax.imshow(artifact_params[..., factor_idx[i]], origin='lower', cmap='magma')
            ax.set_title(f"Artifact Corr. Factor - {label}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

            ax = axs[1, i]
            vmax = np.nanmax(np.abs(artifact_params[..., leak_idx[i]]))
            im = ax.imshow(artifact_params[..., leak_idx[i]], origin='lower', cmap='BrBG', vmin=-vmax, vmax=vmax)
            ax.set_title(f"Artifact Corr. Leak/Bias - {label}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, "Artifact Correction Parameters")
        plt.close(fig)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint['state_dict']
        # keep new lambdas
        for k, v in self.lambdas.items():
            if f"lambdas.{k}" not in state_dict:
                print(f'Add lambda {k}: {v.data}')
                state_dict[f'lambdas.{k}'] = v
                continue
            checkpoint_v = state_dict[f"lambdas.{k}"]
            if k in self.scheduled_lambda_config or checkpoint_v == v:  # skip scheduled lambdas or same values
                continue
            print(f'Update lambda {k}: {checkpoint_v} --> {v.data}')
            state_dict[f'lambdas.{k}'] = v
        # remove old lambdas
        remove_keys = []
        for k, v in state_dict.items():
            if 'lambdas' in k and k.split('.')[1] not in self.lambdas.keys():
                print(f'Remove lambda: {k}')
                remove_keys.append(k)
        [state_dict.pop(k) for k in remove_keys]

        self.load_state_dict(state_dict, strict=False)
        self.validation_outputs = {}  # reset validation outputs
