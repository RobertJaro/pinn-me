from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.visualization import ImageNormalize
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from pme.data.differential_rotation import carrington_rotation_velocity
from pme.data.util import cartesian_to_spherical, spherical_to_cartesian
from pme.evaluation.loader import to_spherical, to_cartesian
from pme.model import MESphericalModel, jacobian, INormalizationModule, VelocityCorrectionModel, LimbCorrectionModel
from pme.train.me_atmosphere import HMIMEAtmosphere, PHIMEAtmosphere
from pme.train.util import acos_safe, atan2_safe, log_wandb_image


class MESphericalModule(LightningModule):

    def __init__(self, image_shape, lambda_config,
                 gauss_per_dB, Rs_per_ds, seconds_per_dt,
                 instrument_config,
                 lr_params=None, model_config=None, normalization_config=None,
                 lambda_stokes=None,
                 lambda_induction=0.0, lambda_divergence=0.0, lambda_force_free=0.0,
                 lambda_static=0.0,
                 **kwargs):
        super().__init__()
        lr_params = lr_params if lr_params is not None else {"start": 5e-4, "end": 5e-5, "iterations": 1e5}
        lambda_stokes = lambda_stokes if lambda_stokes is not None else [1, 1, 1, 1]

        self.image_shape = image_shape

        # init model
        model_config = model_config if model_config is not None else {}
        self.parameter_model = MESphericalModel(**model_config)

        # init instrument models
        forward_models = {}
        velocity_correction_models = {}
        limb_correction_models = {}
        for instrument in instrument_config:
            instrument_id = instrument.pop('instrument_id')
            instrument_type = instrument.pop('type')
            lambda0 = lambda_config[instrument_id]['lambda0']
            velocity_correction = instrument.pop('velocity_correction', False)
            limb_correction = instrument.pop('limb_correction', True)

            if instrument_type == 'hmi':
                forward_models[instrument_id] = HMIMEAtmosphere(lambda0=lambda0, **instrument)
            elif instrument_type == 'phi':
                forward_models[instrument_id] = PHIMEAtmosphere(lambda0=lambda0, **instrument)
            else:
                raise ValueError(f"Unknown instrument type: {instrument_type}")

            if velocity_correction:
                velocity_correction_models[instrument_id] = VelocityCorrectionModel()
            if limb_correction:
                limb_correction_models[instrument_id] = LimbCorrectionModel()

        self.forward_models = nn.ModuleDict(forward_models)
        self.velocity_correction_models = nn.ModuleDict(velocity_correction_models)
        self.limb_correction_models = nn.ModuleDict(limb_correction_models)

        self.lr_params = lr_params

        self.validation_outputs = {}
        normalization_config = normalization_config if normalization_config is not None else {}
        self.normalization = INormalizationModule(**normalization_config)
        self.loss_function = nn.MSELoss(reduction='none')
        self.lambda_stokes = nn.Parameter(torch.tensor(lambda_stokes, dtype=torch.float32), requires_grad=False)
        #
        scheduled_lambda_config = {}
        lambdas = {}
        for k, v in [('induction', lambda_induction), ('divergence', lambda_divergence),
                     ('force_free', lambda_force_free), ('static', lambda_static)]:
            if isinstance(v, dict):
                gamma = (v['end'] / v['start']) ** (1 / v['iterations'])
                scheduled_lambda_config[k] = {'end': v['end'], 'gamma': gamma}
                lambdas[k] = nn.Parameter(torch.tensor(v['start'], dtype=torch.float32), requires_grad=False)
            else:
                lambdas[k] = nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
        self.scheduled_lambda_config = scheduled_lambda_config
        self.lambdas = nn.ParameterDict(lambdas)
        #
        self.gauss_per_dB = gauss_per_dB
        self.Rs_per_ds = Rs_per_ds
        self.meters_per_ds = Rs_per_ds * (1 * u.Rsun).to_value(u.m)
        self.seconds_per_dt = seconds_per_dt

    def configure_optimizers(self):
        parameters = (list(self.parameter_model.parameters()) +
                      list(self.forward_models.parameters()) +
                      list(self.velocity_correction_models.parameters()) +
                      list(self.limb_correction_models.parameters()))
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
        optimizer = torch.optim.Adam(parameters, lr=lr_start)
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
        # time_shift = torch.randn_like(coords[..., 0:1]) * 0.01  # small random shift
        # coords = torch.cat([coords[..., 0:1] + time_shift, coords[..., 1:]], dim=-1)

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
            ds_lambda_grid = batch[instrument_id]['lambda_grid']
            ds_stokes_true = batch[instrument_id]['stokes']

            ds_forward_params = {k: v[current_idx:current_idx + n_samples] for k, v in forward_params.items()}
            # apply velocity correction if available
            if instrument_id in self.velocity_correction_models:
                time_coords = batch[instrument_id]['coords'][..., 0:1]
                v_obs_correction = self.velocity_correction_models[instrument_id](time_coords)
                ds_forward_params['vdop'] += v_obs_correction

            if instrument_id in self.limb_correction_models:
                ds_forward_params, _ = self.correct_limb_effects(ds_forward_params, ds_mu, instrument_id)

            I, Q, U, V = self.forward_models[instrument_id](**ds_forward_params, lambda_grid=ds_lambda_grid, mu=ds_mu)
            ds_stokes_pred = torch.stack([I, Q, U, V], dim=-2)

            stokes_true_normalized.append(self.normalization(ds_stokes_true / ds_mu[..., None]))
            stokes_pred_normalized.append(self.normalization(ds_stokes_pred / ds_mu[..., None]))
            current_idx += n_samples

        #################################################
        # compute stokes loss
        stokes_pred_normalized = torch.cat(stokes_pred_normalized, dim=0)
        stokes_true_normalized = torch.cat(stokes_true_normalized, dim=0)
        stokes_loss = self.loss_function(stokes_pred_normalized, stokes_true_normalized)

        # sum over wavelength axis
        stokes_loss = stokes_loss.sum(-1)

        # logging losses
        I_loss, Q_loss, U_loss, V_loss = stokes_loss.mean(dim=0)

        # weighted loss - apply lambda weights for each stokes parameter
        stokes_loss = (stokes_loss * self.lambda_stokes[None, :]).sum(-1)

        #################################################
        # compute physics losses
        if all((v == 0).item() for k, v in self.lambdas.items() if k in ['induction', 'divergence', 'force_free']):
            # skip physics losses if not required
            physics_losses = {'induction': torch.zeros_like(stokes_loss[..., 0]),
                              'divergence': torch.zeros_like(stokes_loss[..., 0]),
                              'force_free': torch.zeros_like(stokes_loss[..., 0]),
                              }
        else:
            spherical_coords = cartesian_to_spherical(coords[..., 1:], torch)
            # get coordinate range for sampling points
            min_t, max_t = torch.min(coords[..., 0]), torch.max(coords[..., 0])
            min_r, max_r = 1.0, 1.01  # define a shell of 0.01 Rs around the sun
            min_th, max_th = torch.min(spherical_coords[..., 1]), torch.max(spherical_coords[..., 1])
            min_phi, max_phi = torch.min(spherical_coords[..., 2]), torch.max(spherical_coords[..., 2])

            # create random sampling points
            random_coords = torch.rand((4096, 4), dtype=torch.float32, device=coords.device, requires_grad=True)
            # scale random coordinates in spherical coordinates
            random_coords_r = (max_r - min_r) * random_coords[..., 1] + min_r
            random_coords_th = (max_th - min_th) * random_coords[..., 2] + min_th
            random_coords_phi = (max_phi - min_phi) * random_coords[..., 3] + min_phi
            random_coords_spherical = torch.stack([random_coords_r, random_coords_th, random_coords_phi], dim=-1)
            random_coords_cartesian = spherical_to_cartesian(random_coords_spherical, torch)

            # scale random coordinates time
            random_coords_t = (max_t - min_t) * random_coords[..., 0] + min_t

            # combine time and cartesian coordinates
            random_coords = torch.cat([random_coords_t[..., None], random_coords_cartesian], dim=-1)

            # forward pass of random coordinates
            physics_out = self.parameter_model(random_coords)
            b = torch.cat([physics_out['b_x'], physics_out['b_y'], physics_out['b_z']], dim=-1)
            v = torch.cat([physics_out['v_x'], physics_out['v_y'], physics_out['v_z']], dim=-1)

            # compute physics losses
            physics_losses = self.compute_physics_losses(b, v, random_coords)

        static_loss = transformed_output['v_rtp'][..., 0:1].mean().pow(2) # average radial velocity should be approx. zero
        #################################################
        # compute total loss
        total_loss = (stokes_loss.mean() +
                      self.lambdas['induction'] * physics_losses['induction'].mean() +
                      self.lambdas['divergence'] * physics_losses['divergence'].mean() +
                      self.lambdas['force_free'] * physics_losses['force_free'].mean() +
                      self.lambdas['static'] * static_loss)

        assert not torch.isnan(total_loss), f"Encountered invalid value. Loss is NaN"

        return {"loss": total_loss,
                "I_loss": I_loss, "Q_loss": Q_loss,
                "U_loss": U_loss, "V_loss": V_loss,
                "stokes_loss": stokes_loss,
                "induction_loss": physics_losses['induction'].mean(),
                "divergence_loss": physics_losses['divergence'].mean(),
                "force_free_loss": physics_losses['force_free'].mean(),
                "static_loss": static_loss,
                }

    def compute_physics_losses(self, b, v, coords):
        # compute B derivatives
        jac_matrix = jacobian(b, coords)
        dBx_dt = jac_matrix[..., 0, 0]
        dBx_dx = jac_matrix[..., 0, 1]
        dBx_dy = jac_matrix[..., 0, 2]
        dBx_dz = jac_matrix[..., 0, 3]
        dBy_dt = jac_matrix[..., 1, 0]
        dBy_dx = jac_matrix[..., 1, 1]
        dBy_dy = jac_matrix[..., 1, 2]
        dBy_dz = jac_matrix[..., 1, 3]
        dBz_dt = jac_matrix[..., 2, 0]
        dBz_dx = jac_matrix[..., 2, 1]
        dBz_dy = jac_matrix[..., 2, 2]
        dBz_dz = jac_matrix[..., 2, 3]
        # compute V derivatives
        v_jac = jacobian(v, coords)
        dVx_dt = v_jac[:, 0, 0]
        dVx_dx = v_jac[:, 0, 1]
        dVx_dy = v_jac[:, 0, 2]
        dVx_dz = v_jac[:, 0, 3]
        dVy_dt = v_jac[:, 1, 0]
        dVy_dx = v_jac[:, 1, 1]
        dVy_dy = v_jac[:, 1, 2]
        dVy_dz = v_jac[:, 1, 3]
        dVz_dt = v_jac[:, 2, 0]
        dVz_dx = v_jac[:, 2, 1]
        dVz_dy = v_jac[:, 2, 2]
        dVz_dz = v_jac[:, 2, 3]

        # compute induction loss
        div_V = (dVx_dx + dVy_dy + dVz_dz)[..., None]
        div_B = (dBx_dx + dBy_dy + dBz_dz)[..., None]
        dB_dt = torch.stack([dBx_dt, dBy_dt, dBz_dt], -1)
        B_nabla_V = torch.stack([b[:, 0] * dVx_dx + b[:, 1] * dVx_dy + b[:, 2] * dVx_dz,
                                 b[:, 0] * dVy_dx + b[:, 1] * dVy_dy + b[:, 2] * dVy_dz,
                                 b[:, 0] * dVz_dx + b[:, 1] * dVz_dy + b[:, 2] * dVz_dz, ], -1)
        V_nabla_B = torch.stack([v[:, 0] * dBx_dx + v[:, 1] * dBx_dy + v[:, 2] * dBx_dz,
                                 v[:, 0] * dBy_dx + v[:, 1] * dBy_dy + v[:, 2] * dBy_dz,
                                 v[:, 0] * dBz_dx + v[:, 1] * dBz_dy + v[:, 2] * dBz_dz, ], -1)
        induction_rhs = B_nabla_V - V_nabla_B - b * div_V + v * div_B

        induction_equation = dB_dt - induction_rhs
        induction_loss = induction_equation.pow(2).sum(-1)

        # compute divergence loss
        divergence_loss = div_B.pow(2).sum(-1)

        # compute force_free
        rot_x = dBz_dy - dBy_dz
        rot_y = dBx_dz - dBz_dx
        rot_z = dBy_dx - dBx_dy
        j = torch.stack([rot_x, rot_y, rot_z], -1)
        # compute force-free condition
        force_free_loss = torch.cross(j, b, dim=-1).pow(2).sum(-1)

        # x = r * cos(t) * cos(p)
        # y = r * cos(t) * sin(p)
        # z = r * sin(t)
        spherical_coords = cartesian_to_spherical(coords[..., 1:], torch)
        dx_dr = torch.sin(spherical_coords[..., 1]) * torch.cos(spherical_coords[..., 2])
        dy_dr = torch.sin(spherical_coords[..., 1]) * torch.sin(spherical_coords[..., 2])
        dz_dr = torch.cos(spherical_coords[..., 1])
        dBx_dr = dBx_dx * dx_dr + dBx_dy * dy_dr + dBx_dz * dz_dr
        dBy_dr = dBy_dx * dx_dr + dBy_dy * dy_dr + dBy_dz * dz_dr
        dBz_dr = dBz_dx * dx_dr + dBz_dy * dy_dr + dBz_dz * dz_dr
        dB_dr = torch.stack([dBx_dr, dBy_dr, dBz_dr], -1)

        return {'divergence': divergence_loss,
                'force_free': force_free_loss,
                'induction': induction_loss,
                'dB_dt': dB_dt.pow(2).sum(-1).pow(0.5),
                'curl_VxB': induction_rhs.pow(2).sum(-1).pow(0.5),
                'dB_dr': dB_dr.pow(2).sum(-1).pow(0.5),
                }

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

        # xi, eta, zeta
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
        inc = torch.pi - inc
        azi = atan2_safe(-b_img[..., 0:1], b_img[..., 1:2])
        azi += torch.pi / 2  # azimuth is flipped in HMI

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

    def correct_limb_effects(self, parameters, mu, instrument_id):
        # apply limb correction
        limb_correction = self.limb_correction_models[instrument_id](mu)
        c_b0 = limb_correction['c_b0']
        c_b1 = limb_correction['c_b1']
        c_vmac = limb_correction['c_vmac']
        c_damping = limb_correction['c_damping']
        c_kl = limb_correction['c_kl']
        c_vdop = limb_correction['c_vdop']

        # apply limb correction to parameters
        b0 = parameters['b0'] * c_b0
        b1 = parameters['b1'] * c_b1
        vmac = parameters['vmac'] * c_vmac
        damping = parameters['damping'] * c_damping
        kl = parameters['kl'] * c_kl
        vdop = parameters['vdop'] + c_vdop

        corrected_output = {k: v for k, v in parameters.items() if
                            k not in ['b0', 'b1', 'vmac', 'damping', 'kl', 'vdop']}
        corrected_output['b0'] = b0
        corrected_output['b1'] = b1
        corrected_output['vmac'] = vmac
        corrected_output['damping'] = damping
        corrected_output['kl'] = kl
        corrected_output['vdop'] = vdop

        return corrected_output, limb_correction

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # update learning rate
        scheduler = self.lr_schedulers()
        if scheduler.get_last_lr()[0] > self.lr_params['end']:
            scheduler.step()
        self.log('Learning Rate', scheduler.get_last_lr()[0])

        self.parameter_model.step(self.global_step)

        for k in self.scheduled_lambda_config.keys():
            value = self.lambdas[k]
            if value > self.scheduled_lambda_config[k]['end']:
                # update lambda value
                gamma = self.scheduled_lambda_config[k]['gamma']
                new_value = value * gamma
                self.lambdas[k].copy_(new_value)
            wandb.log({f"lambda.{k}": self.lambdas[k].item()}, commit=False)

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
        lambda_grid = batch['lambda_grid']
        instrument_id = batch['instrument_id']

        # forward step
        coords.requires_grad = True
        output = self.parameter_model(coords)

        transformed_output = self.transform_parameters(output, coords, cartesian_to_spherical_transform,
                                                       rtp_to_img_transform)
        forward_params = self.scale_parameters(output, transformed_output, v_obs_los)

        limb_correction = None
        if instrument_id in self.limb_correction_models:
            forward_params, limb_correction = self.correct_limb_effects(forward_params, mu, instrument_id)

        v_obs_correction = torch.zeros_like(v_obs_los)
        if instrument_id in self.velocity_correction_models:
            time_coords = coords[..., 0:1]
            v_obs_correction = self.velocity_correction_models[instrument_id](time_coords)
            forward_params['vdop'] += v_obs_correction

        I, Q, U, V = self.forward_models[instrument_id](**forward_params, mu=mu, lambda_grid=lambda_grid)
        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        stokes_true_normalized = self.normalization(stokes_true / mu[..., None])
        stokes_pred_normalized = self.normalization(stokes_pred / mu[..., None])

        diff = torch.abs(stokes_true_normalized - stokes_pred_normalized)

        b = transformed_output['b_xyz']
        v = transformed_output['v_xyz']
        physics_losses = self.compute_physics_losses(b, v, coords)

        res = {'diff': diff.detach(),
               'stokes_true': stokes_true_normalized.detach(), 'stokes_pred': stokes_pred_normalized.detach(),
               **forward_params,
               'b_rtp': transformed_output['b_rtp'] * self.gauss_per_dB,
               'v_rtp': transformed_output['v_rtp'] * self.meters_per_ds / self.seconds_per_dt,
               'b_img': transformed_output['b_img'] * self.gauss_per_dB,
               'v_img': transformed_output['v_img'] * self.meters_per_ds / self.seconds_per_dt,
               'induction': physics_losses['induction'].detach(),
               'dB_dt': physics_losses['dB_dt'].detach(),
               'curl_VxB': physics_losses['curl_VxB'].detach(),
               'divergence': physics_losses['divergence'].detach(),
               'force_free': physics_losses['force_free'].detach(),
               'dB_dr': physics_losses['dB_dr'].detach(),
               'mu': mu.detach(), 'v_obs_los': v_obs_los.detach(),
               'v_obs_correction': v_obs_correction.detach(),
               }
        if limb_correction is not None:
            res['c_b0'] = limb_correction['c_b0'].detach()
            res['c_b1'] = limb_correction['c_b1'].detach()
            res['c_vmac'] = limb_correction['c_vmac'].detach()
            res['c_damping'] = limb_correction['c_damping'].detach()
            res['c_kl'] = limb_correction['c_kl'].detach()
            res['c_vdop'] = limb_correction['c_vdop'].detach()
        return res

    def validation_epoch_end(self, outputs_list):
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps

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
            field = outputs[k].reshape(*self.image_shape[:2], -1).cpu().numpy().squeeze()
            parameters[k] = field

        self.plot_parameter_overview(parameters)
        self.plot_B_rtp(parameters)
        self.plot_B_rtp_scaled(parameters)
        self.plot_v_rtp(parameters)
        self.plot_physics_losses(parameters)
        self.plot_limb_correction(parameters)

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)

        self.plot_stokes(stokes_pred, stokes_true)

        self.log_dict({
            # log total stokes loss
            "valid.diff": torch.nanmean(outputs['diff']),
            # log stokes differences
            'valid.I_diff': I_diff, 'valid.Q_diff': Q_diff, 'valid.U_diff': U_diff, 'valid.V_diff': V_diff,
            # log physics losses
            'valid.induction': torch.nanmean(outputs['induction']),
            'valid.divergence': torch.nanmean(outputs['divergence']),
            'valid.force_free': torch.nanmean(outputs['force_free']),
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
        plt.close('all')

    def plot_limb_correction(self, outputs):
        if 'c_b0' not in outputs or 'c_b1' not in outputs or 'c_vmac' not in outputs:
            return
        c_b0 = outputs['c_b0']
        c_b1 = outputs['c_b1']
        c_vmac = outputs['c_vmac']
        mu = outputs['mu']
        c_damping = outputs['c_damping']
        c_kl = outputs['c_kl']
        c_vdop = outputs['c_vdop']

        fig, axs = plt.subplots(2, 4, figsize=(10, 5), dpi=100)

        ax = axs[0, 0]
        im = ax.imshow(mu, origin='lower', cmap='cividis', vmin=0, vmax=1)
        ax.set_title("mu")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(c_damping, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{damping}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(c_kl, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{kl}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

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
        im = ax.imshow(c_vmac, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{vmac}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

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
        plt.close('all')

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
            plt.close('all')

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
        plt.close('all')

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
        im = ax.imshow(parameters['mu'], origin='lower', vmin=0, vmax=1, cmap='cividis')
        ax.set_title("Mu")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
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
        plt.close('all')

    def plot_B_rtp(self, parameters):
        b_rtp = parameters['b_rtp']
        b_img = parameters['b_img']

        norm = SymLogNorm(linthresh=10, vmin=-5000, vmax=5000)

        fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=100)

        ax = axs[0, 0]
        im = ax.imshow(b_rtp[..., 0], norm=norm, origin='lower', cmap='PuOr')
        ax.set_title("$B_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(b_rtp[..., 1], cmap='PuOr', norm=norm, origin='lower')
        ax.set_title("$B_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(b_rtp[..., 2], cmap='PuOr', norm=norm, origin='lower')
        ax.set_title("$B_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(b_img[..., 0], norm=norm, origin='lower', cmap='PuOr')
        ax.set_title(r"$B_\text{xi}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(b_img[..., 1], norm=norm, origin='lower', cmap='PuOr')
        ax.set_title(r"$B_\text{eta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        im = ax.imshow(b_img[..., 2], norm=norm, origin='lower', cmap='PuOr')
        ax.set_title(r"$B_\text{zeta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        log_wandb_image(fig, 'B')
        plt.close('all')

    def plot_B_rtp_scaled(self, parameters):
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
        plt.close('all')

    def plot_v_rtp(self, parameters):
        v_rtp = parameters['v_rtp']
        v_img = parameters['v_img']
        v_obs_correction = parameters['v_obs_correction']

        v_min_max = 2000  # m/s
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
        plt.close('all')

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # replace checkpoint lambda with the current lambda
        checkpoint['state_dict']['lambda_stokes'] = self.lambda_stokes.detach().cpu()

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
