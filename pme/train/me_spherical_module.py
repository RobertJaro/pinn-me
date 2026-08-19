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
from torch.optim.lr_scheduler import LambdaLR

from pme.evaluation.loader import to_spherical, to_cartesian
from pme.model import (
    LimbCorrectionModel,
    MESphericalModel,
    MESphericalSirenModel,
    NormalizationModule,
    VelocityCorrectionModel,
)
from pme.train.artifact_correction import ArtifactCorrectionModule
from pme.train.me_atmosphere import HMIMEAtmosphere, PHIMEAtmosphere
from pme.train.physics import PhysicsWeightSchedule
from pme.train.physics_regularization import (
    PhysicsRegularization,
    split_stokes_and_physics_config,
)
from pme.train.soap import SOAP
from pme.train.spherical_synthesis import (
    correct_spherical_limb_effects,
    field_free_forward_parameters,
    mix_magnetic_filling_factor,
    scale_spherical_forward_parameters,
    transform_spherical_parameters,
)
from pme.train.stokes_loss import StokesLossModule
from pme.train.util import log_wandb_image, random_time_shift_coords


class MESphericalModule(LightningModule):

    def __init__(self, image_shape, wavelength_config,
                 gauss_per_dB, Rs_per_ds, seconds_per_dt,
                 instrument_config,
                 lr_params=None, model_config=None, normalization_config=None,
                 stokes_loss_config=None,
                 physics_config=None, physics_domain=None,
                 time_shift_config=None,
                 lambda_config=None):
        super().__init__()
        lr_params = lr_params if lr_params is not None else {"start": 1e-3, "end": 1e-4, "iterations": 1e5}

        self.image_shape = image_shape

        # init model
        model_config = copy.deepcopy(model_config) if model_config is not None else {}
        model_type = model_config.pop('type', None)
        # Legacy configurations may still contain the removed exponential
        # magnitude-scale switch. B/A and V now use direct linear outputs.
        model_config.pop('scale', None)
        if model_type == 'mlp':
            self.parameter_model = MESphericalModel(**model_config)
        elif model_type == 'siren':
            self.parameter_model = MESphericalSirenModel(**model_config)
        else:
            raise ValueError("The spherical model configuration must specify type: mlp or siren.")

        # init instrument models
        forward_models = {}
        velocity_correction_models = {}
        limb_correction_models = {}
        artifact_correction_models = {}
        hmi_instrument_ids = []
        for instrument in copy.deepcopy(instrument_config):
            instrument_id = instrument.pop('instrument_id')
            instrument_type = instrument.pop('type')
            wavelength_center = wavelength_config[instrument_id]['wavelength_center']
            velocity_correction = instrument.pop('velocity_correction', False)
            limb_correction = instrument.pop('limb_correction', False)
            artifact_correction = instrument.pop('artifact_correction', False)

            if instrument_type == 'hmi':
                forward_models[instrument_id] = HMIMEAtmosphere(wavelength_center=wavelength_center, **instrument)
                hmi_instrument_ids.append(instrument_id)
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
        self.hmi_instrument_ids = tuple(hmi_instrument_ids)

        self.lr_params = lr_params

        self.validation_outputs = {}
        normalization_config = normalization_config if normalization_config is not None else {}
        self.normalization = NormalizationModule(**normalization_config)
        stokes_loss_config = copy.deepcopy(stokes_loss_config) if stokes_loss_config is not None else {}
        self.stokes_loss = StokesLossModule(**stokes_loss_config)
        self.stokes_loss_config = self.stokes_loss.configuration()
        # Keep the data objective and continuous physics regularization as two
        # independent components. Historical mixed lambda mappings are split by
        # one compatibility adapter before either component is constructed.
        lambda_config, physics_config = split_stokes_and_physics_config(
            lambda_config, physics_config,
        )
        lambda_config = {
            component: lambda_config.get(component, 1.0)
            for component in ('I', 'Q', 'U', 'V')
        }
        self.stokes_lambda_schedules = {
            component: PhysicsWeightSchedule.from_config(value)
            for component, value in lambda_config.items()
        }
        self.stokes_lambda_config = {
            component: schedule.configuration()
            for component, schedule in self.stokes_lambda_schedules.items()
        }
        self.lambdas = nn.ParameterDict({
            component: nn.Parameter(torch.tensor(schedule.start), requires_grad=False)
            for component, schedule in self.stokes_lambda_schedules.items()
        })
        self.physics_regularization = PhysicsRegularization(
            physics_config,
            physics_domain,
            vector_potential=self.parameter_model.vector_potential,
        )
        self.physics_config = self.physics_regularization.configuration()

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

    @torch.no_grad()
    def _set_stokes_lambda_step(self, global_step):
        for component, schedule in self.stokes_lambda_schedules.items():
            self.lambdas[component].fill_(schedule.value_at(global_step))

    def _get_spectral_response(self, batch, instrument_id):
        keys = ('spectral_offsets', 'spectral_weights', 'continuum_weights')
        missing = [key for key in keys if key not in batch]
        if instrument_id in self.hmi_instrument_ids and missing:
            raise KeyError(
                f'HMI batch {instrument_id!r} is missing dataset-sampled response arrays: {missing}.'
            )
        if missing and len(missing) != len(keys):
            raise KeyError(f'Incomplete spectral response for {instrument_id!r}: missing {missing}.')
        return {} if missing else {key: batch[key] for key in keys}

    def _synthesize_stokes(
        self,
        instrument_id,
        forward_params,
        mu,
        wavelength_grid,
        spectral_response,
        filling_factor=None,
    ):
        """Synthesize one instrument and optionally mix a field-free component."""
        magnetic_components = self.forward_models[instrument_id](
            **forward_params,
            mu=mu,
            wavelength_grid=wavelength_grid,
            **spectral_response,
        )
        stokes = torch.stack(magnetic_components, dim=-2)
        if filling_factor is None:
            return stokes

        field_free_components = self.forward_models[instrument_id](
            **field_free_forward_parameters(forward_params),
            mu=mu,
            wavelength_grid=wavelength_grid,
            **spectral_response,
        )
        field_free_stokes = torch.stack(field_free_components, dim=-2)
        stokes = mix_magnetic_filling_factor(
            stokes,
            field_free_stokes,
            filling_factor,
        )
        return stokes

    def _reduce_stokes_objective(self, wavelength_loss):
        """Apply the established wavelength, sample, and Stokes reductions."""
        sample_loss = wavelength_loss.sum(dim=-1)
        component_loss = sample_loss.mean(dim=0)
        weights = torch.stack([self.lambdas[key] for key in ('I', 'Q', 'U', 'V')])
        total = torch.einsum('...i,i->...', sample_loss, weights).mean()
        return component_loss, total

    def configure_optimizers(self):
        parameters = (list(self.parameter_model.parameters()) +
                      list(self.forward_models.parameters()) +
                      list(self.velocity_correction_models.parameters()) +
                      list(self.limb_correction_models.parameters()) +
                      list(self.artifact_correction_models.parameters()))
        if isinstance(self.lr_params, dict):
            learning_rate = float(self.lr_params['start'])
            end_learning_rate = float(self.lr_params['end'])
            decay_iterations = int(self.lr_params['iterations'])
            if learning_rate <= 0 or end_learning_rate <= 0:
                raise ValueError('Learning rates must be positive.')
            if decay_iterations <= 0:
                raise ValueError('lr_params.iterations must be positive.')
        elif isinstance(self.lr_params, (float, int)):
            learning_rate = float(self.lr_params)
        else:
            raise ValueError(f"Invalid lr_params: {self.lr_params}, must be dict or float/int")

        optimizer = SOAP(parameters, lr=learning_rate, weight_decay=0.0)
        if not isinstance(self.lr_params, dict):
            return optimizer

        decay_ratio = end_learning_rate / learning_rate
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: decay_ratio ** min(step / decay_iterations, 1.0),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_nb):
        self._set_stokes_lambda_step(int(self.global_step))
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
        # Physics collocation points manage their own derivative graph below.
        if self.parameter_model.vector_potential:
            coords.requires_grad_(True)
        output = self.parameter_model(coords)

        #################################################
        # Transform the physical Carrington-frame model outputs to each
        # observer's image frame before Stokes synthesis.
        transformed = self.transform_parameters(
            output, coords, cartesian_to_spherical_transform, rtp_to_img_transform,
        )
        forward_params = self.scale_parameters(output, transformed, v_obs_los)

        current_idx = 0
        stokes_pred_profiles = []
        stokes_true_profiles = []
        for instrument_id, n_samples in ds_lengths.items():
            ds_mu = batch[instrument_id]['mu']
            ds_wavelength_grid = batch[instrument_id]['wavelength_grid']
            ds_stokes_true = batch[instrument_id]['stokes']
            ds_spectral_response = self._get_spectral_response(batch[instrument_id], instrument_id)

            ds_forward_params = {
                key: value[current_idx:current_idx + n_samples]
                for key, value in forward_params.items()
            }
            if instrument_id in self.velocity_correction_models:
                time_coords = batch[instrument_id]['coords'][..., 0:1]
                ds_forward_params['vdop'] = ds_forward_params['vdop'] + \
                    self.velocity_correction_models[instrument_id](time_coords)
            limb_correction = None
            if instrument_id in self.limb_correction_models:
                limb_correction = self.limb_correction_models[instrument_id](ds_mu)
                ds_forward_params = self.correct_limb_effects(ds_forward_params, limb_correction)

            ds_stokes_pred = self._synthesize_stokes(
                instrument_id,
                ds_forward_params,
                ds_mu,
                ds_wavelength_grid,
                ds_spectral_response,
                None if limb_correction is None else limb_correction['filling_factor'],
            )
            if instrument_id in self.artifact_correction_models:
                if 'pix' not in batch[instrument_id]:
                    raise KeyError(
                        f"Artifact correction requires 'pix' for instrument {instrument_id}."
                    )
                correction = self.artifact_correction_models[instrument_id](
                    batch[instrument_id]['pix'], ds_stokes_pred
                )
                ds_stokes_pred = correction['stokes_corr']
            stokes_pred_profiles.append(ds_stokes_pred)

            # stokes profiles
            stokes_true_profiles.append(ds_stokes_true)
            current_idx += n_samples

        #################################################
        # compute stokes loss
        stokes_true_profiles = torch.cat(stokes_true_profiles, dim=0)
        stokes_pred_profiles = torch.cat(stokes_pred_profiles, dim=0)
        stokes_loss = self.stokes_loss(
            stokes_pred_profiles, stokes_true_profiles, self.normalization,
        )

        (I_loss, Q_loss, U_loss, V_loss), stokes_total = \
            self._reduce_stokes_objective(stokes_loss)
        logging_loss = {"I": I_loss, "Q": Q_loss, "U": U_loss, "V": V_loss,
                        "stokes_loss": stokes_total}
        logging_loss.update({
            f"lambda.{component}": value.detach()
            for component, value in self.lambdas.items()
        })

        physics = self.physics_regularization.evaluate(
            self.parameter_model,
            global_step=int(self.global_step),
            reference=stokes_total,
        )
        total_loss = stokes_total + physics.total
        logging_loss.update(physics.metrics)
        assert torch.isfinite(total_loss), "Encountered non-finite total loss."
        logging_loss['loss'] = total_loss
        return logging_loss

    def scale_parameters(self, output, transformed_output, v_obs_los):
        return scale_spherical_forward_parameters(
            output, transformed_output, v_obs_los,
            self.gauss_per_dB, self.meters_per_ds, self.seconds_per_dt,
        )

    def transform_parameters(self, output, coords, cartesian_to_spherical_transform, rtp_to_img_transform):
        return transform_spherical_parameters(
            output, coords, cartesian_to_spherical_transform, rtp_to_img_transform,
            self.meters_per_ds, self.seconds_per_dt,
        )

    def correct_limb_effects(self, parameters, limb_correction):
        return correct_spherical_limb_effects(parameters, limb_correction)

    @staticmethod
    def _next_exponential_lambda(value, end, gamma):
        end = value.new_tensor(end)
        candidate = value * gamma
        if value < end:
            return torch.minimum(candidate, end)
        if value > end:
            return torch.maximum(candidate, end)
        return value

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        learning_rate = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', learning_rate)

        # update time shift scale
        new_scale = self.time_shift_scale - self.time_shift_scale_gamma
        if new_scale <= 0.0:
            new_scale = 0.0
        self.time_shift_scale.copy_(new_scale)
        self.log('time_shift_scale', self.time_shift_scale.item() * self.seconds_per_dt)

        # Physics partitions and time-stratified Stokes batches differ by rank.
        # Synchronizing here reports the actual DDP objective rather than rank
        # zero's current frame/partition.
        lambda_metrics = {
            key: value.mean() for key, value in outputs.items()
            if key.startswith('lambda.')
        }
        train_metrics = {
            f'train.{key}': value.mean() for key, value in outputs.items()
            if not key.startswith('lambda.')
        }
        if lambda_metrics:
            self.log_dict(lambda_metrics, sync_dist=True)
        self.log_dict(train_metrics, sync_dist=True)

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb):
        self._set_stokes_lambda_step(int(self.global_step))
        coords = batch['coords']
        mu = batch['mu']
        stokes_true = batch['stokes']
        cartesian_to_spherical_transform = batch['cartesian_to_spherical_transform']
        rtp_to_img_transform = batch['rtp_to_img_transform']
        v_obs_los = batch['v_obs_los']
        wavelength_grid = batch['wavelength_grid']
        instrument_id = batch['instrument_id']
        spectral_response = self._get_spectral_response(batch, instrument_id)

        # forward step
        if self.parameter_model.vector_potential:
            coords.requires_grad_(True)
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

        artifact_correction = None
        stokes_pred = self._synthesize_stokes(
            instrument_id,
            forward_params,
            mu,
            wavelength_grid,
            spectral_response,
            None if limb_correction is None else limb_correction['filling_factor'],
        )
        if instrument_id in self.artifact_correction_models:
            if 'pix' not in batch:
                raise KeyError(
                    f"Artifact correction requires 'pix' for instrument {instrument_id}."
                )
            artifact_correction = self.artifact_correction_models[instrument_id](
                batch['pix'], stokes_pred
            )
            stokes_pred = artifact_correction['stokes_corr']

        objective_wavelength = self.stokes_loss(
            stokes_pred, stokes_true, self.normalization,
        )

        # Keep the established validation plots and MAE metrics in their baseline
        # normalization space, independent of the selected training objective.
        stokes_true_normalized = self.normalization(stokes_true)
        stokes_pred_normalized = self.normalization(stokes_pred)

        diff = torch.abs(stokes_true_normalized - stokes_pred_normalized)
        objective_components = objective_wavelength.sum(-1)
        objective_weights = torch.stack([
            self.lambdas['I'], self.lambdas['Q'], self.lambdas['U'], self.lambdas['V']
        ])
        objective = torch.einsum('...i,i->...', objective_components, objective_weights)
        objective_valid = self.stokes_loss.valid_sample_mask(stokes_true)

        res = {'diff': diff,
               'objective': objective,
               'objective_valid': objective_valid,
               'stokes_true': stokes_true_normalized, 'stokes_pred': stokes_pred_normalized,
               **forward_params,
               'b_rtp': transformed_output['b_rtp'] * self.gauss_per_dB,
               'v_rtp': transformed_output['v_rtp'] * self.meters_per_ds / self.seconds_per_dt,
               'b_img': transformed_output['b_img'] * self.gauss_per_dB,
               'v_img': transformed_output['v_img'] * self.meters_per_ds / self.seconds_per_dt,
               'mu': mu, 'v_obs_los': v_obs_los,
               'v_obs_correction': v_obs_correction,
               }
        if limb_correction is not None:
            res['c_b0'] = limb_correction['c_b0']
            res['c_b1'] = limb_correction['c_b1']
            res['c_vdop'] = limb_correction['c_vdop']
            res['limb_filling_factor'] = limb_correction['filling_factor']
        if artifact_correction is not None:
            res['artifact_correction_params'] = artifact_correction['correction_params']
            res['artifact_filling_factor'] = artifact_correction['filling_factor']

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
        objective_valid = outputs['objective_valid'].bool()
        if torch.any(objective_valid):
            objective_masked = torch.where(
                objective_valid,
                outputs['objective'],
                torch.zeros_like(outputs['objective']),
            )
            objective_total = objective_masked.sum() / objective_valid.sum()
        else:
            objective_total = torch.tensor(torch.nan, dtype=outputs['objective'].dtype)

        parameters = {}
        for k in ['b_field', 'inc', 'azi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl',
                  'v_rtp', 'b_rtp', 'v_img', 'b_img', 'v_obs_correction',
                  'induction', 'dB_dt', 'curl_VxB', 'divergence', 'force_free', 'dB_dr', 'mu', 'v_obs_los',
                  'c_b0', 'c_b1', 'c_vmac', 'c_damping', 'c_kl', 'c_vdop',
                  'limb_filling_factor']:
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
        validation_metrics = {
            # log total stokes loss
            "valid.diff": torch.nanmean(outputs['diff']),
            # log stokes differences
            'valid.I': I_diff, 'valid.Q': Q_diff, 'valid.U': U_diff, 'valid.V': V_diff,
            # exact selected objective, using the same wavelength and Stokes
            # reductions as the training Stokes term
            'valid.objective': objective_total,
            # log physics losses
            # 'valid.induction': torch.nanmean(outputs['induction']),
            # 'valid.divergence': torch.nanmean(outputs['divergence']),
            # 'valid.force_free': torch.nanmean(outputs['force_free']),
        }
        self.log_dict(validation_metrics)

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
        required = {'c_b0', 'c_b1', 'c_vdop', 'limb_filling_factor'}
        if not required <= outputs.keys():
            return
        c_b0 = outputs['c_b0']
        c_b1 = outputs['c_b1']
        mu = outputs['mu']
        c_vdop = outputs['c_vdop']
        filling_factor = outputs['limb_filling_factor']

        fig, axs = plt.subplots(1, 5, figsize=(12.5, 2.5), dpi=100)

        ax = axs[0]
        im = ax.imshow(mu, origin='lower', cmap='cividis', vmin=0, vmax=1)
        ax.set_title("mu")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[4]
        im = ax.imshow(filling_factor, origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_title(r"$f(\mu)$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1]
        im = ax.imshow(c_b0, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{b0}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2]
        im = ax.imshow(c_b1, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{b1}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[3]
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

        if artifact_params.shape[-1] < 9:
            return
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        im = ax.imshow(artifact_params[..., 8], origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_title("Artifact Correction Filling Factor")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        log_wandb_image(fig, "Artifact Correction Filling Factor")
        plt.close(fig)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['pme_stokes_loss_config'] = copy.deepcopy(self.stokes_loss_config)
        checkpoint['pme_stokes_lambda_config'] = copy.deepcopy(
            self.stokes_lambda_config
        )
        checkpoint['pme_normalization_config'] = self._normalization_objective_config(
            self.normalization.asinh_alphas
        )
        checkpoint['pme_physics_config'] = copy.deepcopy(self.physics_config)

    @staticmethod
    def _normalization_objective_config(asinh_alphas):
        if asinh_alphas is None:
            return {'asinh_alphas': None}
        if isinstance(asinh_alphas, dict):
            values = [asinh_alphas[key] for key in ('Q', 'U', 'V')]
        elif isinstance(asinh_alphas, (float, int)):
            values = [asinh_alphas] * 3
        else:
            values = torch.as_tensor(asinh_alphas).detach().cpu().reshape(-1).tolist()
        if len(values) != 3:
            raise ValueError('Normalization objective must provide Q, U, and V asinh alphas.')
        return {'asinh_alphas': [float(value) for value in values]}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint['state_dict']
        checkpoint_loss_config = checkpoint.get('pme_stokes_loss_config')
        if checkpoint_loss_config is None:
            if 'normalization.alphas' in state_dict:
                raise RuntimeError(
                    'This legacy checkpoint predates objective metadata and used a historical '
                    'target-continuum loss that cannot be resumed safely. Use a fresh base_path.'
                )
            checkpoint_loss_config = StokesLossModule().configuration()
        else:
            try:
                checkpoint_loss_config = StokesLossModule(
                    **checkpoint_loss_config
                ).configuration()
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    'This checkpoint uses a removed or unsupported Stokes objective. '
                    'Use a fresh base_path.'
                ) from error

        checkpoint_normalization_config = checkpoint.get('pme_normalization_config')
        if checkpoint_normalization_config is None:
            checkpoint_normalization_config = self._normalization_objective_config(
                state_dict.get('normalization.asinh_alphas')
            )
        else:
            checkpoint_normalization_config = self._normalization_objective_config(
                checkpoint_normalization_config.get('asinh_alphas')
            )
        current_normalization_config = self._normalization_objective_config(
            self.normalization.asinh_alphas
        )

        if (checkpoint_loss_config != self.stokes_loss_config
                or checkpoint_normalization_config != current_normalization_config):
            raise RuntimeError(
                'The configured Stokes objective does not match the checkpoint objective. '
                f'Checkpoint loss/normalization: {checkpoint_loss_config}/'
                f'{checkpoint_normalization_config}; current: {self.stokes_loss_config}/'
                f'{current_normalization_config}. '
                'Use a fresh base_path when changing the objective transform.'
            )

        checkpoint_lambda_config = checkpoint.get('pme_stokes_lambda_config')
        if checkpoint_lambda_config is None:
            checkpoint_lambda_config = {
                component: PhysicsWeightSchedule.from_config(
                    float(state_dict[f'lambdas.{component}'])
                ).configuration()
                for component in ('I', 'Q', 'U', 'V')
            }
        if checkpoint_lambda_config != self.stokes_lambda_config:
            warnings.warn(
                'The configured Stokes-weight schedules differ from the checkpoint; '
                'the configured schedules will replace the checkpoint weights and '
                'continue from its global step. '
                f'Checkpoint: {checkpoint_lambda_config}; '
                f'configured: {self.stokes_lambda_config}.',
                UserWarning,
                stacklevel=2,
            )
        checkpoint_step = int(checkpoint.get('global_step', 0))
        for component, schedule in self.stokes_lambda_schedules.items():
            key = f'lambdas.{component}'
            configured_value = schedule.value_at(checkpoint_step)
            reference = state_dict.get(key, self.lambdas[component].detach())
            state_dict[key] = reference.new_tensor(configured_value)
        checkpoint['pme_stokes_lambda_config'] = copy.deepcopy(
            self.stokes_lambda_config
        )

        checkpoint_physics_config = checkpoint.get('pme_physics_config')
        legacy_physics_keys = {
            key.split('.', 1)[1]
            for key in state_dict
            if key.startswith('lambdas.') and key.split('.', 1)[1] not in ('I', 'Q', 'U', 'V')
        }
        if checkpoint_physics_config is None and legacy_physics_keys:
            raise RuntimeError(
                'This checkpoint used legacy batch-local physics regularization '
                f'({sorted(legacy_physics_keys)}). It cannot be resumed with the new '
                'global collocation objective; use a fresh base_path.'
            )
        if checkpoint_physics_config != self.physics_config:
            warnings.warn(
                'The configured physics objective or collocation domain differs from '
                'the checkpoint; the current configuration will replace the checkpoint '
                'settings and continue from its global step. '
                f'Checkpoint: {checkpoint_physics_config}; '
                f'configured: {self.physics_config}.',
                UserWarning,
                stacklevel=2,
            )
        checkpoint['pme_physics_config'] = copy.deepcopy(self.physics_config)

        # remove old lambdas
        remove_keys = []
        for k, v in state_dict.items():
            if 'lambdas' in k and k.split('.')[1] not in self.lambdas.keys():
                print(f'Remove lambda: {k}')
                remove_keys.append(k)
        [state_dict.pop(k) for k in remove_keys]

        self.load_state_dict(state_dict, strict=False)
        self.validation_outputs = {}  # reset validation outputs
