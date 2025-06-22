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
from pme.evaluation.loader import to_spherical, to_cartesian
from pme.model import NormalizationModule, MESphericalModel, jacobian
from pme.train.me_atmosphere import MEAtmosphere
from pme.train.util import acos_safe, atan2_safe


class MESphericalModule(LightningModule):

    def __init__(self, image_shape, lambda_config, value_range,
                 gauss_per_dB, Rs_per_ds, seconds_per_dt,
                 lr_params=None, model_config=None, normalization_config=None,
                 lambda_stokes=None,
                 lambda_induction=1e-4, lambda_divergence=1e-4, lambda_force_free=1e-4,
                 lambda_static = 1e-4,
                 **kwargs):
        super().__init__()
        lr_params = lr_params if lr_params is not None else {"start": 1e-4, "end": 1e-5, "iterations": 1e5}
        lambda_stokes = lambda_stokes if lambda_stokes is not None else [1, 1, 1, 1]

        self.image_shape = image_shape

        # init model
        model_config = model_config if model_config is not None else {}
        self.parameter_model = MESphericalModel(**model_config)

        self.forward_model = MEAtmosphere(**lambda_config)
        self.lr_params = lr_params
        #
        self.validation_outputs = {}
        normalization_config = normalization_config if normalization_config is not None else {}
        self.normalization = NormalizationModule(value_range, **normalization_config)
        self.loss_function = nn.MSELoss(reduction='none')
        self.lambda_stokes = nn.Parameter(torch.tensor(lambda_stokes, dtype=torch.float32), requires_grad=False)
        self.lambda_induction = lambda_induction
        self.lambda_divergence = lambda_divergence
        self.lambda_force_free = lambda_force_free
        self.lambda_static = lambda_static
        #
        self.gauss_per_dB = gauss_per_dB
        self.Rs_per_ds = Rs_per_ds
        self.meters_per_ds = Rs_per_ds * (1 * u.Rsun).to_value(u.m)
        self.seconds_per_dt = seconds_per_dt

    def configure_optimizers(self):
        parameters = list(self.parameter_model.parameters())
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
        coords = batch['coords']
        mu = batch['mu']
        stokes_true = batch['stokes']
        cartesian_to_spherical_transform = batch['cartesian_to_spherical_transform']
        rtp_to_img_transform = batch['rtp_to_img_transform']
        spherical_to_cartesian_transform = batch['spherical_to_cartesian_transform']
        img_to_rtp_transform = batch['img_to_rtp_transform']
        v_obs_los = batch['v_obs_los']

        # forward step
        coords.requires_grad = True
        output = self.parameter_model(coords)

        disambiguation_mask = output['disambiguation_mask']

        transformed_output = self.transform_parameters(output,
                                                       cartesian_to_spherical_transform, rtp_to_img_transform,
                                                       spherical_to_cartesian_transform, img_to_rtp_transform,
                                                       disambiguation_mask)

        #################################################
        # stokes profile synthesis
        forward_params = self.scale_parameters(output, transformed_output, v_obs_los)
        I, Q, U, V = self.forward_model(**forward_params, mu=mu)
        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        stokes_true = self.normalization(stokes_true)
        stokes_pred = self.normalization(stokes_pred)

        stokes_loss = self.loss_function(stokes_pred, stokes_true)

        # sum over wavelength axis
        stokes_loss = stokes_loss.sum(-1)

        # logging losses
        I_loss, Q_loss, U_loss, V_loss = stokes_loss.mean(dim=0)

        # weighted loss - apply lambda weights for each stokes parameter
        stokes_loss = stokes_loss * self.lambda_stokes[None, :]

        #################################################
        # compute physics loss
        b = transformed_output['b_xyz_disamb']
        v = transformed_output['v_xyz']

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

        div_V = (dVx_dx + dVy_dy + dVz_dz)[..., None]
        dB_dt = torch.stack([dBx_dt, dBy_dt, dBz_dt], -1)
        B_nabla_V = torch.stack([b[:, 0] * dVx_dx + b[:, 1] * dVx_dy + b[:, 2] * dVx_dz,
                                 b[:, 0] * dVy_dx + b[:, 1] * dVy_dy + b[:, 2] * dVy_dz,
                                 b[:, 0] * dVz_dx + b[:, 1] * dVz_dy + b[:, 2] * dVz_dz, ], -1)
        V_nabla_B = torch.stack([v[:, 0] * dBx_dx + v[:, 1] * dBx_dy + v[:, 2] * dBx_dz,
                                 v[:, 0] * dBy_dx + v[:, 1] * dBy_dy + v[:, 2] * dBy_dz,
                                 v[:, 0] * dBz_dx + v[:, 1] * dBz_dy + v[:, 2] * dBz_dz, ], -1)

        curl_VxB = B_nabla_V - V_nabla_B - b * div_V
        induction_equation = dB_dt - curl_VxB
        induction_loss = induction_equation.pow(2).sum(-1)

        div_B = dBx_dx + dBy_dy + dBz_dz
        divergence_loss = div_B.pow(2)

        rot_x = dBz_dy - dBy_dz
        rot_y = dBx_dz - dBz_dx
        rot_z = dBy_dx - dBx_dy
        j = torch.stack([rot_x, rot_y, rot_z], -1)

        force_free = torch.cross(j, b, dim=-1) / (b.norm(dim=-1, keepdim=True) + 1e-8)
        force_free_loss = force_free.pow(2).sum(-1)

        dV_dt = torch.stack([dVx_dt, dVy_dt, dVz_dt], -1)
        static_loss = dV_dt.pow(2).sum(-1)

        #################################################
        # compute total loss
        total_loss = (stokes_loss.mean() +
                      induction_loss.mean() * self.lambda_induction +
                      divergence_loss.mean() * self.lambda_divergence +
                      force_free_loss.mean() * self.lambda_force_free +
                      static_loss.mean() * self.lambda_static
                      )

        assert not torch.isnan(total_loss), f"Encountered invalid value. Loss is NaN"

        return {"loss": total_loss,
                "I_loss": I_loss, "Q_loss": Q_loss,
                "U_loss": U_loss, "V_loss": V_loss,
                "induction_loss": induction_loss.mean(),
                'divergence_loss': divergence_loss.mean(),
                'force_free_loss': force_free_loss.mean(),
                'static_loss': static_loss.mean(),
                }

    def scale_parameters(self, output, transformed_output, v_obs_los):
        v_dop = transformed_output['v_dop'] * self.meters_per_ds / self.seconds_per_dt
        v_dop = v_dop - v_obs_los  # add doppler correction - spacecraft velocity

        forward_params = {'b_field': transformed_output['b_field'] * self.gauss_per_dB,
                          'sin_inc2': transformed_output['sin_inc2'],
                          'cos_inc': transformed_output['cos_inc'],
                          'inc': transformed_output['inc'],
                          'sin2azi': transformed_output['sin2azi'],
                          'cos2azi': transformed_output['cos2azi'],
                          'azi': transformed_output['azi'],
                          'vdop': v_dop,
                          'vmac': output['vmac'], 'damping': output['damping'],
                          'b0': output['b0'], 'b1': output['b1'], 'kl': output['kl']}
        return forward_params

    def transform_parameters(self, output,
                             cartesian_to_spherical_transform, rtp_to_img_transform,
                             spherical_to_cartesian_transform, img_to_rtp_transform,
                             disambiguation_mask):
        # transform B
        b_xyz = torch.cat([output['b_x'], output['b_y'], output['b_z']], dim=-1)
        b_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, b_xyz)
        b_rtp_alt = torch.stack([b_rtp[..., 0], b_rtp[..., 1], b_rtp[..., 2]], dim=-1)
        b_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp_alt)

        # xi, eta, zeta
        # (field, inclination, azimuth) = field, gamma, psi = b_field, inc, azi
        # b_xi = - field * sin(gamma) * sin(psi)
        # b_eta = field * sin(gamma) * cos(psi)
        # b_zeta = field * cos(gamma)
        b_field = torch.norm(b_img, dim=-1, keepdim=True)

        # sin(pi - x) = sin(x)
        # cos(pi - x) = -cos(x)
        sin_inc2 = (b_img[..., 0:1] ** 2 + b_img[..., 1:2] ** 2) / (b_field ** 2 + 1e-8)
        cos_inc = -b_img[..., 2:3] / (b_field + 1e-8) # flipped inclination angle

        # sin(2*(x + pi/2)) = sin(2*x + pi) = -sin(2*x)
        # cos(2*(x + pi/2)) = cos(2*x + pi) = -cos(2*x)
        # 2 * sin(x) * cos(x) = sin(2*x)
        # sin(x)**2 - cos(x)**2 = -cos(2*x)
        sin2azi = 2 * b_img[..., 0:1] * b_img[..., 1:2] / (b_img[..., 0:1] ** 2 + b_img[..., 1:2] ** 2 + 1e-8)
        cos2azi = (b_img[..., 0:1] ** 2 - b_img[..., 1:2] ** 2) / (b_img[..., 0:1] ** 2 + b_img[..., 1:2] ** 2 + 1e-8)

        # Shift polarizer position for HMI
        inc = acos_safe(b_img[..., 2:3] / (b_field + 1e-8))
        inc = torch.pi - inc
        azi = atan2_safe(-b_img[..., 0:1], b_img[..., 1:2])
        azi += torch.pi / 2 # azimuth is flipped in HMI

        # transform to carrington frame --> add rotation velocity
        v_rot = carrington_rotation_velocity()  # in m/s
        v_rot = v_rot / self.meters_per_ds * self.seconds_per_dt  # convert to ds/dt (model units)

        # transform V
        v_xyz = torch.cat([output['v_x'], output['v_y'], output['v_z']], dim=-1)
        v_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, v_xyz)
        v_rtp_alt = torch.stack([v_rtp[..., 0], v_rtp[..., 1], v_rtp[..., 2] + v_rot], dim=-1)
        v_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, v_rtp_alt)

        v_dop = v_img[..., 2:3]

        # compute disambiguated bxyz
        # b_img_flipped = torch.stack([-b_img[..., 0], -b_img[..., 1], b_img[..., 2]], dim=-1)
        # b_rtp_flipped = torch.einsum("...ij,...j->...i", img_to_rtp_transform, b_img_flipped)
        # b_xyz_flipped = torch.einsum("...ij,...j->...i", spherical_to_cartesian_transform, b_rtp_flipped)

        # detach b vector to only optimize disambiguation mask
        b_img_disamb = b_img #disambiguation_mask * b_img.detach() + (1 - disambiguation_mask) * b_img_flipped.detach()
        b_rtp_disamb = b_rtp #disambiguation_mask * b_rtp.detach() + (1 - disambiguation_mask) * b_rtp_flipped.detach()
        b_xyz_disamb = b_xyz #disambiguation_mask * b_xyz.detach() + (1 - disambiguation_mask) * b_xyz_flipped.detach()

        return {'b_field': b_field,
                'sin2azi': sin2azi, 'cos2azi': cos2azi, 'azi': azi,
                'sin_inc2': sin_inc2, 'cos_inc': cos_inc, 'inc': inc,
                'v_dop': v_dop,
                'v_rtp': v_rtp, 'b_rtp': b_rtp, 'v_img': v_img, 'b_img': b_img,
                'b_xyz': b_xyz, 'v_xyz': v_xyz,
                'b_xyz_disamb': b_xyz_disamb, 'b_rtp_disamb': b_rtp_disamb, 'b_img_disamb': b_img_disamb, }

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # update learning rate
        scheduler = self.lr_schedulers()
        if scheduler.get_last_lr()[0] > self.lr_params['end']:
            scheduler.step()
        self.log('Learning Rate', scheduler.get_last_lr()[0])

        # log results to WANDB
        self.log("train", {k: v.mean() for k, v in outputs.items()})

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb):
        coords = batch['coords']
        mu = batch['mu']
        stokes_true = batch['stokes']
        cartesian_to_spherical_transform = batch['cartesian_to_spherical_transform']
        rtp_to_img_transform = batch['rtp_to_img_transform']
        spherical_to_cartesian_transform = batch['spherical_to_cartesian_transform']
        img_to_rtp_transform = batch['img_to_rtp_transform']
        v_obs_los = batch['v_obs_los']

        # forward step
        coords.requires_grad = True
        output = self.parameter_model(coords)

        disambiguation_mask = output['disambiguation_mask']
        binary_disambiguation_mask = torch.round(disambiguation_mask)  # discretize mask to 0 or 1

        transformed_output = self.transform_parameters(output,
                                                       cartesian_to_spherical_transform, rtp_to_img_transform,
                                                       spherical_to_cartesian_transform, img_to_rtp_transform,
                                                       binary_disambiguation_mask)
        forward_params = self.scale_parameters(output, transformed_output, v_obs_los)

        I, Q, U, V = self.forward_model(**forward_params, mu=mu)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        stokes_true = self.normalization(stokes_true)
        stokes_pred = self.normalization(stokes_pred)

        diff = torch.abs(stokes_true - stokes_pred)

        return {'diff': diff.detach(), 'stokes_true': stokes_true.detach(), 'stokes_pred': stokes_pred.detach(),
                **forward_params,
                'b_rtp': transformed_output['b_rtp_disamb'] * self.gauss_per_dB,
                'v_rtp': transformed_output['v_rtp'] * self.meters_per_ds / self.seconds_per_dt,
                'b_img': transformed_output['b_img_disamb'] * self.gauss_per_dB,
                'v_img': transformed_output['v_img'] * self.meters_per_ds / self.seconds_per_dt,
                'disambiguation_mask': disambiguation_mask.detach(),
                }

    def validation_epoch_end(self, outputs_list):
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps

        outputs = {}
        for k in outputs_list[0].keys():
            outputs[k] = torch.cat([o[k] for o in outputs_list], dim=0)

        I_diff, Q_diff, U_diff, V_diff = torch.nanmean(outputs['diff'], dim=(0, 2))
        self.log("valid", {"diff": torch.nanmean(outputs['diff']),
                           'I_diff': I_diff, 'Q_diff': Q_diff, 'U_diff': U_diff, 'V_diff': V_diff})

        parameters = {}
        for k in ['b_field', 'inc', 'azi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl',
                  'v_rtp', 'b_rtp', 'v_img', 'b_img', 'disambiguation_mask']:
            field = outputs[k].reshape(*self.image_shape[:2], -1).cpu().numpy().squeeze()
            parameters[k] = field

        self.plot_parameter_overview(parameters)
        self.plot_B_rtp(parameters)
        self.plot_B_rtp_scaled(parameters)
        self.plot_v_rtp(parameters)

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)

        self.plot_stokes(stokes_pred, stokes_true)

        # self.plot_profile(stokes_pred, stokes_true)

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
            fig.tight_layout()
            wandb.log({f"Profile x:{x:02d} y:{y:02d}": wandb.Image(fig)})
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
        fig.tight_layout()
        wandb.log({"Integrated Stokes vector - Comparison": fig})
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

        fig, axs = plt.subplots(2, 5, figsize=(16, 4), dpi=150)
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
        im = ax.imshow(parameters['disambiguation_mask'], vmin=0, vmax=1, origin='lower', cmap='PuOr')
        ax.set_title("Disambiguation Mask")
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
        plt.tight_layout()
        wandb.log({"Parameter Overview": fig})
        plt.close('all')

    def plot_B_rtp(self, parameters):
        b_rtp = parameters['b_rtp']
        b_img = parameters['b_img']

        b_rtp_min_max = np.nanmax(np.abs(b_rtp))
        norm = SymLogNorm(linthresh=1, vmin=-b_rtp_min_max, vmax=b_rtp_min_max)

        fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=150)

        ax = axs[0, 0]
        im = ax.imshow(b_rtp[..., 0], norm=norm, origin='lower', cmap='RdBu_r')
        ax.set_title("$B_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(b_rtp[..., 1], cmap='RdBu_r', norm=norm, origin='lower')
        ax.set_title("$B_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(b_rtp[..., 2], cmap='RdBu_r', norm=norm, origin='lower')
        ax.set_title("$B_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(b_img[..., 0], norm=norm, origin='lower', cmap='RdBu_r')
        ax.set_title(r"$B_\text{xi}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(b_img[..., 1], norm=norm, origin='lower', cmap='RdBu_r')
        ax.set_title(r"$B_\text{eta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        im = ax.imshow(b_img[..., 2], norm=norm, origin='lower', cmap='RdBu_r')
        ax.set_title(r"$B_\text{zeta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        wandb.log({"B": fig})
        plt.close('all')

    def plot_B_rtp_scaled(self, parameters):
        b_rtp = parameters['b_rtp']

        norm = Normalize(vmin=-500, vmax=500)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=150)

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

        plt.tight_layout()
        wandb.log({"B_rtp": fig})
        plt.close('all')

    def plot_v_rtp(self, parameters):
        v_rtp = parameters['v_rtp']
        v_img = parameters['v_img']

        v_min_max = np.nanmax(np.abs(v_rtp))
        norm = Normalize(vmin=-v_min_max, vmax=v_min_max)

        fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=150)
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
        im = ax.imshow(v_img[..., 0], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{xi}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(v_img[..., 1], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{eta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        im = ax.imshow(v_img[..., 2], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{zeta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        wandb.log({"v": fig})
        plt.close('all')

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # replace checkpoint lambda with the current lambda
        checkpoint['state_dict']['lambda_stokes'] = self.lambda_stokes.detach().cpu()
