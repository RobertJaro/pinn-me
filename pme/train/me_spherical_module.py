import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from astropy.visualization import ImageNormalize
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from pme.evaluation.loader import to_spherical, to_cartesian
from pme.model import NormalizationModule, MESphericalModel
from pme.train.me_atmosphere import MEAtmosphere
from pme.train.util import acos_safe, atan2_safe


class MESphericalModule(LightningModule):

    def __init__(self, image_shape, lambda_config, value_range, lr_params=None,
                 lambda_stokes=None, model_config=None, **kwargs):
        super().__init__()
        lr_params = lr_params if lr_params is not None else {"start": 5e-4, "end": 5e-5, "iterations": 1e5}
        lambda_stokes = lambda_stokes if lambda_stokes is not None else [1, 1, 1, 1]

        self.image_shape = image_shape

        # init model
        model_config = model_config if model_config is not None else {}
        self.parameter_model = MESphericalModel(4, **model_config)

        self.forward_model = MEAtmosphere(**lambda_config)
        self.lr_params = lr_params
        #
        self.validation_outputs = {}
        self.normalization = NormalizationModule(value_range)
        self.loss_function = nn.MSELoss(reduction='none')
        self.lambda_stokes = nn.Parameter(torch.tensor(lambda_stokes, dtype=torch.float32), requires_grad=False)

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
        ds_keys = list(batch.keys())
        coords = torch.cat([batch[k]['coords'] for k in ds_keys], 0)
        mu = torch.cat([batch[k]['mu'] for k in ds_keys], 0)
        stokes_true = torch.cat([batch[k]['stokes'] for k in ds_keys], 0)
        cartesian_to_spherical_transform = torch.cat([batch[k]['cartesian_to_spherical_transform'] for k in ds_keys], 0)
        rtp_to_img_transform = torch.cat([batch[k]['rtp_to_img_transform'] for k in ds_keys], 0)

        assert not torch.isnan(coords).any(), "Encountered invalid value. coords is NaN"
        assert not torch.isnan(mu).any(), "Encountered invalid value. mu is NaN"
        assert not torch.isnan(stokes_true).any(), "Encountered invalid value. stokes_true is NaN"
        assert not torch.isnan(
            cartesian_to_spherical_transform).any(), "Encountered invalid value. cartesian_to_spherical_transform is NaN"
        assert not torch.isnan(rtp_to_img_transform).any(), "Encountered invalid value. rtp_to_img_transform is NaN"

        # forward step
        output = self.parameter_model(coords)

        transformed_output = self.transform_parameters(output, cartesian_to_spherical_transform, rtp_to_img_transform)

        forward_params = {'b_field': transformed_output['b_field'],
                          'theta': transformed_output['theta'],
                          'chi': transformed_output['chi'],
                          'vdop': transformed_output['v_dop'],
                          'vmac': output['vmac'], 'damping': output['damping'],
                          'b0': output['b0'], 'b1': output['b1'], 'kl': output['kl']}

        I, Q, U, V = self.forward_model(**forward_params, mu=mu)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        stokes_true = self.normalization(stokes_true)
        stokes_pred = self.normalization(stokes_pred)

        loss = self.loss_function(stokes_pred, stokes_true)
        loss = loss.sum(-1)  # sum over wavelength axis

        # logging losses
        I_loss, Q_loss, U_loss, V_loss = loss.mean(dim=0)

        # weighted loss - apply lambda weights for each stokes parameter
        total_loss = loss * self.lambda_stokes[None, :]
        total_loss = total_loss.mean()

        assert not torch.isnan(total_loss), f"Encountered invalid value. Loss is NaN"

        return {"loss": total_loss,
                "I_loss": I_loss, "Q_loss": Q_loss,
                "U_loss": U_loss, "V_loss": V_loss}

    def transform_parameters(self, output, cartesian_to_spherical_transform, rtp_to_img_transform):
        # transform B
        b_xyz = torch.cat([output['b_x'], output['b_y'], output['b_z']], dim=-1)
        b_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, b_xyz)
        b_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)

        # xi, eta, zeta
        # (field, inclination, azimuth) = field, gamma, psi = b_field, theta, chi
        # b_xi = - field * sin(gamma) * sin(psi)
        # b_eta = field * sin(gamma) * cos(psi)
        # b_zeta = field * cos(gamma)
        b_field = torch.norm(b_img, dim=-1, keepdim=True)
        theta = acos_safe(b_img[..., 2:3] / (b_field + 1e-8))
        chi = atan2_safe(-b_img[..., 0:1], b_img[..., 1:2])

        # transform V
        v_xyz = torch.cat([output['v_x'], output['v_y'], output['v_z']], dim=-1)
        v_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, v_xyz)
        v_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, v_rtp)

        v_dop = v_img[..., 2:3]

        return {'b_field': b_field, 'chi': chi, 'theta': theta, 'v_dop': v_dop,
                'v_rtp': v_rtp, 'b_rtp': b_rtp, 'v_img': v_img, 'b_img': b_img}

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # update learning rate
        scheduler = self.lr_schedulers()
        if scheduler.get_last_lr()[0] > self.lr_params['end']:
            scheduler.step()
        self.log('Learning Rate', scheduler.get_last_lr()[0])

        # log results to WANDB
        self.log("train", {k: v.mean() for k, v in outputs.items()})

    def validation_step(self, batch, batch_nb):
        coords = batch['coords']
        mu = batch['mu']
        stokes_true = batch['stokes']
        cartesian_to_spherical_transform = batch['cartesian_to_spherical_transform']
        rtp_to_img_transform = batch['rtp_to_img_transform']

        # forward step
        output = self.parameter_model(coords)

        transformed_output = self.transform_parameters(output, cartesian_to_spherical_transform,
                                                       rtp_to_img_transform)

        forward_params = {'b_field': transformed_output['b_field'],
                          'theta': transformed_output['theta'],
                          'chi': transformed_output['chi'],
                          'vdop': transformed_output['v_dop'],
                          'vmac': output['vmac'], 'damping': output['damping'],
                          'b0': output['b0'], 'b1': output['b1'], 'kl': output['kl']}

        I, Q, U, V = self.forward_model(**forward_params, mu=mu)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        stokes_true = self.normalization(stokes_true)
        stokes_pred = self.normalization(stokes_pred)

        diff = torch.abs(stokes_true - stokes_pred)

        return {'diff': diff.detach(), 'stokes_true': stokes_true.detach(), 'stokes_pred': stokes_pred.detach(),
                **forward_params, 'b_rtp': transformed_output['b_rtp'], 'v_rtp': transformed_output['v_rtp'],
                'b_img': transformed_output['b_img'], 'v_img': transformed_output['v_img']}

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
        for k in ['b_field', 'theta', 'chi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl',
                  'v_rtp', 'b_rtp', 'v_img', 'b_img']:
            field = outputs[k].reshape(*self.image_shape[:2], -1).cpu().numpy().squeeze()
            parameters[k] = field

        self.plot_parameter_overview(parameters)
        self.plot_B_rtp(parameters)
        self.plot_v_rtp(parameters)

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)

        self.plot_stokes(stokes_pred, stokes_true)

        self.plot_profile(stokes_pred, stokes_true)

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
        theta = parameters['theta']
        chi = parameters['chi']
        # reproject vectors (theta flip with negative B)
        b_xyz = to_cartesian(b, theta, chi)
        b, theta, chi = to_spherical(b_xyz)
        theta = theta % np.pi
        chi = chi % (2 * np.pi)

        fig, axs = plt.subplots(2, 5, figsize=(16, 4), dpi=150)
        ax = axs[0, 0]
        im = ax.imshow(b, cmap='viridis', vmin=.1, origin='lower', norm='log')
        ax.set_title("B")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 1]
        im = ax.imshow(theta, cmap='seismic', vmin=0, vmax=np.pi, origin='lower')
        ax.set_title("Theta")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 2]
        im = ax.imshow(chi, cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower')
        ax.set_title("Chi")
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

    def plot_v_rtp(self, parameters):
        v_rtp = parameters['v_rtp']
        v_img = parameters['v_img']

        norm = SymLogNorm(linthresh=10, vmin=-np.nanmax(np.abs(v_rtp)), vmax=np.nanmax(np.abs(v_rtp)))

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
