import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from astropy.io import fits
from astropy.visualization import ImageNormalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from pme.evaluation.loader import to_spherical, to_cartesian
from pme.model import MEModel, NormalizationModule, ParametersModel
from pme.train.me_atmosphere import MEAtmosphere


class MEModule(LightningModule):

    def __init__(self, cube_shape, lambda_config, value_range, pixel_per_ds, psf_config={'type': None},
                 dim=256, lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5}, lambda_stokes=[1, 1, 1, 1],
                 encoding="positional", **kwargs):
        super().__init__()

        self.cube_shape = cube_shape

        # init model
        self.parameter_model = MEModel(3, dim, encoding=encoding)

        psf_type = psf_config.pop('type')
        if psf_type is None:
            self.psf = NoPSF()
        elif psf_type == 'load':
            self.psf = LoadPSF(**psf_config)
        elif psf_type == 'learn':
            self.psf = PSF(**psf_config)
        else:
            raise ValueError(f"Invalid PSF type: {psf_type}")

        # initialize PSF coordinates
        psf_shape = self.psf.psf_shape
        coords_psf = np.stack(
            np.meshgrid(np.linspace(-(psf_shape[0] // 2), psf_shape[0] // 2, psf_shape[0], dtype=np.float32),
                        np.linspace(-(psf_shape[1] // 2), psf_shape[1] // 2, psf_shape[1], dtype=np.float32),
                        np.zeros(1, dtype=np.float32),
                        indexing='ij'), -1)
        coords_psf = coords_psf[:, :, 0]  # remove time axis
        coords_psf /= pixel_per_ds
        coords_psf = torch.tensor(coords_psf, dtype=torch.float32).reshape((1, *psf_shape, 3))
        self.coords_psf = nn.Parameter(coords_psf, requires_grad=False)

        self.forward_model = MEAtmosphere(**lambda_config)
        self.lr_params = lr_params
        #
        self.validation_outputs = {}
        self.normalization = NormalizationModule(value_range)
        self.loss_function = nn.MSELoss(reduction='none')
        self.lambda_stokes = nn.Parameter(torch.tensor(lambda_stokes, dtype=torch.float32), requires_grad=False)

    def configure_optimizers(self):
        parameters = list(self.parameter_model.parameters()) + list(self.psf.parameters())
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
        coords, mu, stokes_true = batch

        coords = coords[:, None, None, :] + self.coords_psf

        # forward step
        coords_shape = coords.shape
        output = self.parameter_model(coords.reshape(-1, 3))

        # expand mu for PSF (TODO adapt mu for PSF sampling)
        mu = mu[:, None, None, :].repeat(1, *self.coords_psf.shape[1:3], 1).reshape(-1, 1)

        I, Q, U, V = self.forward_model(**output, mu=mu)
        I = I.reshape(*coords_shape[:-1], -1)
        Q = Q.reshape(*coords_shape[:-1], -1)
        U = U.reshape(*coords_shape[:-1], -1)
        V = V.reshape(*coords_shape[:-1], -1)

        I, Q, U, V = self.convolve_psf(I, Q, U, V, coords)
        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        stokes_true = self.normalization(stokes_true)
        stokes_pred = self.normalization(stokes_pred)

        loss = self.loss_function(stokes_pred, stokes_true)
        loss = loss.sum(dim=-1) * self.lambda_stokes[None, :]
        total_loss = loss.mean()
        I_loss, Q_loss, U_loss, V_loss = loss.mean(dim=0)

        return {"loss": total_loss,
                "I_loss": I_loss, "Q_loss": Q_loss,
                "U_loss": U_loss, "V_loss": V_loss}

    def convolve_psf(self, I, Q, U, V, coords):
        # filter_outside = (coords[..., 0] < 0) | (coords[..., 0] > 1) | (coords[..., 1] < 0) | (coords[..., 1] > 1)
        # mask = torch.ones_like(I)
        # mask[filter_outside] = torch.nan

        psf = self.psf()
        psf = psf[None, :, :, None]
        I = (I * psf).sum(dim=(1, 2))
        Q = (Q * psf).sum(dim=(1, 2))
        U = (U * psf).sum(dim=(1, 2))
        V = (V * psf).sum(dim=(1, 2))

        return I, Q, U, V

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
        coords, mu, stokes_true = batch

        coords = coords[:, None, None, :] + self.coords_psf

        coords_shape = coords.shape
        output = self.parameter_model(coords.reshape(-1, 3))

        # expand mu for PSF (TODO adapt mu for PSF sampling)
        mu = mu[:, None, None, :].repeat(1, *self.coords_psf.shape[1:3], 1).reshape(-1, 1)

        I, Q, U, V = self.forward_model(**output, mu=mu)

        # reshape to original coords shape
        I = I.reshape(*coords_shape[:-1], -1)
        Q = Q.reshape(*coords_shape[:-1], -1)
        U = U.reshape(*coords_shape[:-1], -1)
        V = V.reshape(*coords_shape[:-1], -1)

        output = {k: v.reshape(*coords_shape[:-1], -1) for k, v in output.items()}

        I, Q, U, V = self.convolve_psf(I, Q, U, V, coords)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        stokes_true = self.normalization(stokes_true)
        stokes_pred = self.normalization(stokes_pred)

        diff = torch.abs(stokes_true - stokes_pred)

        center = coords.shape[1] // 2, coords.shape[2] // 2
        output = {k: v[:, center[0], center[1], :] for k, v in output.items()}  # select center pixel

        return {'diff': diff.detach(), 'stokes_true': stokes_true.detach(), 'stokes_pred': stokes_pred.detach(),
                **output}

    def validation_epoch_end(self, outputs_list):
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps

        outputs = {}
        for k in outputs_list[0].keys():
            outputs[k] = torch.cat([o[k] for o in outputs_list], dim=0)

        I_diff, Q_diff, U_diff, V_diff = outputs['diff'].mean(dim=(0, 2))
        self.log("valid", {"diff": outputs['diff'].mean(),
                           'I_diff': I_diff, 'Q_diff': Q_diff, 'U_diff': U_diff, 'V_diff': V_diff})

        parameters = {}
        for k in ['b_field', 'theta', 'chi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl']:
            field = outputs[k].reshape(*self.cube_shape[1:3]).cpu().numpy()
            parameters[k] = field

        self.plot_parameter_overview(parameters)
        self.plot_B_cartesian(parameters)

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.cube_shape[1:3], 4, -1)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.cube_shape[1:3], 4, -1)

        self.plot_stokes(stokes_pred, stokes_true)

        self.plot_profile(stokes_pred, stokes_true)

        # plot PSF
        if not isinstance(self.psf, NoPSF):
            self.plot_psf()

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

    def plot_psf(self):
        psf = self.psf()
        psf = psf.detach().cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(psf.T, vmin=0)
        fig.colorbar(im)
        fig.tight_layout()
        wandb.log({"PSF": fig})
        plt.close('all')

    def plot_stokes(self, stokes_pred, stokes_true):
        # plot comparison of integrated stokes vectors
        integerated_stokes_true = np.abs(stokes_true).sum(-1)
        integerated_stokes_pred = np.abs(stokes_pred).sum(-1)
        fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
        for i, label in enumerate(['I', 'Q', 'U', 'V']):
            v_min = integerated_stokes_true[:, :, i].min()
            v_max = integerated_stokes_true[:, :, i].max()
            norm = ImageNormalize(vmin=v_min, vmax=v_max)
            im = ax[0, i].imshow(integerated_stokes_true[:, :, i].T, norm=norm)
            ax[0, i].set_title(f"true - {label}")
            divider = make_axes_locatable(ax[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
            im = ax[1, i].imshow(integerated_stokes_pred[:, :, i].T, norm=norm)
            ax[1, i].set_title(f"pred - {label}")
            divider = make_axes_locatable(ax[1, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        fig.tight_layout()
        wandb.log({"Integrated Stokes vector - Comparison": fig})
        plt.close('all')

    def plot_parameter_overview(self, parameters):
        b = np.abs(parameters['b_field'])
        theta = parameters['theta']
        chi = parameters['chi']
        # reproject vectors (theta flip with negative B)
        chi = chi % np.pi
        theta = theta % np.pi

        fig, axs = plt.subplots(2, 5, figsize=(16, 4), dpi=150)
        ax = axs[0, 0]
        b_max = np.abs(b).max()
        im = ax.imshow(b.T, cmap='RdBu_r', vmin=-b_max, vmax=b_max)
        ax.set_title("B")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 1]
        im = ax.imshow(theta.T, cmap='RdBu_r')
        ax.set_title("Theta")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 2]
        im = ax.imshow(chi.T, cmap='twilight')
        ax.set_title("Chi")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 3]
        im = ax.imshow(parameters['b0'].T)
        ax.set_title("B0")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 4]
        im = ax.imshow(parameters['b1'].T)
        ax.set_title("B1")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 0]
        im = ax.imshow(parameters['vmac'].T)
        ax.set_title("Vmac")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 1]
        im = ax.imshow(parameters['damping'].T)
        ax.set_title("Damping")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 2]
        ax.set_axis_off()

        ax = axs[1, 3]
        vdop_max = np.abs(parameters['vdop']).max()
        im = ax.imshow(parameters['vdop'].T, cmap='RdBu_r', vmin=-vdop_max, vmax=vdop_max)
        ax.set_title("Vdop")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 4]
        im = ax.imshow(parameters['kl'].T)
        ax.set_title("Kl")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        wandb.log({"Parameter Overview": fig})
        plt.close('all')


    def plot_B_cartesian(self, parameters):
        b = np.abs(parameters['b_field'])
        theta = parameters['theta']
        chi = parameters['chi']
        # reproject vectors (theta flip with negative B)
        b_xyz = to_cartesian(b, theta, chi)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
        ax = axs[0]
        bx_max = np.abs(b_xyz[..., 0]).max()
        im = ax.imshow(b_xyz[..., 0].T, cmap='gray', vmin=-bx_max, vmax=bx_max)
        ax.set_title("Bx")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)


        ax = axs[1]
        by_max = np.abs(b_xyz[..., 1]).max()
        im = ax.imshow(b_xyz[..., 1].T, cmap='gray', vmin=-by_max, vmax=by_max)
        ax.set_title("By")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2]
        bz_max = np.abs(b_xyz[..., 2]).max()
        im = ax.imshow(b_xyz[..., 2].T, cmap='gray', vmin=-bz_max, vmax=bz_max)
        ax.set_title("Bz")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        wandb.log({"B": fig})
        plt.close('all')


class PSF(nn.Module):

    def __init__(self, shape, file):
        super().__init__()
        assert len(shape) == 2 and shape[0] % 2 == 1 and shape[1] % 2 == 1, "Invalid PSF shape"
        psf = torch.ones(*shape, dtype=torch.float32) * -1
        psf[shape[0] // 2, shape[1] // 2] = 1
        #
        psf = np.load(file)['PSF']
        psf = torch.tensor(psf, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.activation = nn.Softplus()

    def forward(self):
        return self.psf


class LoadPSF(nn.Module):

    def __init__(self, file, crop=None):
        super().__init__()
        #
        if file.endswith('.npz'):
            psf = np.load(file)['PSF']
        elif file.endswith('.fits'):
            psf = fits.getdata(file).astype(np.float32)
        else:
            raise ValueError(f"Invalid PSF file: {file}")
        if crop is not None:
            psf = psf[crop[0]:crop[1], crop[2]:crop[3]]
        psf = torch.tensor(psf, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.psf_shape = psf.shape

    def forward(self):
        return self.psf


class NoPSF(nn.Module):

    def __init__(self):
        super().__init__()

        psf = torch.ones(1, 1, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.psf_shape = psf.shape

    def forward(self):
        return self.psf
