import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from pme.model import MEModel, NormalizationModule, jacobian
from pme.train.me_atmosphere import MEAtmosphere
from astropy import units as u

class MEModule(LightningModule):

    def __init__(self, cube_shape, lambda_config, value_range, psf_config={'type': None}, dim=256,
                 lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 encoding="positional", plot_profiles=False, **kwargs):
        super().__init__()

        self.plot_profiles = plot_profiles

        self.cube_shape = cube_shape

        # init model
        self.parameter_model = MEModel(3, dim, encoding=encoding)

        self.use_psf = True
        if psf_config['type'] is None:
            self.use_psf = False
        elif psf_config['type'] == 'load':
            self.psf = LoadPSF(psf_config['file'])
            psf_shape = self.psf.psf_shape
            coords_psf = np.stack(
                np.meshgrid(np.linspace(-(psf_shape[0] // 2), psf_shape[0] // 2, psf_shape[0], dtype=np.float32),
                            np.linspace(-(psf_shape[1] // 2), psf_shape[1] // 2, psf_shape[1], dtype=np.float32),
                            np.zeros(1, dtype=np.float32),
                            indexing='ij'), -1)
            coords_psf = coords_psf[:, :, 0]  # remove time axis
            coords_psf *= (2 / cube_shape[0])
            coords_psf = torch.tensor(coords_psf, dtype=torch.float32).reshape((1, *psf_shape, 3))
            self.coords_psf = nn.Parameter(coords_psf, requires_grad=False)
        elif psf_config['type'] == 'learn':
            self.psf = PSF(*psf_config['shape'])
            psf_shape = psf_config['shape']
            coords_psf = np.stack(
                np.meshgrid(np.linspace(-(psf_shape[0] // 2), psf_shape[0] // 2, psf_shape[0], dtype=np.float32),
                            np.linspace(-(psf_shape[1] // 2), psf_shape[1] // 2, psf_shape[1], dtype=np.float32),
                            np.zeros(1, dtype=np.float32),
                            indexing='ij'), -1)
            coords_psf = coords_psf[:, :, 0]  # remove time axis
            coords_psf *= (2 / cube_shape[0])
            coords_psf = torch.tensor(coords_psf, dtype=torch.float32).reshape((1, *psf_shape, 3))
            self.coords_psf = nn.Parameter(coords_psf, requires_grad=False)
        else:
            raise ValueError(f"Invalid PSF type: {psf_config['type']}")

        self.forward_model = MEAtmosphere(**lambda_config)
        self.lr_params = lr_params
        #
        self.validation_outputs = {}
        self.normalization = NormalizationModule(value_range)
        self.loss_function = nn.MSELoss(reduction='none')

    def configure_optimizers(self):
        parameters = list(self.parameter_model.parameters())
        if self.use_psf:
            parameters += list(self.psf.parameters())
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
        coords, stokes_true = batch

        if self.use_psf:
            coords = coords[:, None, None, :] + self.coords_psf

        # forward step
        coords_shape = coords.shape
        output = self.parameter_model(coords.reshape(-1, 3))

        I, Q, U, V = self.forward_model(**output)
        I = I.reshape(*coords_shape[:-1], -1)
        Q = Q.reshape(*coords_shape[:-1], -1)
        U = U.reshape(*coords_shape[:-1], -1)
        V = V.reshape(*coords_shape[:-1], -1)

        if self.use_psf:
            I, Q, U, V = self.convolve_psf(I, Q, U, V, coords)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        # normalize stokes vector by I
        I_normalization = stokes_true[..., 0:1, :]
        normalized_QUV_pred = stokes_pred[..., 1:, :] / (I_normalization + 1e-8)
        normalized_QUV_true = stokes_true[..., 1:, :] / (I_normalization + 1e-8)
        I_pred = stokes_pred[..., 0:1, :]
        I_true = stokes_true[..., 0:1, :]
        stokes_pred = torch.cat([I_pred, normalized_QUV_pred], dim=-2)
        stokes_true = torch.cat([I_true, normalized_QUV_true], dim=-2)

        # stokes_pred = torch.arcsinh(stokes_pred / 1e-3) / np.arcsinh(1 / 1e-3)
        # stokes_true = torch.arcsinh(stokes_true / 1e-3) / np.arcsinh(1 / 1e-3)

        # weight = self.weight.to(stokes_pred.device)
        stokes_true = self.normalization(stokes_true)
        stokes_pred = self.normalization(stokes_pred)

        loss = self.loss_function(stokes_pred, stokes_true)
        loss = loss.sum(dim=-1)
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

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb):
        coords, stokes_true = batch

        if self.use_psf:
            coords = coords[:, None, None, :] + self.coords_psf

        coords_shape = coords.shape
        output = self.parameter_model(coords.reshape(-1, 3))

        I, Q, U, V = self.forward_model(**output)

        # reshape to original coords shape
        I = I.reshape(*coords_shape[:-1], -1)
        Q = Q.reshape(*coords_shape[:-1], -1)
        U = U.reshape(*coords_shape[:-1], -1)
        V = V.reshape(*coords_shape[:-1], -1)

        output = {k: v.reshape(*coords_shape[:-1], -1) for k, v in output.items()}

        if self.use_psf:
            I, Q, U, V = self.convolve_psf(I, Q, U, V, coords)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)
        # norm_stokes_true = self.normalization(stokes_true)
        # norm_stokes_pred = self.normalization(stokes_pred)
        #
        # out_vec = torch.cat(list(output.values()), dim=-1)
        # jac_matrix = jacobian(norm_stokes_true.reshape(), out_vec)
        #
        # diff = (norm_stokes_pred - norm_stokes_true) ** 2
        # diff = diff.mean(dim=(1, 2))


        diff = torch.abs(stokes_true - stokes_pred)

        if self.use_psf:
            center = coords.shape[1] // 2, coords.shape[2] // 2
            output = {k: v[:, center[0], center[1], :] for k, v in output.items()}  # select center pixel
        else:
            output = {k: v for k, v in output.items()}

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
        for k in ['b_field', 'theta', 'chi', 'vmac', 'damping', 'b0', 'b1', 'mu', 'vdop', 'kl']:
            field = outputs[k].reshape(*self.cube_shape[:2]).cpu().numpy()
            plot_settings = {}
            if k == 'theta':
                field = field % np.pi
                plot_settings['vmin'] = 0
                plot_settings['vmax'] = np.pi
                plot_settings['cmap'] = 'RdBu_r'
            if k == 'chi':
                field = field % np.pi
                plot_settings['vmin'] = 0
                plot_settings['vmax'] = np.pi
                plot_settings['cmap'] = 'twilight_shifted'
            if k == 'b_field':
                b_norm = np.abs(field).max()
                plot_settings['vmin'] = -b_norm
                plot_settings['vmax'] = b_norm
                plot_settings['cmap'] = 'RdBu_r'
            if k == "b0" or k == "b1" or k == "mu":
                pass
            if k == "vmac" or k == "damping" or k == "kl":
                plot_settings['vmin'] = 0.

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            im = ax.imshow(field, **plot_settings)
            fig.colorbar(im)

            fig.tight_layout()
            wandb.log({k: fig})
            plt.close('all')

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.cube_shape[:2], 4, 56)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.cube_shape[:2], 4, 56)

        stokes_true[..., 1:, :] = stokes_true[..., 1:, :] / stokes_true[..., 0:1, :]
        stokes_pred[..., 1:, :] = stokes_pred[..., 1:, :] / stokes_pred[..., 0:1, :]

        # plot comparison of integrated stokes vectors
        integerated_stokes_true = np.abs(stokes_true).sum(-1)
        integerated_stokes_pred = np.abs(stokes_pred).sum(-1)
        fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
        for i, label in enumerate(['I', 'Q/I', 'U/I', 'V/I']):
            v_min = integerated_stokes_true[:, :, i].min()
            v_max = integerated_stokes_true[:, :, i].max()
            norm = ImageNormalize(stretch=AsinhStretch(1e-1), vmin=v_min, vmax=v_max)
            im = ax[0, i].imshow(integerated_stokes_true[:, :, i], norm = norm)
            ax[0, i].set_title(f"true - {label}")
            divider = make_axes_locatable(ax[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
            im = ax[1, i].imshow(integerated_stokes_pred[:, :, i], norm=norm)
            ax[1, i].set_title(f"pred - {label}")
            divider = make_axes_locatable(ax[1, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        fig.tight_layout()
        wandb.log({"Integrated Stokes vector - Comparison": fig})
        plt.close('all')

        # stokes_true = np.arcsinh(stokes_true / 1e-4) / np.arcsinh(1 / 1e-4)
        # stokes_pred = np.arcsinh(stokes_pred / 1e-4) / np.arcsinh(1 / 1e-4)

        if self.plot_profiles:
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

        # plot PSF
        if self.use_psf:
            psf = self.psf()
            psf = psf.detach().cpu().numpy()
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            im = ax.imshow(psf, vmin=0)
            fig.colorbar(im)
            fig.tight_layout()
            wandb.log({"PSF": fig})
            plt.close('all')

class PSF(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        assert len(shape) == 2 and shape[0] % 2 == 1 and shape[1] % 2 == 1, "Invalid PSF shape"
        psf = torch.ones(*shape, dtype=torch.float32) * -1
        psf[shape[0] // 2, shape[1] // 2] = 1
        #
        psf = np.load('/glade/work/rjarolim/data/inversion/hinode_psf_0.16.fits')['PSF']
        psf = torch.tensor(psf, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.activation = nn.Softplus()

    def forward(self):
        return self.psf


class LoadPSF(nn.Module):

    def __init__(self, path):
        super().__init__()
        #
        if path.endswith('.npz'):
            psf = np.load(path)['PSF']
        elif path.endswith('.fits'):
            psf = fits.getdata(path).astype(np.float32)
        else:
            raise ValueError(f"Invalid PSF file: {path}")
        psf = torch.tensor(psf, dtype=torch.float32)
        self.psf = nn.Parameter(psf, requires_grad=False)
        self.psf_shape = psf.shape

    def forward(self):
        return self.psf
