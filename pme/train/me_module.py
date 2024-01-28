import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from pme.model import MEModel
from pme.train.me_equations import MEAtmosphere


class MEModule(LightningModule):

    def __init__(self, img_shape, lambda_grid, psf_shape=(5, 5), dim=256,
                 lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 positional_encoding="periodic", use_psf=True, **kwargs):
        super().__init__()

        self.img_shape = img_shape

        # init model
        self.parameter_model = MEModel(2, dim, pos_encoding=positional_encoding)

        self.use_psf = use_psf
        self.psf = PSF(*psf_shape)
        coords_psf = np.stack(np.meshgrid(np.linspace(-(psf_shape[0] // 2), psf_shape[0] // 2, psf_shape[0]),
                                          np.linspace(-(psf_shape[1] // 2), psf_shape[1] // 2, psf_shape[1]),
                                          indexing='ij'), -1)
        print("IMG_shape", img_shape)
        coords_psf /= (img_shape[0] / 2)
        self.coords_psf = nn.Parameter(torch.tensor(coords_psf, dtype=torch.float32).reshape((1, *psf_shape, 2)),
                                       requires_grad=False)

        self.forward_model = MEAtmosphere(lambda0=6301.5080, jUp=2.0, jLow=2.0, gUp=1.5, gLow=1.83,
                                          lambdaGrid=lambda_grid, )
        self.lr_params = lr_params
        #
        self.validation_outputs = {}
        weight = torch.tensor([1., 1e5, 1e5, 1e2], dtype=torch.float32).reshape(1, 4)
        self.weight = weight

    def configure_optimizers(self):
        parameters = list(self.parameter_model.parameters()) #+ list(self.psf.parameters())
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
        output = self.parameter_model(coords)

        I, Q, U, V = self.forward_model(**output)

        if self.use_psf:
            I, Q, U, V = self.convolve_psf(I, Q, U, V, coords)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        # normalize stokes vector by I
        I_pred = stokes_pred[..., 0:1, :]
        I_true = stokes_true[..., 0:1, :]
        normalized_QUV_pred = stokes_pred[..., 1:, :] / (I_pred + 1e-8)
        normalized_QUV_true = stokes_true[..., 1:, :] / (I_true + 1e-8)
        stokes_pred = torch.cat([I_pred, normalized_QUV_pred], dim=-2)
        stokes_true = torch.cat([I_true, normalized_QUV_true], dim=-2)

        # stokes_pred = torch.arcsinh(stokes_pred / 1e-3) / np.arcsinh(1 / 1e-3)
        # stokes_true = torch.arcsinh(stokes_true / 1e-3) / np.arcsinh(1 / 1e-3)

        weight = self.weight.to(stokes_pred.device)
        loss = (stokes_pred - stokes_true).pow(2).sum(-1).pow(0.5)
        loss = torch.mean(loss * weight)

        loss_dict = {'loss': loss}
        return loss_dict

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
        coords, stokes_true = batch

        coords = coords[:, None, None, :] + self.coords_psf

        output = self.parameter_model(coords)

        I, Q, U, V = self.forward_model(**output)

        I, Q, U, V = self.convolve_psf(I, Q, U, V, coords)

        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

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

        self.log("valid", {"diff": outputs['diff'].mean()})

        for k in ['b_field', 'theta', 'chi', 'vmac', 'damping', 'b0', 'b1', 'mu', 'vdop', 'kl']:
            field = outputs[k].reshape(*self.img_shape).cpu().numpy()
            plot_settings = {}
            if k == 'theta' or k == 'chi':
                field = np.cos(field) # reproject angles to cosine
                plot_settings['vmin'] = -1
                plot_settings['vmax'] = 1
            if k == 'b_field':
                v_min_max = np.abs(field).max()
                plot_settings['vmin'] = -v_min_max
                plot_settings['vmax'] = v_min_max
                plot_settings['cmap'] = 'RdBu_r'
            if k == "b0" or k == "b1" or k == "mu":
                plot_settings['vmin'] = 0
                plot_settings['vmax'] = 1
            if k == "vmac" or k == "damping" or k == "kl":
                plot_settings['vmin'] = 0

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            im = ax.imshow(field, **plot_settings)
            fig.colorbar(im)

            fig.tight_layout()
            wandb.log({k: fig})
            plt.close('all')

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.img_shape, 4, 50)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.img_shape, 4, 50)

        stokes_true[..., 1:, :] = stokes_true[..., 1:, :] / stokes_true[..., 0:1, :]
        stokes_pred[..., 1:, :] = stokes_pred[..., 1:, :] / stokes_pred[..., 0:1, :]

        # stokes_true = np.arcsinh(stokes_true / 1e-4) / np.arcsinh(1 / 1e-4)
        # stokes_pred = np.arcsinh(stokes_pred / 1e-4) / np.arcsinh(1 / 1e-4)

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
        psf = torch.ones(*shape, dtype=torch.float32) * 1e-2
        psf[shape[0] // 2, shape[1] // 2] = 1
        self.psf = nn.Parameter(psf, requires_grad=True)

    def forward(self):
        psf = torch.softmax(self.psf)
        psf = psf / psf.sum()
        return psf
