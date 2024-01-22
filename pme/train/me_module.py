import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from pme.model import MEModel
from pme.train.me_equations import MEAtmosphere


class MEModule(LightningModule):

    def __init__(self, lambda_grid, dim=256,
                 lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 use_positional_encoding=False, **kwargs):
        super().__init__()

        # init model
        self.parameter_model = MEModel(2, dim, pos_encoding=use_positional_encoding)

        self.forward_model = MEAtmosphere(lambda0=6301.5080, jUp=2.0, jLow=2.0, gUp=1.5, gLow=1.83,
                                          lambdaGrid=lambda_grid, )
        self.lr_params = lr_params
        #
        self.validation_outputs = {}
        weight = torch.tensor([1., 1e5, 1e5, 1e2], dtype=torch.float32).reshape(1, 4)
        weight = weight  # / weight.sum()
        self.weight = weight

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
        coords, stokes_true = batch

        # forward step
        output = self.parameter_model(coords)

        I, Q, U, V = self.forward_model(**output)
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
        loss = (stokes_pred - stokes_true).pow(2).sum(-1)
        loss = (loss * weight).mean()

        loss_dict = {'loss': loss}
        return loss_dict

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

        output = self.parameter_model(coords)

        I, Q, U, V = self.forward_model(**output)
        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        diff = torch.abs(stokes_true - stokes_pred)

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
            field = outputs[k].reshape(41, 41).cpu().numpy()
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            im = ax.imshow(field)
            fig.colorbar(im)

            fig.tight_layout()
            wandb.log({k: fig})
            plt.close('all')

        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(41, 41, 4, 50)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(41, 41, 4, 50)

        stokes_true[..., 1:, :] = stokes_true[..., 1:, :] / stokes_true[..., 0:1, :]
        stokes_pred[..., 1:, :] = stokes_pred[..., 1:, :] / stokes_pred[..., 0:1, :]

        # stokes_true = np.arcsinh(stokes_true / 1e-4) / np.arcsinh(1 / 1e-4)
        # stokes_pred = np.arcsinh(stokes_pred / 1e-4) / np.arcsinh(1 / 1e-4)

        pos = np.stack(np.meshgrid(np.linspace(10, 30, 3, dtype=int), np.linspace(10, 30, 3, dtype=int)), -1)
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
