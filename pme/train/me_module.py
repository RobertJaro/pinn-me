import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from pme.model import MEModel
from pme.train.me_equations import MEAtmosphere


class MEModule(LightningModule):

    def __init__(self, lambda_grid, voigt_function_files, dim=256,
                 lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 use_positional_encoding=True, **kwargs):
        super().__init__()

        # init model
        self.parameter_model = MEModel(2, dim, pos_encoding=use_positional_encoding)

        self.forward_model = MEAtmosphere(lambda0 = 6301.5080, jUp = 2.0, jLow = 2.0, gUp = 1.5, gLow = 1.83,
                                          lambdaGrid=lambda_grid,
                                          voigt_pt=voigt_function_files['voigt'],
                                          faraday_voigt_pt=voigt_function_files['faraday_voigt'])
        self.lr_params = lr_params
        #
        self.validation_outputs = {}

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
        coords,  stokes_true = batch

        # forward step
        output = self.parameter_model(coords)

        I, Q, U, V = self.forward_model(**output)
        stokes_pred = torch.stack([I, Q, U, V], dim=-2)

        loss = (stokes_pred - stokes_true).pow(2).mean()

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
        coords,  stokes_true = batch

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

        b_field = outputs['b_field'].reshape(41, 41).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(b_field)
        fig.colorbar(im)

        fig.tight_layout()
        wandb.log({"B Field": fig})
        plt.close('all')

        fig, axs = plt.subplots(4, 1, figsize=(8, 8))
        stokes_true = outputs['stokes_true'].cpu().numpy().reshape(41, 41, 4, 50)
        stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(41, 41, 4, 50)
        for i, label in enumerate(['I', 'Q', 'U', 'V']):
            axs[i].plot(stokes_true[20, 20, i], label=f'true - {label}')
            axs[i].plot(stokes_pred[20, 20, i], label=f'pred - {label}')
            if i == 0:
                continue
            v_min_max = np.abs(stokes_true[20, 20, i]).max()
            v_min_max = max(0.1, v_min_max)
            axs[i].set_ylim([-v_min_max, v_min_max])

        [ax.legend(loc='upper right') for ax in axs[1:]]
        # set minimum y axis to 0.1


        # log figure
        fig.tight_layout()
        wandb.log({"Profile": wandb.Image(fig)})
        plt.close('all')


