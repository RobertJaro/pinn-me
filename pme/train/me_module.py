import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from pme.model import MEModel


class MEModule(LightningModule):

    def __init__(self, dim=256, lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 use_positional_encoding=True, **kwargs):

        super().__init__()
        # init model

        model = MEModel(3, 3, dim, pos_encoding=use_positional_encoding)

        self.model = model
        self.validation_settings = validation_settings
        assert 'boundary' not in validation_settings['names'], "'boundary' is a reserved callback name!"
        self.lr_params = lr_params
        #
        self.validation_outputs = {}

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
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
        stokes_true = batch['values']

        random_coords = batch['random']
        random_coords.requires_grad = True

        # forward step
        parameters = self.model(coords)

        stokes_pred = self.to_profile(parameters)

        loss = (stokes_pred - stokes_true).pow(2).mean()
        loss_dict = {'loss': loss}
        return loss_dict

    def to_profile(self, parameters):
        # do stuff here
        return parameters

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
    def validation_step(self, batch, batch_nb, dataloader_idx):
        coords = batch['coords']
        stokes_true = batch['values']

        parameters = self.model(coords)
        stokes_pred = self.to_profile(parameters)

        diff = torch.abs(stokes_true - stokes_pred)
        diff = diff.mean()

        return {'diff': diff.detach()}

    def validation_epoch_end(self, outputs_list):
        self.validation_outputs = {}  # reset validation outputs
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps

        outputs = outputs_list[0]  # unpack data loader 0
        diff = torch.stack([o['diff'] for o in outputs]).mean()
        self.validation_outputs['boundary'] = {'diff': diff}

        self.log("valid", {"diff": diff})

        return {'progress_bar': {'diff': diff}}
