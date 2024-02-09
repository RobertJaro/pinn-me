import argparse
import json
import os
import shutil

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from pme.train.data_loader import TestDataModule
from pme.train.me_module import MEModule

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

base_path = args.base_path
os.makedirs(base_path, exist_ok=True)

if not hasattr(args, 'work_directory'):
    setattr(args, 'work_directory', base_path)
os.makedirs(args.work_directory, exist_ok=True)





# init logging
wandb_id = args.logging['wandb_id'] if 'wandb_id' in args.logging else None
log_model = args.logging['wandb_log_model'] if 'wandb_log_model' in args.logging else False
wandb_logger = WandbLogger(project=args.logging['wandb_project'], name=args.logging['wandb_name'], offline=False,
                           entity=args.logging['wandb_entity'], id=wandb_id, dir=base_path, log_model=log_model,
                           save_dir=args.work_directory)
wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

# restore model checkpoint from wandb
if wandb_id is not None:
    checkpoint_reference = f"{args.logging['wandb_entity']}/{args.logging['wandb_project']}/model-{args.logging['wandb_id']}:latest"
    artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
    artifact.download(root=base_path)
    shutil.move(os.path.join(base_path, 'model.ckpt'), os.path.join(base_path, 'last.ckpt'))
    args.data['plot_overview'] = False  # skip overview plot for restored model

data_module = TestDataModule(**args.data)

me_module = MEModule(data_module.cube_shape, data_module.lambda_grid, **args.model, **args.training)

config = {'data': args.data, 'model': args.model, 'training': args.training}
checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                      every_n_epochs=args.training[
                                          'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else None,
                                      save_last=True)

# save callback
save_path = os.path.join(base_path, 'inversion.pinnme')
def save(*args, **kwargs):
    torch.save({
                'parameter_model': me_module.parameter_model,
                'cube_shape': data_module.cube_shape, 'lambda_grid': data_module.lambda_grid,
                'data_range': data_module.data_range}, save_path)

save_callback = LambdaCallback(on_validation_epoch_end=save)


torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=int(args.training['epochs']) if 'epochs' in args.training else 1,
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0,
                  check_val_every_n_epoch=args.training[
                      'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else None,
                  gradient_clip_val=0.1,
                  reload_dataloaders_every_n_epochs=1, # reload dataloaders every epoch to avoid oscillating loss
                  callbacks=[checkpoint_callback, save_callback], )

trainer.fit(me_module, data_module, ckpt_path='last')
