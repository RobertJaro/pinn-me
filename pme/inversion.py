import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from pme.train.data_loader import TestDataModule, SHARPDataModule, HinodeDataModule
from pme.train.me_module import MEModule
from pme.train.util import load_yaml_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args, unknown_args = parser.parse_known_args()

config = load_yaml_config(args.config, unknown_args)

base_path = config['base_path']
os.makedirs(base_path, exist_ok=True)

work_directory = config['work_directory'] if 'work_directory' in config else base_path
os.makedirs(work_directory, exist_ok=True)

logging_config = config['logging'] if 'logging' in config else {}
# init logging
wandb_logger = WandbLogger(**logging_config, save_dir=work_directory)
wandb_logger.experiment.config.update(config, allow_val_change=True)

data_config = config['data']
type = data_config.pop('type')
if type == 'test':
    data_module = TestDataModule(**data_config)
elif type == 'sharp':
    data_module = SHARPDataModule(**data_config)
elif type == 'hinode':
    data_module = HinodeDataModule(**data_config)
else:
    raise ValueError(f'Unknown data type {type}')

model_config = config['model'] if 'model' in config else {}
training_config = config['training'] if 'training' in config else {}
check_val_every_n_epoch = training_config.pop('check_val_every_n_epoch', None)
epochs = training_config.pop('epochs', 50)

me_module = MEModule(data_module.cube_shape, data_module.lambda_config,
                     data_module.value_range, data_module.pixel_per_ds,
                     model_config=model_config,
                     **training_config)

checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                      every_n_epochs=check_val_every_n_epoch,
                                      save_last=True)

# save callback
save_path = os.path.join(base_path, 'inversion.pme')


def save(*args, **kwargs):
    torch.save({
        'parameter_model': me_module.parameter_model,
        'cube_shape': data_module.cube_shape, 'lambda_config': data_module.lambda_config,
        'data_range': data_module.data_range,
        'ref_time': data_module.ref_time, 'times': data_module.times, 'seconds_per_dt': data_module.seconds_per_dt,
        'pixel_per_ds': data_module.pixel_per_ds
    }, save_path)


save_callback = LambdaCallback(on_validation_epoch_end=save)

torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=epochs,
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0,
                  check_val_every_n_epoch=check_val_every_n_epoch,
                  gradient_clip_val=0.1,
                  reload_dataloaders_every_n_epochs=1,  # reload dataloaders every epoch to avoid oscillating loss
                  callbacks=[checkpoint_callback, save_callback], )

trainer.fit(me_module, data_module, ckpt_path='last')
