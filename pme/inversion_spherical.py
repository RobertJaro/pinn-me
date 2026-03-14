import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from pme.loader.spherical import SphericalDataModule
from pme.train.me_spherical_module import MESphericalModule
from pme.train.util import load_yaml_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
parser.add_argument('--reload', action='store_true')
args, unknown_args = parser.parse_known_args()

config = load_yaml_config(args.config, unknown_args)

base_path = config['base_path']
os.makedirs(base_path, exist_ok=True)

work_directory = config['work_directory'] if 'work_directory' in config else base_path
os.makedirs(work_directory, exist_ok=True)

logging_config = config['logging'] if 'logging' in config else {}
# init logging
wandb_logger = WandbLogger(**logging_config, save_dir=work_directory)

@rank_zero_only
def _log_hparams(cfg):
    wandb_logger.log_hyperparams(cfg)

_log_hparams(config)

data_config = config['data']
# type = data_config.pop('type')

data_module_save_path = os.path.join(work_directory, 'data_module.pt')
if os.path.exists(data_module_save_path) and not args.reload:
    data_module = torch.load(data_module_save_path)
    # update batch settings
    if 'batch_size' in data_config:
        data_module.batch_size = data_config['batch_size']
    if 'dataset_batch_size' in data_config:
        data_module.dataset_batch_size = data_config['dataset_batch_size']
else:
    data_module = SphericalDataModule(**data_config, work_directory=work_directory)
    torch.save(data_module, data_module_save_path)

model_config = config['model'] if 'model' in config else {}
training_config = config['training'] if 'training' in config else {}
check_val_every_n_epoch = training_config.pop('check_val_every_n_epoch', None)
val_check_interval = training_config.pop('val_check_interval', None)
epochs = training_config.pop('epochs', 50)
instrument_config = config['instrument'] if 'instrument' in config else {}
lambda_config = config['lambda'] if 'lambda' in config else {}
normalization_config = config['normalization'] if 'normalization' in config else {}
normalization_config['value_range'] = data_module.value_range
artifact_correction_config = config['artifact_correction'] if 'artifact_correction' in config else {}

me_module = MESphericalModule(image_shape=data_module.image_shape, wavelength_config=data_module.wavelength_config,
                              model_config=model_config, normalization_config=normalization_config,
                              instrument_config=instrument_config,
                              Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                              gauss_per_dB=data_module.gauss_per_dB, lambda_config=lambda_config,
                              **training_config)

checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                      every_n_epochs=check_val_every_n_epoch,
                                      save_last=True)

# save callback
save_path = os.path.join(base_path, 'inversion.pme')

@rank_zero_only
def save(*args, **kwargs):
    torch.save({
        'parameter_model': me_module.parameter_model,
        'cube_shape': data_module.image_shape, 'wavelength_config': data_module.wavelength_config,
        'data_range': data_module.data_range,
        'ref_time': data_module.ref_time, 'times': data_module.times,
        'seconds_per_dt': data_module.seconds_per_dt,
        'Rs_per_ds': data_module.Rs_per_ds,
        'gauss_per_dB': data_module.gauss_per_dB,
    }, save_path)


save_callback = LambdaCallback(on_validation_epoch_end=save)

torch.set_float32_matmul_precision('medium')  # for A100 GPUs
torch.multiprocessing.set_sharing_strategy("file_system")

n_gpus = torch.cuda.device_count()

# from torchrun:
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))          # total processes (16)
LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", str(n_gpus)))  # procs on this node (4)

# fallbacks & safety
devices_per_node = max(1, LOCAL_WORLD_SIZE if n_gpus > 0 else 0)
num_nodes = max(1, WORLD_SIZE // devices_per_node)

trainer = Trainer(max_epochs=int(epochs),
                  logger=wandb_logger,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  devices=devices_per_node if n_gpus > 0 else None,
                  num_nodes=num_nodes,
                  strategy=DDPStrategy(find_unused_parameters=False) if (num_nodes > 1 or devices_per_node > 1) else 'auto',
                  num_sanity_val_steps=-1,
                  check_val_every_n_epoch=check_val_every_n_epoch,
                  val_check_interval=val_check_interval,
                  gradient_clip_val=0.5,
                  callbacks=[checkpoint_callback, save_callback])

trainer.fit(me_module, data_module, ckpt_path='last')
