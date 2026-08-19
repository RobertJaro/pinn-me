import argparse
import copy
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from pme.loader.spherical import SphericalDataModule, spherical_data_cache_signature
from pme.train.me_spherical_module import MESphericalModule
from pme.train.physics_regularization import (
    build_spherical_collocation_domain,
    split_stokes_and_physics_config,
)
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

data_config = copy.deepcopy(config['data'])
data_cache_signature = spherical_data_cache_signature(data_config)

data_module_save_path = os.path.join(work_directory, 'data_module.pt')
use_cached_data = os.path.exists(data_module_save_path) and not args.reload
if use_cached_data:
    data_module = torch.load(data_module_save_path, weights_only=False)
    use_cached_data = (
        getattr(data_module, 'cache_version', 0) == SphericalDataModule.CACHE_VERSION
        and getattr(data_module, 'cache_signature', None) == data_cache_signature
    )

if use_cached_data:
    # update batch settings
    if 'batch_size' in data_config:
        data_module.batch_size = data_config['batch_size']
    if 'dataset_batch_size' in data_config:
        data_module.dataset_batch_size = data_config['dataset_batch_size']
else:
    data_module = SphericalDataModule(**data_config, work_directory=work_directory)
    data_module.cache_signature = data_cache_signature
    torch.save(data_module, data_module_save_path)

model_config = copy.deepcopy(config.get('model', {}))
training_config = copy.deepcopy(config.get('training', {}))
check_val_every_n_epoch = training_config.pop('check_val_every_n_epoch', None)
val_check_interval = training_config.pop('val_check_interval', None)
num_sanity_val_steps = training_config.pop('num_sanity_val_steps', 0)
epochs = training_config.pop('epochs', 50)
instrument_config = copy.deepcopy(config.get('instrument', []))
lambda_config = copy.deepcopy(config.get('lambda', {}))
normalization_config = copy.deepcopy(config.get('normalization', {}))
stokes_loss_config = copy.deepcopy(config.get('stokes_loss', {}))
physics_config = copy.deepcopy(config.get('physics'))
lambda_config, physics_config = split_stokes_and_physics_config(
    lambda_config, physics_config,
)
physics_domain = None
if physics_config is not None:
    physics_domain = build_spherical_collocation_domain(
        data_module.train_datasets,
        data_module.Rs_per_ds,
        physics_config.get('radius_range_Rs'),
    )

me_module = MESphericalModule(image_shape=data_module.image_shape, wavelength_config=data_module.wavelength_config,
                              model_config=model_config, normalization_config=normalization_config,
                              stokes_loss_config=stokes_loss_config,
                              physics_config=physics_config, physics_domain=physics_domain,
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
        'forward_models': me_module.forward_models,
        'velocity_correction_models': me_module.velocity_correction_models,
        'limb_correction_models': me_module.limb_correction_models,
        'artifact_correction_models': me_module.artifact_correction_models,
        'normalization': me_module.normalization,
        'model_config': model_config,
        'instrument_config': instrument_config,
        'normalization_config': normalization_config,
        'stokes_loss_config': me_module.stokes_loss_config,
        'stokes_lambda_config': me_module.stokes_lambda_config,
        'physics_config': me_module.physics_config,
        'spectral_response_files': data_module.spectral_response_files,
        'spectral_response_by_acquisition': data_module.spectral_response_by_acquisition,
        'stokes_normalization_by_instrument': data_module.stokes_normalization_by_instrument,
        'data_cache_signature': data_cache_signature,
    }, save_path)


save_callback = LambdaCallback(on_validation_epoch_end=save)

torch.set_float32_matmul_precision('medium')  # for A100 GPUs
torch.multiprocessing.set_sharing_strategy("file_system")

n_gpus = torch.cuda.device_count()

# from torchrun:
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))          # total processes (16)
LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", str(n_gpus)))  # procs on this node (4)

devices_per_node = max(1, LOCAL_WORLD_SIZE if n_gpus > 0 else 0)
num_nodes = max(1, WORLD_SIZE // devices_per_node)

trainer = Trainer(max_epochs=int(epochs),
                  logger=wandb_logger,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  devices=devices_per_node if n_gpus > 0 else None,
                  num_nodes=num_nodes,
                  strategy=DDPStrategy(find_unused_parameters=False) if (num_nodes > 1 or devices_per_node > 1) else 'auto',
                  num_sanity_val_steps=num_sanity_val_steps,
                  check_val_every_n_epoch=check_val_every_n_epoch,
                  val_check_interval=val_check_interval,
                  use_distributed_sampler=False,
                  gradient_clip_val=0.5,
                  callbacks=[checkpoint_callback, save_callback])

trainer.fit(me_module, data_module, ckpt_path='last')
