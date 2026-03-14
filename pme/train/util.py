import os
import tempfile

import torch
import wandb
import yaml
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from pme.data.differential_rotation import carrington_rotation_rate, differential_rotation_rate
from pme.data.util import cartesian_to_spherical, spherical_to_cartesian


def load_yaml_config(yaml_config_file, overwrite_args=None):
    overwrite_args = [] if overwrite_args is None else overwrite_args
    assert all([k.startswith('--') for k in overwrite_args[::2]]), \
        'Only accept --config and overwrite arguments (must start with --)'
    overwrite_args = {k.replace('--', ''): v for k, v in zip(overwrite_args[::2], overwrite_args[1::2])}
    with open(yaml_config_file) as f:
        config_str = f.read()
    for overwrite_key, overwrite_value in overwrite_args.items():
        config_str = config_str.replace('{%s}' % overwrite_key, overwrite_value)
    config = yaml.safe_load(config_str)
    return config


def atan2_safe(numerator, denominator):
    epsilon = 1e-7
    nudge = (denominator == 0) * epsilon
    denominator = denominator + nudge
    out = torch.atan2(numerator, denominator)
    return out


def acos_safe(x):
    epsilon = 1e-7
    nudge_pos = (x == 1) * epsilon
    nudge_neg = (x == -1) * epsilon
    x = x - nudge_pos + nudge_neg
    out = torch.acos(x)
    return out


@rank_zero_only
def log_wandb_image(fig, name: str, dpi: int = 100):
    """
    Save a matplotlib figure as a JPEG to a temporary file and log it to Weights & Biases.

    Parameters:
        fig (matplotlib.figure.Figure): The matplotlib figure to log.
        name (str): The wandb key under which the image is logged.
        dpi (int): Resolution of the saved image. Default is 150.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        tmpfile_path = tmpfile.name

    try:
        fig.savefig(tmpfile_path, format='jpg', dpi=dpi, bbox_inches='tight', facecolor='white')
        wandb.log({name: wandb.Image(tmpfile_path)}, commit=False)
    finally:
        os.remove(tmpfile_path)


def get_random_coordinates(coords: Tensor, n_samples: int = 4096) -> Tensor:
    spherical_coords = cartesian_to_spherical(coords[..., 1:], torch)
    # get coordinate range for sampling points
    min_t, max_t = torch.min(coords[..., 0]), torch.max(coords[..., 0])
    min_r, max_r = 1.0, 1.01  # define a shell of 0.01 Rs around the sun
    min_th, max_th = torch.min(spherical_coords[..., 1]), torch.max(spherical_coords[..., 1])
    min_phi, max_phi = torch.min(spherical_coords[..., 2]), torch.max(spherical_coords[..., 2])

    # create random sampling points
    random_coords = torch.rand((n_samples, 4), dtype=torch.float32, device=coords.device, requires_grad=True)
    # scale random coordinates in spherical coordinates
    random_coords_r = (max_r - min_r) * random_coords[..., 1] + min_r
    random_coords_th = (max_th - min_th) * random_coords[..., 2] + min_th
    random_coords_phi = (max_phi - min_phi) * random_coords[..., 3] + min_phi
    random_coords_spherical = torch.stack([random_coords_r, random_coords_th, random_coords_phi], dim=-1)
    random_coords_cartesian = spherical_to_cartesian(random_coords_spherical, torch)

    # scale random coordinates time
    random_coords_t = (max_t - min_t) * random_coords[..., 0] + min_t

    # combine time and cartesian coordinates
    random_coords = torch.cat([random_coords_t[..., None], random_coords_cartesian], dim=-1)
    return random_coords


def random_time_shift_coords(coords, seconds_per_dt, scale=0.1, ):
    time_shift = torch.randn_like(coords[..., 0:1]) * scale  # small random shift
    spherical_coords = cartesian_to_spherical(coords[..., 1:], torch)
    lat = torch.pi / 2 - spherical_coords[..., 1:2]
    lon = spherical_coords[..., 2:3]
    omega_carrington = carrington_rotation_rate()
    omega_differential = differential_rotation_rate(lat, torch)
    lon_shift = (omega_differential - omega_carrington) * time_shift * seconds_per_dt
    spherical_coords_shift = torch.cat([spherical_coords[..., 0:1], spherical_coords[..., 1:2],
                                        lon + lon_shift], dim=-1)
    cartesian_coords_shift = spherical_to_cartesian(spherical_coords_shift, torch)
    coords_shift = torch.cat([coords[..., 0:1] + time_shift, cartesian_coords_shift], dim=-1)
    return coords_shift
