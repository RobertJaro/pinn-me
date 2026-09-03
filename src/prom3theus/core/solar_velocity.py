"""Solar-frame velocity terms shared by inversion, export, and validation."""

from __future__ import annotations

import math

import numpy as np
import torch


CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS = 25.38
CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S = (
    2.0 * math.pi / (CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS * 86_400.0)
)


def carrington_rotation_velocity_cartesian(
    position_m, backend=None, angular_velocity_rad_per_s=None
):
    """Return ``Omega_Carrington x r`` in Carrington Cartesian coordinates.

    The rotation axis is positive Carrington ``Z`` and the result is positive
    in the direction of increasing Carrington longitude. The input and output
    are in metres and metres per second, respectively.
    """

    if backend is None:
        backend = torch if isinstance(position_m, torch.Tensor) else np
    if backend is torch:
        position = torch.as_tensor(position_m)
        if not position.is_floating_point():
            position = position.to(torch.get_default_dtype())
        if position.shape[-1:] != (3,) or not torch.isfinite(position).all():
            raise ValueError("position_m must end in a finite Cartesian three-vector.")
        omega = torch.as_tensor(
            CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
            if angular_velocity_rad_per_s is None
            else angular_velocity_rad_per_s,
            dtype=position.dtype,
            device=position.device,
        )
        if omega.numel() != 1 or not torch.isfinite(omega):
            raise ValueError("angular_velocity_rad_per_s must be one finite scalar.")
        zeros = torch.zeros_like(position[..., 0])
        return torch.stack(
            (-omega * position[..., 1], omega * position[..., 0], zeros), dim=-1
        )
    if backend is np:
        position = np.asarray(position_m)
        if not np.issubdtype(position.dtype, np.floating):
            position = position.astype(np.float64)
        if position.shape[-1:] != (3,) or not np.isfinite(position).all():
            raise ValueError("position_m must end in a finite Cartesian three-vector.")
        zeros = np.zeros_like(position[..., 0])
        omega = (
            CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
            if angular_velocity_rad_per_s is None
            else np.asarray(angular_velocity_rad_per_s, dtype=position.dtype)
        )
        if np.size(omega) != 1 or not np.isfinite(omega).all():
            raise ValueError("angular_velocity_rad_per_s must be one finite scalar.")
        return np.stack(
            (-omega * position[..., 1], omega * position[..., 0], zeros), axis=-1
        )
    raise TypeError("backend must be numpy or torch.")


__all__ = [
    "CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S",
    "CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS",
    "carrington_rotation_velocity_cartesian",
]
