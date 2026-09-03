"""Differentiable formal solutions with explicit radiative-transfer paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn


@dataclass(frozen=True)
class OpticalDepthPath:
    """Plane-parallel transfer on vertical ``tau500`` with ray cosine ``mu``."""

    mu: torch.Tensor | float = 1.0


@dataclass(frozen=True)
class GeometricHeightPath:
    """Transfer through a top-to-bottom physical-height grid in metres."""

    height_m: torch.Tensor
    mu: torch.Tensor | float = 1.0


@dataclass(frozen=True)
class RayDistancePath:
    """Transfer through points whose distances along an exact ray are in metres."""

    distance_m: torch.Tensor


TransferPath: TypeAlias = OpticalDepthPath | GeometricHeightPath | RayDistancePath


def _validate_depth_grid(log_tau500: torch.Tensor, depth: int) -> torch.Tensor:
    grid = torch.as_tensor(log_tau500)
    if grid.ndim != 1 or grid.numel() != depth:
        raise ValueError(
            f"log_tau500 must have shape [{depth}], received {tuple(grid.shape)}."
        )
    if depth < 2:
        raise ValueError("At least two depth points are required.")
    if grid.dtype not in (torch.float32, torch.float64):
        raise TypeError("log_tau500 must use float32 or float64.")
    if not torch.isfinite(grid).all() or not torch.all(grid[1:] > grid[:-1]):
        raise ValueError("log_tau500 must be finite and strictly increase inward.")
    return grid


def _broadcast_depth_coordinates(
    coordinates,
    *,
    name: str,
    depth: int,
    batch_shape: torch.Size,
    like: torch.Tensor,
) -> torch.Tensor:
    """Broadcast finite physical coordinates without enforcing their ordering.

    Physical path coordinates can be trainable.  A transient folded or
    zero-width interval must therefore remain part of the differentiable
    forward calculation instead of terminating an optimization step.  The
    caller forms signed intervals from this tensor; in particular, this helper
    must not hide a reversal with an absolute value.
    """

    value = torch.as_tensor(coordinates, dtype=like.dtype, device=like.device)
    if value.shape == (depth,):
        value = torch.broadcast_to(value, (*batch_shape, depth))
    elif value.shape != (*batch_shape, depth):
        raise ValueError(
            f"{name} must have shape [{depth}] or {(*batch_shape, depth)}, "
            f"received {tuple(value.shape)}."
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _broadcast_mu(
    mu: torch.Tensor | float,
    batch_shape: torch.Size,
    like: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(mu, dtype=like.dtype, device=like.device)
    if value.ndim == len(batch_shape) + 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    try:
        value = torch.broadcast_to(value, batch_shape)
    except RuntimeError as error:
        raise ValueError(
            f"mu with shape {tuple(value.shape)} cannot broadcast to "
            f"batch shape {tuple(batch_shape)}."
        ) from error
    if not torch.isfinite(value).all() or torch.any((value <= 0) | (value > 1)):
        raise ValueError("mu must be finite and satisfy 0 < mu <= 1.")
    return value


class PolarizedFormalSolver(nn.Module):
    """Frozen-layer matrix-exponential formal solver.

    ``OpticalDepthPath`` expects a propagation matrix normalized per vertical
    ``tau500``. ``GeometricHeightPath`` expects the same normalization plus a
    separately supplied 5000-Angstrom reference extinction in inverse metres.
    ``RayDistancePath`` expects a dimensional propagation matrix in inverse
    metres. The caller must choose one path explicitly.
    """

    def forward(
        self,
        propagation_matrix: torch.Tensor,
        source_vector: torch.Tensor,
        log_tau500: torch.Tensor,
        *,
        path: TransferPath,
        bottom_boundary: torch.Tensor | None = None,
        reference_extinction_m1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return emergent Stokes vectors with shape ``[..., wavelength, 4]``."""

        matrix = torch.as_tensor(propagation_matrix)
        source = torch.as_tensor(
            source_vector, dtype=matrix.dtype, device=matrix.device
        )
        if matrix.ndim < 4 or matrix.shape[-2:] != (4, 4):
            raise ValueError(
                "propagation_matrix must have shape [..., depth, wavelength, 4, 4]."
            )
        if source.shape != matrix.shape[:-1]:
            raise ValueError(
                f"source_vector shape {tuple(source.shape)} does not match "
                f"propagation matrix shape {tuple(matrix.shape)}."
            )
        if not matrix.is_floating_point() or not source.is_floating_point():
            raise TypeError(
                "Propagation matrix and source vector must be floating point."
            )
        if matrix.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "Polarized transfer supports float32 and float64 tensors only."
            )
        if not torch.isfinite(matrix).all() or not torch.isfinite(source).all():
            raise ValueError("Propagation matrix and source vector must be finite.")

        depth = matrix.shape[-4]
        grid = _validate_depth_grid(log_tau500, depth).to(matrix)
        batch_shape = matrix.shape[:-4]

        if isinstance(path, OpticalDepthPath):
            if reference_extinction_m1 is not None:
                raise ValueError(
                    "reference_extinction_m1 is only valid for GeometricHeightPath."
                )
            tau = torch.pow(matrix.new_tensor(10.0), grid)
            if not torch.isfinite(tau).all() or not torch.all(tau[1:] > tau[:-1]):
                raise ValueError(
                    "log_tau500 cannot be represented as a finite, strictly "
                    "increasing optical-depth grid."
                )
            mu = _broadcast_mu(path.mu, batch_shape, matrix)
            interval = (tau[1:] - tau[:-1]).expand(*batch_shape, depth - 1)
            interval = interval / mu[..., None]
            working_matrix = matrix
        elif isinstance(path, GeometricHeightPath):
            if reference_extinction_m1 is None:
                raise ValueError(
                    "GeometricHeightPath requires reference_extinction_m1."
                )
            height = _broadcast_depth_coordinates(
                path.height_m,
                name="height_m",
                depth=depth,
                batch_shape=batch_shape,
                like=matrix,
            )
            mu = _broadcast_mu(path.mu, batch_shape, matrix)
            extinction = torch.as_tensor(
                reference_extinction_m1, dtype=matrix.dtype, device=matrix.device
            )
            if extinction.shape != (*batch_shape, depth):
                raise ValueError(
                    "reference_extinction_m1 must have shape "
                    f"{(*batch_shape, depth)}, received {tuple(extinction.shape)}."
                )
            if not torch.isfinite(extinction).all() or torch.any(extinction <= 0):
                raise ValueError("reference_extinction_m1 must be finite and positive.")
            working_matrix = matrix * extinction[..., :, None, None, None]
            interval = (height[..., :-1] - height[..., 1:]) / mu[..., None]
        elif isinstance(path, RayDistancePath):
            if reference_extinction_m1 is not None:
                raise ValueError(
                    "RayDistancePath uses a dimensional propagation matrix and "
                    "does not accept reference_extinction_m1."
                )
            distance = _broadcast_depth_coordinates(
                path.distance_m,
                name="distance_m",
                depth=depth,
                batch_shape=batch_shape,
                like=matrix,
            )
            working_matrix = matrix
            interval = distance[..., 1:] - distance[..., :-1]
        else:
            raise TypeError(
                "path must be OpticalDepthPath, GeometricHeightPath, or RayDistancePath."
            )

        if bottom_boundary is None:
            stokes = source[..., -1, :, :]
        else:
            stokes = torch.as_tensor(
                bottom_boundary, dtype=matrix.dtype, device=matrix.device
            )
        expected_boundary = (*batch_shape, matrix.shape[-3], 4)
        if stokes.shape != expected_boundary:
            raise ValueError(
                f"bottom_boundary must have shape {expected_boundary}, "
                f"received {tuple(stokes.shape)}."
            )
        if not torch.isfinite(stokes).all():
            raise ValueError("bottom_boundary must contain only finite values.")

        midpoint_matrix = 0.5 * (
            working_matrix[..., :-1, :, :, :] + working_matrix[..., 1:, :, :, :]
        )
        attenuation = torch.matrix_exp(
            -midpoint_matrix * interval[..., None, None, None]
        )
        midpoint_source = 0.5 * (source[..., :-1, :, :] + source[..., 1:, :, :])
        for upper in range(depth - 2, -1, -1):
            layer_source = midpoint_source[..., upper, :, :]
            stokes = layer_source + torch.matmul(
                attenuation[..., upper, :, :, :],
                (stokes - layer_source).unsqueeze(-1),
            ).squeeze(-1)
        return stokes


def scalar_formal_solution(
    opacity_ratio: torch.Tensor,
    source_function: torch.Tensor,
    log_tau500: torch.Tensor,
    *,
    path: OpticalDepthPath,
    bottom_boundary: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar formal solution on an explicitly selected optical-depth path."""

    if not isinstance(path, OpticalDepthPath):
        raise TypeError("scalar_formal_solution requires OpticalDepthPath.")
    opacity = torch.as_tensor(opacity_ratio)
    source = torch.as_tensor(
        source_function, dtype=opacity.dtype, device=opacity.device
    )
    if opacity.shape != source.shape or opacity.ndim < 2:
        raise ValueError(
            "opacity_ratio and source_function must share shape [..., depth, wavelength]."
        )
    if not opacity.is_floating_point() or not source.is_floating_point():
        raise TypeError("Opacity and source tensors must be floating point.")
    if opacity.dtype not in (torch.float32, torch.float64):
        raise TypeError("Scalar transfer supports float32 and float64 tensors only.")
    if (
        not torch.isfinite(opacity).all()
        or not torch.isfinite(source).all()
        or torch.any(opacity < 0)
    ):
        raise ValueError(
            "Opacity must be finite and non-negative, and source_function must be finite."
        )
    depth = opacity.shape[-2]
    grid = _validate_depth_grid(log_tau500, depth).to(opacity)
    tau = torch.pow(opacity.new_tensor(10.0), grid)
    if not torch.isfinite(tau).all() or not torch.all(tau[1:] > tau[:-1]):
        raise ValueError(
            "log_tau500 cannot be represented as a finite, strictly increasing "
            "optical-depth grid."
        )
    batch_shape = opacity.shape[:-2]
    mu = _broadcast_mu(path.mu, batch_shape, opacity)
    if bottom_boundary is None:
        intensity = source[..., -1, :]
    else:
        intensity = torch.as_tensor(
            bottom_boundary, dtype=opacity.dtype, device=opacity.device
        )
    if intensity.shape != (*batch_shape, opacity.shape[-1]):
        raise ValueError("bottom_boundary has an incompatible shape.")
    if not torch.isfinite(intensity).all():
        raise ValueError("bottom_boundary must contain only finite values.")

    for upper in range(depth - 2, -1, -1):
        lower = upper + 1
        mean_opacity = 0.5 * (opacity[..., upper, :] + opacity[..., lower, :])
        delta_tau = (tau[lower] - tau[upper]) / mu
        attenuation = torch.exp(-mean_opacity * delta_tau[..., None])
        mean_source = 0.5 * (source[..., upper, :] + source[..., lower, :])
        intensity = attenuation * intensity + (1.0 - attenuation) * mean_source
    return intensity


__all__ = [
    "GeometricHeightPath",
    "OpticalDepthPath",
    "PolarizedFormalSolver",
    "RayDistancePath",
    "TransferPath",
    "scalar_formal_solution",
]
