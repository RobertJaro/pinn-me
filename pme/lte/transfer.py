"""Differentiable formal solutions of polarized radiative transfer."""

from __future__ import annotations

import torch
from torch import nn


def _validate_depth_grid(log_tau500: torch.Tensor, depth: int) -> torch.Tensor:
    grid = torch.as_tensor(log_tau500)
    if grid.ndim != 1 or grid.numel() != depth:
        raise ValueError(
            f"log_tau500 must have shape [{depth}], received {tuple(grid.shape)}"
        )
    if depth < 2:
        raise ValueError("At least two depth points are required")
    if not bool(torch.all(grid[1:] > grid[:-1])):
        raise ValueError("log_tau500 must increase from the top to the bottom boundary")
    return grid


def _broadcast_mu(mu: torch.Tensor | float, batch_shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(mu, dtype=like.dtype, device=like.device)
    if value.ndim == len(batch_shape) + 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    try:
        value = torch.broadcast_to(value, batch_shape)
    except RuntimeError as error:
        raise ValueError(
            f"mu with shape {tuple(value.shape)} cannot broadcast to batch shape {tuple(batch_shape)}"
        ) from error
    if torch.any((value <= 0) | (value > 1)):
        raise ValueError("mu must satisfy 0 < mu <= 1")
    return value


class PolarizedFormalSolver(nn.Module):
    """Frozen-layer matrix-exponential formal solver.

    Within every interval the propagation matrix and LTE source vector are
    evaluated at the midpoint.  The matrix exponential then solves that layer
    exactly.  Midpoint interpolation makes the overall method second order in
    depth while retaining a compact, fully differentiable reference
    implementation.  It is deliberately kept independent of atmosphere, EOS,
    and neural-network code.

    The transfer equation is

    ``mu dI/dtau_500 = K_lambda (I - S)``

    with optical depth increasing inward.  ``K_lambda`` is therefore normalized
    per unit vertical ``tau_500``.
    """

    def forward(
        self,
        propagation_matrix: torch.Tensor,
        source_vector: torch.Tensor,
        log_tau500: torch.Tensor,
        mu: torch.Tensor | float = 1.0,
        bottom_boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return emergent Stokes vectors with shape ``[..., wavelength, 4]``.

        Parameters
        ----------
        propagation_matrix:
            Tensor with shape ``[..., depth, wavelength, 4, 4]``.
        source_vector:
            Tensor with shape ``[..., depth, wavelength, 4]``.  In LTE this is
            normally ``(B_lambda, 0, 0, 0)``.
        log_tau500:
            Common one-dimensional depth grid ordered top to bottom.
        mu:
            Ray cosine, scalar or broadcastable over the leading batch axes.
        bottom_boundary:
            Optional incident Stokes vector ``[..., wavelength, 4]``.  The
            bottom source vector is used when omitted.
        """

        matrix = torch.as_tensor(propagation_matrix)
        source = torch.as_tensor(source_vector)
        if matrix.ndim < 4 or matrix.shape[-2:] != (4, 4):
            raise ValueError(
                "propagation_matrix must have shape [..., depth, wavelength, 4, 4]"
            )
        if source.shape != matrix.shape[:-1]:
            raise ValueError(
                f"source_vector shape {tuple(source.shape)} does not match "
                f"propagation matrix shape {tuple(matrix.shape)}"
            )
        if not matrix.is_floating_point() or not source.is_floating_point():
            raise TypeError("Propagation matrix and source vector must be floating point")

        depth = matrix.shape[-4]
        grid = _validate_depth_grid(log_tau500, depth).to(matrix)
        tau = torch.pow(matrix.new_tensor(10.0), grid)
        batch_shape = matrix.shape[:-4]
        ray_mu = _broadcast_mu(mu, batch_shape, matrix)

        stokes = source[..., -1, :, :] if bottom_boundary is None else bottom_boundary
        expected_boundary = matrix.shape[:-4] + matrix.shape[-3:-2] + (4,)
        if stokes.shape != expected_boundary:
            raise ValueError(
                f"bottom boundary must have shape {expected_boundary}, got {tuple(stokes.shape)}"
            )

        identity = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
        identity = identity.expand(*matrix.shape[:-4], matrix.shape[-3], 4, 4)
        for upper in range(depth - 2, -1, -1):
            lower = upper + 1
            delta_tau = tau[lower] - tau[upper]
            midpoint_matrix = 0.5 * (
                matrix[..., upper, :, :, :] + matrix[..., lower, :, :, :]
            )
            path_scale = delta_tau / ray_mu
            attenuation = torch.matrix_exp(
                -midpoint_matrix * path_scale[..., None, None, None]
            )
            midpoint_source = 0.5 * (
                source[..., upper, :, :] + source[..., lower, :, :]
            )
            stokes = (
                torch.matmul(attenuation, stokes.unsqueeze(-1)).squeeze(-1)
                + torch.matmul(identity - attenuation, midpoint_source.unsqueeze(-1)).squeeze(-1)
            )
        return stokes


def scalar_formal_solution(
    opacity_ratio: torch.Tensor,
    source_function: torch.Tensor,
    log_tau500: torch.Tensor,
    mu: torch.Tensor | float = 1.0,
    bottom_boundary: torch.Tensor | None = None,
) -> torch.Tensor:
    """Efficient scalar counterpart of :class:`PolarizedFormalSolver`.

    Inputs have shape ``[..., depth, wavelength]`` and the result has shape
    ``[..., wavelength]``.  This is used for continuum normalization and scalar
    validation without constructing diagonal 4x4 matrices.
    """

    opacity = torch.as_tensor(opacity_ratio)
    source = torch.as_tensor(source_function)
    if opacity.shape != source.shape or opacity.ndim < 2:
        raise ValueError("opacity_ratio and source_function must share shape [..., depth, wavelength]")
    depth = opacity.shape[-2]
    grid = _validate_depth_grid(log_tau500, depth).to(opacity)
    tau = torch.pow(opacity.new_tensor(10.0), grid)
    batch_shape = opacity.shape[:-2]
    ray_mu = _broadcast_mu(mu, batch_shape, opacity)
    intensity = source[..., -1, :] if bottom_boundary is None else bottom_boundary
    if intensity.shape != opacity.shape[:-2] + opacity.shape[-1:]:
        raise ValueError("bottom_boundary has an incompatible shape")

    for upper in range(depth - 2, -1, -1):
        lower = upper + 1
        mean_opacity = 0.5 * (opacity[..., upper, :] + opacity[..., lower, :])
        delta_tau = (tau[lower] - tau[upper]) / ray_mu
        attenuation = torch.exp(-mean_opacity * delta_tau[..., None])
        mean_source = 0.5 * (source[..., upper, :] + source[..., lower, :])
        intensity = attenuation * intensity + (1.0 - attenuation) * mean_source
    return intensity
