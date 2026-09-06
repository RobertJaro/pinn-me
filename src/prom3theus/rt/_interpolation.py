"""Shared differentiable interpolation primitives for runtime lookup tables."""

from __future__ import annotations

import torch


def uniform_axis_coordinate(samples: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Map a uniform stored axis to node coordinates without endpoint drift."""

    return (samples - axis[0]) * (axis.numel() - 1) / (axis[-1] - axis[0])


def clamped_catmull_rom(
    samples: torch.Tensor,
    weight: torch.Tensor,
    *,
    dimension: int = -1,
) -> torch.Tensor:
    """Interpolate four samples with the established clamped Catmull--Rom rule.

    Callers clamp neighborhood indices at table boundaries. Consequently the
    outer derivative is one half of the corresponding one-sided secant. Keeping
    this primitive shared preserves the historical STiC opacity interpolation
    exactly while allowing the EoS continuation to match its actual tangent.
    """

    p0, p1, p2, p3 = samples.unbind(dim=dimension)
    while weight.ndim < p1.ndim:
        weight = weight[..., None]
    return p1 + 0.5 * weight * (
        p2
        - p0
        + weight
        * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + weight * (3.0 * (p1 - p2) + p3 - p0))
    )


__all__ = ["clamped_catmull_rom", "uniform_axis_coordinate"]
