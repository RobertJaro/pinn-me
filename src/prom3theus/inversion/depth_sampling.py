"""Depth-grid sampling and opacity-guided refinement."""

from __future__ import annotations

import torch


def jitter_depth_grid(base: torch.Tensor, *, randomize: bool) -> torch.Tensor:
    """Return an ordered grid with bounded stochastic interior jitter."""

    if base.ndim != 1 or base.numel() < 2:
        raise ValueError(
            "The depth grid must be one-dimensional with at least two points."
        )
    if not torch.is_floating_point(base) or not torch.isfinite(base).all():
        raise ValueError("The depth grid must contain finite floating-point values.")
    intervals = base[1:] - base[:-1]
    if torch.any(intervals <= 0):
        raise ValueError("The depth grid must be strictly increasing.")
    if not randomize or base.numel() <= 2:
        return base
    interior = base[1:-1]
    # Bounding each displacement by both adjacent intervals keeps arbitrary
    # non-uniform grids ordered even when neighboring points move toward one
    # another by their maximum amount.
    maximum_shift = 0.4 * torch.minimum(intervals[:-1], intervals[1:])
    shift = (2.0 * torch.rand_like(interior) - 1.0) * maximum_shift
    return torch.cat((base[:1], interior + shift, base[-1:]))


@torch.no_grad()
def importance_fine_distances(
    alpha500: torch.Tensor,
    distance_m: torch.Tensor,
    fine_sample_count: int,
    uniform_weight_floor: float,
) -> torch.Tensor:
    """Place deterministic fine samples using detached contribution weights."""

    if alpha500.shape != distance_m.shape or alpha500.ndim < 2:
        raise ValueError(
            "alpha500 and distance_m must have matching [..., depth] shapes."
        )
    if not torch.is_floating_point(alpha500) or not torch.is_floating_point(distance_m):
        raise TypeError(
            "Refinement opacity and distance tensors must be floating point."
        )
    if not torch.isfinite(alpha500).all() or torch.any(alpha500 < 0):
        raise ValueError("Reference extinction must be finite and non-negative.")
    if not torch.isfinite(distance_m).all():
        raise ValueError("Ray distances must be finite.")
    if (
        type(fine_sample_count) is not int
        or fine_sample_count < 1
        or alpha500.shape[-1] < 2
    ):
        raise ValueError(
            "Refinement requires a positive sample count and two coarse points."
        )
    if not 0.0 <= uniform_weight_floor <= 1.0:
        raise ValueError("uniform_weight_floor must lie between zero and one.")

    # This proposal is detached from the trainable forward graph.  Ordering it
    # here prevents a transient folded coarse path from aborting refinement,
    # while gathering the opacity with the same permutation preserves their
    # physical pairing.  The formal solver itself still receives signed path
    # intervals and never replaces them with absolute distances.
    distance_m, order = torch.sort(distance_m, dim=-1, stable=True)
    alpha500 = torch.gather(alpha500, -1, order)
    interval_m = distance_m[..., 1:] - distance_m[..., :-1]
    delta_tau = 0.5 * (alpha500[..., 1:] + alpha500[..., :-1]) * interval_m
    tau_edge = torch.cat(
        (torch.zeros_like(delta_tau[..., :1]), torch.cumsum(delta_tau, dim=-1)),
        dim=-1,
    )
    # The attenuation integrated over an interval is the transmittance drop.
    # This form is both more accurate than a midpoint approximation and stays
    # finite when a physically opaque interval has very large optical depth.
    contribution = (
        torch.exp(-tau_edge[..., :-1].clamp_max(80.0))
        - torch.exp(-tau_edge[..., 1:].clamp_max(80.0))
    ).detach()
    interval_count = contribution.shape[-1]
    total = contribution.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        total > torch.finfo(contribution.dtype).tiny,
        contribution / total.clamp_min(torch.finfo(contribution.dtype).tiny),
        torch.full_like(contribution, 1.0 / interval_count),
    )
    probability = (
        1.0 - uniform_weight_floor
    ) * normalized + uniform_weight_floor / interval_count
    cdf = torch.cumsum(probability, dim=-1)
    quantiles = (
        torch.arange(
            fine_sample_count,
            dtype=distance_m.dtype,
            device=distance_m.device,
        )
        + 0.5
    ) / fine_sample_count
    quantiles = quantiles.expand(*distance_m.shape[:-1], fine_sample_count)
    interval = torch.searchsorted(
        cdf.contiguous(), quantiles.contiguous(), right=False
    ).clamp_max(interval_count - 1)
    before = torch.cat((torch.zeros_like(cdf[..., :1]), cdf[..., :-1]), dim=-1)
    lower_cdf = torch.gather(before, -1, interval)
    upper_cdf = torch.gather(cdf, -1, interval)
    fraction = (quantiles - lower_cdf) / (upper_cdf - lower_cdf).clamp_min(
        torch.finfo(distance_m.dtype).eps
    )
    lower = torch.gather(distance_m[..., :-1], -1, interval)
    upper = torch.gather(distance_m[..., 1:], -1, interval)
    fine = lower + fraction * (upper - lower)
    # Rounded CDF values can make a quantile coincide exactly with an interval
    # edge in float32.  A fine point on that edge duplicates a coarse point and
    # creates a zero-width transfer layer.  Keep refinement points in the open
    # interval using the nearest representable values; this changes only the
    # rounding of edge cases, not the opacity-weighted sampling distribution.
    lower_open = torch.nextafter(lower, upper)
    upper_open = torch.nextafter(upper, lower)
    return torch.maximum(lower_open, torch.minimum(fine, upper_open))


@torch.no_grad()
def refined_distances(
    alpha500: torch.Tensor,
    distance_m: torch.Tensor,
    fine_sample_count: int,
    uniform_weight_floor: float,
) -> torch.Tensor:
    """Return a merged and ordered coarse-to-fine ray grid."""

    fine = importance_fine_distances(
        alpha500,
        distance_m,
        fine_sample_count,
        uniform_weight_floor,
    )
    return torch.sort(torch.cat((distance_m, fine), dim=-1), dim=-1).values


def merge_depth_samples(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    """Merge scalar or vector samples along their depth dimension."""

    if coarse.ndim == order.ndim:
        return torch.gather(torch.cat((coarse, fine), dim=-1), -1, order)
    if coarse.ndim == order.ndim + 1:
        combined = torch.cat((coarse, fine), dim=-2)
        index = order[..., None].expand(*order.shape, combined.shape[-1])
        return torch.gather(combined, -2, index)
    raise ValueError("Depth samples must be scalar or vector valued.")


__all__ = [
    "importance_fine_distances",
    "jitter_depth_grid",
    "merge_depth_samples",
    "refined_distances",
]
