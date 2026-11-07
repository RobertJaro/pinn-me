"""Shared bounded jitter, contribution-guided sampling, and sample merging."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prom3theus.core.validation import require_tensor


def reference_height_grid(
    atmosphere_model, reference_log_tau500, bounds=(-5.0, 1.0), *,
    height_sampling="reference",
):
    """Map Stokes reference coordinates to physical heights in the line shell.

    The reference coordinate labels quadrature placement only, not the learned
    optical depth. "reference" uses the FALC-shaped distribution; "uniform"
    distributes the same coordinate grid uniformly in physical height, giving
    depressed photospheric layers coverage even before their opacity is learned.
    The atmosphere and ray tracer consume metres in both modes.
    """
    q = torch.as_tensor(reference_log_tau500).to(atmosphere_model.solar_radius_m)
    lower, upper = bounds
    if not lower < 0 < upper:
        raise ValueError("Reference sampling bounds must straddle zero.")
    tolerance = 2 * torch.finfo(q.dtype).eps
    if (
        not torch.isfinite(q).all()
        or torch.any(q < lower - tolerance)
        or torch.any(q > upper + tolerance)
    ):
        raise ValueError(
            "Stokes reference coordinates must stay inside the sampling bounds."
        )
    outer, inner = atmosphere_model.line_formation_height_bounds_megameter
    if not outer > 0 > inner:
        raise ValueError(
            "Stokes sampling requires line-formation heights straddling zero."
        )
    if height_sampling == "uniform":
        fraction = (q - lower) / (upper - lower)
        return (outer + (inner - outer) * fraction) * 1e6
    if height_sampling != "reference":
        raise ValueError("height_sampling must be 'reference' or 'uniform'.")
    reference = atmosphere_model.reference_atmosphere
    endpoints = reference.height_from_log_tau(q.new_tensor(bounds))
    height = reference.height_from_log_tau(q)
    return torch.where(
        q <= 0,
        height * (outer * 1e6 / endpoints[0]),
        height * (inner * 1e6 / endpoints[1]),
    )


@dataclass(frozen=True, slots=True)
class DepthRefinement:
    """Shared contribution-guided fine-ray sampling configuration."""

    enabled: bool
    sample_count: int
    uniform_weight_floor: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("DepthRefinement.enabled must be boolean.")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("DepthRefinement.sample_count must be a positive integer.")
        if not 0.0 <= self.uniform_weight_floor <= 1.0:
            raise ValueError("DepthRefinement.uniform_weight_floor must lie in [0, 1].")


def jitter_depth_grid(base: torch.Tensor, *, randomize: bool) -> torch.Tensor:
    """Return an ordered grid with bounded stochastic interior jitter."""

    if base.ndim < 1 or base.shape[-1] < 2:
        raise ValueError(
            "The depth grid must be have at least two points on its last axis."
        )
    if not torch.is_floating_point(base) or not torch.isfinite(base).all():
        raise ValueError("The depth grid must contain finite floating-point values.")
    intervals = base[..., 1:] - base[..., :-1]
    if torch.any(intervals <= 0):
        raise ValueError("The depth grid must be strictly increasing.")
    if not randomize or base.shape[-1] <= 2:
        return base
    interior = base[..., 1:-1]
    # Bounding each displacement by both adjacent intervals keeps arbitrary
    # non-uniform grids ordered even when neighboring points move toward one
    # another by their maximum amount.
    maximum_shift = 0.4 * torch.minimum(intervals[..., :-1], intervals[..., 1:])
    shift = (2.0 * torch.rand_like(interior) - 1.0) * maximum_shift
    return torch.cat((base[..., :1], interior + shift, base[..., -1:]), dim=-1)


def _unit_mass(contribution: torch.Tensor) -> torch.Tensor:
    """Normalize each channel's interval masses to sum to one."""

    total = contribution.sum(dim=-1, keepdim=True)
    tiny = torch.finfo(contribution.dtype).tiny
    return torch.where(
        total > tiny,
        contribution / total.clamp_min(tiny),
        torch.full_like(contribution, 1.0 / contribution.shape[-1]),
    )


@torch.no_grad()
def importance_fine_distances(
    alpha500: torch.Tensor,
    distance_m: torch.Tensor,
    fine_sample_count: int,
    uniform_weight_floor: float,
    *,
    line_extinction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Place deterministic fine samples using detached contribution weights.

    ``alpha500`` alone concentrates refinement where the continuum forms.  A
    line core reaches unit optical depth while ``tau500`` is still orders of
    magnitude below one, so the layers that dominate the polarized signal
    receive almost no continuum contribution weight.  Supplying the rest-frame
    line-centre extinction adds a second proposal built from
    ``alpha500 + line_extinction``; the two are normalized separately and
    averaged, so line-core coverage is gained without giving up the continuum
    and wing-forming layers.
    """

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
    if line_extinction is not None:
        if line_extinction.shape != alpha500.shape:
            raise ValueError("line_extinction must share the alpha500 shape.")
        if not torch.is_floating_point(line_extinction):
            raise TypeError("line_extinction must be a floating-point tensor.")
        if not torch.isfinite(line_extinction).all() or torch.any(line_extinction < 0):
            raise ValueError("Line extinction must be finite and non-negative.")
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
    # Both proposals share one channel axis so neither the optical-depth
    # integral nor the quantile search grows a Python loop over channels.
    extinction = alpha500[..., None, :]
    if line_extinction is not None:
        extinction = torch.stack(
            (alpha500, alpha500 + torch.gather(line_extinction, -1, order)),
            dim=-2,
        )
    interval_m = (distance_m[..., 1:] - distance_m[..., :-1])[..., None, :]
    delta_tau = 0.5 * (extinction[..., 1:] + extinction[..., :-1]) * interval_m
    tau_edge = torch.cat(
        (torch.zeros_like(delta_tau[..., :1]), torch.cumsum(delta_tau, dim=-1)),
        dim=-1,
    )
    # The attenuation integrated over an interval is the transmittance drop.
    # Factor it instead of subtracting two exponentials: in an optically thin
    # ray both exponentials round to one in float32, silently discarding the
    # opacity proposal and falling back to a uniform interval distribution.
    # exp(-inf) and expm1(-inf) also retain finite weights for opaque layers.
    contribution = (
        torch.exp(-tau_edge[..., :-1]) * (-torch.expm1(-delta_tau))
    ).detach()
    # Each channel carries unit mass before averaging.  Without this an
    # optically thin channel would be outvoted by an opaque one purely through
    # its smaller total transmittance drop rather than its sampling relevance.
    mixed = _unit_mass(contribution).mean(dim=-2)
    return contribution_fine_distances(
        mixed, distance_m, fine_sample_count, uniform_weight_floor, randomize=False
    )


@torch.no_grad()
def stokes_interval_proposal(propagation, intervals):
    """Sum normalized scalar attenuation masses over detector wavelengths.

    Eta_I includes continuum and Doppler/Zeeman-shifted line extinction. Each
    wavelength contributes equally to the proposal so continuum and line-core
    formation regions both receive samples. Polarized transfer still uses full K.
    """
    extinction = propagation[..., 0, 0].transpose(-1, -2)
    delta_tau = (0.5 * extinction[..., :-1] + 0.5 * extinction[..., 1:]) * intervals[..., None, :]
    tau_before = torch.cat((torch.zeros_like(delta_tau[..., :1]),
                            delta_tau[..., :-1].cumsum(dim=-1)), dim=-1)
    masses = torch.exp(-tau_before) * (-torch.expm1(-delta_tau))
    return _unit_mass(masses).sum(dim=-2)


@torch.no_grad()
def contribution_fine_distances(
    contribution,
    distance_m,
    fine_sample_count,
    uniform_weight_floor,
    *,
    randomize=False,
    minimum_separation=None,
):
    """Sample detached interval masses at fixed or stratified-random quantiles.

    Optional minimum_separation reserves space between fine samples as well as
    coarse edges; callers must supply coarse intervals wide enough to hold all
    requested fine points at that separation.
    """
    if contribution.shape != (*distance_m.shape[:-1], distance_m.shape[-1] - 1):
        raise ValueError("Contribution weights must describe each ray interval.")
    if (
        type(fine_sample_count) is not int
        or fine_sample_count < 1
        or not 0 <= uniform_weight_floor <= 1
    ):
        raise ValueError("Invalid fine sample count or proposal floor.")
    require_tensor(torch.isfinite(contribution).all() & (contribution >= 0).all(),
                   "Contribution weights must be finite and nonnegative.")
    require_tensor(torch.isfinite(distance_m).all() & (distance_m.diff(dim=-1) > 0).all(),
                   "Proposal distances must be finite and strictly increasing.")
    interval_count = contribution.shape[-1]
    normalized = _unit_mass(contribution)
    probability = (
        1.0 - uniform_weight_floor
    ) * normalized + uniform_weight_floor / interval_count
    cdf = torch.cumsum(probability, dim=-1)
    strata = torch.arange(
        fine_sample_count, dtype=distance_m.dtype, device=distance_m.device
    )
    shape = (*distance_m.shape[:-1], fine_sample_count)
    offset = (
        torch.rand(shape, dtype=distance_m.dtype, device=distance_m.device)
        if randomize
        else 0.5
    )
    quantiles = ((strata + offset) / fine_sample_count).expand(shape)
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
    if minimum_separation is not None:
        # Quantiles, hence interval IDs, are sorted. Reserve a gap for each
        # fine point and both coarse endpoints, including coincident quantiles.
        # The caller's coarse jitter must reserve room for these gaps.
        left = torch.searchsorted(interval.contiguous(), interval.contiguous(), right=False)
        right = torch.searchsorted(interval.contiguous(), interval.contiguous(), right=True)
        rank = strata - left + 1
        gap = minimum_separation / (upper - lower)
        fraction = fraction.clamp(0, 1) * (1 - (right - left + 1) * gap) + rank * gap
    fine = lower + fraction * (upper - lower)
    # Rounded CDF values can make a quantile coincide exactly with an interval
    # edge in float32.  A fine point on that edge duplicates a coarse point and
    # creates a zero-width transfer layer.  Keep refinement points in the open
    # interval using the nearest representable values; this changes only the
    # rounding of edge cases, not the opacity-weighted sampling distribution.
    lower_open = torch.nextafter(lower, upper)
    upper_open = torch.nextafter(upper, lower)
    return torch.maximum(lower_open, torch.minimum(fine, upper_open))


def merge_depth_samples(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    """Merge depth samples with arbitrary trailing physical/spectral axes."""
    depth_axis = order.ndim - 1
    combined = torch.cat((coarse, fine), dim=depth_axis)
    trailing = combined.shape[order.ndim:]
    index = order.reshape(*order.shape, *((1,) * len(trailing))).expand(*order.shape, *trailing)
    return torch.gather(combined, depth_axis, index)


__all__ = [
    "DepthRefinement",
    "contribution_fine_distances",
    "importance_fine_distances",
    "jitter_depth_grid",
    "merge_depth_samples",
    "stokes_interval_proposal",
]
