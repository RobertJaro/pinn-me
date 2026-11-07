"""Differentiable polarized LTE transfer with an ordered parallel reduction."""

from __future__ import annotations

import torch

from .polarized_exponential import polarized_matrix_exp
from prom3theus.core.validation import require_tensor


def affine_transfer_reduce(attenuation, source, boundary, *, increment=None):
    """Compose frozen layers ordered from observer to lower boundary.

    Each layer maps I to S + A(I-S). Work relative to the lower-boundary
    intensity to preserve a constant LTE equilibrium without subtracting two
    large absolute source terms. Adjacent maps compose as
    (A_near A_far, b_near + A_near b_far). Composition is associative but
    NOT commutative; no layer is reordered. Only log2(depth) batched stages
    are needed, with O(depth) work and autograd storage, including backward.
    """
    anchor = boundary.unsqueeze(-3)
    relative_source = source - anchor
    a = attenuation
    b = (relative_source - (a @ relative_source.unsqueeze(-1)).squeeze(-1)
         if increment is None else -(increment @ relative_source.unsqueeze(-1)).squeeze(-1))
    while a.shape[-4] > 1:
        count = a.shape[-4]
        paired = count - count % 2
        near = a[..., :paired:2, :, :, :]
        far = a[..., 1:paired:2, :, :, :]
        merged_a = near @ far
        merged_b = b[..., :paired:2, :, :] + (
            near @ b[..., 1:paired:2, :, :].unsqueeze(-1)
        ).squeeze(-1)
        if count % 2:
            merged_a = torch.cat((merged_a, a[..., -1:, :, :, :]), dim=-4)
            merged_b = torch.cat((merged_b, b[..., -1:, :, :]), dim=-3)
        a, b = merged_a, merged_b
    return boundary + b.squeeze(-3)


def formal_stokes(propagation_m1, source, distance_m, *, bottom_boundary=None,
                  precision="float32", polarized=False, intervals_m=None):
    """Integrate [..., depth, wavelength, 4] to [..., wavelength, 4].

    Distances increase inward. K and the thermal source are averaged at
    interval midpoints; their constant-layer solution is exact. Geometry and
    coefficients remain differentiable. Float32 is the production default;
    float64 can be selected explicitly for reference checks. The output retains
    the coefficient dtype. No inverse propagation matrices are formed. ``polarized``
    selects the analytic exponential for physical seven-coefficient LTE matrices;
    arbitrary matrices retain the general matrix-exponential implementation.
    Optional ``intervals_m`` supplies authoritative local quadrature widths,
    avoiding subtraction of nearly coincident large float32 distances.
    """
    if precision not in ("float32", "float64"):
        raise ValueError("Transfer precision must be float32 or float64")
    if propagation_m1.ndim < 4 or propagation_m1.shape[-2:] != (4, 4):
        raise ValueError("Propagation must have shape [..., depth, wavelength, 4, 4]")
    if source.shape != propagation_m1.shape[:-1] or source.shape[-3] < 2:
        raise ValueError("Source must match propagation and have at least two depth points")
    if propagation_m1.dtype not in (torch.float32, torch.float64):
        raise TypeError("Formal transfer requires float32 or float64 coefficients")
    dtype = getattr(torch, precision)
    k, s = propagation_m1.to(dtype), source.to(dtype)
    # Preserve supplied geometry precision until local differences are formed;
    # ordinary float32 geometry is never promoted to float64.
    distance = torch.as_tensor(distance_m, device=k.device)
    if not distance.is_floating_point():
        distance = distance.to(dtype)
    distance = torch.broadcast_to(distance, (*k.shape[:-4], k.shape[-4]))
    if intervals_m is None:
        intervals = (distance[..., 1:] - distance[..., :-1]).to(dtype)
    else:
        intervals = torch.as_tensor(intervals_m, device=k.device, dtype=dtype)
        expected = (*distance.shape[:-1], distance.shape[-1] - 1)
        try:
            intervals = torch.broadcast_to(intervals, expected)
        except RuntimeError as error:
            raise ValueError("intervals_m must match the ray batch and depth intervals") from error
    require_tensor(torch.isfinite(k).all() & torch.isfinite(s).all()
                   & torch.isfinite(distance).all() & torch.isfinite(intervals).all()
                   & (intervals > 0).all(),
                   "Finite coefficients and strictly increasing ray distances are required")
    boundary = s[..., -1, :, :] if bottom_boundary is None else bottom_boundary.to(k)
    if boundary.shape != (*k.shape[:-4], k.shape[-3], 4):
        raise ValueError("Bottom boundary must be finite with shape [..., wavelength, 4]")
    require_tensor(torch.isfinite(boundary).all(), "Bottom boundary must be finite with shape [..., wavelength, 4]")
    midpoint_k = .5 * (k[..., :-1, :, :, :] + k[..., 1:, :, :, :])
    midpoint_s = .5 * (s[..., :-1, :, :] + s[..., 1:, :, :])
    argument = -midpoint_k * intervals[..., None, None, None]
    if polarized:
        attenuation, increment = polarized_matrix_exp(argument, return_increment=True)
    else:
        attenuation = torch.matrix_exp(argument)
        identity = torch.eye(4, dtype=argument.dtype, device=argument.device)
        # For general matrices, use the short expm1 series only in thin layers.
        # Mask before powers to keep unused thick-layer branches finite.
        thin = argument.abs().amax(dim=(-2, -1)) < .01
        small_argument = torch.where(thin[..., None, None], argument, 0.)
        term = small_argument
        series = term
        for order in range(2, 9):
            term = term @ small_argument / order
            series = series + term
        increment = torch.where(thin[..., None, None], series, attenuation - identity)
    return affine_transfer_reduce(attenuation, midpoint_s, boundary,
                                  increment=increment).to(propagation_m1.dtype)


__all__ = ["affine_transfer_reduce", "formal_stokes"]
