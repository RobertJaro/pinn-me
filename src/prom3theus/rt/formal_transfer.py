"""Differentiable LTE transfer along an ordered ray."""

from __future__ import annotations

import torch

from .polarized_exponential import independent_coefficients
from .stokes_sweep import polarized_sweep
from .transmission import transmission_above as _transmission_above
from prom3theus.core.validation import require_tensor


def affine_transfer_reduce(attenuation, source, boundary, *, increment=None):
    """Sum each layer's emission weighted by transmission through the layers above.

    With P[i] = A[0] ... A[i-1], the formal solution is
    P[N] @ boundary + sum_i P[i] @ (Id - A[i]) @ S[i].
    Since P[i] @ (Id - A[i]) = P[i] - P[i+1], shifting every S[i] by
    the boundary folds the transmitted boundary term into one final addition.
    This gives exact thermal equilibrium when all sources equal the boundary.
    """
    relative_source = (source - boundary.unsqueeze(-3)).unsqueeze(-1)
    if increment is None:
        emission = relative_source - attenuation @ relative_source
    else:
        # The exponential solver supplies exp(-K ds) - Id accurately in thin layers.
        emission = -(increment @ relative_source)
    # Keep the batched matrix-vector products in the native ray/depth/wavelength layout.
    above = _transmission_above(attenuation.movedim(-4, 0)).movedim(0, -4)
    contributions = (above @ emission).squeeze(-1)
    return boundary + contributions.cumsum(dim=-3).select(-3, -1)


def formal_stokes(propagation_m1, source, distance_m, *, bottom_boundary=None,
                  precision="float32", polarized=False, intervals_m=None):
    """Integrate [..., depth, wavelength, 4] to [..., wavelength, 4].

    Distances increase inward. K and the thermal source are averaged at
    interval midpoints; their constant-layer solution is exact. Geometry and
    coefficients remain differentiable. Float32 is the production default;
    float64 can be selected explicitly for reference checks. The output retains
    the coefficient dtype. No inverse propagation matrices are formed. ``polarized``
    accepts packed (eta_I,Q,U,V, rho_Q,U,V) values or their physical 4x4 matrices;
    arbitrary matrices retain the general matrix-exponential implementation.
    Optional ``intervals_m`` supplies authoritative local quadrature widths,
    avoiding subtraction of nearly coincident large float32 distances.
    """
    if precision not in ("float32", "float64"):
        raise ValueError("Transfer precision must be float32 or float64")
    compact = polarized and propagation_m1.ndim >= 3 and propagation_m1.shape[-1] == 7
    if compact:
        point_shape = propagation_m1.shape[:-1]
    elif propagation_m1.ndim >= 4 and propagation_m1.shape[-2:] == (4, 4):
        point_shape = propagation_m1.shape[:-2]
    else:
        raise ValueError("Propagation must be a 4x4 matrix or seven polarized coefficients per point")
    if source.shape != (*point_shape, 4) or source.shape[-3] < 2:
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
    distance = torch.broadcast_to(distance, point_shape[:-1])
    if intervals_m is None:
        intervals = (distance[..., 1:] - distance[..., :-1]).to(dtype)
    else:
        intervals = torch.as_tensor(intervals_m, device=k.device, dtype=dtype)
        expected = (*distance.shape[:-1], distance.shape[-1] - 1)
        try:
            intervals = torch.broadcast_to(intervals, expected)
        except RuntimeError as error:
            raise ValueError("intervals_m must match the ray batch and depth intervals") from error
    require_tensor(lambda: torch.isfinite(k).all() & torch.isfinite(s).all()
                   & torch.isfinite(distance).all() & torch.isfinite(intervals).all()
                   & (intervals > 0).all(),
                   "Finite coefficients and strictly increasing ray distances are required")
    boundary = s[..., -1, :, :] if bottom_boundary is None else bottom_boundary.to(k)
    if boundary.shape != (*point_shape[:-2], point_shape[-1], 4):
        raise ValueError("Bottom boundary must be finite with shape [..., wavelength, 4]")
    require_tensor(lambda: torch.isfinite(boundary).all(), "Bottom boundary must be finite with shape [..., wavelength, 4]")
    midpoint_s = .5 * (s[..., :-1, :, :] + s[..., 1:, :, :])
    if polarized:
        components = k if compact else independent_coefficients(k)
        midpoint_k = .5 * (components[..., :-1, :, :] + components[..., 1:, :, :])
        argument = -midpoint_k * intervals[..., None, None]
        return polarized_sweep(argument, midpoint_s, boundary).to(propagation_m1.dtype)
    midpoint_k = .5 * (k[..., :-1, :, :, :] + k[..., 1:, :, :, :])
    argument = -midpoint_k * intervals[..., None, None, None]
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
