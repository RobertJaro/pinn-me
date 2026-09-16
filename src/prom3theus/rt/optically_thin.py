"""Generic differentiable quadrature for optically thin ray emission."""

from __future__ import annotations

import math

import torch


# If ne is supplied in m^-3 and ds in m, ne^2 ds has units m^-5.  One cm^-5
# equals 1e10 m^-5, hence EM_cm^-5 = 1e-10 EM_m^-5.
SI_EMISSION_MEASURE_TO_CGS = 1.0e-10
NE2_RESPONSE_SCALE_CM5 = 1.0e-30


def ne2_response_emissivity(
    response_cm5: torch.Tensor,
    electron_density_m3: torch.Tensor,
    *,
    response_scale_cm5: float = 1.0,
) -> torch.Tensor:
    """Return count rate per metre without forming the large SI density square.

    With n0=1e20 m^-3 and K0=1e-30, n0^2 K0 * 1e-10 = 1.
    Multiplying the scaled response first also preserves exact zero emission.
    These are numerical units, not bounds on either physical input.
    ``response_scale_cm5`` is the physical unit of the supplied response values;
    pass NE2_RESPONSE_SCALE_CM5 for an already-normalized table evaluation.
    """
    if not math.isfinite(response_scale_cm5) or response_scale_cm5 <= 0:
        raise ValueError("response_scale_cm5 must be finite and positive.")
    density = (electron_density_m3 / 1.0e20).unsqueeze(-1)
    response = response_cm5 * (response_scale_cm5 / NE2_RESPONSE_SCALE_CM5)
    return (response * density) * density


def _validate_ray_inputs(
    values: torch.Tensor,
    distance_m: torch.Tensor,
    sample_dim: int,
) -> tuple[int, torch.Tensor]:
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError("Ray-integrand values must be a floating-point torch.Tensor.")
    if not isinstance(distance_m, torch.Tensor) or not distance_m.is_floating_point():
        raise TypeError("distance_m must be a floating-point torch.Tensor.")
    if values.ndim < 1:
        raise ValueError("Ray-integrand values must have at least one dimension.")
    sample_axis = sample_dim if sample_dim >= 0 else values.ndim + sample_dim
    if sample_axis < 0 or sample_axis >= values.ndim:
        raise ValueError(
            f"sample_dim={sample_dim} is invalid for shape {values.shape}."
        )
    if values.shape[sample_axis] < 2:
        raise ValueError("Optically thin quadrature needs at least two ray samples.")
    if values.device != distance_m.device:
        raise ValueError("Ray-integrand values and distance_m must share a device.")
    if not torch.isfinite(values).all() or not torch.isfinite(distance_m).all():
        raise ValueError("Optically thin quadrature inputs must be finite.")

    if distance_m.ndim == 1:
        if distance_m.numel() != values.shape[sample_axis]:
            raise ValueError("The distance axis does not match the ray sample count.")
        differences = distance_m[1:] - distance_m[:-1]
        coordinate = distance_m
    elif distance_m.ndim == values.ndim - 1 and sample_axis < values.ndim - 1:
        expected = values.shape[:-1]
        if distance_m.shape != expected:
            raise ValueError("A per-ray distance tensor must match values.shape[:-1].")
        differences = torch.diff(distance_m, dim=sample_axis)
        coordinate = distance_m.unsqueeze(-1)
    elif distance_m.shape == values.shape:
        differences = torch.diff(distance_m, dim=sample_axis)
        coordinate = distance_m
    else:
        raise ValueError(
            "distance_m must be a shared one-dimensional axis, values.shape, "
            "or values.shape[:-1] for channel-last values."
        )
    if torch.any(differences <= 0.0):
        raise ValueError("distance_m must increase strictly along every ray.")
    return sample_axis, coordinate


def trapezoid_weights(
    distance_m: torch.Tensor, *, sample_dim: int = -1
) -> torch.Tensor:
    """Nodal line elements in metres for a nonuniform, increasing ray grid.

    Compute differences in the geometry dtype before any conversion to the
    field dtype. Endpoints receive half one interval; interior nodes receive
    half each adjacent interval. Their sum is the complete ray length.
    """
    if not isinstance(distance_m, torch.Tensor) or not distance_m.is_floating_point():
        raise TypeError("distance_m must be a floating-point torch.Tensor.")
    if distance_m.ndim == 0 or not -distance_m.ndim <= sample_dim < distance_m.ndim:
        raise ValueError("Invalid distance sample dimension.")
    axis = sample_dim % distance_m.ndim
    if distance_m.shape[axis] < 2 or not torch.isfinite(distance_m).all():
        raise ValueError("Ray quadrature requires at least two finite distances.")
    intervals = torch.diff(distance_m, dim=axis)
    if torch.any(intervals <= 0):
        raise ValueError("distance_m must increase strictly along every ray.")
    zero = torch.zeros_like(distance_m.narrow(axis, 0, 1))
    return 0.5 * (
        torch.cat((zero, intervals), dim=axis) + torch.cat((intervals, zero), dim=axis)
    )


def integrate_optically_thin(
    emissivity_per_m: torch.Tensor,
    distance_m: torch.Tensor,
    *,
    sample_dim: int = -1,
) -> torch.Tensor:
    """Integrate a local optically thin emissivity along increasing rays."""

    sample_axis, coordinate = _validate_ray_inputs(
        emissivity_per_m, distance_m, sample_dim
    )
    if coordinate.ndim == 1:
        weights = trapezoid_weights(coordinate)
        shape = [1] * emissivity_per_m.ndim
        shape[sample_axis] = coordinate.numel()
        weights = weights.reshape(shape)
    else:
        weights = trapezoid_weights(coordinate, sample_dim=sample_axis)
    return (emissivity_per_m * weights).sum(dim=sample_axis)


def integrate_ne2_response(
    response_cm5: torch.Tensor,
    electron_density_m3: torch.Tensor,
    distance_m: torch.Tensor,
    *,
    sample_dim: int = -2,
    response_scale_cm5: float = 1.0,
) -> torch.Tensor:
    """Integrate ``ne^2 K(T)`` for an ``ne2`` response table.

    ``response_cm5`` is channel-last with units such as
    ``DN s^-1 pixel^-1 cm^5``. ``electron_density_m3`` must match all axes
    except that final channel axis.  The returned value therefore has the
    response's count-rate units and uses the exact SI-to-cgs emission-measure
    conversion ``1e-10``.
    For pre-normalized responses, supply their physical ``response_scale_cm5``.
    """

    if not isinstance(response_cm5, torch.Tensor) or response_cm5.ndim < 2:
        raise TypeError("response_cm5 must be a channel-last torch.Tensor.")
    if not isinstance(electron_density_m3, torch.Tensor):
        raise TypeError("electron_density_m3 must be a torch.Tensor.")
    if response_cm5.shape[:-1] != electron_density_m3.shape:
        raise ValueError("electron_density_m3 must match response_cm5.shape[:-1].")
    if (
        not response_cm5.is_floating_point()
        or not electron_density_m3.is_floating_point()
    ):
        raise TypeError("AIA response and electron density must be floating point.")
    if response_cm5.device != electron_density_m3.device:
        raise ValueError("AIA response and electron density must share a device.")
    if (
        not torch.isfinite(response_cm5).all()
        or not torch.isfinite(electron_density_m3).all()
        or torch.any(response_cm5 < 0.0)
        or torch.any(electron_density_m3 < 0.0)
    ):
        raise ValueError(
            "AIA response and electron density must be finite and non-negative."
        )
    sample_axis = sample_dim if sample_dim >= 0 else response_cm5.ndim + sample_dim
    if sample_axis == response_cm5.ndim - 1:
        raise ValueError("sample_dim cannot select the final channel axis.")
    local_emissivity = ne2_response_emissivity(
        response_cm5, electron_density_m3, response_scale_cm5=response_scale_cm5
    )
    return integrate_optically_thin(
        local_emissivity,
        distance_m,
        sample_dim=sample_dim,
    ).to(local_emissivity)


__all__ = [
    "SI_EMISSION_MEASURE_TO_CGS",
    "NE2_RESPONSE_SCALE_CM5",
    "trapezoid_weights",
    "integrate_ne2_response",
    "integrate_optically_thin",
    "ne2_response_emissivity",
]
