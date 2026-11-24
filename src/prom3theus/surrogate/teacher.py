"""Exact LTE training targets in the local Stokes basis."""

import torch

from prom3theus.rt.coefficient_mapper import independent_coefficients


def _prepare(teacher, points):
    t, p, xi, los = points[..., :4].unbind(-1)
    basis = torch.eye(3, device=t.device, dtype=t.dtype)
    atmosphere = dict(
        temperature=t,
        gas_pressure=p,
        microturbulence=xi,
        velocity_field=-los[..., None] * basis[2],
        magnetic_field=points[..., 4:],
    )
    return teacher.prepare_atmosphere(
        atmosphere, direction=basis[2], stokes_basis=basis
    )


def direct_local(teacher, points, wavelength, radiance_scale=1.0):
    return teacher.evaluate_prepared(
        _prepare(teacher, points), wavelength, radiance_scale=radiance_scale
    )


def density_inputs(pressure_points, mass_density):
    """Replace pressure (Pa) with the corresponding EOS mass density (kg/m³)."""
    return torch.cat(
        (pressure_points[..., :1], mass_density[..., None], pressure_points[..., 2:]),
        -1,
    )


def training_pair(teacher, pressure_points, wavelength):
    """Reuse the exact LTE plasma state for both physical inputs and labels."""
    prepared = _prepare(teacher, pressure_points)
    local = teacher.evaluate_prepared(prepared, wavelength)
    return (
        density_inputs(pressure_points, prepared.plasma.mass_density),
        independent_coefficients(local.propagation_m1),
    )


def targets(teacher, points, wavelength):
    return independent_coefficients(
        direct_local(teacher, points, wavelength).propagation_m1
    )
