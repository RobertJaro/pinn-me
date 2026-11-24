"""Polarized LTE transfer on sampled Cartesian rays with explicit inflow."""

from __future__ import annotations

import math

import torch

from prom3theus.rt.atmosphere import StratifiedAtmosphere
from prom3theus.rt.opacity import planck_lambda
from prom3theus.rt.wavelength import air_to_vacuum_angstrom
from prom3theus.validation.reference_synthesis import LTESynthesizer
from prom3theus.validation.reference_transfer import RayDistancePath


class RayStokesRenderer:
    """Render already projected vector fields along observer-to-boundary rays.

    Inputs contain only atmosphere inside the box. Repeated endpoint samples
    may pad a batch; their zero-width intervals contribute no transport. The
    caller owns geometry, vector projection, rotation velocity and instrumental
    polarization conventions. No such transformation is repeated here.
    """

    def __init__(self, line_ids, wavelength_angstrom, *, device="cpu", dtype=torch.float32,
                 radiance_scale=1e13):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Ray transfer dtype must be float32 or float64.")
        if not math.isfinite(radiance_scale) or radiance_scale <= 0:
            raise ValueError("radiance_scale must be finite and positive.")
        self.device, self.dtype = torch.device(device), dtype
        self.radiance_scale = radiance_scale
        wavelength = torch.as_tensor(wavelength_angstrom, dtype=torch.float64)
        self.synthesizer = LTESynthesizer(line_ids=tuple(line_ids)).to(device=self.device, dtype=dtype)
        self.synthesizer.prepare_wavelength_grid(wavelength)
        self.synthesizer.to(device=self.device, dtype=dtype)
        self.wavelength_vacuum = air_to_vacuum_angstrom(wavelength).to(device=self.device, dtype=dtype)

    @torch.no_grad()
    def __call__(self, fields, distance_m, bottom_boundary_mask):
        """Return [ray, 4, wavelength] Stokes in radiance_scale units.

        fields uses temperature, pressure, bx/by/bz and vx/vy/vz, each [ray,
        depth]. Vector +z points toward the observer. Distances increase inward.
        Only true bottom-face inflow gets an unpolarized Planck boundary; side
        or shell inflow gets zero incident radiation.
        """
        values = {name: torch.as_tensor(fields[name], device=self.device, dtype=self.dtype)
                  for name in ("temperature", "pressure", "bx", "by", "bz", "vx", "vy", "vz")}
        temperature = values["temperature"]
        if temperature.ndim != 2 or temperature.shape[-1] < 2:
            raise ValueError("Ray fields must have shape [ray, depth>=2].")
        if any(value.shape != temperature.shape for value in values.values()):
            raise ValueError("All sampled ray fields must have the same shape.")
        distance = torch.as_tensor(distance_m, device=self.device, dtype=torch.float64)
        if distance.shape != temperature.shape:
            raise ValueError("distance_m must match the sampled ray fields.")
        if (not torch.isfinite(distance).all() or torch.any(distance.diff(dim=-1) < 0)
                or torch.any(distance[:, -1] <= distance[:, 0])):
            raise ValueError("Ray distances must be finite, nondecreasing, and span a positive length.")
        # Remove any common offset before conversion to transfer precision.
        distance = (distance - distance[:, :1]).to(dtype=self.dtype)
        thermal = torch.as_tensor(bottom_boundary_mask, device=self.device)
        if thermal.dtype != torch.bool or thermal.shape != temperature.shape[:1]:
            raise ValueError("bottom_boundary_mask must be boolean with one value per ray.")
        atmosphere = StratifiedAtmosphere(
            depth_coordinate=torch.arange(temperature.shape[-1], device=self.device, dtype=self.dtype),
            temperature=temperature,
            gas_pressure=values["pressure"],
            velocity_field=torch.stack([values[name] for name in ("vx", "vy", "vz")], dim=-1),
            magnetic_field=torch.stack([values[name] for name in ("bx", "by", "bz")], dim=-1),
            microturbulence=torch.zeros_like(temperature),
        )
        intensity = planck_lambda(temperature[:, -1], self.wavelength_vacuum,
                                  radiance_scale=self.radiance_scale)
        intensity = torch.where(thermal[:, None], intensity, 0.)
        zero = torch.zeros_like(intensity)
        boundary = torch.stack((intensity, zero, zero, zero), dim=-1)
        return self.synthesizer(atmosphere, path=RayDistancePath(distance),
                                bottom_boundary=boundary, radiance_scale=self.radiance_scale)
