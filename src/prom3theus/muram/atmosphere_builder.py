"""Assemble full native columns in top-to-bottom geometric transfer order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prom3theus.rt.atmosphere import StratifiedAtmosphere
from prom3theus.rt.ray_geometry import sample_shell_characteristic, shell_characteristic
from .geometry import SOLAR_RADIUS_M


NATIVE_ATMOSPHERE_FIELDS = ("temperature", "pressure", "vx", "vy", "vz", "bx", "by", "bz")
DEFAULT_STOKES_HEIGHT_MEGAMETERS = 3.0


def stokes_depth_count(nz, dz_m, height_megameters):
    """Keep native cell centers up to a height above the bottom cell center."""
    if height_megameters is None:
        return nz
    if isinstance(height_megameters, bool) or not np.isfinite(height_megameters) or height_megameters <= 0:
        raise ValueError("stokes_height_megameters must be finite and positive, or None for full height.")
    count = min(nz, int(np.floor(height_megameters * 1e6 / dz_m + 1e-9)) + 1)
    if count < 2:
        raise ValueError("stokes_height_megameters must include at least two native cell centers.")
    return count


@dataclass(frozen=True)
class UniformShellGrid:
    depth_count: int
    shell_height_bounds_megameter: tuple[float, float]
    photosphere_height_megameters: float

    def __post_init__(self):
        bounds = self.shell_height_bounds_megameter
        if type(self.depth_count) is not int or self.depth_count < 2:
            raise ValueError("Uniform sampling requires at least two depth points.")
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Uniform sampling requires increasing finite shell heights.")
        if not np.isfinite(self.photosphere_height_megameters):
            raise ValueError("Photosphere height must be finite.")

    @property
    def native_height_range_m(self):
        return [(self.photosphere_height_megameters + h) * 1e6 for h in self.shell_height_bounds_megameter]

    def retained_depth(self, nz, dz_m):
        lower, upper = self.native_height_range_m
        if lower < 0 or upper > (nz - 1) * dz_m:
            raise ValueError("The registered inversion shell extends outside the native cell centers.")
        return int(np.ceil(upper / dz_m)) + 1

    def sample(self, geometry, ix, iy):
        coordinates = geometry.coordinates[iy, ix].double()
        ray = shell_characteristic(
            coordinates, geometry.ray_direction[iy, ix],
            scene_basis=geometry.stokes_basis[0, 0], solar_radius_m=SOLAR_RADIUS_M,
            height_bounds_m=coordinates.new_tensor(self.shell_height_bounds_megameter) * 1e6,
        )
        fraction = torch.linspace(0, 1, self.depth_count, dtype=coordinates.dtype)
        distance = ray["path_length_m"][:, None] * fraction
        _, height, _ = sample_shell_characteristic(ray, distance, solar_radius_m=SOLAR_RADIUS_M)
        return distance, height + self.photosphere_height_megameters * 1e6


def build_native_atmosphere(quantities, *, dtype=torch.float64, device=None, sample_height_m=None):
    """Read native values top to bottom, or interpolate fields at the given native heights."""
    nx, ny, nz = quantities["temperature"].shape
    dz = float(quantities["ds_m"][2])
    if nz < 2 or not np.isfinite(dz) or dz <= 0:
        raise ValueError("Native columns require at least two layers and positive finite dz.")

    if sample_height_m is None:
        height = (torch.arange(nz - 1, -1, -1, dtype=dtype, device=device) * dz).expand(nx * ny, -1)
    else:
        height = sample_height_m.to(device=device, dtype=dtype)
        # Shell endpoints may differ from an exact native center by roundoff.
        index = (height / dz).clamp(0, nz - 1)
        lower = index.long().clamp_max(nz - 2)
        weight = index - lower

    def field(name):
        if sample_height_m is not None:
            native = np.ascontiguousarray(quantities[name].reshape(nx * ny, nz))
            values = torch.as_tensor(native, dtype=dtype, device=device)
            return torch.lerp(values.gather(1, lower), values.gather(1, lower + 1), weight)
        # The reader orients z upward. Transfer walks from the top cell to the
        # bottom cell; an explicit contiguous copy supports negative NumPy strides.
        values = np.ascontiguousarray(quantities[name][..., ::-1].reshape(nx * ny, nz))
        return torch.as_tensor(values, dtype=dtype, device=device)

    temperature = field("temperature")
    return StratifiedAtmosphere(
        # Labels only: GeometricHeightPath supplies all physical path lengths.
        depth_coordinate=torch.arange(height.shape[-1], dtype=dtype, device=device),
        temperature=temperature,
        gas_pressure=field("pressure"),
        velocity_field=torch.stack([field(name) for name in ("vx", "vy", "vz")], dim=-1),
        magnetic_field=torch.stack([field(name) for name in ("bx", "by", "bz")], dim=-1),
        microturbulence=torch.zeros_like(temperature),
        geometric_height_m=height,
    )
