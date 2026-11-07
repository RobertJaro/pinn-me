"""Spatial block means for synthetic observations and evaluation arrays.

Only parallel MURaM rays with a common Stokes basis are supported. Spectra
are averaged before the objective's per-profile normalization. Invalid input
pixels invalidate their entire output block; partial blocks are rejected.
"""

from copy import deepcopy
from dataclasses import replace

import torch

from prom3theus.rt.geometry import chart_to_direction, direction_to_chart_megameter


def validate_binning(factor):
    if type(factor) is not int or factor < 1:
        raise ValueError("binning must be a positive integer")
    return factor


def bin_spatial(values, factor, *, reduction="mean"):
    """Reduce spatial blocks, reading disk-backed inputs in bounded row strips.

    Trailing dimensions are preserved. Tensor inputs retain their autograd
    graph, so the same mean can be used on native-resolution predictions.
    """
    validate_binning(factor)
    ny, nx = values.shape[:2]
    if ny % factor or nx % factor:
        raise ValueError("Spatial dimensions must be divisible by binning")
    if reduction not in {"mean", "all", "uncertainty"}:
        raise ValueError(f"Unknown spatial reduction: {reduction}")
    strips = []
    rows = max(1, 256 // factor) * factor
    for start in range(0, ny, rows):
        part = values[start:start + rows]
        blocks = part.reshape(part.shape[0] // factor, factor, nx // factor,
                              factor, *part.shape[2:])
        if reduction == "all":
            result = blocks.all(dim=3).all(dim=1)
        elif reduction == "uncertainty":
            # Independent standard errors: sqrt(sum(sigma**2)) / count.
            result = blocks.square().mean(dim=(1, 3)).sqrt() / factor
        else:
            result = blocks.mean(dim=(1, 3))
        strips.append(result)
    return torch.cat(strips)


def _geometry(raster, factor, basis):
    radius = float(raster.metadata["ray_geometry"]["solar_radius_m"])
    ray = raster.ray_direction[:]
    if not torch.allclose(ray, -basis[2].to(ray).expand_as(ray), atol=2e-5, rtol=0):
        raise ValueError("MURaM binning requires parallel rays in one common basis")
    if hasattr(raster, "coordinates"):
        coordinates = bin_spatial(raster.coordinates, factor)
        xy = coordinates[..., :2]
    else:
        xy = bin_spatial(direction_to_chart_megameter(
            raster.surface_position_m[:], basis, radius), factor)
        coordinates = None
    surface = chart_to_direction(xy, basis, radius) * radius
    return coordinates, surface, -basis[2].to(surface).expand_as(surface)


def _sampling(raster, factor):
    return {
        "factor": factor,
        "source_shape": list(raster.spatial_shape),
        "operation": "spatial block mean before objective normalization",
        "validity": "all contributing pixels valid",
    }


def bin_stokes_raster(raster, factor=1):
    """Return a binned synthetic Stokes raster without modifying its source."""
    validate_binning(factor)
    if factor == 1:
        return raster
    if "muram_synthesis" not in raster.metadata:
        raise ValueError("Loading-time Stokes binning is supported only for MURaM data")
    basis = raster.stokes_basis[0, 0]
    if not torch.allclose(raster.stokes_basis[:], basis.expand(*raster.spatial_shape, 3, 3)):
        raise ValueError("MURaM binning requires a common Stokes basis")
    coordinates, surface, ray = _geometry(raster, factor, basis)
    metadata = deepcopy(dict(raster.metadata))
    metadata["spatial_binning"] = _sampling(raster, factor)
    for name in ("dx_m", "dy_m"):
        if name in metadata["muram_synthesis"]:
            metadata["spatial_binning"][name] = metadata["muram_synthesis"][name] * factor
    correction = metadata.get("observer_velocity_correction", {})
    velocities = correction.get("removed_solar_los_velocity_m_per_s")
    if velocities is not None:
        correction["removed_solar_los_velocity_m_per_s"] = torch.as_tensor(
            velocities, dtype=torch.float64).reshape(-1, factor).mean(-1).tolist()
    auxiliary = {}
    for name, values in raster.auxiliary.items():
        if values.dtype == torch.bool:
            reduction = "all"
        elif values.dtype.is_floating_point:
            reduction = "uncertainty" if name == "uncertainty" else "mean"
        else:
            raise ValueError(f"No spatial binning rule for auxiliary field {name!r}")
        auxiliary[name] = bin_spatial(values, factor, reduction=reduction)
    return replace(
        raster, stokes=bin_spatial(raster.stokes, factor),
        wavelength_angstrom=raster.wavelength_angstrom[:],
        coordinates=coordinates, surface_position_m=surface, ray_direction=ray,
        stokes_basis=basis.expand(*surface.shape[:2], 3, 3).clone(),
        valid_mask=bin_spatial(raster.valid_mask, factor, reduction="all"),
        auxiliary=auxiliary, metadata=metadata,
    )


def bin_image_raster(raster, factor=1):
    """Bin synthetic AIA mean intensities (reference-pixel count rates)."""
    validate_binning(factor)
    if factor == 1:
        return raster
    if not raster.metadata.get("provenance", {}).get("synthetic", False):
        raise ValueError("Loading-time image binning is supported only for MURaM data")
    basis = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
                         dtype=raster.surface_position_m.dtype)
    _, surface, ray = _geometry(raster, factor, basis)
    valid = bin_spatial(raster.valid_mask, factor, reduction="all")
    metadata = deepcopy(dict(raster.metadata))
    metadata["provenance"]["spatial_binning"] = _sampling(raster, factor)
    for name in ("dx_m", "dy_m"):
        if name in metadata["provenance"]:
            metadata["provenance"]["spatial_binning"][name] = metadata["provenance"][name] * factor
    metadata["mask"] = {
        "definition": "all contributing native pixels valid and on disk",
        "on_disk_pixel_count": valid.numel(),
        "valid_on_disk_pixel_count": int(valid.sum()),
    }
    return replace(
        raster, intensity=bin_spatial(raster.intensity, factor),
        surface_position_m=surface, ray_direction=ray, valid_mask=valid,
        uncertainty=None if raster.uncertainty is None else bin_spatial(
            raster.uncertainty, factor, reduction="uncertainty"),
        metadata=metadata,
    )
