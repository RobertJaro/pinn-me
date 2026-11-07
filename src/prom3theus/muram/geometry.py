"""Disk-center observation geometry for a flat MURaM patch.

A MURaM box is a small, flat, local Cartesian patch (tens of megameter) placed at
disk center. This reuses the same instrument-neutral scene-chart primitives
real HMI/Hinode-SP rasters use (:mod:`prom3theus.rt.geometry`), just without
the sunpy/WCS layer those instruments need for real heliographic pointing:
box ``x`` -> solar west, box ``y`` -> solar north, box height -> radial
at Carrington longitude=0, latitude=0. The basis rows are west=(0,1,0),
north=(0,0,1), radial=(1,0,0); identity would put the patch at the pole.

The transfer approximation is a flat, disk-center box: parallel vertical
rays, mu=1, and a common polarization basis. Surface positions are embedded
on the spherical scene chart for the inversion; this does not turn the box
into curved columns or introduce unmodeled inclined-ray transport.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prom3theus.rt.geometry import direction_to_chart_megameter, validate_scene_basis

SOLAR_RADIUS_M = 6.96e8


@dataclass(frozen=True)
class DiskCenterGeometry:
    """Per-pixel geometry fields for a flat, disk-center observation raster."""

    ray_direction: torch.Tensor  # [ny, nx, 3], observer -> surface
    surface_position_m: torch.Tensor  # [ny, nx, 3]
    stokes_basis: torch.Tensor  # [ny, nx, 3, 3]
    coordinates: torch.Tensor  # [ny, nx, 3] (chart_x_megameter, chart_y_megameter, time_hours)
    mu: torch.Tensor  # [ny, nx]


def build_disk_center_geometry(
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    *,
    stride: int = 1,
    binning: int = 1,
    solar_radius_m: float = SOLAR_RADIUS_M,
    scene_basis: torch.Tensor | None = None,
    reference_time_hours: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> DiskCenterGeometry:
    scene_basis = (
        torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=dtype)
        if scene_basis is None
        else validate_scene_basis(scene_basis)
    ).to(dtype)

    # Preserve the selected native pixel positions; recentering the smaller
    # image would shift the rays (especially for non-divisible dimensions).
    x = (torch.arange(0, nx, stride, dtype=dtype) - (nx - 1) / 2.0) * dx_m
    y = (torch.arange(0, ny, stride, dtype=dtype) - (ny - 1) / 2.0) * dy_m
    # Embed each output pixel at the center of its contributing columns.
    if binning != 1:
        x = x.reshape(-1, binning).mean(-1)
        y = y.reshape(-1, binning).mean(-1)
    nx, ny = x.numel(), y.numel()
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")  # both [ny, nx]

    local = torch.stack(
        (grid_x, grid_y, torch.full_like(grid_x, solar_radius_m)), dim=-1
    )  # [ny, nx, 3] in the (chart_x, chart_y, chart_normal) scene_basis frame
    direction = local / torch.linalg.vector_norm(local, dim=-1, keepdim=True)
    surface_position_m = (
        torch.einsum("ij,...j->...i", scene_basis.T, direction) * solar_radius_m
    )
    toward_observer = scene_basis[2].expand_as(surface_position_m)
    ray_direction = -toward_observer
    stokes_basis = scene_basis.expand(ny, nx, 3, 3).clone()
    mu = torch.ones((ny, nx), dtype=dtype)

    chart_megameter = direction_to_chart_megameter(
        surface_position_m / solar_radius_m, scene_basis, solar_radius_m
    )
    time_column = torch.full(
        chart_megameter.shape[:-1] + (1,), float(reference_time_hours), dtype=dtype
    )
    coordinates = torch.cat((chart_megameter, time_column), dim=-1)

    return DiskCenterGeometry(
        ray_direction=ray_direction,
        surface_position_m=surface_position_m,
        stokes_basis=stokes_basis,
        coordinates=coordinates,
        mu=mu,
    )
