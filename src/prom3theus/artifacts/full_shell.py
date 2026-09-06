"""Physical full-shell evaluation for LTE inversion artifacts.

The ordinary atmosphere export follows the observed rays through the LTE
line-formation domain.  Extrapolation runs also constrain the same coordinate
network above that domain.  This module exposes that additional solution on
radial Carrington columns without assigning an optical-depth coordinate or
attempting LTE synthesis in the upper atmosphere.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from prom3theus.core import (
    cartesian_to_spherical,
    project_cartesian_to_spherical,
)
from prom3theus.observations import ObservationRaster
from prom3theus.training.lightning import LTEInversionModule

from .errors import ArtifactExportError
from .evaluation import resolve_storage_dtype


def full_shell_height_grid(
    module: LTEInversionModule,
    height_samples: int,
) -> torch.Tensor:
    """Return an outer-to-inner geometric-height grid for the trained shell."""

    if type(height_samples) is not int or height_samples < 2:
        raise ValueError("full_shell_samples must be an integer of at least two.")
    model = module.atmosphere_model
    bounds = getattr(model, "shell_height_bounds_Mm", None)
    if (
        not isinstance(bounds, (tuple, list))
        or len(bounds) != 2
        or not all(math.isfinite(float(value)) for value in bounds)
        or not float(bounds[0]) > float(bounds[1])
    ):
        raise ArtifactExportError(
            "The artifact atmosphere does not define finite outer-to-inner "
            "shell_height_bounds_Mm."
        )
    parameter = next(model.parameters())
    return torch.linspace(
        float(bounds[0]) * 1.0e6,
        float(bounds[1]) * 1.0e6,
        height_samples,
        dtype=parameter.dtype,
        device=parameter.device,
    )


def full_shell_metadata(height_grid_m: torch.Tensor) -> dict[str, object]:
    """Describe the explicit physical sampling used by a full-shell product."""

    height = height_grid_m.detach().cpu()
    if (
        height.ndim != 1
        or height.numel() < 2
        or not torch.isfinite(height).all()
        or not torch.all(height[:-1] > height[1:])
    ):
        raise ValueError(
            "full-shell geometric heights must be a strictly outer-to-inner vector."
        )
    return {
        "coordinate": "geometric_height_m",
        "coordinate_array": "full_shell_geometric_height_m",
        "height_samples": int(height.numel()),
        "height_bounds_m": [float(height[0]), float(height[-1])],
        "height_order": "outer_to_inner",
        "spatial_sampling": "radial_carrington_columns",
        "spatial_coordinates": [
            "carrington_chart_x_mm",
            "carrington_chart_y_mm",
            "time_hours",
        ],
        "valid_mask": "valid_mask",
        "vector_frames": {
            "cartesian": "heliocentric_carrington_cartesian",
            "spherical": "local_carrington_radial_colatitude_longitude",
            "velocity": "carrington_corotating",
        },
        "stokes_synthesis": False,
        "opacity_and_tau500": False,
    }


@torch.inference_mode()
def evaluate_full_shell_atmosphere(
    module: LTEInversionModule,
    raster: ObservationRaster,
    *,
    height_grid_m: torch.Tensor,
    batch_size: int,
    storage_dtype: str,
) -> dict[str, np.ndarray]:
    """Evaluate radial columns spanning the complete configured physical shell.

    The two spatial chart coordinates and time come from each valid pixel of
    the artifact's validation raster.  The trailing dimension is the shared
    geometric-height grid, ordered from the outer boundary to the inner one.
    Invalid raster pixels remain NaN and use the ordinary export ``valid_mask``.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    torch_dtype, numpy_dtype = resolve_storage_dtype(storage_dtype)
    model = module.atmosphere_model
    parameter = next(model.parameters())
    height_grid = torch.as_tensor(
        height_grid_m,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    metadata = full_shell_metadata(height_grid)
    expected_bounds = tuple(
        float(value) * 1.0e6 for value in model.shell_height_bounds_Mm
    )
    tolerance = max(
        1.0,
        32.0 * torch.finfo(height_grid.dtype).eps * abs(expected_bounds[0]),
    )
    if not (
        math.isclose(float(height_grid[0]), expected_bounds[0], abs_tol=tolerance)
        and math.isclose(float(height_grid[-1]), expected_bounds[1], abs_tol=tolerance)
    ):
        raise ValueError(
            "full-shell geometric heights must span the configured shell boundaries."
        )

    valid_flat = raster.valid_mask.reshape(-1)
    flat_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        raise ArtifactExportError("The canonical observation contains no valid pixels.")
    coordinates = raster.coordinates.reshape(-1, 3).index_select(0, flat_indices)
    spatial_shape = raster.spatial_shape
    height_count = int(metadata["height_samples"])
    flat_size = int(valid_flat.numel())

    scalar_fields = {
        name: np.full((flat_size, height_count), np.nan, dtype=numpy_dtype)
        for name in (
            "temperature",
            "microturbulence",
            "gas_pressure",
            "mass_density",
            "radius",
            "colatitude",
            "longitude",
        )
    }
    vector_fields = {
        name: np.full((flat_size, height_count, 3), np.nan, dtype=numpy_dtype)
        for name in (
            "position",
            "magnetic",
            "magnetic_spherical",
            "velocity",
            "velocity_spherical",
        )
    }

    was_training = model.training
    model.eval()
    try:
        for start in range(0, coordinates.shape[0], batch_size):
            stop = min(start + batch_size, coordinates.shape[0])
            batch_coordinates = coordinates[start:stop].to(parameter)
            batch_heights = height_grid.expand(batch_coordinates.shape[0], -1)
            fields = model.evaluate_at_height(batch_coordinates, batch_heights)
            expanded_coordinates = batch_coordinates[:, None, :].expand(
                -1, height_count, -1
            )
            position = model.position_from_coords_height(
                expanded_coordinates,
                batch_heights,
            )
            spherical = cartesian_to_spherical(position, torch)
            magnetic_spherical = project_cartesian_to_spherical(
                fields["magnetic_field"], spherical, torch
            )
            velocity_spherical = project_cartesian_to_spherical(
                fields["velocity_field"], spherical, torch
            )
            mass_density = model.thermodynamic_eos.mass_density(
                fields["temperature"], fields["gas_pressure"]
            )
            selected = flat_indices[start:stop].cpu().numpy()

            def store_scalar(name: str, value: torch.Tensor) -> None:
                scalar_fields[name][selected] = (
                    value.detach().to(torch_dtype).cpu().numpy()
                )

            def store_vector(name: str, value: torch.Tensor) -> None:
                vector_fields[name][selected] = (
                    value.detach().to(torch_dtype).cpu().numpy()
                )

            store_scalar("temperature", fields["temperature"])
            store_scalar("microturbulence", fields["microturbulence"])
            store_scalar("gas_pressure", fields["gas_pressure"])
            store_scalar("mass_density", mass_density)
            store_scalar("radius", spherical[..., 0])
            store_scalar("colatitude", spherical[..., 1])
            store_scalar("longitude", spherical[..., 2])
            store_vector("position", position)
            store_vector("magnetic", fields["magnetic_field"])
            store_vector("magnetic_spherical", magnetic_spherical)
            store_vector("velocity", fields["velocity_field"])
            store_vector("velocity_spherical", velocity_spherical)
    finally:
        model.train(was_training)

    shape = (*spatial_shape, height_count)
    vector_shape = (*shape, 3)
    return {
        "full_shell_geometric_height_m": height_grid.detach()
        .to(torch_dtype)
        .cpu()
        .numpy(),
        "full_shell_temperature_k": scalar_fields["temperature"].reshape(shape),
        "full_shell_microturbulence_m_per_s": scalar_fields[
            "microturbulence"
        ].reshape(shape),
        "full_shell_gas_pressure_pa": scalar_fields["gas_pressure"].reshape(shape),
        "full_shell_mass_density_kg_m3": scalar_fields["mass_density"].reshape(shape),
        "full_shell_position_carrington_m": vector_fields["position"].reshape(
            vector_shape
        ),
        "full_shell_radius_m": scalar_fields["radius"].reshape(shape),
        "full_shell_carrington_colatitude_rad": scalar_fields["colatitude"].reshape(
            shape
        ),
        "full_shell_carrington_longitude_rad": scalar_fields["longitude"].reshape(
            shape
        ),
        "full_shell_magnetic_field_gauss": vector_fields["magnetic"].reshape(
            vector_shape
        ),
        "full_shell_magnetic_field_spherical_gauss": vector_fields[
            "magnetic_spherical"
        ].reshape(vector_shape),
        "full_shell_velocity_field_m_per_s": vector_fields["velocity"].reshape(
            vector_shape
        ),
        "full_shell_velocity_field_spherical_m_per_s": vector_fields[
            "velocity_spherical"
        ].reshape(vector_shape),
    }


__all__ = [
    "evaluate_full_shell_atmosphere",
    "full_shell_height_grid",
    "full_shell_metadata",
]
