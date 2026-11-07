"""Physical atmosphere evaluation for validated LTE inversion artifacts."""

from __future__ import annotations

from prom3theus.observations.arrays import materialize_array, read_flat_samples

import math

import numpy as np
import torch

from prom3theus.core import (
    SPEED_OF_LIGHT,
    carrington_rotation_velocity_cartesian,
    cartesian_to_spherical,
    project_cartesian_to_spherical,
    project_spherical_to_observer,
)
from prom3theus.observations import (
    CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    ObservationRaster,
    resolve_velocity_synthesis_mode,
)
from typing import Any

from .errors import ArtifactExportError
from .validation import (
    removed_solar_los_velocity,
    scene_basis,
    validate_raster_velocity_contract,
)


_STORAGE_DTYPES = {
    "float32": (torch.float32, np.dtype(np.float32)),
    "float64": (torch.float64, np.dtype(np.float64)),
}


def _spatial_los_offset(
    value: object,
    spatial_shape: tuple[int, int],
    *,
    name: str,
) -> torch.Tensor:
    """Normalize one observation LOS offset to a CPU ``[Y, X]`` tensor."""

    offset = torch.as_tensor(materialize_array(value)).detach().cpu()
    if tuple(offset.shape) == (*spatial_shape, 1):
        offset = offset[..., 0]
    if tuple(offset.shape) != spatial_shape:
        raise ArtifactExportError(
            f"{name} must have shape {spatial_shape} or {(*spatial_shape, 1)}."
        )
    return offset


def _observation_los_offset(
    raster: ObservationRaster,
    velocity_synthesis_mode: str,
) -> torch.Tensor:
    """Resolve the authoritative per-pixel detector spectral-frame offset."""

    if velocity_synthesis_mode == CARRINGTON_OBSERVER_RELATIVE_VELOCITY:
        return _spatial_los_offset(
            raster.auxiliary["observer_los_velocity_m_per_s"],
            raster.spatial_shape,
            name="observer_los_velocity_m_per_s",
        )

    removed_per_column = torch.as_tensor(
        removed_solar_los_velocity(raster), dtype=torch.float64
    )
    authoritative = removed_per_column.unsqueeze(0).expand(raster.spatial_shape)
    duplicate = raster.auxiliary.get("removed_solar_los_velocity_m_per_s")
    if duplicate is not None:
        duplicate = _spatial_los_offset(
            duplicate,
            raster.spatial_shape,
            name="removed_solar_los_velocity_m_per_s auxiliary",
        )
        if not torch.equal(
            duplicate,
            authoritative.to(dtype=duplicate.dtype),
        ):
            raise ArtifactExportError(
                "The removed_solar_los_velocity_m_per_s auxiliary does not match "
                "the authoritative observer_velocity_correction metadata."
            )
    return authoritative


def resolve_storage_dtype(name: str) -> tuple[torch.dtype, np.dtype]:
    """Resolve the exact supported archive storage dtype."""

    try:
        return _STORAGE_DTYPES[str(name)]
    except KeyError as error:
        raise ValueError("storage_dtype must be 'float32' or 'float64'.") from error


def select_export_device(
    name: str,
    compute_dtype: torch.dtype,
    *,
    include_stokes: bool,
) -> torch.device:
    """Select an evaluation device supporting the model precision."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps":
        raise ValueError("LTE ray geometry requires float64; use CPU or CUDA for export.")
    return device


def photospheric_height_grid(module: Any, sample_count: int | None = 101) -> torch.Tensor:
    """Independent diagnostic height samples; these do not discretize transfer."""
    if sample_count is None:
        sample_count = 101
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("depth_samples must be an integer of at least two.")
    parameter = next(module.atmosphere_model.parameters())
    inner, outer = module.synthesis_window.height_bounds_m.detach().cpu().tolist()
    return torch.linspace(outer, inner, sample_count, dtype=parameter.dtype, device=parameter.device)


@torch.inference_mode()
def evaluate_atmosphere(
    module: Any,
    raster: ObservationRaster,
    *,
    height_grid_m: torch.Tensor,
    batch_size: int,
    storage_dtype: str,
) -> dict[str, np.ndarray]:
    """Evaluate the continuous atmosphere on every valid raster coordinate."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    torch_dtype, numpy_dtype = resolve_storage_dtype(storage_dtype)
    model = module.atmosphere_model
    continuum_opacity = module.synthesis_window.coefficients.plasma_provider
    parameter = next(model.parameters())
    mode = resolve_velocity_synthesis_mode(module.velocity_synthesis_mode).value
    validate_raster_velocity_contract(raster, mode)
    line_of_sight_velocity_correction = float(
        module.instrument_line_of_sight_velocity_correction_m_per_s.detach().cpu()
    )
    if not math.isfinite(line_of_sight_velocity_correction):
        raise ArtifactExportError(
            "The trained instrument LOS-velocity correction is not finite."
        )

    valid_flat = materialize_array(raster.valid_mask).reshape(-1)
    flat_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        raise ArtifactExportError("The canonical observation contains no valid pixels.")
    coordinates = read_flat_samples(raster.coordinates, flat_indices)
    rays = read_flat_samples(raster.ray_direction, flat_indices)
    bases = read_flat_samples(raster.stokes_basis, flat_indices)
    spatial_shape = raster.spatial_shape
    observation_los_offset = _observation_los_offset(raster, mode)
    # Observer/registration and instrument shifts act on detector wavelengths;
    # they are never added to the local material-velocity vector.
    beta_observer = observation_los_offset.to(torch.float64) / SPEED_OF_LIGHT
    beta_calibration = line_of_sight_velocity_correction / SPEED_OF_LIGHT
    if torch.any(beta_observer.abs() >= 1) or abs(beta_calibration) >= 1:
        raise ArtifactExportError("Spectral-frame velocities must be subluminal.")
    wavelength_factor = (
        torch.sqrt((1 - beta_observer) / (1 + beta_observer))
        * math.sqrt((1 - beta_calibration) / (1 + beta_calibration))
    )
    depth = int(height_grid_m.numel())
    flat_size = int(valid_flat.numel())

    scalar_fields = {
        name: np.full((flat_size, depth), np.nan, dtype=numpy_dtype)
        for name in (
            "temperature",
            "microturbulence",
            "gas_pressure",
            "mass_density",
            "geometric_height",
            "alpha500",
            "tau500_radial",
        )
    }
    vector_fields = {
        name: np.full((flat_size, depth, 3), np.nan, dtype=numpy_dtype)
        for name in (
            "magnetic",
            "velocity",
            "spherical_position",
            "magnetic_spherical",
            "velocity_spherical",
            "rotation",
            "inertial_velocity",
            "inertial_velocity_spherical",
            "magnetic_observer",
            "corotating_velocity_observer",
            "inertial_velocity_observer",
        )
    }
    was_training = model.training
    model.eval()
    try:
        for start in range(0, coordinates.shape[0], batch_size):
            stop = min(start + batch_size, coordinates.shape[0])
            module.synthesis_window.boundary_queries(
                coordinates[start:stop].to(parameter), rays[start:stop].to(parameter)
            )
            atmosphere, trace = model.trace_rays(
                coordinates[start:stop].to(parameter),
                rays[start:stop].to(parameter),
                height_grid_m,
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

            store_scalar("temperature", atmosphere.temperature)
            store_scalar("microturbulence", atmosphere.microturbulence)
            store_scalar("gas_pressure", atmosphere.gas_pressure)
            store_scalar("geometric_height", atmosphere.geometric_height_m)
            alpha500 = continuum_opacity.volume_extinction_at_5000(
                atmosphere.temperature,
                atmosphere.gas_pressure,
            )
            intervals_m = (
                atmosphere.geometric_height_m[..., :-1]
                - atmosphere.geometric_height_m[..., 1:]
            )
            increments = 0.5 * (alpha500[..., :-1] + alpha500[..., 1:]) * intervals_m
            tau500_radial = torch.cat(
                (
                    torch.zeros_like(alpha500[..., :1]),
                    torch.cumsum(increments, dim=-1),
                ),
                dim=-1,
            )
            store_scalar("alpha500", alpha500)
            store_scalar("tau500_radial", tau500_radial)
            store_scalar(
                "mass_density",
                model.thermodynamic_eos.mass_density(
                    atmosphere.temperature,
                    atmosphere.gas_pressure,
                ),
            )
            spherical = cartesian_to_spherical(trace.position_m, torch)
            magnetic_spherical = project_cartesian_to_spherical(
                atmosphere.magnetic_field, spherical, torch
            )
            velocity_spherical = project_cartesian_to_spherical(
                atmosphere.velocity_field, spherical, torch
            )
            rotation = (
                carrington_rotation_velocity_cartesian(trace.position_m, torch)
                if mode
                in {
                    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
                    CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
                }
                else torch.zeros_like(atmosphere.velocity_field)
            )
            inertial_velocity = atmosphere.velocity_field + rotation
            inertial_spherical = project_cartesian_to_spherical(
                inertial_velocity, spherical, torch
            )
            observer_basis = bases[start:stop].to(parameter)
            store_vector("magnetic", atmosphere.magnetic_field)
            store_vector("velocity", atmosphere.velocity_field)
            store_vector("spherical_position", spherical)
            store_vector("magnetic_spherical", magnetic_spherical)
            store_vector("velocity_spherical", velocity_spherical)
            store_vector("rotation", rotation)
            store_vector("inertial_velocity", inertial_velocity)
            store_vector("inertial_velocity_spherical", inertial_spherical)
            store_vector(
                "magnetic_observer",
                project_spherical_to_observer(
                    magnetic_spherical, spherical, observer_basis, torch
                ),
            )
            corotating_observer = project_spherical_to_observer(
                velocity_spherical, spherical, observer_basis, torch
            )
            inertial_observer = project_spherical_to_observer(
                inertial_spherical, spherical, observer_basis, torch
            )
            store_vector("inertial_velocity_observer", inertial_observer)
            store_vector("corotating_velocity_observer", corotating_observer)
    finally:
        model.train(was_training)

    shape = (*spatial_shape, depth)
    vector_shape = (*shape, 3)
    coordinates_grid = materialize_array(raster.coordinates).detach().cpu().numpy()
    scene_basis_matrix = scene_basis(raster).astype(numpy_dtype)
    velocity_scene = np.einsum(
        "ij,ndj->ndi", scene_basis_matrix, vector_fields["velocity"]
    )
    magnetic_scene = np.einsum(
        "ij,ndj->ndi", scene_basis_matrix, vector_fields["magnetic"]
    )
    inertial_observer_velocity = vector_fields["inertial_velocity_observer"].reshape(
        vector_shape
    )
    solar_inertial_los = -inertial_observer_velocity[..., 2]
    result = {
        "photospheric_height_grid_m": height_grid_m.detach()
        .to(torch_dtype)
        .cpu()
        .numpy(),
        "temperature_k": scalar_fields["temperature"].reshape(shape),
        "microturbulence_m_per_s": scalar_fields["microturbulence"].reshape(shape),
        "gas_pressure_pa": scalar_fields["gas_pressure"].reshape(shape),
        "mass_density_kg_m3": scalar_fields["mass_density"].reshape(shape),
        "geometric_height_m": scalar_fields["geometric_height"].reshape(shape),
        "alpha500_m_inv": scalar_fields["alpha500"].reshape(shape),
        "tau500_radial": scalar_fields["tau500_radial"].reshape(shape),
        "velocity_field_m_per_s": vector_fields["velocity"].reshape(vector_shape),
        "velocity_field_scene_m_per_s": velocity_scene.reshape(vector_shape),
        "velocity_field_spherical_m_per_s": vector_fields["velocity_spherical"].reshape(
            vector_shape
        ),
        "carrington_rotation_velocity_m_per_s": vector_fields["rotation"].reshape(
            vector_shape
        ),
        "velocity_field_inertial_m_per_s": vector_fields["inertial_velocity"].reshape(
            vector_shape
        ),
        "velocity_field_inertial_spherical_m_per_s": vector_fields[
            "inertial_velocity_spherical"
        ].reshape(vector_shape),
        "instrument_line_of_sight_velocity_correction_m_per_s": np.asarray(
            line_of_sight_velocity_correction, dtype=numpy_dtype
        ),
        "velocity_field_corotating_observer_m_per_s": vector_fields[
            "corotating_velocity_observer"
        ].reshape(vector_shape),
        "velocity_field_inertial_observer_m_per_s": inertial_observer_velocity,
        "v_los_solar_inertial_m_per_s": solar_inertial_los,
        "detector_to_inertial_wavelength_factor": wavelength_factor.to(torch_dtype)
        .numpy().astype(numpy_dtype, copy=True),
        "magnetic_field_gauss": vector_fields["magnetic"].reshape(vector_shape),
        "magnetic_field_scene_gauss": magnetic_scene.reshape(vector_shape),
        "magnetic_field_spherical_gauss": vector_fields["magnetic_spherical"].reshape(
            vector_shape
        ),
        "magnetic_field_observer_gauss": vector_fields["magnetic_observer"].reshape(
            vector_shape
        ),
        "radius_m": vector_fields["spherical_position"][..., 0].reshape(shape),
        "carrington_colatitude_rad": vector_fields["spherical_position"][
            ..., 1
        ].reshape(shape),
        "carrington_longitude_rad": vector_fields["spherical_position"][..., 2].reshape(
            shape
        ),
        "valid_mask": materialize_array(raster.valid_mask).detach().cpu().numpy(),
        "carrington_chart_x_megameter": coordinates_grid[..., 0].astype(
            numpy_dtype, copy=False
        ),
        "carrington_chart_y_megameter": coordinates_grid[..., 1].astype(
            numpy_dtype, copy=False
        ),
        "time_hours": coordinates_grid[..., 2].astype(numpy_dtype, copy=False),
    }
    result["detector_to_inertial_wavelength_factor"][~result["valid_mask"]] = np.nan
    if mode == CARRINGTON_OBSERVER_RELATIVE_VELOCITY:
        observer_los = (
            observation_los_offset.to(torch_dtype)
            .numpy()
            .astype(numpy_dtype, copy=True)
        )
        observer_los[~result["valid_mask"]] = np.nan
        result["observer_los_velocity_m_per_s"] = observer_los
    elif mode == CARRINGTON_REGISTERED_RELATIVE_VELOCITY:
        removed = (
            observation_los_offset.to(torch_dtype)
            .numpy()
            .astype(numpy_dtype, copy=True)
        )
        removed[~result["valid_mask"]] = np.nan
        result["removed_solar_los_velocity_m_per_s"] = removed
    return result


__all__ = [
    "photospheric_height_grid",
    "evaluate_atmosphere",
    "resolve_storage_dtype",
    "select_export_device",
]
