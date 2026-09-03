"""Atmosphere evaluation for validated LTE inversion artifacts."""

from __future__ import annotations

import math

import numpy as np
import torch

from prom3theus.core import (
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
from prom3theus.training.lightning import LTEInversionModule

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


def _apply_positive_redshift_los_offset(
    base_los_velocity: np.ndarray,
    toward_observer_offset: np.ndarray,
) -> np.ndarray:
    """Apply an offset subtracted from ``v_toward`` to ``v_los=-v_toward``."""

    return base_los_velocity + toward_observer_offset[..., None]


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
    """Select an evaluation device subject to polarized-synthesis constraints."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if (
            not include_stokes
            and compute_dtype != torch.float64
            and mps is not None
            and mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and compute_dtype == torch.float64:
        raise ValueError(
            "MPS does not support float64 export; use CPU/CUDA or a float32 model."
        )
    if device.type == "mps" and include_stokes:
        raise ValueError(
            "MPS cannot run polarized LTE synthesis; use CPU or CUDA with "
            "include_stokes=True."
        )
    return device


def depth_grid(
    module: LTEInversionModule,
    depth_samples: int | None,
) -> torch.Tensor:
    """Build the requested fixed or model-native optical-depth grid."""

    parameter = next(module.atmosphere_model.parameters())
    if depth_samples is None:
        depth = module.sample_depth_grid(randomize=False)
    else:
        if type(depth_samples) is not int or depth_samples < 2:
            raise ValueError("depth_samples must be an integer of at least two.")
        source = module.atmosphere_model.log_tau500
        depth = torch.linspace(
            source[0],
            source[-1],
            depth_samples,
            dtype=parameter.dtype,
            device=parameter.device,
        )
    return depth.to(device=parameter.device, dtype=parameter.dtype)


@torch.inference_mode()
def evaluate_atmosphere(
    module: LTEInversionModule,
    raster: ObservationRaster,
    *,
    depth_grid: torch.Tensor,
    batch_size: int,
    storage_dtype: str,
) -> dict[str, np.ndarray]:
    """Evaluate the continuous atmosphere on every valid raster coordinate."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    torch_dtype, numpy_dtype = resolve_storage_dtype(storage_dtype)
    model = module.atmosphere_model
    continuum_opacity = module.synthesizer.continuum_opacity
    parameter = next(model.parameters())
    mode = resolve_velocity_synthesis_mode(module.velocity_synthesis_mode).value
    validate_raster_velocity_contract(raster, mode)
    radial_velocity_correction = float(
        module.instrument_radial_velocity_correction_m_per_s.detach().cpu()
    )
    if not math.isfinite(radial_velocity_correction):
        raise ArtifactExportError(
            "The trained instrument radial-velocity correction is not finite."
        )

    valid_flat = raster.valid_mask.reshape(-1)
    flat_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        raise ArtifactExportError("The canonical observation contains no valid pixels.")
    coordinates = raster.coordinates.reshape(-1, 3).index_select(0, flat_indices)
    rays = raster.ray_direction.reshape(-1, 3).index_select(0, flat_indices)
    bases = raster.stokes_basis.reshape(-1, 3, 3).index_select(0, flat_indices)
    spatial_shape = raster.spatial_shape
    depth = int(depth_grid.numel())
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
            "instrument_velocity",
            "radial_correction",
            "magnetic_observer",
            "velocity_observer",
            "inertial_velocity_observer",
        )
    }
    was_training = model.training
    model.eval()
    try:
        for start in range(0, coordinates.shape[0], batch_size):
            stop = min(start + batch_size, coordinates.shape[0])
            atmosphere, trace = model.trace_rays(
                coordinates[start:stop].to(parameter),
                rays[start:stop].to(parameter),
                depth_grid,
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
                continuum_opacity.reference_mass_density(
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
            radial_unit = trace.position_m / torch.linalg.vector_norm(
                trace.position_m, dim=-1, keepdim=True
            )
            radial_correction = radial_unit * radial_velocity_correction
            instrument_velocity = inertial_velocity + radial_correction
            instrument_spherical = project_cartesian_to_spherical(
                instrument_velocity, spherical, torch
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
            store_vector("instrument_velocity", instrument_velocity)
            store_vector("radial_correction", radial_correction)
            store_vector(
                "magnetic_observer",
                project_spherical_to_observer(
                    magnetic_spherical, spherical, observer_basis, torch
                ),
            )
            store_vector(
                "inertial_velocity_observer",
                project_spherical_to_observer(
                    inertial_spherical, spherical, observer_basis, torch
                ),
            )
            store_vector(
                "velocity_observer",
                project_spherical_to_observer(
                    instrument_spherical, spherical, observer_basis, torch
                ),
            )
    finally:
        model.train(was_training)

    shape = (*spatial_shape, depth)
    vector_shape = (*shape, 3)
    coordinates_grid = raster.coordinates.detach().cpu().numpy()
    scene_basis_matrix = scene_basis(raster).astype(numpy_dtype)
    velocity_scene = np.einsum(
        "ij,ndj->ndi", scene_basis_matrix, vector_fields["velocity"]
    )
    magnetic_scene = np.einsum(
        "ij,ndj->ndi", scene_basis_matrix, vector_fields["magnetic"]
    )
    observer_velocity = vector_fields["velocity_observer"].reshape(vector_shape)
    inertial_observer_velocity = vector_fields["inertial_velocity_observer"].reshape(
        vector_shape
    )
    solar_inertial_los = -inertial_observer_velocity[..., 2]
    instrument_corrected_los = -observer_velocity[..., 2]
    result = {
        "log_tau500": depth_grid.detach().to(torch_dtype).cpu().numpy(),
        "depth_coordinate": depth_grid.detach().to(torch_dtype).cpu().numpy(),
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
        "instrument_radial_velocity_correction_m_per_s": np.asarray(
            radial_velocity_correction, dtype=numpy_dtype
        ),
        "instrument_radial_velocity_correction_cartesian_m_per_s": vector_fields[
            "radial_correction"
        ].reshape(vector_shape),
        "velocity_field_synthesis_m_per_s": vector_fields[
            "instrument_velocity"
        ].reshape(vector_shape),
        "velocity_field_observer_m_per_s": observer_velocity,
        "v_los_solar_inertial_m_per_s": solar_inertial_los,
        "v_los_instrument_corrected_m_per_s": instrument_corrected_los,
        "v_los_m_per_s": instrument_corrected_los.copy(),
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
        "valid_mask": raster.valid_mask.detach().cpu().numpy(),
        "carrington_chart_x_mm": coordinates_grid[..., 0].astype(
            numpy_dtype, copy=False
        ),
        "carrington_chart_y_mm": coordinates_grid[..., 1].astype(
            numpy_dtype, copy=False
        ),
        "time_hours": coordinates_grid[..., 2].astype(numpy_dtype, copy=False),
    }
    if mode == CARRINGTON_OBSERVER_RELATIVE_VELOCITY:
        observer_los = (
            torch.as_tensor(raster.auxiliary["observer_los_velocity_m_per_s"])
            .detach()
            .to(torch_dtype)
            .cpu()
            .numpy()
        )
        if observer_los.shape == (*spatial_shape, 1):
            observer_los = observer_los[..., 0]
        observer_los = observer_los.astype(numpy_dtype, copy=True)
        observer_los[~result["valid_mask"]] = np.nan
        result["observer_los_velocity_m_per_s"] = observer_los
        result["v_los_m_per_s"] = _apply_positive_redshift_los_offset(
            instrument_corrected_los, observer_los
        )
    elif mode == CARRINGTON_REGISTERED_RELATIVE_VELOCITY:
        removed = np.asarray(
            removed_solar_los_velocity(raster),
            dtype=numpy_dtype,
        )
        removed = np.broadcast_to(removed[None, :], spatial_shape).copy()
        removed[~result["valid_mask"]] = np.nan
        result["removed_solar_los_velocity_m_per_s"] = removed
        result["v_los_m_per_s"] = _apply_positive_redshift_los_offset(
            instrument_corrected_los, removed
        )
    return result


__all__ = [
    "depth_grid",
    "evaluate_atmosphere",
    "resolve_storage_dtype",
    "select_export_device",
]
