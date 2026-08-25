"""Checkpoint-safe evaluation and export for LTE neural atmospheres."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from pme.coordinates import (
    cartesian_to_spherical,
    project_cartesian_to_spherical,
    project_spherical_to_observer,
)
from pme.lte.hinode import HinodeRaster, load_hinode_raster
from pme.solar_velocity import (
    CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
    CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS,
    carrington_rotation_velocity_cartesian,
)
from pme.lte.resources import validate_resource_bundle
from pme.train.lte_module import LTEModule
from pme.train.util import load_yaml_config


_DATAMODULE_ONLY_DATA_KEYS = frozenset({
    "batch_size",
    "validation_batch_size",
    "validation_stride",
    "num_workers",
    "pin_memory",
})
_STORAGE_DTYPES = {
    "float32": (torch.float32, np.dtype(np.float32)),
    "float64": (torch.float64, np.dtype(np.float64)),
}


def _storage_dtype(name: str) -> tuple[torch.dtype, np.dtype]:
    try:
        return _STORAGE_DTYPES[str(name)]
    except KeyError as error:
        raise ValueError("storage_dtype must be 'float32' or 'float64'.") from error


def _export_device(
    name: str,
    compute_dtype: torch.dtype,
    *,
    include_stokes: bool,
) -> torch.device:
    """Resolve an explicit export device without changing the RT implementation."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if not include_stokes and compute_dtype != torch.float64:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and compute_dtype == torch.float64:
        raise ValueError(
            "MPS does not support the checkpoint's float64 compute dtype; "
            "use CPU/CUDA or a float32 checkpoint."
        )
    if device.type == "mps" and include_stokes:
        raise ValueError(
            "MPS has no torch.matrix_exp kernel for polarized LTE synthesis; "
            "use CPU or CUDA with --include-stokes."
        )
    return device


def _raster_loader_config(data_config: dict) -> dict:
    """Return only arguments accepted by :func:`load_hinode_raster`."""

    resolved = dict(data_config)
    data_type = resolved.pop("type", "hinode_lte")
    if data_type not in ("hinode_lte", "hinode-lte"):
        raise ValueError(f"Expected Hinode LTE data, got {data_type!r}.")
    for key in _DATAMODULE_ONLY_DATA_KEYS:
        resolved.pop(key, None)
    return resolved


def _normalization_contract(metadata: dict) -> dict:
    """Extract the data-independent Stokes radiometric-calibration definition."""

    keys = (
        "operation",
        "type",
        "edge_samples",
        "windows_angstrom",
        "indices",
        "wavelength_angstrom",
    )
    contract = {key: metadata.get(key) for key in keys if key in metadata}
    calibration = metadata.get("radiometric_calibration")
    if isinstance(calibration, dict):
        quiet_sun = calibration.get("quiet_sun", {})
        contract["radiometric_calibration"] = {
            key: calibration[key]
            for key in (
                "type",
                "operation",
                "stored_stokes_unit",
                "physical_radiance_unit",
                "atlas_disk_center_continuum_radiance_w_m3_sr",
                "reference",
            )
            if key in calibration
        }
        if isinstance(quiet_sun, dict):
            contract["radiometric_calibration"]["quiet_sun"] = {
                key: quiet_sun[key]
                for key in (
                    "selection",
                    "maximum_mean_fractional_polarization",
                    "continuum_trim_quantiles",
                    "minimum_pixel_count",
                )
                if key in quiet_sun
            }
    return contract


def _relocated_resource_configs(
    checkpoint_payload: dict,
    resource_metadata: dict,
) -> tuple[dict, dict]:
    """Retarget saved module paths to a checksum-equivalent prepared bundle."""

    saved_resources = (
        checkpoint_payload.get("lte_metadata", {})
        .get("data", {})
        .get("lte_resources")
    )
    if not isinstance(saved_resources, dict):
        raise RuntimeError(
            "LTE checkpoint does not record its prepared-resource provenance."
        )
    if saved_resources.get("instrument") != resource_metadata.get("instrument"):
        raise RuntimeError(
            "Configured export resources use a different instrument than the checkpoint."
        )
    saved_hashes = saved_resources.get("resource_sha256", {})
    current_hashes = resource_metadata.get("resource_sha256", {})
    for filename in resource_metadata.get("required_production_files", ()):
        if saved_hashes.get(filename) != current_hashes.get(filename):
            raise RuntimeError(
                "Configured export resources differ from the checkpoint's production "
                f"resource {filename!r}."
            )
    hyperparameters = checkpoint_payload.get("hyper_parameters")
    if not isinstance(hyperparameters, dict):
        raise RuntimeError("LTE checkpoint has no reconstructible hyperparameters.")
    synthesizer_config = deepcopy(hyperparameters.get("synthesizer_config", {}))
    instrument_config = deepcopy(hyperparameters.get("instrument_config", {}))
    resource_directory = resource_metadata["directory"]
    synthesizer_config["atomic_data_directory"] = resource_directory
    instrument_config["data_directory"] = resource_directory
    return synthesizer_config, instrument_config


@torch.inference_mode()
def evaluate_atmosphere_model(
    model,
    raster: HinodeRaster,
    *,
    log_tau500=None,
    batch_size: int = 4096,
    continuum_opacity,
    storage_dtype: str = "float32",
) -> dict[str, np.ndarray]:
    """Evaluate a continuous LTE atmosphere on every valid physical raster coordinate."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    parameter = next(model.parameters())
    torch_storage_dtype, numpy_storage_dtype = _storage_dtype(storage_dtype)
    if log_tau500 is None:
        depth_grid = model.log_tau500
    else:
        depth_grid = torch.as_tensor(
            log_tau500,
            dtype=parameter.dtype,
            device=parameter.device,
        )
    # Model validation enforces ordering and domain bounds.
    valid_flat = raster.valid_mask.reshape(-1)
    flat_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        raise ValueError("The raster contains no valid pixels to evaluate.")
    coords = raster.coords.reshape(-1, 3)[flat_indices]
    ray_origin_m = raster.ray_origin_m.reshape(-1, 3)[flat_indices]
    ray_direction = raster.ray_direction.reshape(-1, 3)[flat_indices]
    stokes_basis_valid = raster.stokes_basis.reshape(-1, 3, 3)[flat_indices]
    spatial_shape = raster.spatial_shape
    depth = int(depth_grid.numel())
    flat_size = int(valid_flat.numel())
    scalar_names = [
        "temperature", "microturbulence", "gas_pressure", "mass_density",
        "geometric_height", "alpha500", "tau500_radial",
    ]
    scalar_fields = {
        name: np.full((flat_size, depth), np.nan, dtype=numpy_storage_dtype)
        for name in scalar_names
    }
    magnetic_field = np.full(
        (flat_size, depth, 3), np.nan, dtype=numpy_storage_dtype
    )
    velocity_field = np.full(
        (flat_size, depth, 3), np.nan, dtype=numpy_storage_dtype
    )
    spherical_position = np.full(
        (flat_size, depth, 3), np.nan, dtype=numpy_storage_dtype
    )
    magnetic_spherical = np.full_like(magnetic_field, np.nan)
    velocity_spherical = np.full_like(velocity_field, np.nan)
    rotation_velocity = np.full_like(velocity_field, np.nan)
    velocity_inertial = np.full_like(velocity_field, np.nan)
    velocity_inertial_spherical = np.full_like(velocity_field, np.nan)
    magnetic_observer = np.full_like(magnetic_field, np.nan)
    velocity_observer = np.full_like(velocity_field, np.nan)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, coords.shape[0], batch_size):
            stop = min(start + batch_size, coords.shape[0])
            atmosphere, trace = model.trace_rays(
                coords[start:stop].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                ray_origin_m[start:stop].to(device=parameter.device),
                ray_direction[start:stop].to(device=parameter.device),
                depth_grid,
            )
            selected = flat_indices[start:stop].cpu().numpy()

            def store(name, values):
                scalar_fields[name][selected] = (
                    values.detach().to(dtype=torch_storage_dtype).cpu().numpy()
                )

            store("temperature", atmosphere.temperature)
            store("microturbulence", atmosphere.microturbulence)
            store("gas_pressure", atmosphere.gas_pressure)
            store("geometric_height", atmosphere.geometric_height_m)
            alpha500 = continuum_opacity.volume_extinction_at_5000(
                atmosphere.temperature, atmosphere.gas_pressure
            )
            interval_m = (
                atmosphere.geometric_height_m[..., :-1]
                - atmosphere.geometric_height_m[..., 1:]
            )
            increments = 0.5 * (
                alpha500[..., :-1] + alpha500[..., 1:]
            ) * interval_m
            tau500_radial = torch.cat(
                (
                    torch.zeros_like(alpha500[..., :1]),
                    torch.cumsum(increments, dim=-1),
                ),
                dim=-1,
            )
            store("alpha500", alpha500)
            store("tau500_radial", tau500_radial)
            store(
                "mass_density",
                continuum_opacity.reference_mass_density(
                    atmosphere.temperature, atmosphere.gas_pressure
                ),
            )
            magnetic_field[selected] = (
                atmosphere.magnetic_field.detach()
                .to(dtype=torch_storage_dtype)
                .cpu()
                .numpy()
            )
            velocity_field[selected] = (
                atmosphere.velocity_field.detach()
                .to(dtype=torch_storage_dtype)
                .cpu()
                .numpy()
            )
            spherical = cartesian_to_spherical(trace.position_m, torch)
            magnetic_local = project_cartesian_to_spherical(
                atmosphere.magnetic_field, spherical, torch
            )
            velocity_local = project_cartesian_to_spherical(
                atmosphere.velocity_field, spherical, torch
            )
            rotation = carrington_rotation_velocity_cartesian(
                trace.position_m, torch
            )
            inertial_velocity = atmosphere.velocity_field + rotation
            inertial_velocity_local = project_cartesian_to_spherical(
                inertial_velocity, spherical, torch
            )
            observer_basis = stokes_basis_valid[start:stop].to(
                device=parameter.device, dtype=parameter.dtype
            )
            magnetic_view = project_spherical_to_observer(
                magnetic_local, spherical, observer_basis, torch
            )
            velocity_view = project_spherical_to_observer(
                inertial_velocity_local, spherical, observer_basis, torch
            )
            for target, value in (
                (spherical_position, spherical),
                (magnetic_spherical, magnetic_local),
                (velocity_spherical, velocity_local),
                (rotation_velocity, rotation),
                (velocity_inertial, inertial_velocity),
                (velocity_inertial_spherical, inertial_velocity_local),
                (magnetic_observer, magnetic_view),
                (velocity_observer, velocity_view),
            ):
                target[selected] = (
                    value.detach().to(dtype=torch_storage_dtype).cpu().numpy()
                )
    finally:
        model.train(was_training)

    physical_coords = raster.coords.cpu().numpy()
    scene_basis = np.asarray(
        raster.metadata["ray_geometry"]["scene_basis_rows"], dtype=numpy_storage_dtype
    )
    velocity_scene = np.einsum("ij,ndj->ndi", scene_basis, velocity_field)
    magnetic_scene = np.einsum("ij,ndj->ndi", scene_basis, magnetic_field)
    result = {
        "log_tau500": depth_grid.detach().to(
            dtype=torch_storage_dtype
        ).cpu().numpy(),
        "temperature_k": scalar_fields["temperature"].reshape(*spatial_shape, depth),
        "velocity_field_m_per_s": velocity_field.reshape(
            *spatial_shape, depth, 3
        ),
        "velocity_field_scene_m_per_s": velocity_scene.reshape(
            *spatial_shape, depth, 3
        ),
        "velocity_field_spherical_m_per_s": velocity_spherical.reshape(
            *spatial_shape, depth, 3
        ),
        "carrington_rotation_velocity_m_per_s": rotation_velocity.reshape(
            *spatial_shape, depth, 3
        ),
        "velocity_field_inertial_m_per_s": velocity_inertial.reshape(
            *spatial_shape, depth, 3
        ),
        "velocity_field_inertial_spherical_m_per_s": (
            velocity_inertial_spherical.reshape(*spatial_shape, depth, 3)
        ),
        "velocity_field_observer_m_per_s": velocity_observer.reshape(
            *spatial_shape, depth, 3
        ),
        "v_los_m_per_s": (-velocity_observer[..., 2]).reshape(*spatial_shape, depth),
        "microturbulence_m_per_s": scalar_fields["microturbulence"].reshape(
            *spatial_shape, depth
        ),
        "gas_pressure_pa": scalar_fields["gas_pressure"].reshape(*spatial_shape, depth),
        "magnetic_field_gauss": magnetic_field.reshape(*spatial_shape, depth, 3),
        "magnetic_field_scene_gauss": magnetic_scene.reshape(
            *spatial_shape, depth, 3
        ),
        "magnetic_field_spherical_gauss": magnetic_spherical.reshape(
            *spatial_shape, depth, 3
        ),
        "magnetic_field_observer_gauss": magnetic_observer.reshape(
            *spatial_shape, depth, 3
        ),
        "radius_m": spherical_position[..., 0].reshape(*spatial_shape, depth),
        "carrington_colatitude_rad": spherical_position[..., 1].reshape(
            *spatial_shape, depth
        ),
        "carrington_longitude_rad": spherical_position[..., 2].reshape(
            *spatial_shape, depth
        ),
        "valid_mask": raster.valid_mask.cpu().numpy(),
        "time_hours": physical_coords[..., 0].astype(numpy_storage_dtype, copy=False),
        "carrington_chart_x_mm": physical_coords[..., 1].astype(
            numpy_storage_dtype, copy=False
        ),
        "carrington_chart_y_mm": physical_coords[..., 2].astype(
            numpy_storage_dtype, copy=False
        ),
    }
    result["mass_density_kg_m3"] = scalar_fields["mass_density"].reshape(
        *spatial_shape, depth
    )
    if "geometric_height" in scalar_fields:
        result["geometric_height_m"] = scalar_fields["geometric_height"].reshape(
            *spatial_shape, depth
        )
    if "alpha500" in scalar_fields:
        result["depth_coordinate"] = result["log_tau500"].copy()
        result["alpha500_m_inv"] = scalar_fields["alpha500"].reshape(
            *spatial_shape, depth
        )
        result["tau500_radial"] = scalar_fields["tau500_radial"].reshape(
            *spatial_shape, depth
        )
    return result


@torch.inference_mode()
def evaluate_stokes_model(
    module: LTEModule,
    raster: HinodeRaster,
    *,
    batch_size: int = 16,
    storage_dtype: str = "float32",
) -> dict[str, np.ndarray]:
    """Synthesize fitted Stokes profiles and residuals on valid raster pixels."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    parameter = next(module.atmosphere_model.parameters())
    torch_storage_dtype, numpy_storage_dtype = _storage_dtype(storage_dtype)
    valid_flat = raster.valid_mask.reshape(-1)
    flat_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        raise ValueError("The raster contains no valid pixels to synthesize.")
    coords = raster.coords.reshape(-1, 3)[flat_indices]
    mu = raster.mu.reshape(-1, 1)[flat_indices]
    ray_origin_m = raster.ray_origin_m.reshape(-1, 3)[flat_indices]
    ray_direction = raster.ray_direction.reshape(-1, 3)[flat_indices]
    stokes_basis = raster.stokes_basis.reshape(-1, 3, 3)[flat_indices]
    wavelength_count = int(raster.wavelength_angstrom.numel())
    predicted_flat = np.full(
        (valid_flat.numel(), 4, wavelength_count),
        np.nan,
        dtype=numpy_storage_dtype,
    )
    was_training = module.training
    module.eval()
    try:
        for start in range(0, coords.shape[0], batch_size):
            stop = min(start + batch_size, coords.shape[0])
            prediction = module.synthesize(
                coords[start:stop].to(
                    device=parameter.device, dtype=parameter.dtype
                ),
                mu[start:stop].to(
                    device=parameter.device, dtype=parameter.dtype
                ),
                ray_origin_m=ray_origin_m[start:stop].to(device=parameter.device),
                ray_direction=ray_direction[start:stop].to(device=parameter.device),
                stokes_basis=stokes_basis[start:stop].to(
                    device=parameter.device, dtype=parameter.dtype
                ),
            )["stokes"]
            predicted_flat[flat_indices[start:stop].cpu().numpy()] = (
                prediction.detach().to(dtype=torch_storage_dtype).cpu().numpy()
            )
    finally:
        module.train(was_training)

    spatial_shape = raster.spatial_shape
    predicted = predicted_flat.reshape(*spatial_shape, 4, wavelength_count)
    observed = raster.stokes.to(dtype=torch_storage_dtype).cpu().numpy()
    return {
        "wavelength_angstrom": raster.wavelength_angstrom.to(
            dtype=torch_storage_dtype
        ).cpu().numpy(),
        "predicted_stokes": predicted,
        "observed_stokes": observed,
        "stokes_residual": predicted - observed,
    }


def export_lte_checkpoint(
    checkpoint,
    config,
    output,
    *,
    depth_samples: int | None = None,
    batch_size: int = 4096,
    strict_raster_match: bool = True,
    include_stokes: bool = False,
    stokes_batch_size: int = 16,
    storage_dtype: str = "float32",
    device: str = "auto",
) -> Path:
    """Load an LTE checkpoint and export its atmosphere as compressed NPZ."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    parsed_config = (
        load_yaml_config(config) if isinstance(config, (str, Path)) else dict(config)
    )
    resources_config = parsed_config.get("resources")
    if not isinstance(resources_config, dict) or "directory" not in resources_config:
        raise KeyError(
            "Export configuration must define resources.directory for the prepared bundle."
        )
    resource_metadata = validate_resource_bundle(resources_config["directory"])
    expected_instrument = resources_config.get("instrument")
    if expected_instrument is not None and expected_instrument != (
        resource_metadata.get("instrument")
    ):
        raise ValueError(
            "Prepared export bundle instrument does not match resources.instrument."
        )
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    synthesizer_config, instrument_config = _relocated_resource_configs(
        checkpoint_payload, resource_metadata
    )
    raster_config = _raster_loader_config(parsed_config["data"])
    configured_calibration_directory = raster_config.get(
        "calibration_data_directory"
    )
    if configured_calibration_directory is not None and (
        Path(configured_calibration_directory).expanduser().resolve()
        != Path(resource_metadata["directory"]).expanduser().resolve()
    ):
        raise ValueError(
            "data.calibration_data_directory must match resources.directory."
        )
    raster_config["calibration_data_directory"] = resource_metadata["directory"]
    raster = load_hinode_raster(**raster_config)
    module = LTEModule.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        synthesizer_config=synthesizer_config,
        instrument_config=instrument_config,
    )
    export_device = _export_device(
        device,
        next(module.parameters()).dtype,
        include_stokes=include_stokes,
    )
    module = module.to(export_device).eval()
    checkpoint_wavelength = module.wavelength_angstrom.detach().cpu()
    if checkpoint_wavelength.shape != raster.wavelength_angstrom.shape or not torch.allclose(
        checkpoint_wavelength.to(torch.float64),
        raster.wavelength_angstrom.to(torch.float64),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError(
            "Export raster wavelength grid differs from the checkpoint synthesis grid."
        )
    trained = module.checkpoint_metadata
    current = raster.metadata
    if _normalization_contract(trained.get("normalization", {})) != (
        _normalization_contract(current.get("normalization", {}))
    ):
        raise ValueError(
            "Export raster atlas radiometric calibration differs from the checkpoint. "
            "Raster-mismatch mode relaxes spatial identity only, never the Stokes "
            "calibration contract."
        )
    if trained.get("stokes_order") != current.get("stokes_order"):
        raise ValueError("Export raster Stokes-component order differs from the checkpoint.")
    if strict_raster_match:
        for key in (
            "scan_indices",
            "slit_indices",
            "ref_time",
            "coordinates",
            "data_fingerprints",
        ):
            if trained.get(key) != current.get(key):
                raise ValueError(
                    f"Export raster differs from the training raster in {key!r}. "
                    "Pass strict_raster_match=False only for deliberate coordinate extrapolation."
                )

    depth_grid = None
    if depth_samples is not None:
        if depth_samples < 2:
            raise ValueError("depth_samples must be at least two.")
        reference = module.atmosphere_model.log_tau500
        depth_grid = torch.linspace(
            reference[0],
            reference[-1],
            int(depth_samples),
            dtype=reference.dtype,
            device=reference.device,
        )
    arrays = evaluate_atmosphere_model(
        module.atmosphere_model,
        raster,
        log_tau500=depth_grid,
        batch_size=batch_size,
        continuum_opacity=module.synthesizer.continuum_opacity,
        storage_dtype=storage_dtype,
    )
    if include_stokes:
        arrays.update(
            evaluate_stokes_model(
                module,
                raster,
                batch_size=stokes_batch_size,
                storage_dtype=storage_dtype,
            )
        )
    checkpoint_probe = {}
    module.on_save_checkpoint(checkpoint_probe)
    metadata = {
        "schema_version": 6,
        "checkpoint": str(checkpoint),
        "representation": (
            "continuous F(x,y,z) atmosphere evaluated on fixed spherical shell "
            "radii; tau500 is derived by integrating absolute opacity"
        ),
        "vector_fields": {
            "magnetic_field_gauss": (
                "[B_Xc, B_Yc, B_Zc] in Heliographic Carrington Cartesian; "
                "+Zc is solar north and Xc/Yc span the equatorial plane"
            ),
            "magnetic_field_spherical_gauss": (
                "[B_r, B_theta, B_phi] in the local Carrington spherical basis"
            ),
            "magnetic_field_observer_gauss": (
                "[B_Q, B_U, B_LOS] in each pixel's observer Stokes basis"
            ),
            "magnetic_field_scene_gauss": (
                "[B_chart_x, B_chart_y, B_chart_normal] in the fixed "
                "raster-centred scene basis; these axes match the saved "
                "Carrington gnomonic chart at its tangent point"
            ),
            "velocity_field_m_per_s": (
                "learned co-rotating residual [v_Xc, v_Yc, v_Zc] in "
                "Heliographic Carrington Cartesian"
            ),
            "carrington_rotation_velocity_m_per_s": (
                "rigid sidereal Carrington velocity Omega cross r in Cartesian m/s"
            ),
            "velocity_field_inertial_m_per_s": (
                "learned velocity plus rigid Carrington rotation in Cartesian m/s"
            ),
            "velocity_field_scene_m_per_s": (
                "[v_chart_x, v_chart_y, v_chart_normal] in the fixed "
                "raster-centred scene basis"
            ),
            "velocity_field_spherical_m_per_s": (
                "learned co-rotating [v_r, v_theta, v_phi] in the local "
                "Carrington spherical basis"
            ),
            "velocity_field_inertial_spherical_m_per_s": (
                "learned velocity plus rigid rotation in local spherical components"
            ),
            "velocity_field_observer_m_per_s": (
                "[v_Q, v_U, v_toward] in each pixel's observer Stokes basis"
            ),
            "v_los_m_per_s": (
                "negative projection onto the per-pixel toward-observer axis; "
                "positive values are redshifts"
            ),
            "velocity_observability": (
                "the two observer-transverse velocity combinations are not "
                "constrained by single-view LTE Stokes synthesis without "
                "additional dynamical physics"
            ),
        },
        "coordinate_arrays": {
            "time_hours": "hours since source_raster.coordinates.time_origin",
            "carrington_chart_x_mm": "observer-independent gnomonic chart X in Mm",
            "carrington_chart_y_mm": "observer-independent gnomonic chart Y in Mm",
            "radius_m": "heliocentric radius at each traced shell point",
            "carrington_colatitude_rad": (
                "Carrington colatitude at each traced shell point"
            ),
            "carrington_longitude_rad": (
                "Carrington longitude at each traced shell point"
            ),
            "mass_density_kg_m3": (
                "pinned STiC/Wittmann lookup evaluated at inferred T and Pgas; "
                "this is the density used by the configured MHS residual"
            ),
        },
        "carrington_rotation": {
            "sidereal_period_days": CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS,
            "angular_velocity_rad_per_s": CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
            "axis": "+Zc",
        },
        "source_raster": raster.metadata,
        "checkpoint_data": module.checkpoint_metadata,
        "export_resources": resource_metadata,
        "lte_metadata": checkpoint_probe["lte_metadata"],
        "strict_raster_match": bool(strict_raster_match),
        "includes_stokes": bool(include_stokes),
        "storage_dtype": str(storage_dtype),
        "compute_device": str(export_device),
    }
    if "geometric_height_m" in arrays:
        metadata["coordinate_arrays"]["geometric_height_m"] = (
            "height increasing upward; the checkpoint's fixed valid-FOV "
            "quadrature has mean z=0 at log_tau500=0"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output


def main(argv=None) -> None:
    """Command-line entry point for checkpoint-safe LTE export."""

    parser = argparse.ArgumentParser(
        description="Export a trained LTE neural atmosphere to a dense optical-depth grid."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-samples", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--include-stokes", action="store_true")
    parser.add_argument("--stokes-batch-size", type=int, default=16)
    parser.add_argument(
        "--storage-dtype",
        choices=tuple(_STORAGE_DTYPES),
        default="float32",
        help="numeric dtype stored in the NPZ (float32 limits full-raster memory)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "export compute device (auto uses CUDA when available, MPS only for "
            "float32 atmosphere-only export, otherwise CPU)"
        ),
    )
    parser.add_argument(
        "--allow-raster-mismatch",
        action="store_true",
        help="allow deliberate coordinate extrapolation beyond the training raster",
    )
    args = parser.parse_args(argv)
    path = export_lte_checkpoint(
        args.checkpoint,
        args.config,
        args.output,
        depth_samples=args.depth_samples,
        batch_size=args.batch_size,
        strict_raster_match=not args.allow_raster_mismatch,
        include_stokes=args.include_stokes,
        stokes_batch_size=args.stokes_batch_size,
        storage_dtype=args.storage_dtype,
        device=args.device,
    )
    print(path)


__all__ = [
    "evaluate_atmosphere_model",
    "evaluate_stokes_model",
    "export_lte_checkpoint",
    "main",
]


if __name__ == "__main__":
    main()
