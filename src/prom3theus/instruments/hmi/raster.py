"""Assembly of one calibrated HMI acquisition into the observation contract."""

from __future__ import annotations

import numpy as np
import torch
from astropy import units as u
from astropy.time import Time
from sunpy.map import Map

from prom3theus.observations import ObservationRaster
from prom3theus.rt.geometry import direction_to_chart_mm
from prom3theus.rt.radiometry import load_solar_reference

from .acquisition import (
    SEGMENT_KEYS,
    hmi_observation_wcs_header,
    read_acquisition_header,
    read_stokes_cube,
    resolve_segment_files,
)
from .calibration import array_sha256, calibrate_stokes, sample_response_grid
from .constants import (
    HMI_OBSERVER_VELOCITY_KEYS,
    HMI_WAVELENGTH_CENTER,
    HMI_WAVELENGTH_GRID,
    SPEED_OF_LIGHT_M_PER_S,
)
from .geometry import build_geometry, detector_coordinates, observer_velocity_rwn
from .response import HMIResponseArchive, resolve_response_profile


HMI_SOLAR_REFERENCE_RESOURCE = "hmi_stokes/solar_reference_617nm.json"


def load_raster(
    files,
    transmission_profile_directory,
    require_quality_zero: bool = True,
    quiet_sun_max_fractional_polarization: float = 0.01,
    quiet_sun_trim_quantiles=(0.05, 0.95),
    minimum_quiet_sun_pixels: int = 32,
    calibration_sample_limit: int = 4096,
    reference_time=None,
    scene_basis_rows=None,
    response_sampler: HMIResponseArchive | None = None,
    solar_reference=None,
    acquisition: dict | None = None,
):
    """Load, calibrate, and geometrically register one HMI acquisition."""

    segments = resolve_segment_files(files)
    headers, stokes = read_stokes_cube(
        segments, require_quality_zero=require_quality_zero
    )
    observer_velocity = observer_velocity_rwn(headers)
    reference_file = segments["I0"]
    if acquisition is None:
        acquisition = read_acquisition_header(reference_file)
    if response_sampler is None:
        profile_file = resolve_response_profile(
            transmission_profile_directory, reference_file
        )
        response_sampler = HMIResponseArchive(profile_file)
    else:
        profile_file = response_sampler.profile_file

    s_map = Map(stokes[..., 0, 5], hmi_observation_wcs_header(headers[0]))
    detector_grid = detector_coordinates(s_map)
    valid = np.isfinite(stokes).all(axis=(-2, -1)) & (stokes[..., 0, :].mean(-1) > 0)
    (
        rays,
        surface,
        bases,
        mu,
        observer_los_velocity,
        ray_metadata,
    ) = build_geometry(
        s_map,
        valid,
        observer_velocity,
        scene_basis_rows=scene_basis_rows,
    )
    valid &= np.isfinite(mu) & (mu > 0)
    if not np.any(valid):
        raise ValueError(
            "The selected HMI acquisition contains no valid on-disk pixels."
        )
    first_valid = tuple(np.argwhere(valid)[0])
    for geometry_array in (rays, surface, bases):
        geometry_array[~valid] = geometry_array[first_valid]
    mu[~valid] = 0.0
    observer_los_velocity[~valid] = 0.0

    if solar_reference is None:
        solar_reference = load_solar_reference(
            resource_name=HMI_SOLAR_REFERENCE_RESOURCE,
        )
    stokes, radiometry = calibrate_stokes(
        stokes,
        mu,
        valid,
        detector_grid,
        response_sampler,
        solar_reference,
        observer_los_velocity,
        quiet_sun_max_fractional_polarization=quiet_sun_max_fractional_polarization,
        quiet_sun_trim_quantiles=quiet_sun_trim_quantiles,
        minimum_quiet_sun_pixels=minimum_quiet_sun_pixels,
        calibration_sample_limit=calibration_sample_limit,
    )
    stokes[~valid] = np.nan

    surface_unit = surface / np.linalg.norm(surface, axis=-1, keepdims=True)
    scene_basis = torch.tensor(ray_metadata["scene_basis_rows"], dtype=torch.float64)
    chart = direction_to_chart_mm(
        torch.from_numpy(surface_unit), scene_basis, ray_metadata["solar_radius_m"]
    ).numpy()
    ref_time = s_map.date if reference_time is None else Time(reference_time)
    time_hours = float((s_map.date - ref_time).to_value(u.hour))
    time_grid = np.full((*chart.shape[:-1], 1), time_hours, dtype=np.float64)
    coordinates = np.concatenate((chart, time_grid), axis=-1).astype(np.float32)
    centre = np.mean(chart[valid], axis=0)
    scale = np.max(np.abs(chart[valid] - centre), axis=0)
    scale = np.maximum(scale, 1.0e-6)
    wavelength = (
        HMI_WAVELENGTH_CENTER.to_value(u.AA) + HMI_WAVELENGTH_GRID.to_value(u.AA)
    )[::-1].copy()
    response_spectral, response_continuum = sample_response_grid(
        detector_grid, response_sampler
    )
    metadata = {
        "instrument": "SDO/HMI",
        "files": [str(segments[key]) for key in SEGMENT_KEYS],
        "file_count": len(segments),
        "acquisition_key": acquisition["acquisition_key"],
        "times": [str(s_map.date.isot)],
        "ref_time": str(ref_time.isot),
        "coordinates": {
            "order": [
                "carrington_chart_x_mm",
                "carrington_chart_y_mm",
                "time_hours",
            ],
            "units": ["Mm", "Mm", "hour"],
            "time_origin": str(ref_time.isot),
            "time_scale": str(ref_time.scale).upper(),
            "time_formula": "time_hours=(T_OBS-time_origin)/3600 s",
            "spatial_frame": "observer-independent Carrington gnomonic scene chart",
            "network_affine": {
                "center_mm": centre.tolist(),
                "scale_mm": scale.tolist(),
            },
        },
        "normalization": {
            "operation": "no per-pixel or raster continuum normalization",
            "indices": [0, 5],
            "radiometric_calibration": radiometry,
        },
        "wavelength": {
            "atomic_line_air_angstrom": 6173.3352,
            "instrument_tuning_reference_air_angstrom": float(
                HMI_WAVELENGTH_CENTER.to_value(u.AA)
            ),
            "observed_filter_centres_air_angstrom": wavelength.tolist(),
            "frame": "instrument frame; observer motion is not applied to filter wavelengths",
            "storage_to_training_reversal_applied": True,
        },
        "observer_velocity_correction": {
            "source_keywords": list(HMI_OBSERVER_VELOCITY_KEYS),
            "drms_components_m_per_s": observer_velocity.tolist(),
            "component_order": [
                "radial_away_from_sun",
                "solar_west",
                "solar_north",
            ],
            "projected_quantity": "observer velocity along each toward-observer LOS",
            "projected_los_velocity_range_m_per_s": [
                float(observer_los_velocity[valid].min()),
                float(observer_los_velocity[valid].max()),
            ],
            "sign_convention": (
                "positive is observer motion toward the observer/away from the Sun at "
                "disk center and produces a redshift"
            ),
            "inversion_action": (
                "subtract projected observer LOS velocity from the solar velocity's "
                "toward-observer component before v_los=-v_toward"
            ),
            "calibration_action": (
                "shift only the solar atlas lookup for detector-gain calibration; "
                "keep HMI filter wavelengths in the instrument frame"
            ),
        },
        "stokes_order": ["I", "Q", "U", "V"],
        "ray_geometry": ray_metadata,
        "spectral_response": {
            "file": str(profile_file),
            "metadata": response_sampler.metadata,
            "filter_order_reversed_to_increasing_wavelength": True,
        },
        "quality_mask": {
            "require_quality_zero": bool(require_quality_zero),
            "quality_values": [int(header["QUALITY"]) for header in headers],
            "rejected_pixel_count": int(valid.size - np.count_nonzero(valid)),
        },
        "data_fingerprints": {
            "algorithm": "sha256(dtype, shape, C-order bytes)",
            "selected_stokes": array_sha256(stokes),
            "wavelength_angstrom": array_sha256(wavelength),
            "coordinates": array_sha256(coordinates),
            "ray_direction": array_sha256(rays),
            "surface_position_m": array_sha256(surface),
            "stokes_basis": array_sha256(bases),
            "observer_los_velocity_m_per_s": array_sha256(observer_los_velocity),
            "instrument_response:spectral_weights": array_sha256(
                response_spectral.numpy()
            ),
            "instrument_response:continuum_weights": array_sha256(
                response_continuum.numpy()
            ),
            "valid_mask": array_sha256(valid),
        },
    }
    observer_velocity_tensor = torch.from_numpy(
        observer_los_velocity.astype(np.float32)
    )
    if not torch.isfinite(observer_velocity_tensor).all() or torch.any(
        observer_velocity_tensor.abs() >= SPEED_OF_LIGHT_M_PER_S
    ):
        raise ValueError("HMI observer LOS velocity must be finite and subluminal.")
    raster = ObservationRaster(
        stokes=torch.from_numpy(stokes.astype(np.float32)),
        wavelength_angstrom=torch.from_numpy(wavelength.astype(np.float32)),
        coordinates=torch.from_numpy(coordinates),
        ray_direction=torch.from_numpy(rays.astype(np.float32)),
        surface_position_m=torch.from_numpy(surface.astype(np.float32)),
        stokes_basis=torch.from_numpy(bases.astype(np.float32)),
        valid_mask=torch.from_numpy(valid),
        metadata=metadata,
        auxiliary={
            "observer_los_velocity_m_per_s": observer_velocity_tensor,
            "instrument_response:spectral_weights": response_spectral,
            "instrument_response:continuum_weights": response_continuum,
        },
    )
    return raster


__all__ = ["HMI_SOLAR_REFERENCE_RESOURCE", "load_raster"]
