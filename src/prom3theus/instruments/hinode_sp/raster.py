"""Assembly of one calibrated Hinode/SOT-SP observation raster."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
from astropy import units as u
from astropy.constants import R_sun
from dateutil.parser import parse as parse_datetime

from prom3theus.observations import ObservationRaster
from prom3theus.observations.io import ordered_parallel_fits_map
from prom3theus.rt.geometry import direction_to_chart_mm
from prom3theus.rt.radiometry import load_solar_reference

from .calibration import (
    array_sha256,
    calibrate_stokes,
    continuum_selection,
    observer_los_velocity_metadata,
)
from .constants import SP_PREP_SOURCE_SHA256, SP_PREP_SOURCE_URL
from .fits_io import (
    read_column,
    resolve_files,
    slit_coordinates_arcsec,
    slice_from_config,
)
from .geometry import (
    carrington_rays,
    coordinate_affine_metadata,
    observer_distance_m,
    solar_radius_arcsec,
)


def load_raster(
    files,
    slit_slice=None,
    continuum_edge_samples: int = 8,
    solar_reference=None,
    quiet_sun_max_fractional_polarization: float = 0.01,
    quiet_sun_continuum_trim_quantiles=(0.05, 0.95),
    minimum_quiet_sun_pixels: int = 1,
    stokes_reference_angle_deg: float = 0.0,
    reference_time: datetime | str | None = None,
    scene_basis_rows=None,
    *,
    fits_read_workers: int = 1,
    progress: bool = False,
    defer_radiometric_calibration: bool = False,
) -> ObservationRaster:
    """Load one prepared Level-1 raster into the canonical observation contract."""

    paths = resolve_files(files)
    slit_selection = slice_from_config(slit_slice, name="slit_slice")
    headers = []
    columns = []
    detector_quality = []
    wavelength = None
    wavelength_solution = None
    full_slit_count = None
    selected_slit_indices = None
    reads = ordered_parallel_fits_map(
        lambda path: read_column(path, slit_selection),
        paths,
        fits_read_workers,
        progress=progress,
        description="Hinode FITS columns",
        unit="column",
    )
    for path, read in zip(paths, reads, strict=True):
        header = read["header"]
        file_wavelength = read["wavelength"]
        file_slit_count = read["full_slit_count"]
        if full_slit_count is None:
            full_slit_count = file_slit_count
            selected_slit_indices = np.arange(full_slit_count)[slit_selection]
            if selected_slit_indices.size == 0:
                raise ValueError("slit_slice selected no detector rows.")
        elif file_slit_count != full_slit_count:
            raise ValueError(
                "All selected Hinode files must have the same slit length."
            )
        if wavelength is None:
            wavelength = file_wavelength
            wavelength_solution = read["wavelength_solution"]
        elif not np.allclose(file_wavelength, wavelength, rtol=0.0, atol=1e-9):
            raise ValueError("Selected Hinode files do not share one wavelength grid.")
        columns.append(read["column"])
        detector_quality.append(read["detector_quality"])
        headers.append(header)

    stokes = np.stack(columns, axis=1)
    continuum_mask, normalization = continuum_selection(
        wavelength, continuum_edge_samples
    )
    continuum = np.mean(
        stokes[..., 0, continuum_mask], axis=-1, dtype=np.float64
    ).astype(np.float32)
    valid = (
        np.isfinite(stokes).all(axis=(-2, -1))
        & np.isfinite(continuum)
        & (continuum > 0)
    )
    valid &= np.stack(detector_quality, axis=1)

    times = [parse_datetime(header["DATE_OBS"]) for header in headers]
    ref_time = (
        min(times) if reference_time is None else parse_datetime(str(reference_time))
    )
    if ref_time > min(times):
        raise ValueError("reference_time cannot be later than the first exposure.")
    time_hours = np.asarray(
        [(time - ref_time).total_seconds() / 3_600.0 for time in times],
        dtype=np.float64,
    )
    scan_indices = np.asarray(
        [header["SLITINDX"] for header in headers],
        dtype=np.float32,
    )
    if scan_indices.size > 1 and np.any(np.diff(scan_indices) <= 0):
        raise ValueError(
            "Selected Hinode files must have strictly increasing, unique SLITINDX "
            "values; lexical file order is not a valid raster order."
        )
    acquisition_seconds = np.asarray(
        [(time - times[0]).total_seconds() for time in times], dtype=np.float64
    )
    if acquisition_seconds.size > 1 and np.any(np.diff(acquisition_seconds) <= 0):
        raise ValueError(
            "Selected Hinode files must have strictly increasing DATE_OBS values."
        )

    x_arcsec, y_arcsec, pointing_metadata = slit_coordinates_arcsec(
        headers, selected_slit_indices
    )
    distances_m = observer_distance_m(times)
    angular_radius = solar_radius_arcsec(times, distances_m)
    tx_rad = x_arcsec * u.arcsec.to(u.rad)
    ty_rad = y_arcsec * u.arcsec.to(u.rad)
    cos_rho = np.cos(tx_rad) * np.cos(ty_rad)
    sin_rho = np.sqrt(np.clip(1.0 - np.square(cos_rho), 0.0, 1.0))
    impact_m = distances_m[None, :] * sin_rho
    mu = np.sqrt(
        np.clip(1.0 - np.square(impact_m / R_sun.to_value(u.m)), 0.0, 1.0)
    ).astype(np.float32)
    valid &= mu > 0
    if not np.any(valid):
        raise ValueError("The selected Hinode raster contains no valid on-disk pixels.")

    if solar_reference is None:
        solar_reference = load_solar_reference()
    if defer_radiometric_calibration:
        radiometric_calibration = {
            "type": "pending shared Hinode/SOT-SP instrument dataset calibration"
        }
    else:
        stokes, radiometric_calibration = calibrate_stokes(
            stokes,
            wavelength,
            continuum_mask,
            continuum,
            mu,
            valid,
            solar_reference=solar_reference,
            quiet_sun_max_fractional_polarization=(
                quiet_sun_max_fractional_polarization
            ),
            quiet_sun_continuum_trim_quantiles=(quiet_sun_continuum_trim_quantiles),
            minimum_quiet_sun_pixels=minimum_quiet_sun_pixels,
        )
    stokes[~valid] = np.nan

    ray_direction, surface_position_m, stokes_basis, spherical_geometry = (
        carrington_rays(
            x_arcsec,
            y_arcsec,
            times,
            stokes_reference_angle_deg=stokes_reference_angle_deg,
            valid_mask=valid,
            scene_basis_rows=scene_basis_rows,
        )
    )
    surface_direction = surface_position_m / np.linalg.norm(
        surface_position_m, axis=-1, keepdims=True
    )
    chart_xy_mm = direction_to_chart_mm(
        torch.from_numpy(surface_direction),
        torch.tensor(spherical_geometry["scene_basis_rows"]),
        spherical_geometry["solar_radius_m"],
    ).numpy()
    x_mm, y_mm = chart_xy_mm[..., 0], chart_xy_mm[..., 1]
    coords = np.stack(
        (x_mm, y_mm, np.broadcast_to(time_hours[None, :], x_mm.shape)), axis=-1
    ).astype(np.float32)
    coordinate_affine = coordinate_affine_metadata(
        x_mm,
        y_mm,
        valid,
        headers=headers,
        observer_distances_m=distances_m,
        slit_indices=selected_slit_indices,
        scan_indices=scan_indices,
    )

    normalization.update(
        {
            "operation": "no raster or per-pixel continuum normalization",
            "scope": "none",
            "estimator": None,
            "radiometric_calibration": radiometric_calibration,
            "continuum_intensity_quantiles": {
                str(percentile): float(np.percentile(continuum[valid], percentile))
                for percentile in (1, 5, 50, 95, 99)
            },
        }
    )
    wavelength_metadata = {
        **wavelength_solution,
        "runtime_positive_wcs_required": True,
        "detector_order_preserved": True,
        "jointly_reordered_with_stokes": False,
        "sign_convention_provenance": {
            "method": "SolarSoft sp_prep calibrated-axis convention",
            "evidence": (
                "thermd_sbsp reverses the output spectral direction so increasing "
                "spectral pixel means increasing wavelength; calibrated sp_prep "
                "products record positive CDELT1 to match"
            ),
            "source_url": SP_PREP_SOURCE_URL,
            "source_sha256": SP_PREP_SOURCE_SHA256,
        },
        "spwlshft": [header["SPWLSHFT"] for header in headers],
        "spwlsft0": [header["SPWLSFT0"] for header in headers],
        "shift_application": (
            "not reapplied to spectra: SPWLSHFT and SPWLSFT0 are used only to "
            "reconstruct the removed solar LOS velocity gauge"
        ),
    }
    velocity_metadata = observer_los_velocity_metadata(
        headers, wavelength_solution["effective_cdelt1_angstrom"]
    )
    metadata = {
        "instrument": "Hinode/SOT-SP",
        "files": [str(path) for path in paths],
        "file_count": len(paths),
        "slit_indices": selected_slit_indices.tolist(),
        "scan_indices": scan_indices.astype(int).tolist(),
        "times": [time.isoformat() for time in times],
        "ref_time": ref_time.isoformat(),
        "coordinates": {
            "order": [
                "carrington_chart_x_mm",
                "carrington_chart_y_mm",
                "time_hours",
            ],
            "units": ["Mm", "Mm", "hour"],
            "time_origin": ref_time.isoformat(),
            "time_formula": "time_hours=(DATE_OBS-time_origin)/3600 s",
            "spatial_frame": "observer-independent Carrington gnomonic scene chart",
            "spatial_projection": (
                "gnomonic chart tangent to the raster-centre solar direction"
            ),
            "spatial_formula": (
                "xy_mm=R_sun*(u dot chart_xy)/(u dot chart_normal)/1e6"
            ),
            "observer": (
                "Astropy geocentric solar ephemeris proxy; Level-1 headers do not "
                "provide the Hinode spacecraft distance"
            ),
            "observer_distance_m": distances_m.tolist(),
            "chart_x_range_mm": [float(x_mm[valid].min()), float(x_mm[valid].max())],
            "chart_y_range_mm": [float(y_mm[valid].min()), float(y_mm[valid].max())],
            "surface_deprojection_applied": True,
            "network_affine": coordinate_affine,
            "source_keywords": [
                "DATE_OBS",
                "XCEN",
                "YCEN",
                "CRPIX2",
                "CDELT2",
                "CROTA2",
            ],
        },
        "normalization": normalization,
        "wavelength": wavelength_metadata,
        "observer_velocity_correction": velocity_metadata,
        "stokes_order": ["I", "Q", "U", "V"],
        "quality_mask": {
            "reject_nonfinite": True,
            "reject_nonpositive_continuum": True,
            "reject_integer_detector_limits": True,
            "rejected_pixel_count": int(valid.size - np.count_nonzero(valid)),
        },
        "ray_geometry": {
            **pointing_metadata,
            **spherical_geometry,
            "mu_definition": (
                "sqrt(max(0, 1 - (D/R_sun)^2*(1 - (cos(Tx)*cos(Ty))^2)))"
            ),
            "rho_definition": ("cos(rho)=cos(Tx)*cos(Ty) for HPC longitude/latitude"),
            "impact_parameter_formula": ("b=D*sqrt(1-(cos(Tx)*cos(Ty))^2)"),
            "apparent_solar_radius_formula": "asin(R_sun / geocentric_sun_distance)",
            "apparent_solar_radius_arcsec": angular_radius.tolist(),
            "observer_distance": (
                "Astropy geocentric solar ephemeris; Level-1 headers do not provide "
                "a spacecraft observer distance"
            ),
            "transfer_model": (
                "differentiable 3-D ray intersections with learned corrugated "
                "tau500 surfaces; formal transfer uses exact delta-s"
            ),
        },
        "data_fingerprints": {
            "algorithm": "sha256(dtype, shape, C-order bytes)",
            "selected_stokes": array_sha256(stokes),
            "wavelength_angstrom": array_sha256(wavelength),
            "coordinates": array_sha256(coords),
            "ray_direction": array_sha256(ray_direction),
            "surface_position_m": array_sha256(surface_position_m),
            "stokes_basis": array_sha256(stokes_basis),
            "removed_solar_los_velocity_m_per_s": array_sha256(
                np.asarray(
                    velocity_metadata["removed_solar_los_velocity_m_per_s"],
                    dtype=np.float32,
                )
            ),
            "valid_mask": array_sha256(valid),
        },
    }
    removed_velocity = (
        torch.as_tensor(
            velocity_metadata["removed_solar_los_velocity_m_per_s"], dtype=torch.float32
        )
        .unsqueeze(0)
        .expand(stokes.shape[0], -1)
    )
    return ObservationRaster(
        stokes=torch.from_numpy(np.asarray(stokes, dtype=np.float32)),
        wavelength_angstrom=torch.from_numpy(np.asarray(wavelength, dtype=np.float32)),
        coordinates=torch.from_numpy(coords),
        ray_direction=torch.from_numpy(ray_direction.astype(np.float32)),
        surface_position_m=torch.from_numpy(surface_position_m.astype(np.float32)),
        stokes_basis=torch.from_numpy(stokes_basis.astype(np.float32)),
        valid_mask=torch.from_numpy(valid),
        metadata=metadata,
        auxiliary={"removed_solar_los_velocity_m_per_s": removed_velocity},
    )


__all__ = ["load_raster"]
