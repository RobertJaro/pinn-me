"""Radiometric calibration and response materialization for HMI Stokes data."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from astropy import units as u

from prom3theus.rt.radiometry import (
    disk_center_continuum_radiance,
    disk_center_intensity_radiance,
    neckel_continuum_limb_darkening,
    reference_summary,
)

from .constants import (
    HMI_OBSERVER_VELOCITY_KEYS,
    HMI_WAVELENGTH_CENTER,
    HMI_WAVELENGTH_GRID,
    SPEED_OF_LIGHT_M_PER_S,
)
from .geometry import HMIDetectorGrid
from .response import HMIResponseArchive


def array_sha256(value: np.ndarray) -> str:
    """Hash an array together with its dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def sample_response_grid(
    detector_coordinates: HMIDetectorGrid | np.ndarray,
    response_archive: HMIResponseArchive,
    *,
    row_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize exportable per-pixel response tensors in bounded chunks."""

    if type(row_chunk) is not int or row_chunk < 1:
        raise ValueError("row_chunk must be a positive integer.")
    if isinstance(detector_coordinates, HMIDetectorGrid):
        spatial_shape = detector_coordinates.spatial_shape
    else:
        dense_coordinates = np.asarray(detector_coordinates)
        if (
            dense_coordinates.ndim != 3
            or dense_coordinates.shape[-1] != 2
            or not np.isfinite(dense_coordinates).all()
            or np.any((dense_coordinates < 0) | (dense_coordinates > 1))
        ):
            raise ValueError(
                "Dense HMI detector coordinates must be finite [y,x,2] values "
                "inside [0,1]^2."
            )
        spatial_shape = tuple(dense_coordinates.shape[:2])
    filters, quadrature = response_archive.offsets.shape
    spectral = torch.empty((*spatial_shape, filters, quadrature), dtype=torch.float32)
    continuum = torch.empty((*spatial_shape, filters), dtype=torch.float32)
    for start in range(0, spatial_shape[0], row_chunk):
        stop = min(start + row_chunk, spatial_shape[0])
        if isinstance(detector_coordinates, HMIDetectorGrid):
            rows, columns = np.indices((stop - start, spatial_shape[1]), dtype=np.int64)
            rows += start
            coordinates = detector_coordinates.at_indices(
                np.stack((rows, columns), axis=-1)
            )
        else:
            coordinates = np.asarray(detector_coordinates)[start:stop]
        sampled = response_archive.sample(coordinates)
        spectral[start:stop].copy_(sampled["spectral_weights"])
        continuum[start:stop].copy_(sampled["continuum_weights"])
    return spectral, continuum


def calibrate_stokes(
    stokes,
    mu,
    valid,
    detector_coordinates,
    response_archive,
    solar_reference,
    observer_los_velocity_m_per_s,
    *,
    quiet_sun_max_fractional_polarization,
    quiet_sun_trim_quantiles,
    minimum_quiet_sun_pixels,
    calibration_sample_limit,
):
    """Fit one absolute detector gain against the disk-center solar atlas."""

    stokes = np.asarray(stokes)
    mu = np.asarray(mu, dtype=np.float64)
    valid = np.asarray(valid)
    if (
        stokes.shape != (*mu.shape, 4, 6)
        or not np.issubdtype(stokes.dtype, np.floating)
        or valid.shape != mu.shape
        or valid.dtype != np.bool_
        or not np.any(valid)
        or not np.isfinite(stokes[valid]).all()
        or not np.isfinite(mu).all()
        or np.any((mu < 0) | (mu > 1))
        or np.any(mu[valid] <= 0)
    ):
        raise ValueError(
            "HMI calibration requires finite [y,x,4,6] Stokes data at valid "
            "pixels, a boolean mask, and 0 < mu <= 1 on that mask."
        )
    threshold = float(quiet_sun_max_fractional_polarization)
    try:
        lower_quantile, upper_quantile = map(float, quiet_sun_trim_quantiles)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "quiet_sun_trim_quantiles must contain exactly two values."
        ) from error
    if (
        not np.isfinite(threshold)
        or threshold <= 0
        or not np.isfinite((lower_quantile, upper_quantile)).all()
        or not 0 <= lower_quantile < upper_quantile <= 1
    ):
        raise ValueError("HMI quiet-Sun calibration parameters are invalid.")
    if type(minimum_quiet_sun_pixels) is not int or minimum_quiet_sun_pixels < 1:
        raise ValueError("minimum_quiet_sun_pixels must be a positive integer.")
    if type(calibration_sample_limit) is not int or calibration_sample_limit < 1:
        raise ValueError("calibration_sample_limit must be a positive integer.")
    observer_los_velocity = np.asarray(observer_los_velocity_m_per_s, dtype=np.float64)
    if (
        observer_los_velocity.shape != np.shape(mu)
        or not np.isfinite(observer_los_velocity).all()
    ):
        raise ValueError(
            "HMI observer LOS velocity must be finite and match the spatial image shape."
        )
    if np.any(np.abs(observer_los_velocity) >= SPEED_OF_LIGHT_M_PER_S):
        raise ValueError("HMI observer LOS velocity must be subluminal.")
    continuum_scale = float(
        disk_center_continuum_radiance(
            solar_reference, [float(HMI_WAVELENGTH_CENTER.to_value(u.AA))]
        )[0]
    )
    i_mean = np.mean(stokes[..., 0, :], axis=-1)
    polarized = np.sqrt(np.sum(np.square(stokes[..., 1:, :]), axis=-2)).mean(axis=-1)
    fractional = polarized / i_mean
    candidates = (
        valid
        & np.isfinite(fractional)
        & np.all(stokes[..., 0, :] > 0, axis=-1)
        & (fractional <= threshold)
    )
    limb = neckel_continuum_limb_darkening(
        float(HMI_WAVELENGTH_CENTER.to_value(u.AA)), mu
    )
    disk_center_equivalent = i_mean / limb
    if np.count_nonzero(candidates) >= 20:
        low, high = np.quantile(
            disk_center_equivalent[candidates], (lower_quantile, upper_quantile)
        )
        candidates &= (disk_center_equivalent >= low) & (disk_center_equivalent <= high)
    indices = np.argwhere(candidates)
    if len(indices) < int(minimum_quiet_sun_pixels):
        raise ValueError(
            f"HMI atlas calibration found {len(indices)} quiet-Sun pixels; "
            f"{minimum_quiet_sun_pixels} are required."
        )
    if len(indices) > int(calibration_sample_limit):
        selection = np.linspace(
            0, len(indices) - 1, int(calibration_sample_limit)
        ).astype(int)
        indices = indices[selection]
    if isinstance(detector_coordinates, HMIDetectorGrid):
        selected_coordinates = detector_coordinates.at_indices(indices)
    else:
        selected_coordinates = np.asarray(detector_coordinates)[
            indices[:, 0], indices[:, 1]
        ]
    response = response_archive.sample(selected_coordinates)
    observed_wavelength = (
        HMI_WAVELENGTH_CENTER.to_value(u.AA) + HMI_WAVELENGTH_GRID.to_value(u.AA)
    )[::-1]
    quadrature_wavelength = response_archive.quadrature_wavelength(
        observed_wavelength
    ).numpy()
    selected_observer_velocity = observer_los_velocity[indices[:, 0], indices[:, 1]]
    beta = selected_observer_velocity / SPEED_OF_LIGHT_M_PER_S
    frequency_doppler_factor = np.sqrt((1.0 - beta) / (1.0 + beta))
    # The filter grid remains in the instrument frame. Shift only the atlas
    # lookup into the solar frame when estimating the detector gain.
    atlas_lookup_wavelength = (
        quadrature_wavelength[None, :] * frequency_doppler_factor[:, None]
    )
    atlas_quadrature = disk_center_intensity_radiance(
        solar_reference, atlas_lookup_wavelength
    )
    expected = (
        atlas_quadrature[:, None, :] * response["spectral_weights"].numpy()
    ).sum(axis=-1)
    expected += response["continuum_weights"].numpy() * continuum_scale
    expected *= limb[indices[:, 0], indices[:, 1], None]
    observed = stokes[indices[:, 0], indices[:, 1], 0, :]
    gain = float(np.median(expected / observed))
    if not np.isfinite(gain) or gain <= 0:
        raise FloatingPointError("HMI detector-to-radiance calibration is invalid.")
    stokes *= gain / continuum_scale
    return stokes, {
        "type": "absolute quiet-Sun FTS/filter-profile calibration",
        "operation": (
            "one detector-to-radiance scalar is fitted to quiet-Sun Stokes I after "
            "convolving the disk-center FTS atlas with each pixel's phase-map response; "
            "the same scalar is applied to I,Q,U,V"
        ),
        "stored_stokes_unit": "I_c,atlas(mu=1)",
        "physical_radiance_unit": "W m^-3 sr^-1",
        "atlas_disk_center_continuum_radiance_w_m3_sr": continuum_scale,
        "detector_to_radiance_w_m3_sr_per_raw_unit": gain,
        "quiet_sun_pixel_count": int(len(indices)),
        "maximum_mean_fractional_polarization": threshold,
        "continuum_trim_quantiles": [lower_quantile, upper_quantile],
        "line_center_to_limb_approximation": (
            "the Neckel continuum factor at 6173.3433 A multiplies the complete "
            "disk-center atlas profile during detector-gain estimation"
        ),
        "observer_velocity_doppler_correction": {
            "source_keywords": list(HMI_OBSERVER_VELOCITY_KEYS),
            "positive_los_velocity": (
                "observer motion along the toward-observer LOS, producing a redshift"
            ),
            "atlas_lookup_formula": (
                "lambda_atlas_air=lambda_instrument_air*sqrt((1-beta)/(1+beta)), "
                "beta=v_observer_los/c"
            ),
            "filter_wavelength_frame": "instrument frame (unchanged)",
            "selected_los_velocity_range_m_per_s": [
                float(selected_observer_velocity.min()),
                float(selected_observer_velocity.max()),
            ],
        },
        "reference": reference_summary(solar_reference),
    }


__all__ = ["array_sha256", "calibrate_stokes", "sample_response_grid"]
