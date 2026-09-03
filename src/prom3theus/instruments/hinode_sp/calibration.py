"""Radiometric and velocity-gauge calibration for Hinode/SOT-SP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

import numpy as np
from astropy.io import fits

from prom3theus.rt.radiometry import (
    disk_center_continuum_radiance,
    neckel_continuum_limb_darkening,
    reference_summary,
)

from .constants import (
    HINODE_6301_LAB_AIR_WAVELENGTH_ANGSTROM,
    SP_PREP_REFERENCE_LINE_ANGSTROM,
    SP_PREP_SOURCE_SHA256,
    SP_PREP_SOURCE_URL,
    THERMD_SBSP_SOURCE_SHA256,
    THERMD_SBSP_SOURCE_URL,
)


_C_LIGHT_M_PER_S = 299_792_458.0


def array_sha256(value: np.ndarray) -> str:
    """Hash a numeric array including dtype, shape, and C-order bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def observer_los_velocity_metadata(
    headers: Sequence[fits.Header], dispersion_angstrom_per_pixel: float
) -> dict:
    """Recover the solar LOS shift removed by Level-1 line registration."""

    required = ("DOP_RCV", "SPWLSHFT", "SPWLSFT0")
    missing = [
        (index, key)
        for index, header in enumerate(headers)
        for key in required
        if key not in header
    ]
    if missing:
        raise KeyError(
            "Hinode Level-1 velocity reconstruction requires DOP_RCV, "
            "SPWLSHFT, and SPWLSFT0 in every selected FITS header; missing "
            f"(scan, keyword) entries: {missing[:8]}."
        )
    velocity = np.asarray([header["DOP_RCV"] for header in headers], dtype=np.float64)
    registration_pixels = np.asarray(
        [header["SPWLSHFT"] for header in headers], dtype=np.float64
    )
    orbital_plus_thermal_pixels = np.asarray(
        [header["SPWLSFT0"] for header in headers], dtype=np.float64
    )
    dispersion = float(dispersion_angstrom_per_pixel)
    if (
        not np.isfinite(velocity).all()
        or not np.isfinite(registration_pixels).all()
        or not np.isfinite(orbital_plus_thermal_pixels).all()
        or not np.isfinite(dispersion)
        or dispersion <= 0
    ):
        raise ValueError("Hinode velocity-registration inputs must be finite.")
    if np.any(np.abs(velocity) >= _C_LIGHT_M_PER_S):
        raise ValueError("Hinode observer velocities must be subluminal.")

    orbital_pixels = (
        velocity / _C_LIGHT_M_PER_S * SP_PREP_REFERENCE_LINE_ANGSTROM / dispersion
    )
    thermal_pixels = orbital_plus_thermal_pixels - orbital_pixels
    removed_solar_shift_pixels = -registration_pixels - thermal_pixels
    removed_solar_wavelength = (
        SP_PREP_REFERENCE_LINE_ANGSTROM + removed_solar_shift_pixels * dispersion
    )
    if not np.isfinite(removed_solar_wavelength).all() or np.any(
        removed_solar_wavelength <= 0
    ):
        raise ValueError(
            "Recovered Hinode solar wavelengths must be finite and positive."
        )
    ratio_squared = np.square(
        removed_solar_wavelength / HINODE_6301_LAB_AIR_WAVELENGTH_ANGSTROM
    )
    removed_solar_velocity = (
        _C_LIGHT_M_PER_S * (ratio_squared - 1.0) / (ratio_squared + 1.0)
    )
    if not np.isfinite(removed_solar_velocity).all() or np.any(
        np.abs(removed_solar_velocity) >= _C_LIGHT_M_PER_S
    ):
        raise ValueError("Recovered Hinode solar LOS velocities must be subluminal.")
    return {
        "source_keyword": "DOP_RCV",
        "observer_los_velocity_m_per_s": velocity.tolist(),
        "sign_convention": (
            "positive means spacecraft-Sun relative motion producing a redshift"
        ),
        "calibration_stage": "already removed from the Level-1 spectra by sp_prep",
        "inversion_action": (
            "do not apply DOP_RCV again; add rigid Carrington rotation to the "
            "learned co-rotating velocity, then subtract the recovered solar LOS "
            "shift removed by spectral registration"
        ),
        "removed_solar_los_velocity_m_per_s": removed_solar_velocity.tolist(),
        "removed_solar_shift_pixels": removed_solar_shift_pixels.tolist(),
        "applied_registration_shift_pixels": registration_pixels.tolist(),
        "predicted_thermal_shift_pixels": thermal_pixels.tolist(),
        "orbital_shift_pixels": orbital_pixels.tolist(),
        "dispersion_angstrom_per_pixel": dispersion,
        "sp_prep_registered_air_wavelength_angstrom": SP_PREP_REFERENCE_LINE_ANGSTROM,
        "synthesis_lab_air_wavelength_angstrom": (
            HINODE_6301_LAB_AIR_WAVELENGTH_ANGSTROM
        ),
        "reconstruction_formula": (
            "thermal_pix=SPWLSFT0-DOP_RCV/c*lambda_ref/CDELT1; "
            "solar_pix=-SPWLSHFT-thermal_pix; "
            "lambda_solar=lambda_ref+solar_pix*CDELT1; velocity uses the exact "
            "relativistic wavelength ratio relative to the synthesis laboratory line"
        ),
        "inferred_velocity_frame": (
            "per-scan registered spectra with a recovered solar LOS zero point; "
            "spacecraft motion remains removed"
        ),
        "source": {
            "routine": "SolarSoft Hinode/SOT thermd_sbsp.pro called by sp_prep.pro",
            "url": THERMD_SBSP_SOURCE_URL,
            "sha256_retrieved_2026_08_20": THERMD_SBSP_SOURCE_SHA256,
            "caller_url": SP_PREP_SOURCE_URL,
            "caller_sha256_retrieved_2026_08_19": SP_PREP_SOURCE_SHA256,
        },
    }


def continuum_selection(
    wavelength: np.ndarray,
    continuum_edge_samples: int,
) -> tuple[np.ndarray, dict]:
    """Select the symmetric detector-edge continuum samples in schema v1."""

    wavelength = np.asarray(wavelength)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or not np.issubdtype(wavelength.dtype, np.floating)
        or not np.isfinite(wavelength).all()
        or not np.all(np.diff(wavelength) > 0)
    ):
        raise ValueError("wavelength must be a finite, increasing floating-point grid.")
    if type(continuum_edge_samples) is not int:
        raise TypeError("continuum_edge_samples must be an integer.")
    count = continuum_edge_samples
    if count < 1 or 2 * count >= wavelength.size:
        raise ValueError(
            "continuum_edge_samples must select at least one and fewer than "
            "half the samples."
        )
    mask = np.zeros(wavelength.size, dtype=bool)
    mask[:count] = True
    mask[-count:] = True
    method = {"type": "edge_mean", "edge_samples": count}
    method.update(
        {
            "indices": np.flatnonzero(mask).tolist(),
            "wavelength_angstrom": wavelength[mask].tolist(),
        }
    )
    return mask, method


def _calibration_parameters(
    maximum_fractional_polarization: float,
    trim_quantiles,
    minimum_pixels: int,
) -> tuple[float, float]:
    if (
        not np.isfinite(float(maximum_fractional_polarization))
        or maximum_fractional_polarization <= 0
    ):
        raise ValueError("quiet_sun_max_fractional_polarization must be positive.")
    if type(minimum_pixels) is not int or minimum_pixels < 1:
        raise ValueError("minimum_quiet_sun_pixels must be positive.")
    try:
        lower, upper = map(float, trim_quantiles)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "quiet_sun_continuum_trim_quantiles must contain two quantiles."
        ) from error
    if not np.isfinite((lower, upper)).all() or not 0 <= lower < upper <= 1:
        raise ValueError(
            "quiet_sun_continuum_trim_quantiles must satisfy 0 <= low < high <= 1."
        )
    return lower, upper


def _raster_calibration_payload(
    stokes: np.ndarray,
    wavelength: np.ndarray,
    continuum_indices: np.ndarray,
    mu: np.ndarray,
    valid: np.ndarray,
    solar_reference: Mapping,
    maximum_fractional_polarization: float,
) -> dict:
    continuum_wavelength = wavelength[continuum_indices]
    continuum = np.mean(stokes[..., 0, continuum_indices], axis=-1, dtype=np.float64)
    disk_center_samples = disk_center_continuum_radiance(
        solar_reference, continuum_wavelength
    )
    atlas_continuum = float(np.mean(disk_center_samples))
    limb_samples = neckel_continuum_limb_darkening(
        continuum_wavelength[None, None, :], mu[..., None]
    )
    expected = np.mean(disk_center_samples[None, None, :] * limb_samples, axis=-1)
    limb_darkening = expected / atlas_continuum
    disk_center_equivalent = continuum / limb_darkening
    polarization = np.sqrt(np.sum(np.square(stokes[..., 1:, :]), axis=-2))
    fractional_polarization = np.mean(polarization, axis=-1) / continuum
    candidates = (
        valid
        & np.isfinite(fractional_polarization)
        & (fractional_polarization <= maximum_fractional_polarization)
    )
    return {
        "stokes": stokes,
        "continuum": continuum,
        "expected": expected,
        "limb_darkening": limb_darkening,
        "disk_center_equivalent": disk_center_equivalent,
        "candidates": candidates,
        "atlas_continuum": atlas_continuum,
        "mu": mu,
    }


def _calibration_metadata(
    payloads: Sequence[dict],
    solar_reference: Mapping,
    *,
    maximum_fractional_polarization: float,
    trim_quantiles: tuple[float, float],
    minimum_pixels: int,
    detector_to_radiance: float,
) -> dict:
    selected = lambda name: np.concatenate(  # noqa: E731
        [payload[name][payload["candidates"]] for payload in payloads]
    )
    lower, upper = trim_quantiles
    return {
        "type": "absolute quiet-Sun atlas calibration",
        "scope": "one shared Hinode/SOT-SP instrument dataset calibration",
        "raster_count": len(payloads),
        "operation": (
            "one detector-to-radiance scalar derived from all selected rasters is "
            "applied uniformly to every raster and to I,Q,U,V, then expressed in "
            "fixed disk-center atlas-continuum radiance units"
        ),
        "stored_stokes_unit": "I_c,atlas(mu=1)",
        "physical_radiance_unit": "W m^-3 sr^-1",
        "atlas_disk_center_continuum_radiance_w_m3_sr": payloads[0]["atlas_continuum"],
        "detector_to_radiance_w_m3_sr_per_raw_unit": detector_to_radiance,
        "quiet_sun": {
            "selection": (
                "valid pixels across all selected rasters below the configured "
                "mean fractional-polarization threshold, trimmed once by global "
                "disk-center-equivalent continuum quantiles"
            ),
            "maximum_mean_fractional_polarization": float(
                maximum_fractional_polarization
            ),
            "continuum_trim_quantiles": [lower, upper],
            "pixel_count": sum(
                int(np.count_nonzero(payload["candidates"])) for payload in payloads
            ),
            "minimum_pixel_count": int(minimum_pixels),
            "median_mu": float(np.median(selected("mu"))),
            "median_raw_continuum": float(np.median(selected("continuum"))),
            "median_disk_center_equivalent_raw_continuum": float(
                np.median(selected("disk_center_equivalent"))
            ),
            "mean_limb_darkening_range": [
                float(np.min(selected("limb_darkening"))),
                float(np.max(selected("limb_darkening"))),
            ],
            "median_expected_continuum_radiance_w_m3_sr": float(
                np.median(selected("expected"))
            ),
        },
        "reference": reference_summary(solar_reference),
    }


def calibrate_stokes(
    stokes: np.ndarray,
    wavelength: np.ndarray,
    continuum_mask: np.ndarray,
    continuum: np.ndarray,
    mu: np.ndarray,
    valid: np.ndarray,
    *,
    solar_reference: Mapping,
    quiet_sun_max_fractional_polarization: float,
    quiet_sun_continuum_trim_quantiles,
    minimum_quiet_sun_pixels: int,
) -> tuple[np.ndarray, dict]:
    """Apply one atlas-derived detector scale to a single raster."""

    stokes = np.asarray(stokes)
    wavelength = np.asarray(wavelength)
    continuum_mask = np.asarray(continuum_mask)
    continuum = np.asarray(continuum)
    mu = np.asarray(mu)
    valid = np.asarray(valid)
    spatial_shape = tuple(stokes.shape[:2])
    if (
        stokes.ndim != 4
        or stokes.shape[-2] != 4
        or not np.issubdtype(stokes.dtype, np.floating)
        or wavelength.shape != (stokes.shape[-1],)
        or not np.issubdtype(wavelength.dtype, np.floating)
        or not np.isfinite(wavelength).all()
        or not np.all(np.diff(wavelength) > 0)
        or continuum_mask.shape != wavelength.shape
        or continuum_mask.dtype != np.bool_
        or not np.any(continuum_mask)
        or continuum.shape != spatial_shape
        or mu.shape != spatial_shape
        or valid.shape != spatial_shape
        or valid.dtype != np.bool_
        or not np.any(valid)
        or not np.isfinite(stokes[valid]).all()
        or not np.isfinite(continuum[valid]).all()
        or np.any(continuum[valid] <= 0)
        or not np.isfinite(mu).all()
        or np.any((mu < 0) | (mu > 1))
        or np.any(mu[valid] <= 0)
    ):
        raise ValueError(
            "Hinode calibration requires finite floating-point Stokes spectra, "
            "an increasing wavelength grid, a boolean continuum selection and "
            "valid mask, positive continuum, and 0 < mu <= 1 at valid pixels."
        )

    lower, upper = _calibration_parameters(
        quiet_sun_max_fractional_polarization,
        quiet_sun_continuum_trim_quantiles,
        minimum_quiet_sun_pixels,
    )
    payload = _raster_calibration_payload(
        stokes,
        wavelength,
        np.flatnonzero(continuum_mask),
        mu,
        valid,
        solar_reference,
        quiet_sun_max_fractional_polarization,
    )
    # Reuse the caller's precomputed continuum to make disagreement impossible
    # to hide in calibration metadata.
    if not np.allclose(payload["continuum"], continuum, equal_nan=True):
        raise ValueError("The supplied continuum does not match the selected samples.")
    candidates = payload["candidates"]
    if np.count_nonzero(candidates) >= 20:
        low, high = np.quantile(
            payload["disk_center_equivalent"][candidates], (lower, upper)
        )
        candidates &= (payload["disk_center_equivalent"] >= low) & (
            payload["disk_center_equivalent"] <= high
        )
    candidate_count = int(np.count_nonzero(candidates))
    if candidate_count < minimum_quiet_sun_pixels:
        raise ValueError(
            f"Atlas calibration found only {candidate_count} quiet-Sun pixels; at "
            f"least {minimum_quiet_sun_pixels} are required."
        )
    detector_to_radiance = float(
        np.median(payload["expected"][candidates] / payload["continuum"][candidates])
    )
    if not np.isfinite(detector_to_radiance) or detector_to_radiance <= 0:
        raise FloatingPointError(
            "The Hinode detector-to-radiance calibration must be finite and positive."
        )
    metadata = _calibration_metadata(
        [payload],
        solar_reference,
        maximum_fractional_polarization=quiet_sun_max_fractional_polarization,
        trim_quantiles=(lower, upper),
        minimum_pixels=minimum_quiet_sun_pixels,
        detector_to_radiance=detector_to_radiance,
    )
    metadata["scope"] = "one Hinode/SOT-SP raster calibration"
    scale = detector_to_radiance / payload["atlas_continuum"]
    stokes *= scale
    return stokes, metadata


__all__ = [
    "array_sha256",
    "calibrate_stokes",
    "continuum_selection",
    "observer_los_velocity_metadata",
]
