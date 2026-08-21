"""Side-effect-free Hinode/SOT-SP raster ingestion for LTE inversions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from datetime import datetime
import glob
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from astropy import units as u
from astropy.constants import R_sun
from astropy.coordinates import get_sun
from astropy.io import fits
from astropy.time import Time
from dateutil.parser import parse as parse_datetime
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from pme.lte.radiometry import (
    disk_center_continuum_radiance,
    load_solar_reference,
    neckel_continuum_limb_darkening,
    reference_summary,
)


SP_PREP_SOURCE_URL = (
    "https://sohoftp.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/sp_prep.pro"
)
SP_PREP_SOURCE_SHA256 = "63bd10f742fae21cb32f62fb11abb29f00c82c2445eb8fabd65976f7d22f60a9"
THERMD_SBSP_SOURCE_URL = (
    "https://sohoftp.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/thermd_sbsp.pro"
)
THERMD_SBSP_SOURCE_SHA256 = (
    "e539150351e99a7ed99e5d97c8a9435022b2c3215fb75816630f3ba6a79e69fe"
)
SP_PREP_REFERENCE_LINE_ANGSTROM = 6301.5091
SP_PREP_REFERENCE_RAW_PIXEL = 138.0
HINODE_COORDINATE_NORMALIZATION_PIXELS = 512.0
HINODE_KEYWORD_REFERENCE_URL = (
    "https://hinode.nao.ac.jp/uploads/2016/04/22/SB_MW_Key13.pdf"
)
SOLAR_WCS_REFERENCE_URL = "https://fits.gsfc.nasa.gov/wcs/coordinates.pdf"
def _observer_los_velocity_metadata(headers: Sequence[fits.Header]) -> dict:
    """Describe the spacecraft-Sun Doppler correction already in Level 1.

    ``thermd_sbsp.pro`` defines ``DOP_RCV`` as the spacecraft-Sun relative
    velocity in m/s, positive for a redshift. It converts that velocity to a
    detector-pixel displacement and applies the opposite spectral shift before
    the final Level-1 wavelength-axis reversal. Applying ``DOP_RCV`` again in
    the LTE forward model would therefore be a double correction.
    """

    missing = [index for index, header in enumerate(headers) if "DOP_RCV" not in header]
    if missing:
        raise KeyError(
            "Hinode Level-1 observer-motion provenance requires DOP_RCV in every "
            f"selected FITS header; missing scan indices: {missing[:8]}."
        )
    velocity = np.asarray([header["DOP_RCV"] for header in headers], dtype=np.float64)
    if not np.isfinite(velocity).all():
        raise ValueError("Hinode DOP_RCV values must be finite m/s velocities.")
    return {
        "source_keyword": "DOP_RCV",
        "observer_los_velocity_m_per_s": velocity.tolist(),
        "sign_convention": (
            "positive means spacecraft-Sun relative motion producing a redshift"
        ),
        "calibration_stage": "already removed from the Level-1 spectra by sp_prep",
        "inversion_action": (
            "no additional wavelength or atmosphere-velocity correction; applying "
            "DOP_RCV again would double-correct the observations"
        ),
        "inferred_velocity_frame": (
            "solar-relative after the per-exposure spacecraft Doppler removal, but "
            "with no absolute solar velocity zero because sp_prep also registers the "
            "slit-averaged Fe I 6301.5 line centre"
        ),
        "dopvused_raw": [header.get("DOPVUSED") for header in headers],
        "dopvused_interpretation": (
            "retained as an uninterpreted instrument-control header; it is not used "
            "as a physical velocity"
        ),
        "source": {
            "routine": "SolarSoft Hinode/SOT thermd_sbsp.pro called by sp_prep.pro",
            "url": THERMD_SBSP_SOURCE_URL,
            "sha256_retrieved_2026_08_20": THERMD_SBSP_SOURCE_SHA256,
            "caller_url": SP_PREP_SOURCE_URL,
            "caller_sha256_retrieved_2026_08_19": SP_PREP_SOURCE_SHA256,
        },
    }


def _array_sha256(value: np.ndarray) -> str:
    """Hash a numeric array including dtype, shape, and C-order bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _slice_from_config(value, *, name: str) -> slice:
    if value is None:
        return slice(None)
    if isinstance(value, slice):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a slice or [start, stop, optional step].")
    if len(value) not in (2, 3):
        raise ValueError(f"{name} must contain two or three entries.")
    return slice(*[None if item is None else int(item) for item in value])


def _resolve_files(files, scan_slice=None) -> list[Path]:
    patterns = [files] if isinstance(files, (str, Path)) else list(files)
    resolved = []
    for pattern in patterns:
        pattern = str(pattern)
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.extend(Path(match) for match in matches)
    # Preserve a deterministic scan order while rejecting accidental duplicates.
    resolved = sorted(dict.fromkeys(resolved))
    if not resolved:
        raise FileNotFoundError(f"No Hinode FITS files match {patterns}.")
    resolved = resolved[_slice_from_config(scan_slice, name="scan_slice")]
    if not resolved:
        raise ValueError("scan_slice selected no Hinode files.")
    return resolved


def _fits_wavelength_angstrom(header, count: int) -> tuple[np.ndarray, dict]:
    """Evaluate the calibrated Level-1 grid and repair pre-1.05 headers.

    SolarSoft ``sp_prep`` reverses the calibrated spectral axis so increasing
    array index means increasing wavelength.  Version 1.05 added the matching
    positive ``CDELT1`` and an ROI-dependent ``CRVAL1`` correction.  Earlier
    Level-1 products, including the supplied version-1.04 raster, contain the
    already-reversed data but retain the stale Level-0 WCS keywords.  The
    repair below is the formula in the authoritative ``sp_prep.pro`` source.
    """

    required = ("CRVAL1", "CRPIX1", "CDELT1", "CUNIT1")
    missing = [key for key in required if key not in header]
    if missing:
        raise KeyError(f"Hinode FITS header is missing wavelength keys: {missing}.")
    source_unit_label = str(header["CUNIT1"]).strip()
    try:
        source_unit = u.Unit(source_unit_label)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Hinode CUNIT1={source_unit_label!r} is not a recognized FITS unit."
        ) from error
    if not source_unit.is_equivalent(u.AA):
        raise ValueError(
            f"Hinode CUNIT1={source_unit_label!r} is not wavelength-equivalent "
            "to Angstrom."
        )
    unit_to_angstrom = float(source_unit.to(u.AA))
    raw_increment = float(header["CDELT1"]) * unit_to_angstrom
    increment = abs(raw_increment)
    if not np.isfinite(increment) or increment == 0:
        raise ValueError("Hinode CDELT1 must be finite and non-zero.")
    raw_reference = float(header["CRVAL1"]) * unit_to_angstrom
    repair_applied = raw_increment < 0
    if repair_applied:
        history = header.get("HISTORY", [])
        history = [history] if isinstance(history, str) else list(history)
        if not any("sp_prep" in str(card).lower() for card in history):
            raise ValueError(
                "Negative CDELT1 can only be repaired for a Level-1 product "
                "whose HISTORY records sp_prep calibration."
            )
        required_roi = ("SPCCDIY0", "SPCCDIY1")
        missing_roi = [key for key in required_roi if key not in header]
        if missing_roi:
            raise KeyError(
                "Pre-1.05 sp_prep wavelength repair requires spectral ROI keys: "
                f"{missing_roi}."
            )
        roi_center = 0.5 * (float(header["SPCCDIY0"]) + float(header["SPCCDIY1"]))
        reference = SP_PREP_REFERENCE_LINE_ANGSTROM + increment * (
            SP_PREP_REFERENCE_RAW_PIXEL - roi_center
        )
        method = "SolarSoft sp_prep version-1.05 legacy-header repair"
    else:
        reference = raw_reference
        method = "positive calibrated FITS WCS"
    pixel = np.arange(count, dtype=np.float64) + 1.0
    wavelength = reference + (pixel - float(header["CRPIX1"])) * increment
    return wavelength, {
        "unit": "angstrom",
        "source_cunit1": source_unit_label,
        "cunit1_conversion_factor_to_angstrom": unit_to_angstrom,
        "fits_formula": "effective_CRVAL1 + ((zero_based_pixel + 1) - CRPIX1) * effective_CDELT1",
        "raw_cdelt1_angstrom": raw_increment,
        "effective_cdelt1_angstrom": increment,
        "raw_crval1_angstrom": raw_reference,
        "effective_crval1_angstrom": reference,
        "legacy_header_repair_applied": repair_applied,
        "method": method,
        "sp_prep_source": {
            "url": SP_PREP_SOURCE_URL,
            "sha256_retrieved_2026_08_19": SP_PREP_SOURCE_SHA256,
            "repair_introduced": "version 1.05, 2008-04-18",
        },
    }


def _observer_distance_m(times: Sequence[datetime]) -> np.ndarray:
    """Return the geocentric Sun distance used by the Level-1 geometry proxy."""

    distances = np.asarray(
        [get_sun(Time(time)).distance.to_value(u.m) for time in times],
        dtype=np.float64,
    )
    if not np.isfinite(distances).all() or np.any(distances <= R_sun.to_value(u.m)):
        raise ValueError("Solar ephemeris returned an invalid Sun-observer distance.")
    return distances


def _solar_radius_arcsec(
    times: Sequence[datetime], observer_distance_m: np.ndarray | None = None
) -> np.ndarray:
    distances = (
        _observer_distance_m(times)
        if observer_distance_m is None
        else np.asarray(observer_distance_m, dtype=np.float64)
    )
    if distances.shape != (len(times),):
        raise ValueError("observer_distance_m must contain one value per scan time.")
    radii = []
    for distance in distances:
        ratio = R_sun.to_value(u.m) / distance
        if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError("Solar ephemeris returned an invalid Sun-observer distance.")
        # The angular semidiameter of a sphere whose centre is at distance D is
        # asin(R/D).  atan(R/D) instead describes a plane at the centre of the
        # sphere and is not the exact limb geometry.
        radius = np.arcsin(ratio) * u.rad
        radii.append(radius.to_value(u.arcsec))
    return np.asarray(radii, dtype=np.float64)


def _tangent_plane_mm(
    solar_x_arcsec: np.ndarray,
    solar_y_arcsec: np.ndarray,
    observer_distance_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert HPC longitude/latitude to one Sun-centred tangent plane in Mm."""

    tx_rad = np.asarray(solar_x_arcsec, dtype=np.float64) * u.arcsec.to(u.rad)
    ty_rad = np.asarray(solar_y_arcsec, dtype=np.float64) * u.arcsec.to(u.rad)
    distance = np.asarray(observer_distance_m, dtype=np.float64)[None, :]
    if (
        tx_rad.ndim != 2
        or tx_rad.shape != ty_rad.shape
        or tx_rad.shape[1] != distance.shape[1]
    ):
        raise ValueError("Helioprojective coordinates must contain one column per distance.")
    x_mm = distance * np.tan(tx_rad) / 1.0e6
    y_mm = distance * np.tan(ty_rad) / np.cos(tx_rad) / 1.0e6
    return x_mm, y_mm


def _coordinate_affine_metadata(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    valid: np.ndarray,
    *,
    headers: Sequence[fits.Header],
    observer_distance_m: np.ndarray,
    slit_indices: np.ndarray,
    scan_indices: np.ndarray,
) -> dict:
    """Build one saved isotropic affine for both atmospheric coordinate MLPs."""

    valid_x = np.asarray(x_mm, dtype=np.float64)[valid]
    valid_y = np.asarray(y_mm, dtype=np.float64)[valid]
    if valid_x.size == 0:
        raise ValueError("Cannot define a coordinate affine without valid spatial samples.")
    centre = np.asarray(
        (
            0.5 * (float(valid_x.min()) + float(valid_x.max())),
            0.5 * (float(valid_y.min()) + float(valid_y.max())),
        ),
        dtype=np.float64,
    )

    spacings = []
    for axis in (0, 1):
        if x_mm.shape[axis] < 2:
            continue
        separation = np.hypot(np.diff(x_mm, axis=axis), np.diff(y_mm, axis=axis))
        detector_indices = slit_indices if axis == 0 else scan_indices
        detector_steps = np.abs(np.diff(np.asarray(detector_indices, dtype=np.float64)))
        if np.any(detector_steps == 0):
            raise ValueError("Hinode detector indices must be unique for coordinate scaling.")
        step_shape = [1, 1]
        step_shape[axis] = detector_steps.size
        separation = separation / detector_steps.reshape(step_shape)
        paired_valid = np.take(valid, range(valid.shape[axis] - 1), axis=axis) & np.take(
            valid, range(1, valid.shape[axis]), axis=axis
        )
        selected = separation[paired_valid]
        spacings.extend(selected[np.isfinite(selected) & (selected > 0)].tolist())

    if not spacings:
        # A one-pixel crop has no selected-neighbour separation. Fall back to
        # documented native plate-scale keywords rather than inventing a crop
        # extent or making the normalization singular.
        angular_scales = []
        for header in headers:
            for key in ("CDELT2", "YSCALE", "XSCALE"):
                value = abs(float(header.get(key, np.nan)))
                if np.isfinite(value) and value > 0:
                    angular_scales.append(value)
        if not angular_scales:
            raise ValueError(
                "A one-pixel Hinode crop requires CDELT2, YSCALE, or XSCALE "
                "to define its physical coordinate normalization."
            )
        distance = float(np.median(observer_distance_m))
        spacings = [
            distance * math.tan(value * u.arcsec.to(u.rad)) / 1.0e6
            for value in angular_scales
        ]

    native_spacing_mm = float(np.median(np.asarray(spacings, dtype=np.float64)))
    if not np.isfinite(native_spacing_mm) or native_spacing_mm <= 0:
        raise ValueError("Hinode physical neighbour spacing must be finite and positive.")
    scale_mm = HINODE_COORDINATE_NORMALIZATION_PIXELS * native_spacing_mm
    return {
        "center_mm": centre.tolist(),
        "scale_mm": [scale_mm, scale_mm],
        "formula": "normalized_xy=(solar_xy_mm-center_mm)/scale_mm",
        "isotropic": True,
        "native_neighbor_spacing_mm": native_spacing_mm,
        "normalization_pixels": HINODE_COORDINATE_NORMALIZATION_PIXELS,
        "spacing_estimator": (
            "median positive centre separation per detector-index step in scan/slit directions; "
            "native plate-scale fallback for a one-pixel crop"
        ),
    }


def _slit_coordinates_arcsec(
    headers: Sequence[fits.Header], selected_slit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return helioprojective coordinates at every selected detector row.

    FITS ``CROTA2`` is an angle in degrees.  With the standard 2-D FITS
    rotation convention, a positive displacement along pixel axis 2 has
    ``(-sin(theta), cos(theta))`` components in Solar-X/Solar-Y.  ``XCEN`` and
    ``YCEN`` locate the slit reference centre for each scan step.
    """

    x_columns = []
    y_columns = []
    rotations_deg = []
    scales_arcsec = []
    scale_keywords = []
    for header in headers:
        required = ("XCEN", "YCEN", "CRPIX2")
        missing = [key for key in required if key not in header]
        if missing:
            raise KeyError(f"Hinode FITS header is missing pointing keys: {missing}.")
        # CDELT2 is the FITS pixel increment of the SP slit axis.  YSCALE is
        # only a legacy plate-scale keyword and is used as a guarded fallback.
        scale_keyword = "CDELT2" if "CDELT2" in header else "YSCALE"
        scale = float(header.get(scale_keyword, np.nan))
        rotation_deg = float(header.get("CROTA2", 0.0))
        centre_x = float(header["XCEN"])
        centre_y = float(header["YCEN"])
        values = np.asarray((scale, rotation_deg, centre_x, centre_y), dtype=np.float64)
        if not np.isfinite(values).all() or scale == 0.0:
            raise ValueError("Hinode slit pointing keywords must be finite with non-zero scale.")
        offset = (
            selected_slit_indices.astype(np.float64)
            + 1.0
            - float(header["CRPIX2"])
        ) * scale
        rotation_rad = np.deg2rad(rotation_deg)
        x_columns.append(centre_x - offset * np.sin(rotation_rad))
        y_columns.append(centre_y + offset * np.cos(rotation_rad))
        rotations_deg.append(rotation_deg)
        scales_arcsec.append(scale)
        scale_keywords.append(scale_keyword)
    return np.stack(x_columns, axis=1), np.stack(y_columns, axis=1), {
        "coordinate_frame": "helioprojective Solar-X/Solar-Y",
        "slit_pixel_formula": "offset=(zero_based_row+1-CRPIX2)*CDELT2",
        "rotation_formula": "x=XCEN-offset*sin(CROTA2); y=YCEN+offset*cos(CROTA2)",
        "rotation_unit": "degree",
        "crota2_deg": rotations_deg,
        "slit_scale_arcsec_per_pixel": scales_arcsec,
        "slit_scale_keywords": scale_keywords,
        "slit_scale_fallback": "YSCALE is used only when CDELT2 is absent",
        "references": {
            "hinode_mission_wide_keywords": HINODE_KEYWORD_REFERENCE_URL,
            "solar_fits_wcs_rotation": SOLAR_WCS_REFERENCE_URL,
        },
    }


def _continuum_mask(
    wavelength: np.ndarray,
    continuum_windows_angstrom,
    continuum_edge_samples: int,
) -> tuple[np.ndarray, dict]:
    if continuum_windows_angstrom is None:
        count = int(continuum_edge_samples)
        if count < 1 or 2 * count >= wavelength.size:
            raise ValueError(
                "continuum_edge_samples must select at least one and fewer than half the samples."
            )
        mask = np.zeros(wavelength.size, dtype=bool)
        mask[:count] = True
        mask[-count:] = True
        method = {"type": "edge_mean", "edge_samples": count}
    else:
        mask = np.zeros(wavelength.size, dtype=bool)
        windows = []
        for window in continuum_windows_angstrom:
            if len(window) != 2:
                raise ValueError("Each continuum window must be [minimum, maximum] in angstrom.")
            low, high = map(float, window)
            if high <= low:
                raise ValueError("Continuum window maximum must exceed its minimum.")
            mask |= (wavelength >= low) & (wavelength <= high)
            windows.append([low, high])
        if not mask.any():
            raise ValueError("Configured continuum windows contain no observed wavelength samples.")
        method = {"type": "window_mean", "windows_angstrom": windows}
    method.update({
        "indices": np.flatnonzero(mask).tolist(),
        "wavelength_angstrom": wavelength[mask].tolist(),
    })
    return mask, method


def _atlas_calibrate_stokes(
    stokes: np.ndarray,
    wavelength: np.ndarray,
    continuum_mask: np.ndarray,
    continuum: np.ndarray,
    mu: np.ndarray,
    valid: np.ndarray,
    *,
    calibration_data_directory,
    quiet_sun_max_fractional_polarization: float,
    quiet_sun_continuum_trim_quantiles,
    minimum_quiet_sun_pixels: int,
) -> tuple[np.ndarray, dict]:
    """Apply one atlas-derived detector scale uniformly to all Stokes data."""

    if quiet_sun_max_fractional_polarization <= 0:
        raise ValueError("quiet_sun_max_fractional_polarization must be positive.")
    if minimum_quiet_sun_pixels < 1:
        raise ValueError("minimum_quiet_sun_pixels must be positive.")
    try:
        lower_quantile, upper_quantile = map(
            float, quiet_sun_continuum_trim_quantiles
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "quiet_sun_continuum_trim_quantiles must contain two quantiles."
        ) from error
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "quiet_sun_continuum_trim_quantiles must satisfy 0 <= low < high <= 1."
        )

    reference = load_solar_reference(calibration_data_directory)
    continuum_wavelength = wavelength[continuum_mask]
    disk_center_samples = disk_center_continuum_radiance(
        reference, continuum_wavelength
    )
    atlas_disk_center_continuum = float(np.mean(disk_center_samples))
    limb_samples = neckel_continuum_limb_darkening(
        continuum_wavelength[None, None, :], mu[..., None]
    )
    expected_continuum = np.mean(
        disk_center_samples[None, None, :] * limb_samples, axis=-1
    )
    mean_limb_darkening = expected_continuum / atlas_disk_center_continuum
    disk_center_equivalent_raw_continuum = continuum / mean_limb_darkening

    polarized_amplitude = np.sqrt(np.sum(np.square(stokes[..., 1:, :]), axis=-2))
    fractional_polarization = np.mean(polarized_amplitude, axis=-1) / continuum
    candidates = (
        valid
        & np.isfinite(fractional_polarization)
        & (fractional_polarization <= quiet_sun_max_fractional_polarization)
    )
    # Remove the known continuum CLV before intensity-based quiet-Sun
    # selection. Otherwise a sufficiently wide raster would rank pixels by
    # heliocentric angle rather than by intrinsic continuum structure.
    # Quantile trimming is meaningful only for a population, not tiny cropped
    # rasters used for diagnostics or tests.
    if np.count_nonzero(candidates) >= 20:
        low, high = np.quantile(
            disk_center_equivalent_raw_continuum[candidates],
            (lower_quantile, upper_quantile),
        )
        candidates &= (
            (disk_center_equivalent_raw_continuum >= low)
            & (disk_center_equivalent_raw_continuum <= high)
        )
    candidate_count = int(np.count_nonzero(candidates))
    if candidate_count < minimum_quiet_sun_pixels:
        raise ValueError(
            "Atlas calibration found only "
            f"{candidate_count} quiet-Sun pixels; at least "
            f"{minimum_quiet_sun_pixels} are required. Supply a raster containing "
            "quiet Sun or relax the explicit quiet_sun calibration settings."
        )

    detector_to_radiance = float(
        np.median(expected_continuum[candidates] / continuum[candidates])
    )
    if not np.isfinite(detector_to_radiance) or detector_to_radiance <= 0:
        raise FloatingPointError(
            "The Hinode detector-to-radiance calibration must be finite and positive."
        )
    # Store all four components in one fixed physical unit. This is not a
    # raster continuum normalization: the detector calibration is derived from
    # quiet Sun, and the denominator is the immutable disk-center atlas value.
    calibrated = stokes * (detector_to_radiance / atlas_disk_center_continuum)
    calibration = {
        "type": "absolute quiet-Sun atlas calibration",
        "operation": (
            "one detector-to-radiance scalar is applied uniformly to I,Q,U,V, "
            "then expressed in fixed disk-center atlas-continuum radiance units"
        ),
        "stored_stokes_unit": "I_c,atlas(mu=1)",
        "physical_radiance_unit": "W m^-3 sr^-1",
        "atlas_disk_center_continuum_radiance_w_m3_sr": atlas_disk_center_continuum,
        "detector_to_radiance_w_m3_sr_per_raw_unit": detector_to_radiance,
        "quiet_sun": {
            "selection": (
                "valid pixels below the configured mean fractional-polarization "
                "threshold, trimmed by disk-center-equivalent continuum quantiles "
                "after removing the atlas center-to-limb variation"
            ),
            "maximum_mean_fractional_polarization": float(
                quiet_sun_max_fractional_polarization
            ),
            "continuum_trim_quantiles": [lower_quantile, upper_quantile],
            "pixel_count": candidate_count,
            "minimum_pixel_count": int(minimum_quiet_sun_pixels),
            "median_mu": float(np.median(mu[candidates])),
            "median_raw_continuum": float(np.median(continuum[candidates])),
            "median_disk_center_equivalent_raw_continuum": float(
                np.median(disk_center_equivalent_raw_continuum[candidates])
            ),
            "mean_limb_darkening_range": [
                float(np.min(mean_limb_darkening[candidates])),
                float(np.max(mean_limb_darkening[candidates])),
            ],
            "median_expected_continuum_radiance_w_m3_sr": float(
                np.median(expected_continuum[candidates])
            ),
        },
        "reference": reference_summary(reference),
    }
    return calibrated, calibration


@dataclass(frozen=True)
class HinodeRaster:
    """One atlas-calibrated Hinode raster and its physical inversion coordinates.

    ``coords`` is ordered ``[time_hours, Solar-X_Mm, Solar-Y_Mm]``. The
    spatial coordinates are helioprojective image-plane distances, not a
    surface-deprojected heliographic map.
    """

    stokes: torch.Tensor
    wavelength_angstrom: torch.Tensor
    coords: torch.Tensor
    mu: torch.Tensor
    valid_mask: torch.Tensor
    metadata: dict

    def __post_init__(self) -> None:
        if self.stokes.ndim != 4 or self.stokes.shape[-2] != 4:
            raise ValueError("stokes must have shape [slit, scan, 4, wavelength].")
        spatial_shape = self.stokes.shape[:2]
        if self.wavelength_angstrom.shape != (self.stokes.shape[-1],):
            raise ValueError("wavelength_angstrom does not match the Stokes wavelength dimension.")
        if self.coords.shape != (*spatial_shape, 3):
            raise ValueError("coords must have shape [slit, scan, 3].")
        if not torch.isfinite(self.coords).all():
            raise ValueError("coords must contain finite physical coordinates.")
        if self.mu.shape != (*spatial_shape, 1):
            raise ValueError("mu must have shape [slit, scan, 1].")
        if self.valid_mask.shape != spatial_shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean [slit, scan] tensor.")
        if not torch.all(self.wavelength_angstrom[1:] > self.wavelength_angstrom[:-1]):
            raise ValueError("wavelength_angstrom must be strictly increasing.")
        if not torch.isfinite(self.mu).all() or torch.any((self.mu < 0) | (self.mu > 1)):
            raise ValueError("mu must be finite and lie in [0, 1].")
        if torch.any(self.mu[..., 0][self.valid_mask] <= 0):
            raise ValueError("Every valid pixel must have 0 < mu <= 1.")

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.stokes.shape[0]), int(self.stokes.shape[1])


def load_hinode_raster(
    files,
    scan_slice=None,
    slit_slice=None,
    continuum_windows_angstrom=None,
    continuum_edge_samples: int = 8,
    calibration_data_directory=None,
    quiet_sun_max_fractional_polarization: float = 0.01,
    quiet_sun_continuum_trim_quantiles=(0.05, 0.95),
    minimum_quiet_sun_pixels: int = 1,
) -> HinodeRaster:
    """Load a Hinode Level-1 scan without plotting, logging, or global state.

    Scan and slit selections are applied before arrays are stacked, making this
    suitable for the small-patch validation stage of the LTE inversion.
    Wavelengths use the exact one-based FITS pixel convention.  Legacy
    pre-1.05 ``sp_prep`` headers are repaired with the documented SolarSoft
    formula; spectral data remain in their calibrated increasing-wavelength
    order. ``DOP_RCV`` supplies the spacecraft-Sun LOS velocity, positive for
    redshift, but ``sp_prep`` has already removed that shift from Level 1.
    ``SPWLSHFT``/``SPWLSFT0`` likewise record corrections already performed by
    ``sp_prep`` and are never applied a second time. Inversion coordinates are
    ``[hours since the first scan, helioprojective Solar-X Mm, Solar-Y Mm]``;
    the spatial conversion uses the same explicit geocentric ephemeris proxy
    documented for the ray geometry.
    """

    paths = _resolve_files(files, scan_slice=scan_slice)
    slit_selection = _slice_from_config(slit_slice, name="slit_slice")

    headers = []
    columns = []
    detector_quality = []
    wavelength = None
    wavelength_solution = None
    full_slit_count = None
    selected_slit_indices = None
    for path in paths:
        header = fits.getheader(path, 0)
        data = fits.getdata(path, 0, memmap=True)
        if data.ndim != 3 or data.shape[0] != 4:
            raise ValueError(
                f"Expected Hinode [4, slit, wavelength] data in {path}; got {data.shape}."
            )
        if full_slit_count is None:
            full_slit_count = data.shape[1]
            selected_slit_indices = np.arange(full_slit_count)[slit_selection]
            if selected_slit_indices.size == 0:
                raise ValueError("slit_slice selected no detector rows.")
        elif data.shape[1] != full_slit_count:
            raise ValueError("All selected Hinode files must have the same slit length.")

        file_wavelength, file_solution = _fits_wavelength_angstrom(header, data.shape[-1])
        if wavelength is None:
            wavelength = file_wavelength
            wavelength_solution = file_solution
        elif not np.allclose(file_wavelength, wavelength, rtol=0.0, atol=1e-9):
            raise ValueError("Selected Hinode files do not share one wavelength grid.")

        # Crop the slit while the data are still per-file.  Detector order is
        # the calibrated increasing-wavelength order and is deliberately kept.
        selected_data = np.asarray(data[:, slit_selection, :])
        if np.issubdtype(selected_data.dtype, np.integer):
            limits = np.iinfo(selected_data.dtype)
            saturated = np.any(
                (selected_data == limits.min) | (selected_data == limits.max),
                axis=(0, 2),
            )
        else:
            saturated = np.zeros(selected_data.shape[1], dtype=bool)
        column = np.asarray(selected_data, dtype=np.float32)
        column = np.moveaxis(column, 0, -2)  # [slit, Stokes, wavelength]
        columns.append(column)
        detector_quality.append(~saturated)
        headers.append(header)

    stokes = np.stack(columns, axis=1)  # [slit, scan, Stokes, wavelength]
    continuum_mask, normalization_metadata = _continuum_mask(
        wavelength, continuum_windows_angstrom, continuum_edge_samples
    )
    continuum = np.mean(stokes[..., 0, continuum_mask], axis=-1, dtype=np.float64).astype(np.float32)
    valid = np.isfinite(stokes).all(axis=(-2, -1)) & np.isfinite(continuum) & (continuum > 0)
    detector_quality = np.stack(detector_quality, axis=1)
    valid &= detector_quality
    times = [parse_datetime(header["DATE_OBS"]) for header in headers]
    ref_time = min(times)
    time_hours = np.asarray(
        [(time - ref_time).total_seconds() / 3_600.0 for time in times],
        dtype=np.float64,
    )
    scan_indices = np.asarray(
        [header.get("SLITINDX", index) for index, header in enumerate(headers)], dtype=np.float32
    )
    if scan_indices.size > 1 and np.any(np.diff(scan_indices) <= 0):
        raise ValueError(
            "Selected Hinode files must have strictly increasing, unique SLITINDX values; "
            "the lexical file order is not a valid raster order."
        )
    acquisition_seconds = np.asarray(
        [(time - times[0]).total_seconds() for time in times], dtype=np.float64
    )
    if acquisition_seconds.size > 1 and np.any(np.diff(acquisition_seconds) < 0):
        raise ValueError("Selected Hinode files are not in nondecreasing DATE_OBS order.")
    x_arcsec, y_arcsec, pointing_metadata = _slit_coordinates_arcsec(
        headers, selected_slit_indices
    )
    observer_distance_m = _observer_distance_m(times)
    solar_radius = _solar_radius_arcsec(times, observer_distance_m)[None, :]
    tx_rad = x_arcsec * u.arcsec.to(u.rad)
    ty_rad = y_arcsec * u.arcsec.to(u.rad)
    # In helioprojective Cartesian coordinates Tx is longitude and Ty is
    # latitude, so cos(rho)=cos(Tx)*cos(Ty). The corresponding ray impact
    # parameter is D*sin(rho); both hypot(Tx,Ty) and rho/Rsun_obs are only
    # small-angle approximations.
    cos_rho = np.cos(tx_rad) * np.cos(ty_rad)
    sin_rho = np.sqrt(np.clip(1.0 - np.square(cos_rho), 0.0, 1.0))
    impact_parameter_m = observer_distance_m[None, :] * sin_rho
    impact_fraction_squared = np.square(
        impact_parameter_m / R_sun.to_value(u.m)
    )
    mu = np.sqrt(np.clip(1.0 - impact_fraction_squared, 0.0, 1.0)).astype(np.float32)
    valid &= mu > 0
    if not np.any(valid):
        raise ValueError("The selected Hinode raster contains no valid on-disk pixels.")

    calibration_directory = (
        Path(__file__).resolve().parent / "data"
        if calibration_data_directory is None
        else calibration_data_directory
    )
    stokes, radiometric_calibration = _atlas_calibrate_stokes(
        stokes,
        wavelength,
        continuum_mask,
        continuum,
        mu,
        valid,
        calibration_data_directory=calibration_directory,
        quiet_sun_max_fractional_polarization=quiet_sun_max_fractional_polarization,
        quiet_sun_continuum_trim_quantiles=quiet_sun_continuum_trim_quantiles,
        minimum_quiet_sun_pixels=minimum_quiet_sun_pixels,
    )
    stokes[~valid] = np.nan

    x_mm, y_mm = _tangent_plane_mm(
        x_arcsec, y_arcsec, observer_distance_m
    )
    time_grid = np.broadcast_to(time_hours[None, :], x_mm.shape)
    coords = np.stack((time_grid, x_mm, y_mm), axis=-1).astype(np.float32)
    coordinate_affine = _coordinate_affine_metadata(
        x_mm,
        y_mm,
        valid,
        headers=headers,
        observer_distance_m=observer_distance_m,
        slit_indices=selected_slit_indices,
        scan_indices=scan_indices,
    )

    normalization_metadata.update({
        "operation": "no raster or per-pixel continuum normalization",
        "scope": "none",
        "estimator": None,
        "radiometric_calibration": radiometric_calibration,
        "continuum_intensity_quantiles": {
            str(percentile): float(np.percentile(continuum[valid], percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
    })
    wavelength_metadata = {
        **wavelength_solution,
        "raw_cdelt1_sign_overridden": bool(float(headers[0]["CDELT1"]) < 0),
        "detector_order_preserved": True,
        "jointly_reordered_with_stokes": False,
        "sign_convention_provenance": {
            "method": "SolarSoft sp_prep calibrated-axis convention",
            "evidence": (
                "thermd_sbsp reverses the output spectral direction so increasing "
                "spectral pixel means increasing wavelength; sp_prep 1.05 makes "
                "CDELT1 positive to match"
            ),
            "source_url": SP_PREP_SOURCE_URL,
            "source_sha256": SP_PREP_SOURCE_SHA256,
        },
        "spwlshft": [header.get("SPWLSHFT") for header in headers],
        "spwlsft0": [header.get("SPWLSFT0") for header in headers],
        "shift_application": (
            "not reapplied: sp_prep records SPWLSHFT as the pixel shift already "
            "applied in the spectral direction and SPWLSFT0 as its thermal-drift input"
        ),
    }
    observer_velocity_correction = _observer_los_velocity_metadata(headers)
    metadata = {
        "instrument": "Hinode/SOT-SP",
        "files": [str(path) for path in paths],
        "file_count": len(paths),
        "slit_indices": selected_slit_indices.tolist(),
        "scan_indices": scan_indices.astype(int).tolist(),
        "times": [time.isoformat() for time in times],
        "ref_time": ref_time.isoformat(),
        "coordinates": {
            "order": ["time_hours", "solar_x_mm", "solar_y_mm"],
            "units": ["hour", "Mm", "Mm"],
            "time_origin": ref_time.isoformat(),
            "time_formula": "time_hours=(DATE_OBS-time_origin)/3600 s",
            "spatial_frame": "helioprojective Solar-X/Solar-Y image plane",
            "spatial_projection": "geocentric tangent plane through the Sun centre",
            "spatial_formula": (
                "X_mm=D*tan(Tx)/1e6; "
                "Y_mm=D*tan(Ty)/(cos(Tx)*1e6)"
            ),
            "observer": (
                "Astropy geocentric solar ephemeris proxy; Level-1 headers do not "
                "provide the Hinode spacecraft distance"
            ),
            "observer_distance_m": observer_distance_m.tolist(),
            "solar_x_range_mm": [float(x_mm[valid].min()), float(x_mm[valid].max())],
            "solar_y_range_mm": [float(y_mm[valid].min()), float(y_mm[valid].max())],
            "surface_deprojection_applied": False,
            "network_affine": coordinate_affine,
            "source_keywords": [
                "DATE_OBS",
                "XCEN",
                "YCEN",
                "CRPIX2",
                "CDELT2/YSCALE",
                "XSCALE",
                "CROTA2",
            ],
        },
        "normalization": normalization_metadata,
        "wavelength": wavelength_metadata,
        "observer_velocity_correction": observer_velocity_correction,
        "stokes_order": ["I", "Q", "U", "V"],
        "quality_mask": {
            "reject_nonfinite": True,
            "reject_nonpositive_continuum": True,
            "reject_integer_detector_limits": True,
            "rejected_pixel_count": int(valid.size - np.count_nonzero(valid)),
        },
        "ray_geometry": {
            **pointing_metadata,
            "mu_definition": (
                "sqrt(max(0, 1 - (D/R_sun)^2*"
                "(1 - (cos(Tx)*cos(Ty))^2)))"
            ),
            "rho_definition": "cos(rho)=cos(Tx)*cos(Ty) for HPC longitude/latitude",
            "impact_parameter_formula": (
                "b=D*sqrt(1-(cos(Tx)*cos(Ty))^2)"
            ),
            "apparent_solar_radius_formula": "asin(R_sun / geocentric_sun_distance)",
            "apparent_solar_radius_arcsec": solar_radius[0].tolist(),
            "observer_distance": (
                "Astropy geocentric solar ephemeris; the Level-1 headers do not "
                "provide a spacecraft observer distance"
            ),
            "transfer_model": (
                "1.5D plane-parallel: mu lengthens each vertical optical-depth "
                "interval by 1/mu without horizontal ray transport"
            ),
        },
        "data_fingerprints": {
            "algorithm": "sha256(dtype, shape, C-order bytes)",
            "selected_stokes": _array_sha256(stokes),
            "wavelength_angstrom": _array_sha256(wavelength),
            "coords": _array_sha256(coords),
            "mu": _array_sha256(mu),
            "valid_mask": _array_sha256(valid),
        },
    }
    return HinodeRaster(
        stokes=torch.from_numpy(np.asarray(stokes, dtype=np.float32)),
        wavelength_angstrom=torch.from_numpy(np.asarray(wavelength, dtype=np.float64)),
        coords=torch.from_numpy(coords),
        mu=torch.from_numpy(mu[..., None]),
        valid_mask=torch.from_numpy(valid),
        metadata=metadata,
    )


class HinodePixelDataset(Dataset):
    """Flatten the valid pixels of a :class:`HinodeRaster`."""

    def __init__(self, raster: HinodeRaster):
        super().__init__()
        self.raster = raster
        self.pixel_indices = torch.nonzero(raster.valid_mask, as_tuple=False)
        if self.pixel_indices.numel() == 0:
            raise ValueError("The selected Hinode raster contains no valid on-disk pixels.")

    def __len__(self) -> int:
        return int(self.pixel_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pixel = self.pixel_indices[index]
        slit, scan = int(pixel[0]), int(pixel[1])
        return {
            "coords": self.raster.coords[slit, scan],
            "mu": self.raster.mu[slit, scan],
            "stokes": self.raster.stokes[slit, scan],
            "pixel_index": pixel,
        }


class _UniformSpatialSampler:
    """Draw independent uniform Solar-X/Solar-Y samples over the FOV bounds."""

    def __init__(self, raster: HinodeRaster):
        valid = raster.valid_mask.detach()
        if not torch.any(valid):
            raise ValueError("Coordinate sampling requires at least one valid pixel.")
        coordinates = raster.coords.detach()[valid]
        self.time_hours = coordinates[:, 0].mean()
        self.xy_min = coordinates[:, 1:].amin(dim=0)
        self.xy_max = coordinates[:, 1:].amax(dim=0)

    @property
    def metadata(self) -> dict:
        return {
            "type": "independent_uniform_xy_bounds",
            "coordinate_order": ["time_hours", "solar_x_mm", "solar_y_mm"],
            "time_sampling": "fixed valid-pixel mean; static atmosphere ignores time",
            "solar_x_bounds_mm": [self.xy_min[0].item(), self.xy_max[0].item()],
            "solar_y_bounds_mm": [self.xy_min[1].item(), self.xy_max[1].item()],
            "axis_aligned_bounding_box_used": True,
        }

    def sample(self) -> torch.Tensor:
        fraction = torch.rand(2, dtype=self.xy_min.dtype, device=self.xy_min.device)
        xy = self.xy_min + fraction * (self.xy_max - self.xy_min)
        return torch.cat((self.time_hours.reshape(1), xy))


class RandomAtmosphereVolumeDataset(Dataset):
    """Independent spatial samples grouped onto random tau surfaces at collation.

    The dataset samples only coordinates. The physics collate function assigns
    consecutive groups of coordinates one shared random ``log_tau500`` value.
    """

    def __init__(
        self,
        raster: HinodeRaster,
        log_tau_bounds,
        length: int,
    ):
        self.coordinate_sampler = _UniformSpatialSampler(raster)
        self.log_tau_min, self.log_tau_max = map(float, log_tau_bounds)
        self.length = int(length)
        if (
            self.length < 1
            or not self.log_tau_max > self.log_tau_min
        ):
            raise ValueError("Invalid physics-volume sampler configuration.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return {"coords": self.coordinate_sampler.sample()}


def _grouped_tau_collate(
    batch,
    *,
    log_tau_bounds,
    points_per_tau: int,
):
    """Assign each spatial group one freshly sampled optical-depth value."""

    collated = default_collate(batch)
    sample_count = collated["coords"].shape[0]
    if sample_count % points_per_tau:
        raise ValueError(
            "A grouped tau batch must be divisible by points_per_tau; got "
            f"{sample_count} samples and {points_per_tau} points per tau."
        )
    minimum, maximum = map(float, log_tau_bounds)
    group_count = sample_count // points_per_tau
    group_tau = minimum + torch.rand(group_count) * (maximum - minimum)
    collated["log_tau500"] = group_tau.repeat_interleave(points_per_tau)
    return collated


class RandomTopBoundaryDataset(Dataset):
    """Independent pressure-boundary samples on the upper optical-depth face."""

    def __init__(
        self,
        raster: HinodeRaster,
        top_log_tau: float,
        gas_pressure_pa: float,
        length: int,
    ):
        self.coordinate_sampler = _UniformSpatialSampler(raster)
        self.top_log_tau = float(top_log_tau)
        self.gas_pressure_pa = float(gas_pressure_pa)
        self.length = int(length)
        if self.length < 1 or self.gas_pressure_pa <= 0:
            raise ValueError(
                "Top-boundary sampler length and gas pressure must be positive."
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        coordinate = self.coordinate_sampler.sample()
        return {
            "coords": coordinate,
            "log_tau500": coordinate.new_tensor(self.top_log_tau),
            "gas_pressure_pa": coordinate.new_tensor(self.gas_pressure_pa),
        }


class HinodeLTEDataModule(LightningDataModule):
    """Small-batch Lightning data module for Hinode LTE inversion."""

    def __init__(
        self,
        files,
        batch_size: int = 4,
        validation_batch_size: int | None = None,
        validation_stride: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **loader_kwargs,
    ):
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if validation_batch_size is not None and validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive when provided.")
        if validation_stride < 1:
            raise ValueError("validation_stride must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if pin_memory:
            raise ValueError("Hinode LTE data loading requires pin_memory=false.")
        self.files = files
        self.batch_size = int(batch_size)
        self.validation_batch_size = (
            self.batch_size if validation_batch_size is None else int(validation_batch_size)
        )
        self.validation_stride = int(validation_stride)
        self.num_workers = int(num_workers)
        self.loader_kwargs = loader_kwargs
        self.raster: HinodeRaster | None = None
        self.dataset: HinodePixelDataset | None = None
        self.validation_dataset: Subset | None = None
        self.physics_sampling_config: dict | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.raster is None:
            self.raster = load_hinode_raster(self.files, **self.loader_kwargs)
            self.dataset = HinodePixelDataset(self.raster)
            # A flat stride aliases badly with regular raster widths (for
            # example stride 64 on width 1024 selects vertical stripes). Use a
            # two-dimensional lattice with approximately the requested area
            # decimation instead.
            row_stride = max(1, int(math.floor(math.sqrt(self.validation_stride))))
            column_stride = max(
                1, int(math.ceil(self.validation_stride / row_stride))
            )
            pixels = self.dataset.pixel_indices
            selected = (
                (pixels[:, 0].remainder(row_stride) == 0)
                & (pixels[:, 1].remainder(column_stride) == 0)
            )
            subset_indices = torch.nonzero(selected, as_tuple=False).squeeze(-1).tolist()
            if not subset_indices:
                subset_indices = [0]
            self.validation_lattice_stride = (row_stride, column_stride)
            self.validation_dataset = Subset(self.dataset, subset_indices)

    def configure_physics_sampling(
        self,
        log_tau500,
        *,
        volume_points_per_step: int,
        points_per_tau: int,
        boundary_points_per_step: int,
        top_pressure_pa: float | None,
    ) -> None:
        """Attach interior physics samples and an optional pressure boundary stream."""

        grid = torch.as_tensor(log_tau500)
        if grid.ndim != 1 or grid.numel() < 2 or not torch.all(grid[1:] > grid[:-1]):
            raise ValueError("Physics sampling requires an increasing log_tau500 grid.")
        if volume_points_per_step < 1 or points_per_tau < 1 or boundary_points_per_step < 0:
            raise ValueError(
                "Physics volume samples and points_per_tau must be positive; "
                "boundary samples cannot be negative."
            )
        if volume_points_per_step % points_per_tau:
            raise ValueError(
                "volume_points_per_step must be divisible by points_per_tau."
            )
        if boundary_points_per_step > 0 and (
            top_pressure_pa is None or top_pressure_pa <= 0
        ):
            raise ValueError(
                "A positive top pressure is required when the pressure boundary stream "
                "is enabled."
            )
        self.physics_sampling_config = {
            "log_tau_bounds": (float(grid[0]), float(grid[-1])),
            "volume_points_per_step": int(volume_points_per_step),
            "points_per_tau": int(points_per_tau),
            "tau_surfaces_per_step": int(volume_points_per_step // points_per_tau),
            "boundary_points_per_step": int(boundary_points_per_step),
            "top_boundary_stream_enabled": bool(boundary_points_per_step > 0),
            "top_pressure_pa": (
                None if boundary_points_per_step == 0 else float(top_pressure_pa)
            ),
            "collocation_distribution": (
                "independent uniform Solar-X/Y samples grouped onto shared, uniformly "
                "sampled log_tau500 surfaces"
            ),
        }

    @property
    def wavelength_angstrom(self) -> torch.Tensor:
        if self.raster is None:
            raise RuntimeError("Call setup() before accessing wavelength_angstrom.")
        return self.raster.wavelength_angstrom

    @property
    def normalization_metadata(self) -> dict:
        if self.raster is None:
            raise RuntimeError("Call setup() before accessing normalization metadata.")
        return self.raster.metadata["normalization"]

    def checkpoint_metadata(self) -> dict:
        if self.raster is None:
            raise RuntimeError("Call setup() before requesting checkpoint metadata.")
        metadata = dict(self.raster.metadata)
        metadata["validation"] = {
            "type": "deterministic diagnostic subset of inversion pixels",
            "target_area_stride": self.validation_stride,
            "spatial_lattice_stride_slit_scan": list(self.validation_lattice_stride),
            "pixel_count": len(self.validation_dataset),
            "held_out": False,
        }
        metadata["training_sampling"] = {
            "type": "full valid raster once per epoch",
            "samples_per_epoch": len(self.dataset),
            "valid_pixel_count": len(self.dataset),
        }
        metadata["physics_sampling"] = (
            None
            if self.physics_sampling_config is None
            else dict(self.physics_sampling_config)
        )
        if metadata["physics_sampling"] is not None:
            metadata["physics_sampling"]["spatial_sampling"] = (
                _UniformSpatialSampler(self.raster).metadata
            )
        return metadata

    def _data_loader(
        self,
        *,
        dataset,
        shuffle: bool,
        batch_size: int,
        collate_fn=None,
    ) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )

    def _stokes_train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return self._data_loader(
            dataset=self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        stokes_loader = self._stokes_train_dataloader()
        if self.physics_sampling_config is None:
            return stokes_loader
        steps = len(stokes_loader)
        volume_batch = self.physics_sampling_config["volume_points_per_step"]
        boundary_batch = self.physics_sampling_config["boundary_points_per_step"]
        volume = RandomAtmosphereVolumeDataset(
            self.raster,
            self.physics_sampling_config["log_tau_bounds"],
            steps * volume_batch,
        )
        loaders = {
            "stokes": stokes_loader,
            "physics_volume": self._data_loader(
                dataset=volume,
                shuffle=False,
                batch_size=volume_batch,
                collate_fn=partial(
                    _grouped_tau_collate,
                    log_tau_bounds=self.physics_sampling_config["log_tau_bounds"],
                    points_per_tau=self.physics_sampling_config["points_per_tau"],
                ),
            ),
        }
        if boundary_batch > 0:
            boundary = RandomTopBoundaryDataset(
                self.raster,
                self.physics_sampling_config["log_tau_bounds"][0],
                self.physics_sampling_config["top_pressure_pa"],
                steps * boundary_batch,
            )
            loaders["pressure_boundary"] = self._data_loader(
                dataset=boundary, shuffle=False, batch_size=boundary_batch
            )
        return loaders

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            self.setup()
        return self._data_loader(
            dataset=self.validation_dataset,
            shuffle=False,
            batch_size=self.validation_batch_size,
        )
