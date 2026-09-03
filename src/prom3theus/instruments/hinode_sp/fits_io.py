"""Deterministic FITS reading and calibrated WCS evaluation for Hinode/SP."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import glob
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse as parse_datetime

from .constants import HINODE_KEYWORD_REFERENCE_URL, SOLAR_WCS_REFERENCE_URL


def slice_from_config(value, *, name: str) -> slice:
    """Convert one strict two-item half-open selection into a slice."""

    if value is None:
        return slice(None)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be [start, stop].")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two entries.")
    if any(type(item) is not int for item in value):
        raise TypeError(f"{name} entries must be integers.")
    start, stop = value
    if start < 0 or stop <= start:
        raise ValueError(f"{name} must satisfy 0 <= start < stop.")
    return slice(start, stop)


def resolve_files(files) -> list[Path]:
    """Resolve file patterns once and return a unique deterministic scan order."""

    patterns = [files] if isinstance(files, (str, Path)) else list(files)
    resolved: list[Path] = []
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.extend(Path(match) for match in matches)
    resolved = sorted(dict.fromkeys(resolved))
    if not resolved:
        raise FileNotFoundError(f"No Hinode FITS files match {patterns}.")
    return resolved


def wavelength_angstrom(header: fits.Header, count: int) -> tuple[np.ndarray, dict]:
    """Evaluate a positive, calibrated Level-1 wavelength WCS."""

    if type(count) is not int or count < 2:
        raise ValueError("Hinode wavelength grids require at least two samples.")
    required = ("CTYPE1", "CRVAL1", "CRPIX1", "CDELT1", "CUNIT1")
    missing = [key for key in required if key not in header]
    if missing:
        raise KeyError(f"Hinode FITS header is missing wavelength keys: {missing}.")
    if str(header["CTYPE1"]).strip().lower() not in {"wave", "wavelength"}:
        raise ValueError("Hinode CTYPE1 must identify the wavelength axis.")
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
    increment = float(header["CDELT1"]) * unit_to_angstrom
    if not np.isfinite(increment) or increment <= 0:
        raise ValueError("Hinode runtime ingestion requires a finite positive CDELT1.")
    reference = float(header["CRVAL1"]) * unit_to_angstrom
    reference_pixel = float(header["CRPIX1"])
    if not np.isfinite(reference) or not np.isfinite(reference_pixel):
        raise ValueError("Hinode CRVAL1 and CRPIX1 must be finite.")
    pixel = np.arange(int(count), dtype=np.float64) + 1.0
    wavelength = reference + (pixel - reference_pixel) * increment
    return wavelength, {
        "unit": "angstrom",
        "source_cunit1": source_unit_label,
        "cunit1_conversion_factor_to_angstrom": unit_to_angstrom,
        "fits_formula": ("CRVAL1 + ((zero_based_pixel + 1) - CRPIX1) * CDELT1"),
        "raw_cdelt1_angstrom": increment,
        "effective_cdelt1_angstrom": increment,
        "raw_crval1_angstrom": reference,
        "effective_crval1_angstrom": reference,
        "method": "positive calibrated FITS WCS",
    }


def read_primary_date_obs(path: Path) -> datetime:
    """Read the authoritative acquisition timestamp from the primary HDU."""

    header = fits.getheader(path, 0)
    if "DATE_OBS" not in header:
        raise KeyError(f"Hinode FITS file is missing DATE_OBS: {path}.")
    return parse_datetime(header["DATE_OBS"])


def read_column(path: Path, slit_selection: slice) -> dict:
    """Read one selected Level-1 scan column and close its mmap promptly."""

    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data
        if data is None or data.ndim != 3 or data.shape[0] != 4:
            raise ValueError(
                f"Expected Hinode [4, slit, wavelength] data in {path}; "
                f"got {None if data is None else data.shape}."
            )
        missing = sorted({"DATE_OBS", "SLITINDX"} - set(header))
        if missing:
            raise KeyError(
                f"Hinode FITS file {path} is missing raster identity keys: {missing}."
            )
        slit_index = header["SLITINDX"]
        if int(slit_index) != slit_index or int(slit_index) < 0:
            raise ValueError(
                f"Hinode SLITINDX must be a non-negative integer in {path}."
            )
        full_slit_count = int(data.shape[1])
        selected = data[:, slit_selection, :]
        if selected.shape[1] == 0:
            raise ValueError("slit_slice selected no detector rows.")
        if np.issubdtype(selected.dtype, np.integer):
            limits = np.iinfo(selected.dtype)
            saturated = np.any(
                (selected == limits.min) | (selected == limits.max), axis=(0, 2)
            )
        else:
            saturated = np.zeros(selected.shape[1], dtype=bool)
        column = np.array(selected, dtype=np.float32, copy=True)
    wavelength, solution = wavelength_angstrom(header, column.shape[-1])
    return {
        "header": header,
        "column": np.moveaxis(column, 0, -2),
        "detector_quality": ~saturated,
        "wavelength": wavelength,
        "wavelength_solution": solution,
        "full_slit_count": full_slit_count,
    }


def slit_coordinates_arcsec(
    headers: Sequence[fits.Header], selected_slit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate the helioprojective pointing WCS at every selected slit row."""

    x_columns: list[np.ndarray] = []
    y_columns: list[np.ndarray] = []
    rotations_deg: list[float] = []
    scales_arcsec: list[float] = []
    for header in headers:
        required = (
            "CTYPE2",
            "CUNIT2",
            "XCEN",
            "YCEN",
            "CRPIX2",
            "CDELT2",
            "CROTA2",
        )
        missing = [key for key in required if key not in header]
        if missing:
            raise KeyError(f"Hinode FITS header is missing pointing keys: {missing}.")
        if str(header["CTYPE2"]).strip().upper() not in {"SOLAR-Y", "HPLT-TAN"}:
            raise ValueError("Hinode CTYPE2 must identify the solar-latitude axis.")
        try:
            spatial_unit = u.Unit(str(header["CUNIT2"]).strip())
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Hinode CUNIT2 must be a recognized angular unit."
            ) from error
        if not spatial_unit.is_equivalent(u.arcsec) or not np.isclose(
            spatial_unit.to(u.arcsec), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Hinode CUNIT2 must express the slit WCS in arcseconds.")
        scale = float(header["CDELT2"])
        rotation_deg = float(header["CROTA2"])
        centre_x = float(header["XCEN"])
        centre_y = float(header["YCEN"])
        values = np.asarray((scale, rotation_deg, centre_x, centre_y))
        if not np.isfinite(values).all() or scale <= 0.0:
            raise ValueError(
                "Hinode slit pointing keywords must be finite with positive CDELT2."
            )
        offset = (
            selected_slit_indices.astype(np.float64) + 1.0 - float(header["CRPIX2"])
        ) * scale
        rotation_rad = np.deg2rad(rotation_deg)
        x_columns.append(centre_x - offset * np.sin(rotation_rad))
        y_columns.append(centre_y + offset * np.cos(rotation_rad))
        rotations_deg.append(rotation_deg)
        scales_arcsec.append(scale)
    return (
        np.stack(x_columns, axis=1),
        np.stack(y_columns, axis=1),
        {
            "coordinate_frame": "helioprojective Solar-X/Solar-Y",
            "slit_pixel_formula": "offset=(zero_based_row+1-CRPIX2)*CDELT2",
            "rotation_formula": (
                "x=XCEN-offset*sin(CROTA2); y=YCEN+offset*cos(CROTA2)"
            ),
            "rotation_unit": "degree",
            "crota2_deg": rotations_deg,
            "slit_scale_arcsec_per_pixel": scales_arcsec,
            "slit_scale_keyword": "CDELT2",
            "references": {
                "hinode_mission_wide_keywords": HINODE_KEYWORD_REFERENCE_URL,
                "solar_fits_wcs_rotation": SOLAR_WCS_REFERENCE_URL,
            },
        },
    )


__all__ = [
    "read_column",
    "read_primary_date_obs",
    "resolve_files",
    "slit_coordinates_arcsec",
    "slice_from_config",
    "wavelength_angstrom",
]
