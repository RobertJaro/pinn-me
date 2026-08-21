"""Absolute quiet-Sun radiometry for Hinode/SOT-SP calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SOLAR_REFERENCE_FILENAME = "solar_reference_630nm.json"


def load_solar_reference(directory) -> dict:
    """Load and validate the prepared 630 nm absolute solar reference."""

    path = Path(directory).expanduser().resolve() / SOLAR_REFERENCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing absolute solar reference {path}. Run pinn-me-lte-fetch-data."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported absolute solar reference in {path}.")
    wavelength = np.asarray(document.get("wavelength_air_angstrom"), dtype=np.float64)
    continuum = np.asarray(
        document.get("continuum_radiance_w_m3_sr"), dtype=np.float64
    )
    intensity = np.asarray(
        document.get("intensity_radiance_w_m3_sr"), dtype=np.float64
    )
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or continuum.shape != wavelength.shape
        or intensity.shape != wavelength.shape
        or not np.isfinite(wavelength).all()
        or not np.isfinite(continuum).all()
        or not np.isfinite(intensity).all()
        or np.any(np.diff(wavelength) <= 0)
        or np.any(continuum <= 0)
        or np.any(intensity <= 0)
    ):
        raise RuntimeError(f"Invalid absolute solar reference arrays in {path}.")
    return document


def disk_center_continuum_radiance(
    reference: dict, wavelength_air_angstrom
) -> np.ndarray:
    """Interpolate disk-center continuum radiance in W m^-3 sr^-1."""

    wavelength = np.asarray(wavelength_air_angstrom, dtype=np.float64)
    grid = np.asarray(reference["wavelength_air_angstrom"], dtype=np.float64)
    if wavelength.size == 0 or wavelength.min() < grid[0] or wavelength.max() > grid[-1]:
        raise ValueError(
            "Observed continuum wavelengths fall outside the prepared solar-reference window."
        )
    return np.interp(
        wavelength,
        grid,
        np.asarray(reference["continuum_radiance_w_m3_sr"], dtype=np.float64),
    )


def neckel_continuum_limb_darkening(
    wavelength_air_angstrom, mu
) -> np.ndarray:
    """Return I_c(lambda, mu) / I_c(lambda, 1) from Neckel (2005).

    This is the published 422.57--1100 nm branch. Wavelength and ``mu`` are
    broadcast by NumPy; wavelength is supplied in Angstrom and converted to
    micrometres for the coefficient formula.
    """

    wavelength_um = np.asarray(wavelength_air_angstrom, dtype=np.float64) * 1.0e-4
    mu = np.asarray(mu, dtype=np.float64)
    if (
        not np.isfinite(wavelength_um).all()
        or np.any((wavelength_um < 0.42257) | (wavelength_um > 1.1))
    ):
        raise ValueError("Neckel continuum wavelengths must lie in 4225.7--11000 A.")
    if not np.isfinite(mu).all() or np.any((mu < 0.0) | (mu > 1.0)):
        raise ValueError("mu must be finite and lie in [0, 1].")
    wavelength_um, mu = np.broadcast_arrays(wavelength_um, mu)
    inverse = 1.0 / wavelength_um
    inverse_fifth = inverse**5
    coefficients = np.stack(
        (
            0.75267 - 0.265577 * inverse,
            0.93874 + 0.265577 * inverse - 0.004095 * inverse_fifth,
            -1.89287 + 0.012582 * inverse_fifth,
            2.42234 - 0.017117 * inverse_fifth,
            -1.71150 + 0.011977 * inverse_fifth,
            0.49062 - 0.003347 * inverse_fifth,
        ),
        axis=-1,
    )
    result = sum(coefficients[..., index] * mu**index for index in range(6))
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise FloatingPointError("Neckel continuum limb darkening is non-positive.")
    return result


def reference_summary(reference: dict) -> dict:
    """Return compact checkpoint-safe provenance for a loaded reference."""

    return {
        "schema_version": reference["schema_version"],
        "source": reference["source"],
        "units": reference["units"],
        "wavelength_range_air_angstrom": reference[
            "wavelength_range_air_angstrom"
        ],
        "limb_darkening": reference["limb_darkening"],
    }


__all__ = [
    "SOLAR_REFERENCE_FILENAME",
    "disk_center_continuum_radiance",
    "load_solar_reference",
    "neckel_continuum_limb_darkening",
    "reference_summary",
]
