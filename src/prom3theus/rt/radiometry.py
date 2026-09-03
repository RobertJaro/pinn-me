"""Absolute quiet-Sun radiometry for LTE observation calibration."""

from __future__ import annotations

import json

import numpy as np

from prom3theus.resources import (
    load_verified_source_manifest,
    resource_path,
    verify_manifest_resource,
)


HINODE_SOLAR_REFERENCE_RESOURCE = "hinode_sp/solar_reference_630nm.json"
SOLAR_REFERENCE_RESOURCES = frozenset(
    {
        HINODE_SOLAR_REFERENCE_RESOURCE,
        "hmi_stokes/solar_reference_617nm.json",
    }
)
SOLAR_REFERENCE_UNITS = {
    "wavelength": "standard-air angstrom",
    "intensity_radiance": "W m^-3 sr^-1",
    "continuum_radiance": "W m^-3 sr^-1",
    "source_intensity_radiance": "W cm^-2 sr^-1 angstrom^-1",
    "source_to_runtime_factor": 100_000_000_000_000,
}


def load_solar_reference(
    resource_name: str = HINODE_SOLAR_REFERENCE_RESOURCE,
) -> dict:
    """Load one checksum-verified solar reference from the packaged bundle."""

    if resource_name not in SOLAR_REFERENCE_RESOURCES:
        raise ValueError(
            f"Unknown solar reference {resource_name!r}; expected one of "
            f"{sorted(SOLAR_REFERENCE_RESOURCES)}."
        )
    path = resource_path(resource_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing absolute solar reference {path}. Prepare a verified LTE resource bundle."
        )
    manifest = load_verified_source_manifest()
    verify_manifest_resource(
        path,
        manifest,
        resource_name=resource_name,
        kind="absolute-solar-reference",
    )
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported absolute solar reference in {path}.")
    wavelength = np.asarray(document.get("wavelength_air_angstrom"), dtype=np.float64)
    continuum = np.asarray(document.get("continuum_radiance_w_m3_sr"), dtype=np.float64)
    intensity = np.asarray(document.get("intensity_radiance_w_m3_sr"), dtype=np.float64)
    wavelength_range = np.asarray(
        document.get("wavelength_range_air_angstrom"), dtype=np.float64
    )
    if (
        document.get("units") != SOLAR_REFERENCE_UNITS
        or wavelength.ndim != 1
        or wavelength.size < 2
        or continuum.shape != wavelength.shape
        or intensity.shape != wavelength.shape
        or not np.isfinite(wavelength).all()
        or not np.isfinite(continuum).all()
        or not np.isfinite(intensity).all()
        or np.any(np.diff(wavelength) <= 0)
        or np.any(continuum <= 0)
        or np.any(intensity <= 0)
        or wavelength_range.shape != (2,)
        or not np.isfinite(wavelength_range).all()
        or wavelength_range[0] > wavelength[0]
        or wavelength_range[1] < wavelength[-1]
        or wavelength_range[1] <= wavelength_range[0]
    ):
        raise RuntimeError(f"Invalid absolute solar reference arrays in {path}.")
    return document


def disk_center_continuum_radiance(
    reference: dict, wavelength_air_angstrom
) -> np.ndarray:
    """Interpolate disk-center continuum radiance in W m^-3 sr^-1."""

    wavelength = np.asarray(wavelength_air_angstrom, dtype=np.float64)
    grid = np.asarray(reference["wavelength_air_angstrom"], dtype=np.float64)
    if (
        wavelength.size == 0
        or not np.isfinite(wavelength).all()
        or wavelength.min() < grid[0]
        or wavelength.max() > grid[-1]
    ):
        raise ValueError(
            "Observed continuum wavelengths fall outside the prepared solar-reference window."
        )
    return np.interp(
        wavelength,
        grid,
        np.asarray(reference["continuum_radiance_w_m3_sr"], dtype=np.float64),
    )


def disk_center_intensity_radiance(
    reference: dict, wavelength_air_angstrom
) -> np.ndarray:
    """Interpolate disk-center atlas intensity in W m^-3 sr^-1."""

    wavelength = np.asarray(wavelength_air_angstrom, dtype=np.float64)
    grid = np.asarray(reference["wavelength_air_angstrom"], dtype=np.float64)
    if (
        wavelength.size == 0
        or not np.isfinite(wavelength).all()
        or wavelength.min() < grid[0]
        or wavelength.max() > grid[-1]
    ):
        raise ValueError(
            "Requested wavelengths fall outside the prepared solar-reference window."
        )
    return np.interp(
        wavelength,
        grid,
        np.asarray(reference["intensity_radiance_w_m3_sr"], dtype=np.float64),
    )


def neckel_continuum_limb_darkening(wavelength_air_angstrom, mu) -> np.ndarray:
    """Return I_c(lambda, mu) / I_c(lambda, 1) from Neckel (2005).

    This is the published 422.57--1100 nm branch. Wavelength and ``mu`` are
    broadcast by NumPy; wavelength is supplied in Angstrom and converted to
    micrometres for the coefficient formula.
    """

    wavelength_um = np.asarray(wavelength_air_angstrom, dtype=np.float64) * 1.0e-4
    mu = np.asarray(mu, dtype=np.float64)
    if not np.isfinite(wavelength_um).all() or np.any(
        (wavelength_um < 0.42257) | (wavelength_um > 1.1)
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
    """Return compact artifact-safe provenance for a loaded reference."""

    return {
        "schema_version": reference["schema_version"],
        "source": reference["source"],
        "units": reference["units"],
        "wavelength_range_air_angstrom": reference["wavelength_range_air_angstrom"],
        "limb_darkening": reference["limb_darkening"],
    }


__all__ = [
    "HINODE_SOLAR_REFERENCE_RESOURCE",
    "SOLAR_REFERENCE_RESOURCES",
    "SOLAR_REFERENCE_UNITS",
    "disk_center_continuum_radiance",
    "disk_center_intensity_radiance",
    "load_solar_reference",
    "neckel_continuum_limb_darkening",
    "reference_summary",
]
