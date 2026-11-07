"""Idealized (uncalibrated) HMI tunable-filter model for synthetic validation.

The real :class:`~prom3theus.instruments.hmi.operator.HMIFilterProfiles`
integrates through ``spectral_weights``/``continuum_weights`` derived from
per-record JSOC phase-map calibration archives
(:mod:`prom3theus.instruments.hmi.response`), which this validation tool does
not have. This module instead builds a textbook single-Gaussian
approximation of HMI's six tuned filter positions (FWHM ~76 mA, Couvidat et
al. 2012) with a normalized quadrature integral, so the same
``HMIFilterProfiles`` line-integration code can still be exercised. This is a
deliberate physical simplification (no side lobes, no per-pixel phase-map
variation, no far-field blocking-filter leakage) for testing the LTE
inversion, not a substitute for the calibrated instrument response.
"""

from __future__ import annotations

import math

import torch

from prom3theus.instruments.hmi.operator import HMIFilterProfiles

TUNING_REFERENCE_ANGSTROM = 6173.3433
TUNING_OFFSETS_ANGSTROM = (-0.172, -0.1032, -0.0344, 0.0344, 0.1032, 0.172)
IDEALIZED_FILTER_FWHM_ANGSTROM = 0.076
INNER_QUADRATURE_HALF_WIDTH_ANGSTROM = 0.65


def build_idealized_hmi_filter(
    *, n_quadrature: int = 161, dtype: torch.dtype = torch.float64
):
    """Return ``(operator, observed_wavelength, spectral_weights, continuum_weights)``.

    ``operator`` is a real ``HMIFilterProfiles`` instance (so the tested
    filter-integration and Stokes-basis-convention code runs unchanged);
    only the transmission weights themselves are idealized. Absolute wavelength
    metadata and the unprepared operator stay float64; ``dtype`` selects the
    response-weight dtype. Cast the operator only after spectral preparation.
    """

    if (
        isinstance(n_quadrature, bool)
        or int(n_quadrature) != n_quadrature
        or n_quadrature < 3
    ):
        raise ValueError("n_quadrature must be an integer of at least three.")
    half_width = INNER_QUADRATURE_HALF_WIDTH_ANGSTROM
    quadrature = torch.linspace(
        TUNING_REFERENCE_ANGSTROM - 0.98 * half_width,
        TUNING_REFERENCE_ANGSTROM + 0.98 * half_width,
        n_quadrature,
        dtype=torch.float64,
    )
    operator = HMIFilterProfiles(
        quadrature_wavelength_angstrom=quadrature,
        inner_half_width_angstrom=half_width,
        magnetic_azimuth_offset_deg=90.0,
    )

    observed = TUNING_REFERENCE_ANGSTROM + torch.as_tensor(
        TUNING_OFFSETS_ANGSTROM, dtype=torch.float64
    )
    d_quadrature = float(quadrature[1] - quadrature[0])
    sigma = IDEALIZED_FILTER_FWHM_ANGSTROM / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    delta = quadrature[None, :] - observed[:, None]
    transmission = torch.exp(-0.5 * (delta / sigma) ** 2)
    unnormalized_weight = transmission * d_quadrature
    spectral_weights = unnormalized_weight / unnormalized_weight.sum(
        dim=-1, keepdim=True
    )
    continuum_weights = torch.zeros(observed.shape[0], dtype=dtype)

    return operator, observed, spectral_weights.to(dtype), continuum_weights
