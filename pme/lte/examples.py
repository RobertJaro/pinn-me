"""Synthetic atmosphere helpers for examples and regression tests.

These profiles are deliberately outside the inversion parameterization. They
are not an initialization, an EOS table, or a claimed standard atmosphere.
"""

from __future__ import annotations

import torch


_SYNTHETIC_LOG_TAU500 = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0)
_SYNTHETIC_TEMPERATURE_K = (4470.0, 4170.0, 4320.0, 4800.0, 5450.0, 6420.0, 8000.0)


def synthetic_temperature_profile(log_tau500: torch.Tensor) -> torch.Tensor:
    """Return a smooth, coarse photospheric test profile in kelvin."""

    x = log_tau500
    xp = x.new_tensor(_SYNTHETIC_LOG_TAU500)
    fp = x.new_tensor(_SYNTHETIC_TEMPERATURE_K)
    flat = x.reshape(-1)
    clamped = flat.clamp(xp[0], xp[-1])
    intervals = xp[1:] - xp[:-1]
    secants = (fp[1:] - fp[:-1]) / intervals
    interior_slopes = (
        intervals[1:] * secants[:-1] + intervals[:-1] * secants[1:]
    ) / (intervals[:-1] + intervals[1:])
    slopes = torch.cat((secants[:1], interior_slopes, secants[-1:]))
    indices = torch.searchsorted(xp, clamped).clamp(1, xp.numel() - 1)
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]
    m0 = slopes[indices - 1]
    m1 = slopes[indices]
    width = x1 - x0
    fraction = (clamped - x0) / width
    fraction2 = fraction.square()
    fraction3 = fraction2 * fraction
    result = (
        (2.0 * fraction3 - 3.0 * fraction2 + 1.0) * y0
        + (fraction3 - 2.0 * fraction2 + fraction) * width * m0
        + (-2.0 * fraction3 + 3.0 * fraction2) * y1
        + (fraction3 - fraction2) * width * m1
    )
    return result.reshape(x.shape)


__all__ = ["synthetic_temperature_profile"]
