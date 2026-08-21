"""Wavelength-medium conversions used at the LTE physics boundary.

Hinode/SOT-SP Level-1 coordinates and the adopted optical line metadata are
standard-air Angstrom. Photon frequencies, Planck radiance, and continuum
cross sections use the corresponding vacuum wavelength.
"""

from __future__ import annotations

import torch


def air_to_vacuum_angstrom(wavelength_air_angstrom) -> torch.Tensor:
    """Convert standard-air wavelength in Angstrom to vacuum Angstrom.

    This differentiable algebraic expression is the Piskunov inverse adopted
    by VALD for the Morton (2000) standard-air refractive index. Its published
    validity interval is 2000--100000 Angstrom.

    See https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion.
    """

    wavelength = torch.as_tensor(wavelength_air_angstrom)
    if not wavelength.is_floating_point():
        wavelength = wavelength.to(torch.get_default_dtype())
    if not torch.isfinite(wavelength).all():
        raise ValueError("Air wavelengths must be finite.")
    if torch.any((wavelength < 2000.0) | (wavelength > 100000.0)):
        raise ValueError(
            "The adopted standard-air conversion is valid from 2000 to "
            "100000 Angstrom."
        )
    inverse_micrometres_squared = (1.0e4 / wavelength).square()
    refractive_index = (
        1.0
        + 0.00008336624212083
        + 0.02408926869968
        / (130.1065924522 - inverse_micrometres_squared)
        + 0.0001599740894897
        / (38.92568793293 - inverse_micrometres_squared)
    )
    return wavelength * refractive_index


__all__ = ["air_to_vacuum_angstrom"]
