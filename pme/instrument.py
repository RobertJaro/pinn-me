"""Instrument wavelength definitions shared by loaders and synthetic data."""

import numpy as np
from astropy import units as u


HMI_WAVELENGTH_CENTER = 6173.3433 * u.AA

# hmi.S_720s segments I0..I5 are stored from the red to the blue wing.
# The nominal tuning separation is 68.8 mA and the outer positions are
# approximately +/-172 mA from the Fe I reference wavelength.
HMI_WAVELENGTH_GRID = np.array([
    0.1720,
    0.1032,
    0.0344,
    -0.0344,
    -0.1032,
    -0.1720,
]) * u.AA


def hmi_wavelength_config():
    """Return independent quantities so callers cannot mutate the constants."""
    return {
        'wavelength_center': HMI_WAVELENGTH_CENTER.copy(),
        'wavelength_grid': HMI_WAVELENGTH_GRID.copy(),
    }
