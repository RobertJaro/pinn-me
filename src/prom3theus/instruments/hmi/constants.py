"""Physical and acquisition constants for SDO/HMI."""

import numpy as np
from astropy import units as u


HMI_WAVELENGTH_CENTER = 6173.3433 * u.AA
HMI_WAVELENGTH_GRID = (
    np.array([0.1720, 0.1032, 0.0344, -0.0344, -0.1032, -0.1720], dtype=np.float64)
    * u.AA
)
HMI_LINE_ID = "FeI_6173.3352"
HMI_OBSERVER_VELOCITY_KEYS = ("OBS_VR", "OBS_VW", "OBS_VN")
HMI_CCD_SIZE = 4096
HMI_CAMERA = 3
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


__all__ = [
    "HMI_CCD_SIZE",
    "HMI_CAMERA",
    "HMI_LINE_ID",
    "HMI_OBSERVER_VELOCITY_KEYS",
    "HMI_WAVELENGTH_CENTER",
    "HMI_WAVELENGTH_GRID",
    "SPEED_OF_LIGHT_M_PER_S",
]
