"""Numerical foundations for the LTE inversion framework."""

from .constants import (
    ATOMIC_MASS_UNIT,
    ELECTRON_VOLT,
    H_PLANCK,
    K_BOLTZMANN,
    M_ELECTRON,
    SPEED_OF_LIGHT,
)
from .coordinates import (
    cartesian_to_spherical,
    cartesian_to_spherical_matrix,
    project_cartesian_to_spherical,
    project_spherical_to_cartesian,
    project_spherical_to_observer,
    spherical_to_cartesian,
    spherical_to_cartesian_matrix,
    spherical_to_observer_matrix,
)
from .nn import FourierEncoding, MLPModel
from .integrity import sha256_file, sha256_file_set
from .solar_velocity import (
    CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
    CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS,
    carrington_rotation_velocity_cartesian,
)
from .special import Faddeeva, polyval

__all__ = [
    "ATOMIC_MASS_UNIT",
    "CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S",
    "CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS",
    "ELECTRON_VOLT",
    "Faddeeva",
    "FourierEncoding",
    "H_PLANCK",
    "K_BOLTZMANN",
    "MLPModel",
    "M_ELECTRON",
    "SPEED_OF_LIGHT",
    "carrington_rotation_velocity_cartesian",
    "cartesian_to_spherical",
    "cartesian_to_spherical_matrix",
    "polyval",
    "project_cartesian_to_spherical",
    "project_spherical_to_cartesian",
    "project_spherical_to_observer",
    "spherical_to_cartesian",
    "spherical_to_cartesian_matrix",
    "spherical_to_observer_matrix",
    "sha256_file",
    "sha256_file_set",
]
