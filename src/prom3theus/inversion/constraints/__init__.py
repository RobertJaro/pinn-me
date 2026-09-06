"""Optional physical constraints for inversion training."""

from .magnetofluid import (
    BOUNDARY_EQUATIONS,
    EQUATION_NAMES,
    SIDE_BOUNDARY_EQUATIONS,
    UPPER_BOUNDARY_EQUATIONS,
    UPPER_VOLUME_EQUATIONS,
    VOLUME_EQUATIONS,
    MagnetofluidConstraints,
    PhysicsResult,
)

__all__ = [
    "BOUNDARY_EQUATIONS",
    "EQUATION_NAMES",
    "SIDE_BOUNDARY_EQUATIONS",
    "UPPER_BOUNDARY_EQUATIONS",
    "UPPER_VOLUME_EQUATIONS",
    "VOLUME_EQUATIONS",
    "MagnetofluidConstraints",
    "PhysicsResult",
]
