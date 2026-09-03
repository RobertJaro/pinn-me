"""Optional physical constraints for inversion training."""

from .magnetofluid import (
    BOUNDARY_EQUATIONS,
    EQUATION_NAMES,
    VOLUME_EQUATIONS,
    MagnetofluidConstraints,
    PhysicsResult,
)

__all__ = [
    "BOUNDARY_EQUATIONS",
    "EQUATION_NAMES",
    "VOLUME_EQUATIONS",
    "MagnetofluidConstraints",
    "PhysicsResult",
]
