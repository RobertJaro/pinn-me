"""Differentiable instrument operators and their public factory."""

from __future__ import annotations

from .base import InstrumentOperator
from .registry import (
    InstrumentRegistration,
    build_instrument,
    get_instrument_registration,
    resolve_instrument_config,
)


def __getattr__(name: str):
    if name == "HinodeSpectralPSF":
        from .hinode_sp import HinodeSpectralPSF

        return HinodeSpectralPSF
    if name == "HMIFilterProfiles":
        from .hmi import HMIFilterProfiles

        return HMIFilterProfiles
    raise AttributeError(name)


__all__ = [
    "HMIFilterProfiles",
    "HinodeSpectralPSF",
    "InstrumentOperator",
    "InstrumentRegistration",
    "build_instrument",
    "get_instrument_registration",
    "resolve_instrument_config",
]
