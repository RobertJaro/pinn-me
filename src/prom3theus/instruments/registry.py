"""Closed factory for the two supported LTE instrument operators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from torch import nn

from .base import MagneticAzimuthConvention


@dataclass(frozen=True, slots=True)
class InstrumentRegistration:
    """One immutable instrument discriminator and its lazy factory."""

    name: str
    factory: Callable[..., nn.Module]


def _build_hinode_sp(**options: Any) -> nn.Module:
    from .hinode_sp.operator import _build_hinode_sp

    return _build_hinode_sp(**options)


def _build_hmi_filter_profiles(**options: Any) -> nn.Module:
    from .hmi.operator import HMIFilterProfiles

    return HMIFilterProfiles(**options)


_INSTRUMENTS: Mapping[str, InstrumentRegistration] = MappingProxyType(
    {
        "hinode_sp": InstrumentRegistration("hinode_sp", _build_hinode_sp),
        "hmi_filter_profiles": InstrumentRegistration(
            "hmi_filter_profiles", _build_hmi_filter_profiles
        ),
    }
)


def get_instrument_registration(name: str) -> InstrumentRegistration:
    """Return one exact supported instrument discriminator."""

    try:
        return _INSTRUMENTS[name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Unknown instrument type {name!r}; expected one of {sorted(_INSTRUMENTS)}."
        ) from error


def resolve_instrument_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the discriminator while preserving its exact constructor data."""

    resolved = dict(config)
    try:
        name = resolved.pop("type")
    except KeyError as error:
        raise ValueError("instrument.type is required.") from error
    registration = get_instrument_registration(name)
    return {"type": registration.name, **resolved}


def build_instrument(config: Mapping[str, Any]) -> nn.Module:
    """Construct a configured differentiable instrument operator."""

    resolved = dict(config)
    try:
        name = resolved.pop("type")
    except KeyError as error:
        raise ValueError("instrument.type is required.") from error
    registration = get_instrument_registration(name)
    operator = registration.factory(**resolved)
    missing = [
        method
        for method in ("synthesis_grid", "forward", "metadata")
        if not callable(getattr(operator, method, None))
    ]
    convention = getattr(operator, "polarization_convention", None)
    if (
        not isinstance(operator, nn.Module)
        or missing
        or not isinstance(convention, MagneticAzimuthConvention)
    ):
        raise TypeError(
            f"Instrument {registration.name!r} returned an incompatible operator; "
            f"missing methods={missing}, valid polarization_convention="
            f"{isinstance(convention, MagneticAzimuthConvention)}."
        )
    return operator


__all__ = [
    "InstrumentRegistration",
    "build_instrument",
    "get_instrument_registration",
    "resolve_instrument_config",
]
