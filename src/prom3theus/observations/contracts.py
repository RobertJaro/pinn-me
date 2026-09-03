"""Typed contracts shared by every spectropolarimetric observation.

The observation layer owns array layout, units, and velocity-gauge semantics.
Instrument adapters may add named auxiliary arrays, but they do not introduce
instrument-specific raster subclasses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Any, TypedDict

import torch


COORDINATES = "coordinates"
RAY_DIRECTION = "ray_direction"
SURFACE_POSITION_M = "surface_position_m"
STOKES_BASIS = "stokes_basis"
STOKES = "stokes"
VALID_MASK = "valid_mask"
PIXEL_INDEX = "pixel_index"
INSTRUMENT_RESPONSE = "instrument_response"
_RESERVED_SAMPLE_FIELDS = frozenset(
    {
        COORDINATES,
        RAY_DIRECTION,
        SURFACE_POSITION_M,
        STOKES_BASIS,
        STOKES,
        VALID_MASK,
        PIXEL_INDEX,
        INSTRUMENT_RESPONSE,
    }
)


class VelocitySynthesisMode(StrEnum):
    """Explicit reference frame used for the learned velocity."""

    CARRINGTON_REGISTERED_RELATIVE = "carrington_registered_relative"
    CARRINGTON_OBSERVER_RELATIVE = "carrington_observer_relative"


CARRINGTON_REGISTERED_RELATIVE_VELOCITY = (
    VelocitySynthesisMode.CARRINGTON_REGISTERED_RELATIVE.value
)
CARRINGTON_OBSERVER_RELATIVE_VELOCITY = (
    VelocitySynthesisMode.CARRINGTON_OBSERVER_RELATIVE.value
)
VELOCITY_SYNTHESIS_MODES = frozenset(mode.value for mode in VelocitySynthesisMode)


def resolve_velocity_synthesis_mode(
    value: str | VelocitySynthesisMode,
) -> VelocitySynthesisMode:
    try:
        return VelocitySynthesisMode(str(value).strip().lower())
    except ValueError as error:
        raise ValueError(
            f"Unknown velocity synthesis mode {value!r}; expected one of "
            f"{sorted(VELOCITY_SYNTHESIS_MODES)}."
        ) from error


def velocity_synthesis_contract(value: str | VelocitySynthesisMode) -> dict[str, Any]:
    mode = resolve_velocity_synthesis_mode(value)
    if mode is VelocitySynthesisMode.CARRINGTON_REGISTERED_RELATIVE:
        return {
            "mode": mode.value,
            "learned_velocity": "co-rotating Carrington Cartesian velocity",
            "carrington_rotation": "add rigid sidereal Omega_Carrington cross r",
            "observer_velocity": "not applied; Level-1 registration already removed it",
            "spectral_registration": "subtract the recovered solar LOS velocity removed during line registration",
        }
    return {
        "mode": mode.value,
        "learned_velocity": "co-rotating Carrington Cartesian residual",
        "carrington_rotation": "add rigid sidereal Omega_Carrington cross r",
        "observer_velocity": "subtract the per-pixel observer toward-LOS velocity before the radiative-transfer redshift sign",
    }


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Scientific boundary between an observation and an LTE synthesizer."""

    observation_id: str
    observation_type: str
    instrument_type: str
    wavelength_angstrom: torch.Tensor
    continuum_indices: tuple[int, ...]
    radiance_scale_w_m3_sr: float
    velocity_synthesis_mode: VelocitySynthesisMode | str
    required_line_ids: tuple[str, ...] = ()
    line_support_angstrom: tuple[float, float] | None = None
    instrument_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifiers = (self.observation_id, self.observation_type, self.instrument_type)
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers
        ):
            raise ValueError(
                "Observation and instrument identifiers must be non-empty."
            )
        wavelength = torch.as_tensor(self.wavelength_angstrom, dtype=torch.float32)
        if wavelength.ndim != 1 or wavelength.numel() < 2:
            raise ValueError("Observation wavelengths must be a one-dimensional grid.")
        if not torch.isfinite(wavelength).all() or not torch.all(
            wavelength[1:] > wavelength[:-1]
        ):
            raise ValueError(
                "Observation wavelengths must be finite and strictly increasing."
            )
        indices = tuple(int(index) for index in self.continuum_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Continuum indices must be non-empty and unique.")
        if min(indices) < 0 or max(indices) >= wavelength.numel():
            raise ValueError("Continuum indices lie outside the wavelength grid.")
        scale = float(self.radiance_scale_w_m3_sr)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Radiance scale must be finite and positive.")
        line_ids = tuple(self.required_line_ids)
        if any(
            not isinstance(line_id, str) or not line_id.strip() for line_id in line_ids
        ):
            raise ValueError("Required line identifiers must be non-empty strings.")
        if len(set(line_ids)) != len(line_ids):
            raise ValueError("Required line identifiers must be unique.")
        support = self.line_support_angstrom
        if support is not None:
            support = tuple(map(float, support))
            if (
                len(support) != 2
                or not all(math.isfinite(bound) for bound in support)
                or not support[0] < support[1]
            ):
                raise ValueError("Line support must contain finite, increasing bounds.")
        object.__setattr__(self, "wavelength_angstrom", wavelength.detach().clone())
        object.__setattr__(self, "continuum_indices", indices)
        object.__setattr__(self, "radiance_scale_w_m3_sr", scale)
        object.__setattr__(
            self,
            "velocity_synthesis_mode",
            resolve_velocity_synthesis_mode(self.velocity_synthesis_mode),
        )
        object.__setattr__(self, "required_line_ids", line_ids)
        object.__setattr__(self, "line_support_angstrom", support)
        object.__setattr__(
            self, "instrument_options", MappingProxyType(dict(self.instrument_options))
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "instrument_type": self.instrument_type,
            "wavelength_angstrom": self.wavelength_angstrom.tolist(),
            "continuum_indices": list(self.continuum_indices),
            "radiance_scale_w_m3_sr": self.radiance_scale_w_m3_sr,
            "velocity_synthesis_mode": self.velocity_synthesis_mode.value,
            "velocity_synthesis": velocity_synthesis_contract(
                self.velocity_synthesis_mode
            ),
            "required_line_ids": list(self.required_line_ids),
            "line_support_angstrom": None
            if self.line_support_angstrom is None
            else list(self.line_support_angstrom),
            "instrument_options": dict(self.instrument_options),
        }


@dataclass(frozen=True, slots=True)
class ObservationRaster:
    """Canonical calibrated Stokes raster.

    Spatial dimensions are always the first two axes.  Stokes data have shape
    ``[height, width, 4, wavelength]`` and all geometry is expressed in one
    right-handed Cartesian scene frame.  Adapter-only arrays live in
    :attr:`auxiliary` and must begin with the same spatial shape.
    """

    stokes: torch.Tensor
    wavelength_angstrom: torch.Tensor
    coordinates: torch.Tensor
    ray_direction: torch.Tensor
    surface_position_m: torch.Tensor
    stokes_basis: torch.Tensor
    valid_mask: torch.Tensor
    metadata: Mapping[str, Any]
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tensors = {
            "stokes": torch.as_tensor(self.stokes),
            "wavelength_angstrom": torch.as_tensor(self.wavelength_angstrom),
            "coordinates": torch.as_tensor(self.coordinates),
            "ray_direction": torch.as_tensor(self.ray_direction),
            "surface_position_m": torch.as_tensor(self.surface_position_m),
            "stokes_basis": torch.as_tensor(self.stokes_basis),
            "valid_mask": torch.as_tensor(self.valid_mask),
        }
        stokes = tensors["stokes"]
        if stokes.ndim != 4 or stokes.shape[-2] != 4:
            raise ValueError("stokes must have shape [height, width, 4, wavelength].")
        spatial = tuple(stokes.shape[:2])
        wavelength = tensors["wavelength_angstrom"]
        if wavelength.shape != (stokes.shape[-1],) or wavelength.numel() < 2:
            raise ValueError(
                "wavelength_angstrom does not match the Stokes wavelength dimension."
            )
        if not torch.isfinite(wavelength).all() or not torch.all(
            wavelength[1:] > wavelength[:-1]
        ):
            raise ValueError(
                "wavelength_angstrom must be finite and strictly increasing."
            )
        expected = {
            "coordinates": (*spatial, 3),
            "ray_direction": (*spatial, 3),
            "surface_position_m": (*spatial, 3),
            "stokes_basis": (*spatial, 3, 3),
            "valid_mask": spatial,
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != tuple(shape):
                raise ValueError(
                    f"{name} must have shape {shape}; got {tuple(tensors[name].shape)}."
                )
        if tensors["valid_mask"].dtype is not torch.bool:
            raise ValueError("valid_mask must be boolean.")
        if not torch.any(tensors["valid_mask"]):
            raise ValueError("An observation raster requires at least one valid pixel.")
        for name in (
            "stokes",
            "wavelength_angstrom",
            "coordinates",
            "ray_direction",
            "surface_position_m",
            "stokes_basis",
        ):
            if not tensors[name].is_floating_point():
                raise ValueError(f"{name} must use a real floating-point dtype.")
        for name in (
            "coordinates",
            "ray_direction",
            "surface_position_m",
            "stokes_basis",
        ):
            if not torch.isfinite(tensors[name]).all():
                raise ValueError(f"{name} must contain only finite values.")
        valid = tensors["valid_mask"]
        if not torch.isfinite(stokes[valid]).all():
            raise ValueError("Stokes values must be finite at every valid pixel.")
        unit = torch.linalg.vector_norm(tensors["ray_direction"], dim=-1)
        if not torch.allclose(unit, torch.ones_like(unit), rtol=0.0, atol=2e-5):
            raise ValueError("ray_direction must contain unit vectors.")
        basis = tensors["stokes_basis"]
        identity = torch.eye(3, dtype=basis.dtype, device=basis.device).expand(
            *spatial, 3, 3
        )
        if not torch.allclose(
            basis @ basis.transpose(-1, -2), identity, rtol=0.0, atol=2e-5
        ):
            raise ValueError("stokes_basis must be orthonormal.")
        if torch.any(torch.linalg.det(basis) <= 0):
            raise ValueError("stokes_basis must be right-handed.")
        if not torch.allclose(
            basis[..., 2, :],
            -tensors["ray_direction"],
            rtol=0.0,
            atol=2e-5,
        ):
            raise ValueError(
                "The third Stokes-basis axis must point from the surface to the observer."
            )
        if any(not isinstance(name, str) for name in self.auxiliary):
            raise ValueError("Auxiliary array names must be strings.")
        auxiliary = {
            name: torch.as_tensor(value) for name, value in self.auxiliary.items()
        }
        for name, value in auxiliary.items():
            if not name or name.startswith("auxiliary:"):
                raise ValueError(
                    "Auxiliary array names must be non-empty and cannot use the "
                    "reserved 'auxiliary:' prefix."
                )
            if name in _RESERVED_SAMPLE_FIELDS:
                raise ValueError(
                    f"Auxiliary array name {name!r} collides with the canonical sample contract."
                )
            if name.startswith("instrument_response:") and not name.removeprefix(
                "instrument_response:"
            ):
                raise ValueError(
                    "Instrument-response auxiliary names require a field name."
                )
            if tuple(value.shape[:2]) != spatial:
                raise ValueError(
                    f"Auxiliary array {name!r} must begin with spatial shape {spatial}."
                )
            if value.is_complex():
                raise ValueError(f"Auxiliary array {name!r} must be real-valued.")
            if value.is_floating_point() and not torch.isfinite(value[valid]).all():
                raise ValueError(
                    f"Auxiliary array {name!r} must be finite at every valid pixel."
                )
        for name, value in tensors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "auxiliary", MappingProxyType(auxiliary))
        selected_mu = self.mu[..., 0][tensors["valid_mask"]]
        if (
            not torch.isfinite(selected_mu).all()
            or torch.any(selected_mu <= 0)
            or torch.any(selected_mu > 1.0 + 2e-5)
        ):
            raise ValueError("Every valid pixel must have 0 < mu <= 1.")

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.stokes.shape[0]), int(self.stokes.shape[1])

    @property
    def mu(self) -> torch.Tensor:
        surface = self.surface_position_m / torch.linalg.vector_norm(
            self.surface_position_m, dim=-1, keepdim=True
        )
        return (surface * -self.ray_direction).sum(dim=-1, keepdim=True)


class ObservationSample(TypedDict, total=False):
    coordinates: torch.Tensor
    ray_direction: torch.Tensor
    surface_position_m: torch.Tensor
    stokes_basis: torch.Tensor
    stokes: torch.Tensor
    pixel_index: torch.Tensor
    instrument_response: Mapping[str, torch.Tensor]


class ObservationBatch(ObservationSample, total=False):
    """Batched canonical sample; adapter auxiliary keys remain extensible."""


__all__ = [
    "CARRINGTON_OBSERVER_RELATIVE_VELOCITY",
    "CARRINGTON_REGISTERED_RELATIVE_VELOCITY",
    "COORDINATES",
    "INSTRUMENT_RESPONSE",
    "ObservationBatch",
    "ObservationRaster",
    "ObservationSample",
    "ObservationSpec",
    "PIXEL_INDEX",
    "RAY_DIRECTION",
    "STOKES",
    "STOKES_BASIS",
    "SURFACE_POSITION_M",
    "VALID_MASK",
    "VELOCITY_SYNTHESIS_MODES",
    "VelocitySynthesisMode",
    "resolve_velocity_synthesis_mode",
    "velocity_synthesis_contract",
]
