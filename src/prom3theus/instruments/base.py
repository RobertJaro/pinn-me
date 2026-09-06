"""Public protocol implemented by differentiable instrument operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class MagneticAzimuthConvention:
    """Instrument-owned mapping from physical to synthesis-frame azimuth."""

    name: str
    offset_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MagneticAzimuthConvention.name must be non-empty.")
        if not math.isfinite(float(self.offset_deg)):
            raise ValueError("MagneticAzimuthConvention.offset_deg must be finite.")
        object.__setattr__(self, "offset_deg", float(self.offset_deg))

    def to_synthesis_frame(self, physical_observer_field: torch.Tensor) -> torch.Tensor:
        """Add the convention offset to observer-frame magnetic azimuth."""

        field = torch.as_tensor(physical_observer_field)
        if field.shape[-1:] != (3,) or not field.is_floating_point():
            raise ValueError(
                "physical_observer_field must be a floating tensor ending in three."
            )
        if not torch.isfinite(field).all():
            raise ValueError("physical_observer_field must contain only finite values.")
        angle = math.radians(self.offset_deg)
        cosine = field.new_tensor(math.cos(angle))
        sine = field.new_tensor(math.sin(angle))
        x = cosine * field[..., 0] - sine * field[..., 1]
        y = sine * field[..., 0] + cosine * field[..., 1]
        return torch.stack((x, y, field[..., 2]), dim=-1)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "magnetic_azimuth_offset_deg": self.offset_deg,
            "application": (
                "added to physical observer-frame magnetic azimuth immediately "
                "before polarized spectral synthesis"
            ),
        }


@runtime_checkable
class InstrumentOperator(Protocol):
    polarization_convention: MagneticAzimuthConvention

    def synthesis_grid(self, observed_wavelength_angstrom) -> torch.Tensor: ...
    def forward(
        self,
        stokes: torch.Tensor,
        synthesis_wavelength_angstrom,
        observed_wavelength_angstrom,
        **response,
    ) -> torch.Tensor: ...
    def metadata(self) -> dict: ...


__all__ = ["InstrumentOperator", "MagneticAzimuthConvention"]
