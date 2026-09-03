"""Public protocol implemented by differentiable instrument operators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class InstrumentOperator(Protocol):
    def synthesis_grid(self, observed_wavelength_angstrom) -> torch.Tensor: ...
    def forward(
        self,
        stokes: torch.Tensor,
        synthesis_wavelength_angstrom,
        observed_wavelength_angstrom,
        **response,
    ) -> torch.Tensor: ...
    def metadata(self) -> dict: ...


__all__ = ["InstrumentOperator"]
