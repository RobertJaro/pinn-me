"""Modality-neutral observation-term contract.

Data terms own observation-specific forward operators, objectives, and nuisance
parameters.  They never own the shared atmosphere or schedule optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class DataTermBatchResult:
    """One reduced observation likelihood plus diagnostic components."""

    likelihood_loss: torch.Tensor
    component_losses: Mapping[str, torch.Tensor]
    metrics: Mapping[str, torch.Tensor]
    sample_count: int
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.likelihood_loss.ndim != 0:
            raise ValueError("likelihood_loss must be a scalar tensor.")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive.")
        for collection_name, values in (
            ("component_losses", self.component_losses),
            ("metrics", self.metrics),
        ):
            if not isinstance(values, Mapping):
                raise TypeError(f"{collection_name} must be a mapping.")
            if any(not isinstance(name, str) or not name for name in values):
                raise ValueError(f"{collection_name} keys must be non-empty strings.")
            if any(not isinstance(value, torch.Tensor) for value in values.values()):
                raise TypeError(f"{collection_name} values must be tensors.")


class ObservationDataTerm(nn.Module, ABC):
    """Base class for one prepared-observation likelihood."""

    observation_kind: str

    @abstractmethod
    def evaluate_batch(self, batch: Mapping[str, Any]) -> DataTermBatchResult:
        """Synthesize and score one deterministic or scheduled batch."""

    def nuisance_losses(self) -> Mapping[str, torch.Tensor]:
        """Return unweighted nuisance priors, evaluated once per joint step."""

        return {}


@dataclass(frozen=True, slots=True)
class SharedTermResult:
    """One physics or atmosphere objective evaluated once per joint step."""

    loss: torch.Tensor
    component_losses: Mapping[str, torch.Tensor]
    metrics: Mapping[str, torch.Tensor]
    active: bool = True

    def __post_init__(self) -> None:
        if self.loss.ndim != 0:
            raise ValueError("A shared-term loss must be scalar.")
        for values in (self.component_losses, self.metrics):
            if not isinstance(values, Mapping) or any(
                not isinstance(name, str) or not name for name in values
            ):
                raise TypeError("Shared-term outputs must use non-empty string keys.")
            if any(not isinstance(value, torch.Tensor) for value in values.values()):
                raise TypeError("Shared-term outputs must contain tensors.")


class SharedObjectiveTerm(nn.Module, ABC):
    """Base class for an objective on the one shared atmosphere."""

    @abstractmethod
    def evaluate(self, atmosphere_model: nn.Module) -> SharedTermResult:
        """Evaluate one complete, deterministic shared-atmosphere objective."""


__all__ = [
    "DataTermBatchResult",
    "ObservationDataTerm",
    "SharedObjectiveTerm",
    "SharedTermResult",
]
