"""Typed contracts for joint application services."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from prom3theus.config.joint_schema import JointInversionConfig
from prom3theus.inversion.data_terms import ObservationDataTerm, SharedObjectiveTerm
from prom3theus.inversion.joint import JointEvaluation, JointForwardModel
from prom3theus.observations import PreparedObservationStream, SceneContract


@dataclass(frozen=True, slots=True)
class LoadedJointStream:
    """Prepared observation plus the runtime-only objects needed for assembly."""

    prepared: PreparedObservationStream
    specification: Any
    rasters: tuple[Any, ...]
    store_metadata: Mapping[str, Any] = field(default_factory=dict)
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)
    scene_contract: SceneContract | None = None
    sampling_support: Callable | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedObservationStream):
            raise TypeError("prepared must be a PreparedObservationStream.")
        if not self.rasters and (
            self.scene_contract is None or self.sampling_support is None
        ):
            raise ValueError(
                "A non-raster stream must provide a scene contract and sampling support."
            )
        for name in ("store_metadata", "setup_metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping.")
            object.__setattr__(self, name, MappingProxyType(dict(value)))

    @property
    def data_module(self):
        return self.prepared.data_module


@dataclass(frozen=True, slots=True)
class SharedTermAssembly:
    """Shared objectives and the deterministic spatial sampling description."""

    terms: Mapping[str, SharedObjectiveTerm]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", MappingProxyType(dict(self.terms)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class JointRunnerFactories:
    """Replaceable construction seams for setup and runtime tests.

    The scene builder receives only the configured reference stream. Each later
    stream is prepared and persisted against that established scene in turn.
    """

    resource_validator: Callable[[Iterable[str]], Mapping[str, Mapping]] | None = None
    stream_loader: Callable[..., LoadedJointStream] | None = None
    scene_builder: Callable[..., SceneContract] | None = None
    atmosphere_builder: Callable[..., nn.Module] | None = None
    term_builder: Callable[..., ObservationDataTerm] | None = None
    shared_terms_builder: Callable[..., SharedTermAssembly] | None = None


@dataclass(frozen=True, slots=True)
class JointRuntime:
    """Shared runtime state for joint evaluation and optimization."""

    config: JointInversionConfig
    resources: Mapping[str, Mapping[str, Any]]
    streams: Mapping[str, LoadedJointStream]
    scene: SceneContract
    model: JointForwardModel
    validation_batches: Mapping[str, Mapping[str, Any]]
    device: torch.device
    shared_metadata: Mapping[str, Any]
    state_sha256_at_build: str | None
    data_module: Any = None


@dataclass(frozen=True, slots=True)
class JointRuntimeEvaluation:
    """Tensor diagnostics and JSON-ready audit report from one evaluation."""

    evaluation: JointEvaluation
    report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class JointDryRun:
    """Result of a setup, forward render, gradient smoke test, and report write."""

    runtime: JointRuntime
    evaluation: JointRuntimeEvaluation
    report_path: Path
