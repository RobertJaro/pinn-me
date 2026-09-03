"""Typed values exchanged across the artifact export pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prom3theus.observations import ObservationRaster, ObservationSpec

from .model import ArtifactManifest

if TYPE_CHECKING:
    from prom3theus.training.lightning import LTEInversionModule


@dataclass(frozen=True, slots=True)
class ObservationContract:
    """Validated manifest contract for the embedded observation store."""

    path: Path
    source_signature: str
    spec: ObservationSpec
    manifest_value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StoredRasterSelection:
    """The exact validation raster selected from the embedded store."""

    raster: ObservationRaster
    name: str
    index: int
    raster_count: int


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    """Fully checked inputs required by numerical artifact evaluation."""

    root: Path
    manifest: ArtifactManifest
    observation: ObservationContract
    resources: dict[str, Any]
    raster_selection: StoredRasterSelection
    module: LTEInversionModule


__all__ = ["ObservationContract", "StoredRasterSelection", "ValidatedArtifact"]
