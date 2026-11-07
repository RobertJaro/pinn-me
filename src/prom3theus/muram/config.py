"""Config schema extension: load pre-synthesized MURaM rasters as an
``lte_stokes`` observation stream, without any real HMI FITS/JSOC data.

Importing this module registers the ``muram_stored`` observation type with
the strict config decoder (see ``prom3theus.config.registry.register_stream_schema``).
A caller must import this module before parsing a yaml config that uses
``observation: {type: muram_stored, ...}`` — see
:mod:`prom3theus.muram.cli`, the small wrapper that does this before
delegating to the normal ``prom3theus`` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from prom3theus.config.joint_schema import LTEStokesDataTermConfig
from prom3theus.config.registry import register_stream_schema
from prom3theus.config.schema import ConfigNode, DataLoaderConfig


@dataclass(frozen=True, slots=True)
class MuramStoredObservationConfig(ConfigNode):
    """Load an :class:`~prom3theus.observations.store.ObservationStore`
    directory of pre-synthesized MURaM HMI rasters (built by
    ``prom3theus.muram.synthesize`` + ``ObservationStore.save_sequence``)
    directly as an ``lte_stokes`` observation stream."""

    type: Literal["muram_stored"]
    directory: Path
    loader: DataLoaderConfig
    validation_raster_index: int = 0
    binning: int = 1

    def __post_init__(self) -> None:
        if type(self.binning) is not int or self.binning < 1:
            raise ValueError("binning must be a positive integer")
        if self.validation_raster_index < 0:
            raise ValueError("validation_raster_index must be non-negative")


register_stream_schema(
    "muram_stored", MuramStoredObservationConfig, "lte_stokes", LTEStokesDataTermConfig
)

__all__ = ["MuramStoredObservationConfig"]
