"""Metadata construction and atomic NPZ writing for artifact exports."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from prom3theus.core import (
    CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
    CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS,
)
from prom3theus.observations import velocity_synthesis_contract


def _export_metadata(
    observation,
    velocity_mode,
    resources,
    arrays,
    raster_selection,
    *,
    depth_samples,
    include_stokes,
    full_shell,
    storage_dtype,
    device,
):
    evaluation = {
        "depth_coordinate": "geometric_height_m",
        "stokes_sampling_coordinate": "stokes_reference_log_tau500",
        "depth_samples": depth_samples,
        "include_stokes": include_stokes,
        "storage_dtype": storage_dtype,
        "device": str(device),
        "raster_role": "validation",
        "raster_name": raster_selection.name,
        "raster_index": raster_selection.index,
        "raster_count": raster_selection.raster_count,
        "carrington_sidereal_rotation_period_days": (
            CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS
        ),
        "carrington_angular_velocity_rad_per_s": (
            CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
        ),
    }
    if full_shell is not None:
        evaluation["full_shell"] = dict(full_shell)

    return {
        "schema_version": 1,
        "observation": observation,
        "resources": dict(resources),
        "velocity_synthesis": velocity_synthesis_contract(velocity_mode),
        "evaluation": evaluation,
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        },
    }


def build_save_state_export_metadata(
    state,
    arrays,
    raster_selection,
    *,
    save_state_sha256,
    depth_samples,
    include_stokes,
    full_shell,
    storage_dtype,
    device,
):
    """Describe P3S provenance without inventing an archival artifact manifest."""
    from .checkpoint import P3S_FORMAT, P3S_VERSION

    reference = state.observation
    metadata = _export_metadata(
        {
            "spec": reference.spec.metadata(),
            "source_signature": reference.source_signature,
            "raster_names": list(reference.raster_names),
            "validation_raster_index": reference.validation_raster_index,
            "times": list(reference.times),
            "bounds": dict(reference.bounds),
        },
        reference.spec.velocity_synthesis_mode,
        state.resources,
        arrays,
        raster_selection,
        depth_samples=depth_samples,
        include_stokes=include_stokes,
        full_shell=full_shell,
        storage_dtype=storage_dtype,
        device=device,
    )
    metadata["schema_version"] = 2
    metadata["save_state"] = {
        "format": P3S_FORMAT,
        "version": P3S_VERSION,
        "sha256": save_state_sha256,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "resolved_config": state.config.to_dict(),
    }
    return metadata


def write_export_archive(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically write a compressed NPZ export with canonical JSON metadata."""

    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp.npz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "build_save_state_export_metadata",
    "write_export_archive",
]
