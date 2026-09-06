"""Metadata construction and atomic NPZ writing for artifact exports."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from prom3theus.core import (
    CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
    CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS,
)
from prom3theus.observations import velocity_synthesis_contract

from .contracts import ObservationContract, StoredRasterSelection
from .model import ArtifactManifest


def build_export_metadata(
    manifest: ArtifactManifest,
    contract: ObservationContract,
    resources: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    raster_selection: StoredRasterSelection,
    *,
    depth_samples: int,
    include_stokes: bool,
    full_shell: Mapping[str, Any] | None = None,
    storage_dtype: str,
    device: torch.device,
) -> dict[str, Any]:
    """Describe the validated inputs and every exported array."""

    evaluation = {
        "depth_coordinate": "log_tau500",
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
        "artifact": {
            "schema_version": manifest.schema_version,
            "solver": manifest.solver,
            "package_version": manifest.package_version,
            "created_utc": manifest.created_utc,
            "weights_sha256": manifest.weights_sha256,
        },
        "observation": contract.manifest_value,
        "resources": dict(resources),
        "velocity_synthesis": velocity_synthesis_contract(
            contract.spec.velocity_synthesis_mode
        ),
        "evaluation": evaluation,
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        },
    }


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


__all__ = ["build_export_metadata", "write_export_archive"]
