"""Publish and repair native diagnostics independently of instrument products."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import plotting
from .columns import column_batches, strided_columns
from .diagnostic_maps import (
    SCHEMA,
    STATUS,
    DiagnosticAccumulator,
    DiagnosticMaps,
    atomic_path,
    load_diagnostic_maps,
    save_diagnostic_maps,
)
from .reference import open_native_reference, verify_native_reference
from .reporting import timed_stage

logger = logging.getLogger(__name__)


def collect_diagnostic_maps(reference, *, batch_size=256, progress=True):
    """Backfill missing maps in one source pass, without invoking radiative transfer."""
    snapshot = open_native_reference(reference)
    try:
        source = snapshot.synthesis_columns()
        sampling = reference["pixel_mapping"]
        columns, stride = strided_columns(source, sampling["stride"])
        depth = reference.get("column_selection", {}).get("depth_samples", columns.shape[2])
        columns = columns.cropped(depth)
        accumulator = DiagnosticAccumulator(
            columns.shape, columns.ds_m, native_shape=source.shape, stride=stride,
            synthesis_binning=sampling["binning"], native_reference=reference, time_s=snapshot.time_s,
        )
        for ix, iy, batch in column_batches(columns, batch_size, fields=accumulator.fields,
                                           progress=progress, description=f"Iteration {reference.get('iteration', 'unknown')}: diagnostics"):
            accumulator(ix, iy, batch)
        verify_native_reference(reference)
        return accumulator.result()
    finally:
        snapshot.close()


def ensure_diagnostics(root, iteration, reference=None, *, result=None, batch_size=256, progress=True):
    """Repair maps/JPEG independently of the completed instrument-product manifest."""
    directory = Path(root) / "diagnostics"
    maps_path = directory / f"muram_{iteration:06d}.npz"
    overview_path = directory / f"muram_{iteration:06d}.jpg"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": SCHEMA, "snapshots": {}}
    record = {"maps": maps_path.name, "overview": overview_path.name, "state": "pending"}

    def publish():
        manifest["snapshots"][str(iteration)] = record
        with atomic_path(manifest_path) as temporary:
            temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    try:
        if result is None and not maps_path.exists():
            if not reference:
                raise ValueError(f"No native reference for diagnostic iteration {iteration}.")
            logger.info("Iteration %s: diagnostic maps missing; loading native columns without spectral synthesis", iteration)
            result = collect_diagnostic_maps(reference, batch_size=batch_size, progress=progress)
        if result is not None:
            if reference is not None and result.metadata["native_reference"] != reference:
                raise ValueError("Diagnostic source reference differs from the synthesized source.")
            if result.metadata["native_reference"]:
                verify_native_reference(result.metadata["native_reference"])
            metadata = {**result.metadata, "iteration": int(iteration)}
            # Invalidate the old preview first: interruption can never pair new
            # maps with an old image that resume would mistake for complete.
            overview_path.unlink(missing_ok=True)
            with timed_stage(logger, "Iteration %s: saving diagnostic maps to %s", iteration, maps_path):
                save_diagnostic_maps(maps_path, DiagnosticMaps(result.maps, result.status, metadata))
            counts = np.bincount(result.status.ravel(), minlength=len(STATUS))
            logger.info("Iteration %s diagnostics: %s", iteration,
                        ", ".join(f"{label}={counts[code]}" for code, label in STATUS.items()))
        with np.load(maps_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema") != SCHEMA or metadata.get("iteration") != iteration:
            raise ValueError(f"Diagnostic identity mismatch: {maps_path}")
        if reference is not None and metadata["native_reference"] != reference:
            raise ValueError(f"Diagnostic source reference mismatch: {maps_path}")
        record["state"] = "maps_saved"
        publish()
        if not overview_path.is_file():
            with timed_stage(logger, "Iteration %s: rendering overview from saved maps", iteration):
                plotting.write_parameter_overview(overview_path, maps_path)
        else:
            logger.debug("Iteration %s: reusing overview %s", iteration, overview_path)
        record["state"] = "complete"
        publish()
    except BaseException as error:
        logger.error("Iteration %s diagnostics failed: %s", iteration, error)
        record["error"] = f"{type(error).__name__}: {error}"
        publish()
        raise
    return maps_path, overview_path
