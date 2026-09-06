"""Strict, artifact-only export for LTE inversions.

An export is reconstructed exclusively from the versioned artifact manifest,
its tensor-only state dictionary, and its canonical embedded observation
store. The validation, numerical evaluation, and archive-writing concerns live
in focused sibling modules; this module is the stable public facade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .archive import build_export_metadata, write_export_archive
from .errors import ArtifactExportError
from .evaluation import (
    depth_grid,
    evaluate_atmosphere,
    resolve_storage_dtype,
    select_export_device,
)
from .full_shell import (
    evaluate_full_shell_atmosphere,
    full_shell_height_grid,
    full_shell_metadata,
)
from .stokes import evaluate_stokes
from .validation import load_validated_artifact


def _validate_export_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Reject non-finite model results at every scientifically valid pixel."""

    valid_mask = arrays.get("valid_mask")
    if (
        not isinstance(valid_mask, np.ndarray)
        or valid_mask.dtype != np.bool_
        or valid_mask.ndim != 2
        or not valid_mask.any()
    ):
        raise ArtifactExportError("Export arrays must contain a boolean valid_mask.")
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise ArtifactExportError(f"Export array {name!r} is not a NumPy array.")
        if value.dtype.kind not in "biuf":
            raise ArtifactExportError(
                f"Export array {name!r} must use a real numeric or boolean dtype."
            )
        if value.dtype.kind not in "fc":
            continue
        if value.shape[: valid_mask.ndim] == valid_mask.shape:
            scientific_values = value[valid_mask]
        else:
            scientific_values = value
        if not np.isfinite(scientific_values).all():
            raise ArtifactExportError(
                f"Export array {name!r} contains non-finite scientific values."
            )


def export_artifact(
    artifact_directory: str | Path,
    output: str | Path,
    *,
    depth_samples: int | None = 101,
    batch_size: int = 4096,
    include_stokes: bool = False,
    stokes_batch_size: int = 16,
    include_full_shell: bool = False,
    full_shell_samples: int = 101,
    storage_dtype: str = "float32",
    device: str = "auto",
) -> Path:
    """Export one strict LTE artifact as a compressed, self-described NPZ file.

    The observation is never rebuilt from its original FITS inputs. Its exact
    canonical store must be present below ``artifact_directory`` and named by
    ``manifest.observation.store.path``.
    """

    artifact_root = Path(artifact_directory).expanduser().resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_root}")
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output must use the .npz suffix.")
    if batch_size < 1 or stokes_batch_size < 1:
        raise ValueError("batch_size and stokes_batch_size must be positive.")
    if type(full_shell_samples) is not int or full_shell_samples < 2:
        raise ValueError("full_shell_samples must be an integer of at least two.")
    resolve_storage_dtype(storage_dtype)

    artifact = load_validated_artifact(artifact_root)
    module = artifact.module
    compute_dtype = next(module.parameters()).dtype
    export_device = select_export_device(
        device,
        compute_dtype,
        include_stokes=include_stokes,
    )
    module = module.to(export_device).eval()
    evaluation_depth = depth_grid(module, depth_samples)
    raster = artifact.raster_selection.raster
    arrays = evaluate_atmosphere(
        module,
        raster,
        depth_grid=evaluation_depth,
        batch_size=batch_size,
        storage_dtype=storage_dtype,
    )
    full_shell = None
    if include_full_shell:
        shell_height = full_shell_height_grid(module, full_shell_samples)
        arrays.update(
            evaluate_full_shell_atmosphere(
                module,
                raster,
                height_grid_m=shell_height,
                batch_size=batch_size,
                storage_dtype=storage_dtype,
            )
        )
        full_shell = full_shell_metadata(shell_height)
    if include_stokes:
        arrays.update(
            evaluate_stokes(
                module,
                raster,
                artifact.observation.spec,
                batch_size=stokes_batch_size,
                storage_dtype=storage_dtype,
            )
        )
    _validate_export_arrays(arrays)
    metadata = build_export_metadata(
        artifact.manifest,
        artifact.observation,
        artifact.resources,
        arrays,
        artifact.raster_selection,
        depth_samples=int(evaluation_depth.numel()),
        include_stokes=include_stokes,
        full_shell=full_shell,
        storage_dtype=storage_dtype,
        device=export_device,
    )
    return write_export_archive(output_path, arrays, metadata)


__all__ = ["ArtifactExportError", "export_artifact"]
