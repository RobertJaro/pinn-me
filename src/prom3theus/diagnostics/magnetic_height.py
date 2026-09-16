"""Inspect a saved atmosphere's radial magnetic and thermal height dependence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _height_metrics(magnetic_field_gauss, temperature_k, heights_megameter):
    magnetic = np.asarray(magnetic_field_gauss, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    heights = np.asarray(heights_megameter, dtype=np.float64)
    if (magnetic.ndim != 3 or magnetic.shape[-1] != 3 or not magnetic.shape[0]
            or temperature.shape != magnetic.shape[:2]
            or heights.shape != (magnetic.shape[1],) or len(heights) < 2):
        raise ValueError("Expected [pixel,height,3] magnetic and [pixel,height] temperature arrays")
    if not all(np.isfinite(value).all() for value in (magnetic, temperature, heights)):
        raise ValueError("Height consistency cannot be established with non-finite values")
    differences = magnetic - magnetic[:, :1]
    rms = np.sqrt(np.mean(np.sum(magnetic**2, axis=-1), axis=0))
    temperature_range = np.ptp(temperature, axis=1)
    checks = {
        "nonzero_magnetic_field": bool(rms[0] > 1.0),
        "magnetic_height_independence": bool(np.all(
            np.abs(differences) <= 0.001 + 1e-6 * np.abs(magnetic[:, :1])
        )),
        "temperature_height_variation_present": bool(np.median(temperature_range) > 1.0),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "comparison_height_megameter": float(heights[0]),
        "heights_megameter": heights.tolist(),
        "sample_count": len(magnetic),
        "magnetic_rms_gauss_by_height": rms.tolist(),
        "magnetic_vector_rms_difference_gauss_by_height": np.sqrt(
            np.mean(np.sum(differences**2, axis=-1), axis=0)
        ).tolist(),
        "maximum_magnetic_component_difference_gauss": float(np.max(np.abs(differences))),
        "magnetic_constancy_tolerance": {"absolute_gauss": 0.001, "relative": 1e-6},
        "temperature_range_k": {
            "minimum": float(np.min(temperature_range)),
            "median": float(np.median(temperature_range)),
            "maximum": float(np.max(temperature_range)),
            "fraction_exceeding_one_kelvin": float(np.mean(temperature_range > 1.0)),
        },
    }


def check_magnetic_height(
    p3s_path, output_directory, *, sample_count=256,
    heights_megameter=(0.0, 0.1, 0.15, 0.3, 1.0),
):
    """Save an optional radial-column consistency report from real P3S samples.

    A passing result verifies the height-independent magnetic parameterization
    while temperature remains stratified. It does not validate the inversion or
    corona. An unrestricted atmosphere can fail this diagnostic by design.
    """
    import torch

    from prom3theus.application.batches import deterministic_validation_batch
    from prom3theus.artifacts.loader import P3SLoader

    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    heights = np.asarray(heights_megameter, dtype=np.float64)
    if (heights.ndim != 1 or len(heights) < 2 or not np.isfinite(heights).all()
            or not np.all(np.diff(heights) > 0)):
        raise ValueError("Heights must be at least two finite, strictly increasing values")
    source = Path(p3s_path).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    loader = P3SLoader(source, device="cpu")
    model = loader.module.atmosphere_model.eval()
    outer, inner = model.shell_height_bounds_Mm
    if heights[0] < inner or heights[-1] > outer:
        raise ValueError("All diagnostic heights must lie within the atmosphere shell")
    batch = deterministic_validation_batch(loader._observations(), max_samples=sample_count)
    coordinates = batch["coordinates"]
    magnetic, temperature = [], []
    with torch.no_grad():
        for height in heights:
            fields = model.evaluate_chart_height_points(
                coordinates, torch.full_like(coordinates[:, 0], float(height * 1e6)),
            )
            magnetic.append(fields["magnetic_field"].cpu().numpy())
            temperature.append(fields["temperature"].cpu().numpy())
    magnetic, temperature = np.stack(magnetic, axis=1), np.stack(temperature, axis=1)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source P3S changed during the diagnostic; use a stable snapshot")
    report = _height_metrics(magnetic, temperature, heights)
    report.update({
        "format": "prom3theus.magnetic_height_consistency.v1",
        "source": {"path": str(source), "sha256": digest, "global_step": loader.global_step,
                   "stream_id": loader.stream_id,
                   "observation_signature": loader.observation.source_signature},
        "magnetic_reference_height_megameter": model.magnetic_reference_height_megameter,
        "sampling": "fixed angular chart coordinates and time from the validated observation cache",
        "magnetic_basis": "native Cartesian, gauss",
        "scope": "parameterization consistency; not an inversion or coronal accuracy test",
        "interpretation": "Unrestricted magnetic atmospheres may fail this check by design. Oblique rays can cross angular structure even when radial columns are constant.",
    })
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {"report": str(output / "magnetic-height-consistency.json"),
             "arrays": str(output / "magnetic-height-samples.npz")}
    np.savez_compressed(paths["arrays"], heights_megameter=heights,
                        coordinates=coordinates.cpu().numpy(),
                        pixel_index=batch["pixel_index"].cpu().numpy(),
                        magnetic_field_gauss=magnetic, temperature_k=temperature)
    Path(paths["report"]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"report": report, "paths": paths}
