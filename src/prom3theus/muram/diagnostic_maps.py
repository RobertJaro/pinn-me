"""Native tau=1 map accumulation and numeric sidecar storage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .atmosphere_builder import NATIVE_ATMOSPHERE_FIELDS

SCHEMA = "muram_tau1_diagnostics_v1"
STATUS = {0: "valid", 1: "missing_crossing", 2: "ambiguous_crossing",
          3: "invalid_tau", 4: "invalid_surface_values"}
UNITS = {"temperature": "K", "pressure": "Pa", "height_m": "m",
         **{name: "G" for name in ("bx", "by", "bz")},
         **{name: "m s^-1" for name in ("vx", "vy", "vz")}}


@dataclass(frozen=True)
class DiagnosticMaps:
    maps: dict[str, np.ndarray]
    status: np.ndarray
    metadata: dict


class DiagnosticAccumulator:
    """Keep only unbinned 2D maps, in YX order, from each already loaded batch."""

    fields = (*NATIVE_ATMOSPHERE_FIELDS, "tau")

    def __init__(self, shape, spacing_m, *, native_shape, stride, synthesis_binning,
                 native_reference, time_s):
        nx, ny, nz = shape
        if nz < 2:
            raise ValueError("Surface diagnostics require at least two native layers.")
        self.dz = float(spacing_m[2])
        self.maps = {name: np.full((ny, nx), np.nan, dtype=np.float32) for name in UNITS}
        self.status = np.full((ny, nx), 1, dtype=np.uint8)
        self.metadata = {
            "schema": SCHEMA, "native_reference": native_reference, "time_s": float(time_s),
            "surface": {
                "quantity": "native tau", "value": 1.0,
                "interpolation": "linear in log(tau) between adjacent native cells; linear in tau if a bracket endpoint is zero",
                "selection": "exactly one strict crossing or exact native sample; multiple samples at tau=1 are ambiguous",
                "height_origin": "bottom native cell center; not the photosphere",
            },
            "units": UNITS, "status_codes": {str(k): v for k, v in STATUS.items()},
            "vector_frame": "local synthesis axes: x west, y north, z upward; co-rotating velocity; before instrument/observer transforms",
            "sampling": {
                "map_order": "yx", "shape_yx": [ny, nx], "native_shape_xyz": list(native_shape),
                "native_spacing_m": list(spacing_m), "stride": stride, "map_binning": 1,
                "sampled_depth_count": nz, "sampled_height_range_m": [0., (nz - 1) * self.dz],
                "synthesis_binning": synthesis_binning,
                "native_x_index": "map_x * stride", "native_y_index": "map_y * stride",
            },
        }

    def __call__(self, ix, iy, columns):
        tau = columns["tau"][:, 0, :]
        exact = tau == 1
        left, right = tau[:, :-1], tau[:, 1:]
        crossing = ((left > 1) & (right < 1)) | ((left < 1) & (right > 1))
        count = exact.sum(axis=1) + crossing.sum(axis=1)
        status = np.where(count == 0, 1, np.where(count == 1, 0, 2)).astype(np.uint8)
        invalid = ~np.isfinite(tau).all(axis=1) | (tau < 0).any(axis=1)
        status[invalid] = 3
        rows = np.flatnonzero(status == 0)
        if rows.size:
            is_exact = exact[rows].any(axis=1)
            lo = np.where(is_exact, exact[rows].argmax(axis=1), crossing[rows].argmax(axis=1))
            hi = np.where(is_exact, lo, lo + 1)
            a, b = tau[rows, lo].astype(np.float64), tau[rows, hi].astype(np.float64)
            weight = np.zeros(rows.size, dtype=np.float64)
            log = ~is_exact & (a > 0) & (b > 0)
            weight[log] = -np.log(a[log]) / (np.log(b[log]) - np.log(a[log]))
            linear = ~is_exact & ~log
            weight[linear] = (1 - a[linear]) / (b[linear] - a[linear])
            values = {"height_m": (lo + weight) * self.dz}
            for name in NATIVE_ATMOSPHERE_FIELDS:
                field = columns[name][:, 0, :]
                values[name] = (1 - weight) * field[rows, lo] + weight * field[rows, hi]
            good = np.logical_and.reduce([np.isfinite(v) for v in values.values()])
            good &= (values["temperature"] > 0) & (values["pressure"] > 0)
            status[rows[~good]] = 4
            for name, value in values.items():
                self.maps[name][iy[rows[good]], ix[rows[good]]] = value[good]
        self.status[iy, ix] = status

    def result(self):
        return DiagnosticMaps(self.maps, self.status, self.metadata)


@contextmanager
def atomic_path(path):
    """Publish one complete diagnostic without replacing a good file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_diagnostic_maps(path, result):
    with atomic_path(path) as temporary, temporary.open("wb") as handle:
        np.savez(handle, **result.maps, status=result.status, valid_mask=result.status == 0,
                 metadata_json=np.asarray(json.dumps(result.metadata, sort_keys=True, allow_nan=False)))


def load_diagnostic_maps(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported diagnostic schema in {path}")
        maps = {name: archive[name] for name in UNITS}
        status = archive["status"]
        shape = tuple(metadata["sampling"]["shape_yx"])
        if (status.shape != shape or status.dtype != np.uint8
                or not np.isin(status, list(STATUS)).all()
                or not np.array_equal(archive["valid_mask"], status == 0)
                or any(v.shape != shape or v.dtype != np.float32 for v in maps.values())):
            raise ValueError(f"Invalid diagnostic maps in {path}")
    return DiagnosticMaps(maps, status, metadata)
