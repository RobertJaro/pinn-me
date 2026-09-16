"""Preserve the original diagnostic baseline across checkpoint continuations."""

import json
from pathlib import Path

import numpy as np


_ARRAY_KEYS = (
    "initial", "target", "pixel_index", "representative_initial",
    "representative_target", "representative_pixel_index",
)


def preserve_initial_reference(path, arrays, initialization, *, resume):
    """Store a new baseline or restore it after exact observation identity checks."""
    path = Path(path)
    if set(arrays) != set(_ARRAY_KEYS):
        raise ValueError(f"Initial reference requires exactly these arrays: {_ARRAY_KEYS}")
    arrays = {name: np.asarray(arrays[name]) for name in _ARRAY_KEYS}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Initial reference arrays must not contain pickled objects")
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot resume: initial diagnostic reference is missing: {path}")
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {*_ARRAY_KEYS, "initialization_json"}:
                raise ValueError("Stored initial diagnostic reference has invalid array keys")
            previous = {name: stored[name] for name in _ARRAY_KEYS}
            provenance = json.loads(stored["initialization_json"].item())
        for name in ("target", "pixel_index", "representative_target", "representative_pixel_index"):
            if previous[name].dtype != arrays[name].dtype or not np.array_equal(previous[name], arrays[name]):
                raise ValueError(f"Cannot resume: initial diagnostic reference {name} differs from current data")
        return previous, provenance
    provenance = np.asarray(json.dumps(initialization, allow_nan=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, initialization_json=provenance)
    return arrays, initialization


__all__ = ["preserve_initial_reference"]
