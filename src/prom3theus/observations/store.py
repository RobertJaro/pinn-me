"""Versioned, class-free persistence for prepared observation sequences.

Stores contain JSON manifests and independent NumPy arrays only.  Python
classes and pickle payloads never cross this boundary.  Every array is hashed,
and loading can memory-map even a multi-raster time series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from .contracts import ObservationRaster


OBSERVATION_STORE_FORMAT = "prom3theus.observation_store"
OBSERVATION_STORE_VERSION = 2
MANIFEST_FILENAME = "manifest.json"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Observation metadata keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"Observation metadata contains unsupported value {type(value).__name__}."
    )


def _resource_contract(resources: Mapping[str, Any]) -> dict[str, Any]:
    """Remove installation locations while retaining scientific contracts."""

    def strip(value):
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Resource metadata keys must be strings.")
            return {
                key: strip(item)
                for key, item in value.items()
                if key not in {"directory", "installation_directory", "root"}
            }
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return _json_value(value)

    return strip(resources)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def observation_store_signature(
    config: Mapping[str, Any],
    resources: Mapping[str, Any],
    *,
    source_files_sha256: str,
) -> str:
    """Hash preparation inputs without hashing a resource installation path.

    Replacing any raw observation or external calibration file selects a new
    immutable cache.
    """
    _require_sha256(source_files_sha256, "source_files_sha256")
    payload = json.dumps(
        {
            "store_version": OBSERVATION_STORE_VERSION,
            "config": _json_value(config),
            "resources": _resource_contract(resources),
            "source_files_sha256": source_files_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ObservationStore:
    """Read and write canonical single- or multi-raster observation stores."""

    _CORE_FIELDS = (
        "stokes",
        "wavelength_angstrom",
        "coordinates",
        "ray_direction",
        "surface_position_m",
        "stokes_basis",
        "valid_mask",
    )

    @classmethod
    def _write_array(cls, path: Path, tensor: torch.Tensor) -> dict[str, Any]:
        array = tensor.detach().cpu().contiguous().numpy()
        np.save(path, array, allow_pickle=False)
        return {
            "file": path.name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _sha256(path),
        }

    @classmethod
    def save_sequence(
        cls,
        path: str | os.PathLike[str],
        rasters: Sequence[ObservationRaster],
        *,
        raster_names: Sequence[str] | None = None,
        source_signature: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        destination = Path(path).expanduser().resolve()
        rasters = tuple(rasters)
        if not rasters:
            raise ValueError("An observation store requires at least one raster.")
        names = (
            tuple(f"raster_{index:04d}" for index in range(len(rasters)))
            if raster_names is None
            else tuple(map(str, raster_names))
        )
        if (
            len(names) != len(rasters)
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError(
                "raster_names must be unique, non-empty, and match rasters."
            )
        _require_sha256(source_signature, "source_signature")
        if destination.exists():
            raise FileExistsError(
                f"Observation store already exists: {destination}. Use a new signature or rebuild it."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
        )
        try:
            raster_records = []
            for raster_index, (name, raster) in enumerate(
                zip(names, rasters, strict=True)
            ):
                values = {field: getattr(raster, field) for field in cls._CORE_FIELDS}
                values.update(
                    {
                        f"auxiliary:{key}": value
                        for key, value in raster.auxiliary.items()
                    }
                )
                array_records = {}
                for array_index, (key, value) in enumerate(values.items()):
                    filename = f"r{raster_index:04d}_a{array_index:03d}.npy"
                    array_records[key] = cls._write_array(temporary / filename, value)
                raster_records.append(
                    {
                        "name": name,
                        "metadata": _json_value(raster.metadata),
                        "arrays": array_records,
                    }
                )
            manifest = {
                "format": OBSERVATION_STORE_FORMAT,
                "version": OBSERVATION_STORE_VERSION,
                "source_signature": source_signature,
                "metadata": _json_value(metadata or {}),
                "rasters": raster_records,
            }
            (temporary / MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    @classmethod
    def save(
        cls,
        path: str | os.PathLike[str],
        raster: ObservationRaster,
        *,
        source_signature: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return cls.save_sequence(
            path,
            [raster],
            source_signature=source_signature,
            metadata=metadata,
        )

    @classmethod
    def manifest(cls, path: str | os.PathLike[str]) -> dict[str, Any]:
        manifest_path = Path(path).expanduser().resolve() / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Observation-store manifest not found: {manifest_path}."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_fields = {
            "format",
            "version",
            "source_signature",
            "metadata",
            "rasters",
        }
        if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
            raise ValueError(
                f"Invalid observation-store manifest schema in {manifest_path}."
            )
        if manifest.get("format") != OBSERVATION_STORE_FORMAT:
            raise ValueError(
                f"Unsupported observation-store format in {manifest_path}."
            )
        if manifest.get("version") != OBSERVATION_STORE_VERSION:
            raise ValueError(
                f"Unsupported observation-store version {manifest.get('version')!r}; "
                f"expected {OBSERVATION_STORE_VERSION}. Reprepare the observation."
            )
        if not isinstance(manifest.get("rasters"), list) or not manifest["rasters"]:
            raise ValueError(
                f"Observation-store rasters are missing from {manifest_path}."
            )
        try:
            _require_sha256(manifest["source_signature"], "source_signature")
        except ValueError as error:
            raise ValueError(
                f"Invalid observation-store source signature in {manifest_path}."
            ) from error
        if not isinstance(manifest["metadata"], Mapping):
            raise ValueError(
                f"Observation-store metadata must be an object in {manifest_path}."
            )
        return manifest

    @classmethod
    def _load_array(
        cls,
        root: Path,
        name: str,
        record: Mapping[str, Any],
        *,
        mmap: bool,
        verify: bool,
    ) -> torch.Tensor:
        if not isinstance(record, Mapping) or set(record) != {
            "file",
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError(f"Invalid array schema for {name!r} in observation store.")
        filename = record.get("file")
        if not isinstance(filename, str) or not filename.endswith(".npy"):
            raise ValueError(f"Invalid array record {name!r} in observation store.")
        try:
            _require_sha256(record["sha256"], f"array {name!r} sha256")
            dtype = np.dtype(record["dtype"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid array record {name!r} in observation store."
            ) from error
        shape = record["shape"]
        if not isinstance(shape, list) or any(
            type(size) is not int or size < 0 for size in shape
        ):
            raise ValueError(f"Invalid array record {name!r} in observation store.")
        array_path = (root / filename).resolve()
        if array_path.parent != root or not array_path.is_file():
            raise ValueError(f"Unsafe or missing observation array path for {name!r}.")
        if verify and _sha256(array_path) != record.get("sha256"):
            raise ValueError(f"Observation array checksum mismatch for {name!r}.")
        array = np.load(array_path, mmap_mode="c" if mmap else None, allow_pickle=False)
        if list(array.shape) != shape or array.dtype != dtype:
            raise ValueError(f"Observation array schema mismatch for {name!r}.")
        return torch.from_numpy(array)

    @classmethod
    def load_sequence(
        cls,
        path: str | os.PathLike[str],
        *,
        mmap: bool = True,
        verify: bool = True,
        expected_signature: str | None = None,
    ) -> tuple[list[ObservationRaster], list[str], dict[str, Any]]:
        root = Path(path).expanduser().resolve()
        manifest = cls.manifest(root)
        if expected_signature is not None:
            _require_sha256(expected_signature, "expected_signature")
            if manifest["source_signature"] != expected_signature:
                raise ValueError(
                    "Observation-store source signature does not match the requested preparation."
                )
        rasters, names = [], []
        filenames: set[str] = set()
        for raster_record in manifest["rasters"]:
            if (
                not isinstance(raster_record, Mapping)
                or set(raster_record) != {"name", "metadata", "arrays"}
                or not isinstance(raster_record["metadata"], Mapping)
                or not isinstance(raster_record["arrays"], Mapping)
            ):
                raise ValueError("Invalid raster record in observation store.")
            array_names = set(raster_record["arrays"])
            missing = set(cls._CORE_FIELDS) - array_names
            unexpected = {
                name
                for name in array_names - set(cls._CORE_FIELDS)
                if not isinstance(name, str)
                or not name.startswith("auxiliary:")
                or not name.removeprefix("auxiliary:")
            }
            if missing or unexpected:
                raise ValueError(
                    "Observation store array names do not match the canonical contract; "
                    f"missing={sorted(missing)}, unexpected={sorted(map(str, unexpected))}."
                )
            record_filenames = [
                record.get("file")
                for record in raster_record["arrays"].values()
                if isinstance(record, Mapping)
            ]
            if (
                len(record_filenames) != len(raster_record["arrays"])
                or any(not isinstance(filename, str) for filename in record_filenames)
                or len(set(record_filenames)) != len(record_filenames)
                or filenames.intersection(record_filenames)
            ):
                raise ValueError("Observation-store array files must be unique.")
            filenames.update(record_filenames)
            arrays = {
                name: cls._load_array(root, name, record, mmap=mmap, verify=verify)
                for name, record in raster_record["arrays"].items()
            }
            auxiliary = {
                name.removeprefix("auxiliary:"): value
                for name, value in arrays.items()
                if name.startswith("auxiliary:")
            }
            rasters.append(
                ObservationRaster(
                    **{name: arrays[name] for name in cls._CORE_FIELDS},
                    metadata=raster_record["metadata"],
                    auxiliary=auxiliary,
                )
            )
            name = raster_record["name"]
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Observation-store raster names must be non-empty strings."
                )
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("Observation-store raster names must be unique.")
        return rasters, names, dict(manifest["metadata"])

    @classmethod
    def load(cls, path: str | os.PathLike[str], **options) -> ObservationRaster:
        rasters, _, _ = cls.load_sequence(path, **options)
        if len(rasters) != 1:
            raise ValueError(
                "ObservationStore.load() requires one raster; use load_sequence()."
            )
        return rasters[0]


__all__ = [
    "MANIFEST_FILENAME",
    "OBSERVATION_STORE_FORMAT",
    "OBSERVATION_STORE_VERSION",
    "ObservationStore",
    "observation_store_signature",
]
