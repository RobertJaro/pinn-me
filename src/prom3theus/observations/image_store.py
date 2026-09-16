"""Immutable, class-free persistence for scalar image observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from prom3theus.core import sha256_file

from .image_contracts import ImageObservationRaster


IMAGE_OBSERVATION_STORE_FORMAT = "prom3theus.image_observation_store"
IMAGE_OBSERVATION_STORE_VERSION = 1
IMAGE_MANIFEST_FILENAME = "manifest.json"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Image-observation metadata keys must be strings.")
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
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Image-observation metadata cannot contain NaN or infinity.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"Image-observation metadata contains unsupported value {type(value).__name__}."
    )


def _portable_contract(value: Any) -> Any:
    """Remove installation locations while retaining scientific identity."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Preparation-dependency keys must be strings.")
        return {
            key: _portable_contract(item)
            for key, item in value.items()
            if key not in {"directory", "installation_directory", "root"}
        }
    if isinstance(value, (list, tuple)):
        return [_portable_contract(item) for item in value]
    return _json_value(value)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def image_observation_store_signature(
    store_configuration: Mapping[str, Any],
    preparation_dependencies: Mapping[str, Any],
    *,
    source_files_sha256: str,
) -> str:
    """Hash only inputs that determine the prepared image arrays.

    Callers pass an adapter-filtered ``store_configuration``.  Loader batch
    sizes, validation selection, scheduling, and forward-model resources do not
    belong in it, so those changes can reuse the same immutable image store.
    """

    _require_sha256(source_files_sha256, "source_files_sha256")
    payload = json.dumps(
        {
            "store_format": IMAGE_OBSERVATION_STORE_FORMAT,
            "store_version": IMAGE_OBSERVATION_STORE_VERSION,
            "store_configuration": _json_value(store_configuration),
            "preparation_dependencies": _portable_contract(preparation_dependencies),
            "source_files_sha256": source_files_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ImageObservationStore:
    """Read and write one or more native-grid image rasters."""

    _CORE_FIELDS = (
        "intensity",
        "uncertainty",
        "ray_direction",
        "surface_position_m",
        "valid_mask",
    )

    @classmethod
    def _write_array(cls, path: Path, tensor: torch.Tensor) -> dict[str, Any]:
        from .arrays import write_array
        dtype, shape = write_array(path, tensor)
        return {
            "file": path.name,
            "dtype": dtype,
            "shape": list(shape),
            "sha256": sha256_file(path),
        }

    @classmethod
    def write_raster(cls, directory: Path, index: int, name: str,
                     raster: ImageObservationRaster) -> dict:
        """Write one independent raster; return only its manifest entry."""
        arrays = {
            field: cls._write_array(directory / f"i{index:04d}_a{array_index:03d}.npy",
                                    getattr(raster, field))
            for array_index, field in enumerate(cls._CORE_FIELDS)
            if getattr(raster, field) is not None
        }
        return {
            "name": name, "channel_angstrom": raster.channel_angstrom,
            "exposure_group": raster.exposure_group,
            "absolute_tai_seconds": raster.absolute_tai_seconds,
            "metadata": _json_value(raster.metadata), "arrays": arrays,
        }

    @classmethod
    def publish(cls, staging: Path, destination: Path, records: Sequence[dict], *,
                source_signature: str, metadata: Mapping | None = None) -> Path:
        """Publish already-written raster arrays without reading them back."""
        if destination.exists():
            raise FileExistsError(f"Image-observation store already exists: {destination}.")
        cls.write_manifest(staging, records, source_signature=source_signature, metadata=metadata)
        os.replace(staging, destination)
        return destination

    @classmethod
    def write_manifest(cls, directory: Path, records: Sequence[dict], *,
                       source_signature: str, metadata: Mapping | None = None) -> Path:
        """Finish a store in place after all its image arrays have been written."""
        _require_sha256(source_signature, "source_signature")
        manifest = {
            "format": IMAGE_OBSERVATION_STORE_FORMAT,
            "version": IMAGE_OBSERVATION_STORE_VERSION,
            "source_signature": source_signature,
            "metadata": _json_value(metadata or {}), "rasters": list(records),
        }
        (directory / IMAGE_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return directory

    @classmethod
    def save_sequence(
        cls,
        path: str | os.PathLike[str],
        rasters: Sequence[ImageObservationRaster],
        *,
        raster_names: Sequence[str] | None = None,
        source_signature: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        destination = Path(path).expanduser().resolve()
        rasters = tuple(rasters)
        if not rasters:
            raise ValueError("An image-observation store requires at least one raster.")
        names = (
            tuple(f"image_{index:04d}" for index in range(len(rasters)))
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
                f"Image-observation store already exists: {destination}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
        )
        try:
            records = [
                cls.write_raster(temporary, index, name, raster)
                for index, (name, raster) in enumerate(zip(names, rasters, strict=True))
            ]
            cls.publish(temporary, destination, records,
                        source_signature=source_signature, metadata=metadata)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    @classmethod
    def save(
        cls,
        path: str | os.PathLike[str],
        raster: ImageObservationRaster,
        *,
        source_signature: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return cls.save_sequence(
            path,
            (raster,),
            source_signature=source_signature,
            metadata=metadata,
        )

    @classmethod
    def manifest(cls, path: str | os.PathLike[str]) -> dict[str, Any]:
        manifest_path = Path(path).expanduser().resolve() / IMAGE_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Image-observation manifest not found: {manifest_path}."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"format", "version", "source_signature", "metadata", "rasters"}
        if not isinstance(manifest, Mapping) or set(manifest) != expected:
            raise ValueError(
                f"Invalid image-observation manifest schema in {manifest_path}."
            )
        if manifest["format"] != IMAGE_OBSERVATION_STORE_FORMAT:
            raise ValueError(
                f"Unsupported image-observation store format in {manifest_path}."
            )
        if manifest["version"] != IMAGE_OBSERVATION_STORE_VERSION:
            raise ValueError(
                "Unsupported image-observation store version "
                f"{manifest['version']!r}; expected {IMAGE_OBSERVATION_STORE_VERSION}."
            )
        _require_sha256(manifest["source_signature"], "source_signature")
        if not isinstance(manifest["metadata"], Mapping):
            raise ValueError("Image-observation store metadata must be an object.")
        if not isinstance(manifest["rasters"], list) or not manifest["rasters"]:
            raise ValueError("Image-observation store requires raster records.")
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
        expected = {"file", "dtype", "shape", "sha256"}
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError(f"Invalid array schema for {name!r}.")
        filename = record["file"]
        if (
            not isinstance(filename, str)
            or not filename.endswith(".npy")
            or Path(filename).name != filename
        ):
            raise ValueError(f"Unsafe image-observation array path for {name!r}.")
        _require_sha256(record["sha256"], f"array {name!r} sha256")
        try:
            dtype = np.dtype(record["dtype"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid array dtype for {name!r}.") from error
        shape = record["shape"]
        if not isinstance(shape, list) or any(
            type(size) is not int or size < 0 for size in shape
        ):
            raise ValueError(f"Invalid array shape for {name!r}.")
        array_path = (root / filename).resolve()
        if array_path.parent != root or not array_path.is_file():
            raise ValueError(f"Unsafe or missing image-observation array for {name!r}.")
        if verify and sha256_file(array_path) != record["sha256"]:
            raise ValueError(f"Image-observation array checksum mismatch for {name!r}.")
        from .loading import load_array_tensor
        return load_array_tensor(array_path, mmap=mmap, shape=shape, dtype=dtype)

    @classmethod
    def load_sequence(
        cls,
        path: str | os.PathLike[str],
        *,
        mmap: bool = True,
        verify: bool = True,
        expected_signature: str | None = None,
    ) -> tuple[list[ImageObservationRaster], list[str], dict[str, Any]]:
        root = Path(path).expanduser().resolve()
        manifest = cls.manifest(root)
        if expected_signature is not None:
            _require_sha256(expected_signature, "expected_signature")
            if manifest["source_signature"] != expected_signature:
                raise ValueError(
                    "Image-observation source signature does not match the request."
                )

        rasters: list[ImageObservationRaster] = []
        names: list[str] = []
        filenames: set[str] = set()
        record_fields = {
            "name",
            "channel_angstrom",
            "exposure_group",
            "absolute_tai_seconds",
            "metadata",
            "arrays",
        }
        for raster_record in manifest["rasters"]:
            if (
                not isinstance(raster_record, Mapping)
                or set(raster_record) != record_fields
                or not isinstance(raster_record["metadata"], Mapping)
                or not isinstance(raster_record["arrays"], Mapping)
                or set(raster_record["arrays"]) not in (
                    set(cls._CORE_FIELDS), set(cls._CORE_FIELDS) - {"uncertainty"}
                )
            ):
                raise ValueError("Invalid raster record in image-observation store.")
            record_filenames = []
            for record in raster_record["arrays"].values():
                if isinstance(record, Mapping):
                    record_filenames.append(record.get("file"))
            if (
                len(record_filenames) != len(raster_record["arrays"])
                or any(not isinstance(name, str) for name in record_filenames)
                or len(set(record_filenames)) != len(record_filenames)
                or filenames.intersection(record_filenames)
            ):
                raise ValueError("Image-observation array files must be unique.")
            filenames.update(record_filenames)
            arrays = {
                name: cls._load_array(root, name, record, mmap=mmap, verify=verify)
                for name, record in raster_record["arrays"].items()
            }
            name = raster_record["name"]
            if not isinstance(name, str) or not name:
                raise ValueError("Image raster names must be non-empty strings.")
            names.append(name)
            from .catalog import validated_raster, attach_catalog
            rasters.append(
                validated_raster(ImageObservationRaster,
                    **arrays,
                    absolute_tai_seconds=raster_record["absolute_tai_seconds"],
                    channel_angstrom=raster_record["channel_angstrom"],
                    exposure_group=raster_record["exposure_group"],
                    metadata=raster_record["metadata"],
                )
            )
            attach_catalog(rasters[-1], root, raster_record)
        if len(set(names)) != len(names):
            raise ValueError("Image raster names must be unique.")
        identities = [
            (raster.exposure_group, raster.channel_angstrom) for raster in rasters
        ]
        if len(set(identities)) != len(identities):
            raise ValueError(
                "An image-observation store may contain only one raster per "
                "exposure group and channel."
            )
        return rasters, names, dict(manifest["metadata"])

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], **options: Any
    ) -> ImageObservationRaster:
        rasters, _, _ = cls.load_sequence(path, **options)
        if len(rasters) != 1:
            raise ValueError(
                "ImageObservationStore.load() requires one raster; use load_sequence()."
            )
        return rasters[0]


__all__ = [
    "IMAGE_MANIFEST_FILENAME",
    "IMAGE_OBSERVATION_STORE_FORMAT",
    "IMAGE_OBSERVATION_STORE_VERSION",
    "ImageObservationStore",
    "image_observation_store_signature",
]
