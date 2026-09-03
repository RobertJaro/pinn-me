"""Self-describing, versioned inversion artifacts.

Only tensor state is stored with PyTorch.  Python objects, data modules, and
configuration classes are never pickled into an artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch

from prom3theus.core import sha256_file
from prom3theus.observations.store import (
    OBSERVATION_STORE_FORMAT,
    OBSERVATION_STORE_VERSION,
    ObservationStore,
)


ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.pt"
OBSERVATION_DIRECTORY_NAME = "observations"


class UnsupportedArtifactVersion(ValueError):
    """Raised when an artifact does not use the current exact schema."""


@dataclass(frozen=True)
class ArtifactManifest:
    """Portable metadata required to reconstruct one inversion model."""

    schema_version: int
    solver: str
    package_version: str
    created_utc: str
    resolved_config: dict[str, Any]
    resources: dict[str, Any]
    observation: dict[str, Any]
    model: dict[str, Any]
    weights_file: str = WEIGHTS_NAME
    weights_sha256: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        package_version: str,
        resolved_config: Mapping[str, Any],
        resources: Mapping[str, Any],
        observation: Mapping[str, Any],
        model: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
    ) -> "ArtifactManifest":
        return cls(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            solver="lte",
            package_version=str(package_version),
            created_utc=datetime.now(timezone.utc).isoformat(),
            resolved_config=dict(resolved_config),
            resources=dict(resources),
            observation=dict(observation),
            model=dict(model),
            provenance=dict(provenance or {}),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactManifest":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown artifact manifest fields: {sorted(unknown)}")
        version = value.get("schema_version")
        if type(version) is not int or version != ARTIFACT_SCHEMA_VERSION:
            raise UnsupportedArtifactVersion(
                f"Artifact schema {version!r} is unsupported; expected exactly "
                f"{ARTIFACT_SCHEMA_VERSION}."
            )
        if value.get("solver") != "lte":
            raise ValueError("Only solver='lte' artifacts are supported.")
        weights_sha256 = value.get("weights_sha256")
        if not _is_sha256(weights_sha256):
            raise ValueError(
                "Artifact weights_sha256 must be a lowercase SHA-256 digest."
            )
        manifest = cls(**dict(value))
        _validate_weights_filename(manifest.weights_file)
        for name in ("resolved_config", "resources", "observation", "model"):
            if not isinstance(getattr(manifest, name), dict):
                raise TypeError(f"Artifact manifest field {name!r} must be an object.")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_weights_filename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or Path(value).name != value:
        raise ValueError("Artifact weights_file must be a non-empty local filename.")
    return value


def _weights_path(directory: str | Path, filename: Any) -> Path:
    """Resolve one weights file while keeping symlinks inside its artifact."""

    root = Path(directory).expanduser().resolve()
    local_name = _validate_weights_filename(filename)
    path = (root / local_name).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Artifact weights_file must stay inside the artifact directory."
        ) from error
    return path


def _validate_tensor_state(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in value.items()
    ):
        raise TypeError("Artifact weights must be a tensor-only state dictionary.")
    for name, tensor in value.items():
        if tensor.device.type == "meta":
            raise ValueError(
                f"Artifact tensor {name!r} must contain materialized values."
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            raise ValueError(f"Artifact tensor {name!r} contains non-finite values.")
    return value


def save_artifact(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    manifest: ArtifactManifest,
    observation_store: str | Path,
) -> Path:
    """Atomically write a complete, independently exportable LTE artifact."""

    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(
            f"Artifact directory already exists: {directory}. Choose a new output path."
        )
    _validate_weights_filename(manifest.weights_file)
    if (
        type(manifest.schema_version) is not int
        or manifest.schema_version != ARTIFACT_SCHEMA_VERSION
    ):
        raise UnsupportedArtifactVersion(
            f"Artifact schema {manifest.schema_version!r} is unsupported; expected "
            f"exactly {ARTIFACT_SCHEMA_VERSION}."
        )
    if manifest.solver != "lte":
        raise ValueError("Only solver='lte' artifacts are supported.")
    if "directory" in manifest.resources:
        raise ValueError(
            "Artifact resources must not contain an installation-specific directory."
        )
    if manifest.resolved_config.get("schema_version") != 1:
        raise ValueError("Artifact resolved_config must use schema version 1.")
    if manifest.resolved_config.get("resources") != {"bundle": "packaged"}:
        raise ValueError(
            "Artifact resolved_config must use the packaged production resources."
        )
    observation = manifest.observation
    if not isinstance(observation, Mapping) or set(observation) != {"store", "spec"}:
        raise ValueError(
            "Artifact observation must contain exactly 'store' and 'spec'."
        )
    store_contract = observation["store"]
    if not isinstance(store_contract, Mapping) or set(store_contract) != {
        "path",
        "format",
        "version",
        "source_signature",
    }:
        raise ValueError(
            "Artifact observation.store must contain the exact schema-v1 fields."
        )
    if store_contract["path"] != OBSERVATION_DIRECTORY_NAME:
        raise ValueError(
            f"Artifact observation.store.path must be {OBSERVATION_DIRECTORY_NAME!r}."
        )
    if store_contract["format"] != OBSERVATION_STORE_FORMAT:
        raise ValueError("Artifact observation-store format is unsupported.")
    if store_contract["version"] != OBSERVATION_STORE_VERSION:
        raise ValueError("Artifact observation-store version is unsupported.")
    source_signature = store_contract["source_signature"]
    if (
        not isinstance(source_signature, str)
        or len(source_signature) != 64
        or any(character not in "0123456789abcdef" for character in source_signature)
    ):
        raise ValueError(
            "Artifact observation-store source_signature must be a lowercase SHA256 digest."
        )

    source_store = Path(observation_store).expanduser().resolve()
    if not source_store.is_dir():
        raise FileNotFoundError(f"Observation store not found: {source_store}")
    rasters, raster_names, store_metadata = ObservationStore.load_sequence(
        source_store,
        mmap=True,
        verify=True,
        expected_signature=source_signature,
    )
    if not raster_names or len(set(raster_names)) != len(raster_names):
        raise ValueError("Observation-store raster names must be non-empty and unique.")
    validation_index = store_metadata.get("validation_raster_index")
    if (
        type(validation_index) is not int
        or validation_index < 0
        or validation_index >= len(rasters)
    ):
        raise ValueError(
            "Observation-store metadata must identify a valid validation_raster_index."
        )
    if store_metadata.get("observation") != observation["spec"]:
        raise ValueError(
            "Observation-store metadata does not match the artifact observation spec."
        )
    state = _validate_tensor_state(model.state_dict())
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent)
    )
    try:
        weights_path = temporary / manifest.weights_file
        torch.save(state, weights_path)
        stored_manifest = replace(
            manifest,
            weights_sha256=sha256_file(weights_path),
        )
        serialized_manifest = (
            json.dumps(
                stored_manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        )
        embedded_store = temporary / OBSERVATION_DIRECTORY_NAME
        shutil.copytree(source_store, embedded_store)
        # Verify the copied bytes before the atomic directory rename.
        ObservationStore.load_sequence(
            embedded_store,
            mmap=True,
            verify=True,
            expected_signature=source_signature,
        )
        (temporary / MANIFEST_NAME).write_text(serialized_manifest, encoding="utf-8")
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory / MANIFEST_NAME


def load_manifest(directory: str | Path) -> ArtifactManifest:
    """Load and strictly validate an artifact manifest."""

    path = Path(directory).expanduser().resolve() / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Artifact manifest not found: {path}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("Artifact manifest must contain a JSON object.")
    return ArtifactManifest.from_mapping(payload)


def load_state_dict(
    directory: str | Path,
    manifest: ArtifactManifest,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load tensor state without permitting arbitrary pickled objects."""

    path = _weights_path(directory, manifest.weights_file)
    if not _is_sha256(manifest.weights_sha256):
        raise ValueError("Artifact weights_sha256 must be a lowercase SHA-256 digest.")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != manifest.weights_sha256:
        raise RuntimeError(
            "Artifact weights failed SHA-256 integrity verification: "
            f"expected {manifest.weights_sha256}, got {actual_sha256}."
        )
    state = torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    return _validate_tensor_state(state)
