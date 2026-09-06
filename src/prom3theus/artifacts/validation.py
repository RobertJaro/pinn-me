"""Strict validation and reconstruction for self-contained LTE artifacts.

This module owns the complete trust boundary used by export: manifest fields,
the resolved schema-v2 configuration, checksum-pinned packaged resources, the
embedded observation store, its scientific contracts, and tensor-only model
reconstruction all have to agree before evaluation starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prom3theus.config import parse_config
from prom3theus.observations import (
    CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    OBSERVATION_STORE_FORMAT,
    OBSERVATION_STORE_VERSION,
    ObservationRaster,
    ObservationSpec,
    ObservationStore,
    get_observation_adapter,
    resolve_velocity_synthesis_mode,
)
from prom3theus.resources import validate_resource_bundle
from prom3theus.training.lightning import LTEInversionModule

from .contracts import (
    ObservationContract,
    StoredRasterSelection,
    ValidatedArtifact,
)
from .errors import ArtifactExportError
from .model import (
    OBSERVATION_DIRECTORY_NAME,
    ArtifactManifest,
    load_manifest,
    load_state_dict,
)


_OBSERVATION_CONTRACT_FIELDS = frozenset({"store", "spec"})
_OBSERVATION_STORE_FIELDS = frozenset({"path", "format", "version", "source_signature"})
_OBSERVATION_SPEC_FIELDS = frozenset(
    {
        "observation_id",
        "observation_type",
        "instrument_type",
        "wavelength_angstrom",
        "continuum_indices",
        "radiance_scale_w_m3_sr",
        "velocity_synthesis_mode",
        "velocity_synthesis",
        "required_line_ids",
        "line_support_angstrom",
        "instrument_options",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "log_tau500",
        "wavelength_angstrom",
        "atmosphere_config",
        "synthesizer_config",
        "instrument_config",
        "stokes_loss_config",
        "weight_config",
        "wavelength_weights",
        "wavelength_exclude_windows_angstrom",
        "continuum_indices",
        "atlas_continuum_radiance_w_m3_sr",
        "depth_sampling_config",
        "physics_config",
        "learning_rate",
        "run_metadata",
        "observation_id",
        "velocity_synthesis_mode",
        "instrument_line_of_sight_velocity_correction_m_per_s",
        "optimize_instrument_line_of_sight_velocity_correction",
        "vector_regularization_config",
    }
)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ArtifactExportError(f"{context} must be an object with string keys.")
    return dict(value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ArtifactExportError(f"{context} has {' and '.join(details)}.")


def _json_canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    differences = set(left) ^ set(right)
    differences.update(
        key
        for key in set(left) & set(right)
        if _json_canonical(left[key]) != _json_canonical(right[key])
    )
    return sorted(differences)


def _artifact_child(artifact_directory: Path, relative: Any, context: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ArtifactExportError(f"{context} must be a non-empty relative path.")
    supplied = Path(relative)
    if supplied.is_absolute():
        raise ArtifactExportError(
            f"{context} must be relative to the artifact directory."
        )
    resolved = (artifact_directory / supplied).resolve(strict=False)
    try:
        resolved.relative_to(artifact_directory)
    except ValueError as error:
        raise ArtifactExportError(
            f"{context} escapes the artifact directory: {relative!r}."
        ) from error
    return resolved


def _parse_observation_spec(value: Any) -> ObservationSpec:
    raw = _require_mapping(value, "manifest.observation.spec")
    _require_exact_fields(raw, _OBSERVATION_SPEC_FIELDS, "manifest.observation.spec")
    support = raw["line_support_angstrom"]
    try:
        spec = ObservationSpec(
            observation_id=raw["observation_id"],
            observation_type=raw["observation_type"],
            instrument_type=raw["instrument_type"],
            wavelength_angstrom=torch.as_tensor(raw["wavelength_angstrom"]),
            continuum_indices=tuple(raw["continuum_indices"]),
            radiance_scale_w_m3_sr=raw["radiance_scale_w_m3_sr"],
            velocity_synthesis_mode=raw["velocity_synthesis_mode"],
            required_line_ids=tuple(raw["required_line_ids"]),
            line_support_angstrom=(None if support is None else tuple(support)),
            instrument_options=_require_mapping(
                raw["instrument_options"],
                "manifest.observation.spec.instrument_options",
            ),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"manifest.observation.spec is invalid: {error}"
        ) from error
    canonical = spec.metadata()
    if _json_canonical(canonical) != _json_canonical(raw):
        raise ArtifactExportError(
            "manifest.observation.spec is not the exact canonical "
            "ObservationSpec.metadata() representation."
        )
    try:
        adapter = get_observation_adapter(spec.observation_type)
    except ValueError as error:
        raise ArtifactExportError(str(error)) from error
    if adapter.name != spec.observation_type:
        raise ArtifactExportError("Observation adapter identity is inconsistent.")
    return spec


def _observation_contract(
    manifest: ArtifactManifest,
    artifact_directory: Path,
) -> ObservationContract:
    raw = _require_mapping(manifest.observation, "manifest.observation")
    if "store" not in raw:
        raise ArtifactExportError(
            "manifest.observation.store is required by the canonical artifact contract."
        )
    _require_exact_fields(raw, _OBSERVATION_CONTRACT_FIELDS, "manifest.observation")
    store = _require_mapping(raw["store"], "manifest.observation.store")
    _require_exact_fields(
        store, _OBSERVATION_STORE_FIELDS, "manifest.observation.store"
    )
    if store["path"] != OBSERVATION_DIRECTORY_NAME:
        raise ArtifactExportError(
            "manifest.observation.store.path must be exactly "
            f"{OBSERVATION_DIRECTORY_NAME!r}."
        )
    if store["format"] != OBSERVATION_STORE_FORMAT:
        raise ArtifactExportError(
            f"manifest.observation.store.format must be {OBSERVATION_STORE_FORMAT!r}."
        )
    if (
        type(store["version"]) is not int
        or store["version"] != OBSERVATION_STORE_VERSION
    ):
        raise ArtifactExportError(
            "manifest.observation.store.version must be exactly "
            f"{OBSERVATION_STORE_VERSION}."
        )
    signature = store["source_signature"]
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise ArtifactExportError(
            "manifest.observation.store.source_signature must be a lowercase SHA256 digest."
        )
    path = _artifact_child(
        artifact_directory,
        store["path"],
        "manifest.observation.store.path",
    )
    if not path.is_dir():
        raise ArtifactExportError(
            f"Canonical observation store is missing at {path}; export requires the "
            "embedded store and never rebuilds observations from source data."
        )
    return ObservationContract(
        path=path,
        source_signature=signature,
        spec=_parse_observation_spec(raw["spec"]),
        manifest_value=raw,
    )


def _validated_config(manifest: ArtifactManifest, artifact_directory: Path) -> Any:
    try:
        return parse_config(
            manifest.resolved_config,
            base_directory=artifact_directory,
            environ={},
        )
    except (TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"manifest.resolved_config does not satisfy schema version 2: {error}"
        ) from error


def _validated_resources(manifest: ArtifactManifest, config: Any) -> dict[str, Any]:
    recorded = _require_mapping(manifest.resources, "manifest.resources")
    if config.resources.bundle != "packaged":
        raise ArtifactExportError(
            "Schema-v2 export requires resolved_config.resources.bundle='packaged'."
        )
    if "directory" in recorded:
        raise ArtifactExportError(
            "manifest.resources must not record an installation-specific directory."
        )
    try:
        current = validate_resource_bundle()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"The artifact's LTE resource bundle failed validation: {error}"
        ) from error
    portable_current = {
        name: value for name, value in current.items() if name != "directory"
    }
    differences = _mapping_difference(recorded, portable_current)
    if differences:
        raise ArtifactExportError(
            "The current LTE resource bundle differs from manifest.resources in: "
            f"{differences}. Only the checksum-pinned packaged bundle is valid."
        )
    return portable_current


def _load_observation(
    manifest: ArtifactManifest,
    contract: ObservationContract,
    config: Any,
) -> StoredRasterSelection:
    try:
        store_manifest = ObservationStore.manifest(contract.path)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"Invalid canonical observation store: {error}"
        ) from error
    for name, expected in (
        ("format", OBSERVATION_STORE_FORMAT),
        ("version", OBSERVATION_STORE_VERSION),
        ("source_signature", contract.source_signature),
    ):
        if store_manifest.get(name) != expected:
            raise ArtifactExportError(
                f"Observation-store {name} differs from manifest.observation.store."
            )
    try:
        rasters, raster_names, store_metadata = ObservationStore.load_sequence(
            contract.path,
            mmap=True,
            verify=True,
            expected_signature=contract.source_signature,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"Invalid canonical observation store: {error}"
        ) from error
    spec = contract.spec
    if store_metadata.get("adapter") != spec.observation_type:
        raise ArtifactExportError(
            "Observation-store adapter differs from the observation spec."
        )
    if store_metadata.get("observation") != spec.metadata():
        raise ArtifactExportError(
            "Observation-store metadata differs from the observation spec."
        )
    validation_index = store_metadata.get("validation_raster_index")
    if (
        type(validation_index) is not int
        or validation_index < 0
        or validation_index >= len(rasters)
    ):
        raise ArtifactExportError(
            "Observation-store metadata must identify a valid validation_raster_index."
        )
    if (
        len(raster_names) != len(rasters)
        or any(not name for name in raster_names)
        or len(set(raster_names)) != len(raster_names)
    ):
        raise ArtifactExportError(
            "Observation-store raster names must be non-empty and unique."
        )
    raster = rasters[validation_index]
    if config.observation.type != spec.observation_type:
        raise ArtifactExportError(
            "manifest.resolved_config.observation.type differs from the observation spec."
        )
    if config.instrument.type != spec.instrument_type:
        raise ArtifactExportError(
            "manifest.resolved_config.instrument.type differs from the observation spec."
        )
    wavelength = raster.wavelength_angstrom.detach().cpu().to(torch.float64)
    expected_wavelength = spec.wavelength_angstrom.detach().cpu().to(torch.float64)
    if wavelength.shape != expected_wavelength.shape or not torch.allclose(
        wavelength, expected_wavelength, rtol=0.0, atol=1.0e-7
    ):
        raise ArtifactExportError(
            "Canonical observation-store wavelength grid differs from the observation spec."
        )
    validate_raster_scientific_contract(raster, spec)
    validate_raster_velocity_contract(raster, spec.velocity_synthesis_mode.value)
    scene_basis(raster)
    return StoredRasterSelection(
        raster=raster,
        name=raster_names[validation_index],
        index=validation_index,
        raster_count=len(rasters),
    )


def _validate_model_contract(
    manifest: ArtifactManifest,
    config: Any,
    spec: ObservationSpec,
) -> dict[str, Any]:
    model = _require_mapping(manifest.model, "manifest.model")
    _require_exact_fields(model, _MODEL_FIELDS, "manifest.model")
    if model["observation_id"] != spec.observation_id:
        raise ArtifactExportError(
            "manifest.model.observation_id differs from the observation spec."
        )
    try:
        mode = resolve_velocity_synthesis_mode(model["velocity_synthesis_mode"])
    except ValueError as error:
        raise ArtifactExportError(str(error)) from error
    if mode != spec.velocity_synthesis_mode:
        raise ArtifactExportError(
            "manifest.model.velocity_synthesis_mode differs from the observation spec."
        )
    try:
        wavelength = torch.as_tensor(model["wavelength_angstrom"], dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise ArtifactExportError(
            "manifest.model.wavelength_angstrom must be a numeric sequence."
        ) from error
    expected = spec.wavelength_angstrom.to(torch.float64)
    if wavelength.shape != expected.shape or not torch.allclose(
        wavelength, expected, rtol=0.0, atol=1.0e-7
    ):
        raise ArtifactExportError(
            "manifest.model.wavelength_angstrom differs from the observation spec."
        )
    raw_continuum_indices = model["continuum_indices"]
    if not isinstance(raw_continuum_indices, (list, tuple)) or any(
        type(value) is not int for value in raw_continuum_indices
    ):
        raise ArtifactExportError(
            "manifest.model.continuum_indices must be a sequence of integers."
        )
    continuum_indices = tuple(raw_continuum_indices)
    if continuum_indices != spec.continuum_indices:
        raise ArtifactExportError(
            "manifest.model.continuum_indices differs from the observation spec."
        )
    raw_scale = model["atlas_continuum_radiance_w_m3_sr"]
    if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
        raise ArtifactExportError(
            "manifest.model.atlas_continuum_radiance_w_m3_sr must be numeric."
        )
    scale = float(raw_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ArtifactExportError(
            "manifest.model.atlas_continuum_radiance_w_m3_sr must be finite and positive."
        )
    if not math.isclose(
        scale,
        spec.radiance_scale_w_m3_sr,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ArtifactExportError(
            "manifest.model atlas continuum radiance differs from the observation spec."
        )
    instrument = _require_mapping(
        model["instrument_config"], "manifest.model.instrument_config"
    )
    if instrument.get("type") != spec.instrument_type:
        raise ArtifactExportError(
            "manifest.model.instrument_config.type differs from the observation spec."
        )
    if config.instrument.type != instrument["type"]:
        raise ArtifactExportError(
            "manifest.model.instrument_config.type differs from "
            "resolved_config.instrument.type."
        )
    synthesizer = _require_mapping(
        model["synthesizer_config"], "manifest.model.synthesizer_config"
    )
    if set(synthesizer) != {"line_ids"}:
        raise ArtifactExportError(
            "manifest.model.synthesizer_config must contain exactly 'line_ids'."
        )
    if synthesizer["line_ids"] != list(config.synthesis.line_ids):
        raise ArtifactExportError(
            "manifest.model line_ids differ from resolved_config.synthesis.line_ids."
        )
    if not set(spec.required_line_ids).issubset(synthesizer["line_ids"]):
        raise ArtifactExportError(
            "manifest.model line_ids do not cover the observation requirements."
        )
    if "data_directory" in instrument:
        raise ArtifactExportError(
            "manifest.model must not record instrument_config.data_directory; "
            "schema-v1 artifacts reconstruct from packaged resources."
        )
    return model


def _reconstruct_module(
    artifact_directory: Path,
    manifest: ArtifactManifest,
    model_config: Mapping[str, Any],
) -> LTEInversionModule:
    try:
        module = LTEInversionModule(**deepcopy(dict(model_config)))
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"manifest.model could not reconstruct LTEInversionModule exactly: {error}"
        ) from error
    try:
        state = load_state_dict(artifact_directory, manifest, map_location="cpu")
        module.load_state_dict(state, strict=True)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"Artifact tensor weights do not exactly match manifest.model: {error}"
        ) from error
    return module


def scene_basis(raster: ObservationRaster) -> np.ndarray:
    """Return the validated right-handed scene basis stored with a raster."""

    geometry = raster.metadata.get("ray_geometry")
    if not isinstance(geometry, Mapping):
        raise ArtifactExportError(
            "Observation metadata must contain ray_geometry.scene_basis_rows for export."
        )
    basis = np.asarray(geometry.get("scene_basis_rows"), dtype=np.float64)
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise ArtifactExportError(
            "Observation ray_geometry.scene_basis_rows must be a finite 3x3 matrix."
        )
    if (
        not np.allclose(basis @ basis.T, np.eye(3), rtol=0.0, atol=2.0e-5)
        or np.linalg.det(basis) <= 0
    ):
        raise ArtifactExportError(
            "Observation ray_geometry.scene_basis_rows must be right-handed and orthonormal."
        )
    return basis


def removed_solar_los_velocity(raster: ObservationRaster) -> Any | None:
    """Return the stored Hinode registration correction, when present."""

    correction = raster.metadata.get("observer_velocity_correction")
    if not isinstance(correction, Mapping):
        return None
    return correction.get("removed_solar_los_velocity_m_per_s")


def validate_raster_scientific_contract(
    raster: ObservationRaster,
    spec: ObservationSpec,
) -> None:
    """Validate Stokes ordering and radiometric normalization metadata."""

    if raster.metadata.get("stokes_order") != ["I", "Q", "U", "V"]:
        raise ArtifactExportError(
            "Canonical observation metadata must declare stokes_order=['I','Q','U','V']."
        )
    normalization = raster.metadata.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ArtifactExportError(
            "Canonical observation metadata must contain its normalization contract."
        )
    try:
        raw_indices = normalization["indices"]
        radiometry = normalization["radiometric_calibration"]
        raw_radiance_scale = radiometry["atlas_disk_center_continuum_radiance_w_m3_sr"]
    except (KeyError, TypeError) as error:
        raise ArtifactExportError(
            "Canonical observation normalization lacks continuum indices or its "
            "atlas continuum radiance."
        ) from error
    if not isinstance(raw_indices, (list, tuple)) or any(
        type(value) is not int for value in raw_indices
    ):
        raise ArtifactExportError(
            "Canonical observation normalization indices must be integers."
        )
    continuum_indices = tuple(raw_indices)
    if isinstance(raw_radiance_scale, bool) or not isinstance(
        raw_radiance_scale, (int, float)
    ):
        raise ArtifactExportError(
            "Canonical observation atlas continuum radiance must be numeric."
        )
    radiance_scale = float(raw_radiance_scale)
    if not math.isfinite(radiance_scale) or radiance_scale <= 0.0:
        raise ArtifactExportError(
            "Canonical observation atlas continuum radiance must be finite and positive."
        )
    if continuum_indices != spec.continuum_indices:
        raise ArtifactExportError(
            "Observation-store continuum indices differ from the observation spec."
        )
    if not math.isclose(
        radiance_scale,
        spec.radiance_scale_w_m3_sr,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ArtifactExportError(
            "Observation-store atlas continuum radiance differs from the observation spec."
        )


def validate_raster_velocity_contract(
    raster: ObservationRaster,
    velocity_synthesis_mode: str,
) -> None:
    """Validate the observation-specific velocity synthesis inputs."""

    mode = resolve_velocity_synthesis_mode(velocity_synthesis_mode)
    observer = raster.auxiliary.get("observer_los_velocity_m_per_s")
    removed = removed_solar_los_velocity(raster)
    if mode.value == CARRINGTON_OBSERVER_RELATIVE_VELOCITY:
        if observer is None:
            raise ArtifactExportError(
                "carrington_observer_relative export requires observation auxiliary "
                "'observer_los_velocity_m_per_s'."
            )
        if tuple(observer.shape) not in {
            raster.spatial_shape,
            (*raster.spatial_shape, 1),
        }:
            raise ArtifactExportError(
                "observer_los_velocity_m_per_s must match the raster spatial shape."
            )
        if not torch.isfinite(observer).all():
            raise ArtifactExportError(
                "observer_los_velocity_m_per_s must contain only finite values."
            )
        if removed is not None:
            raise ArtifactExportError(
                "Observer-relative observations must not contain a Hinode registration velocity."
            )
    elif mode.value == CARRINGTON_REGISTERED_RELATIVE_VELOCITY:
        values = np.asarray(removed)
        if values.shape != (raster.spatial_shape[1],) or not np.isfinite(values).all():
            raise ArtifactExportError(
                "carrington_registered_relative export requires one finite "
                "removed_solar_los_velocity_m_per_s value per scan column."
            )
        if observer is not None:
            raise ArtifactExportError(
                "Registered-relative observations must not contain an observer LOS array."
            )
    elif observer is not None or removed is not None:
        raise ArtifactExportError(
            "registered_relative export must not apply an observation-specific LOS correction."
        )


def load_validated_artifact(artifact_directory: str | Path) -> ValidatedArtifact:
    """Load and cross-validate every input needed to export an LTE artifact."""

    artifact_root = Path(artifact_directory).expanduser().resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_root}")
    manifest = load_manifest(artifact_root)
    observation = _observation_contract(manifest, artifact_root)
    config = _validated_config(manifest, artifact_root)
    resources = _validated_resources(manifest, config)
    raster_selection = _load_observation(manifest, observation, config)
    model_config = _validate_model_contract(manifest, config, observation.spec)
    module = _reconstruct_module(artifact_root, manifest, model_config)
    return ValidatedArtifact(
        root=artifact_root,
        manifest=manifest,
        observation=observation,
        resources=resources,
        raster_selection=raster_selection,
        module=module,
    )


__all__ = [
    "load_validated_artifact",
    "removed_solar_los_velocity",
    "scene_basis",
    "validate_raster_scientific_contract",
    "validate_raster_velocity_contract",
]
