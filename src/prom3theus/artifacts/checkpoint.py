"""Portable, standalone PROM3THEUS evaluation save states."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import torch

from prom3theus.observations import ObservationSpec
from prom3theus.training.lightning import (
    LTEInversionModule,
    P3S_CONTEXT_KEY,
    P3S_FORMAT,
    P3S_VERSION,
)

from .errors import ArtifactExportError
from .model import ArtifactManifest
from .validation import (
    _parse_observation_spec,
    _validate_model_contract,
    _validated_config,
    _validated_resources,
)


@dataclass(frozen=True, slots=True)
class ObservationReference:
    """Lightweight source-selection and sampling contract stored in P3S."""

    source_signature: str
    spec: ObservationSpec
    raster_names: tuple[str, ...]
    validation_raster_index: int
    times: tuple[dict[str, Any], ...]
    bounds: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ValidatedSaveState:
    """A reconstructed P3S file and its exact post-training evaluation inputs."""

    path: Path
    resolved_config: Any
    observation: ObservationReference
    resources: dict[str, Any]
    module: LTEInversionModule
    epoch: int
    global_step: int


def _safe_metadata(value: Any, *, context: str) -> Any:
    """Reduce P3S metadata to types accepted by the restricted torch loader."""

    if isinstance(value, Enum):
        return _safe_metadata(value.value, context=context)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{context} metadata keys must be strings.")
        return {
            key: _safe_metadata(item, context=f"{context}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_metadata(item, context=f"{context}[]") for item in value
        ]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(
        f"{context} contains unsupported P3S metadata type "
        f"{type(value).__name__}."
    )


def p3s_context(
    *,
    package_version: str,
    resolved_config: Mapping[str, Any],
    resources: Mapping[str, Any],
    observation_spec: Mapping[str, Any],
    source_signature: str,
    raster_names: Sequence[str],
    validation_raster_index: int,
    times: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the versioned metadata stored inside every P3S save state."""

    rendering_model = _safe_metadata(model, context="model")
    # Training/provenance metadata can include long FITS-derived index lists.
    # It does not affect reconstruction or forward rendering.
    rendering_model["run_metadata"] = {}
    return {
        "format": P3S_FORMAT,
        "version": P3S_VERSION,
        "solver": "lte",
        "package_version": str(package_version),
        "resolved_config": _safe_metadata(resolved_config, context="resolved_config"),
        "resources": _safe_metadata(resources, context="resources"),
        "observation": {
            "source_signature": str(source_signature),
            "spec": _safe_metadata(observation_spec, context="observation.spec"),
            "raster_names": _safe_metadata(
                raster_names, context="observation.raster_names"
            ),
            "validation_raster_index": int(validation_raster_index),
            "times": _safe_metadata(times, context="observation.times"),
            "bounds": _safe_metadata(bounds, context="observation.bounds"),
        },
        "model": rendering_model,
    }


def _save_state_manifest(context: Mapping[str, Any]) -> ArtifactManifest:
    expected = {
        "format",
        "version",
        "solver",
        "package_version",
        "resolved_config",
        "resources",
        "observation",
        "model",
    }
    if set(context) != expected:
        raise ArtifactExportError(
            f"P3S context fields do not match format version {P3S_VERSION}."
        )
    if context.get("format") != P3S_FORMAT or context.get("version") != P3S_VERSION:
        raise ArtifactExportError("P3S format or version is unsupported.")
    if context.get("solver") != "lte":
        raise ArtifactExportError("Checkpoint solver must be 'lte'.")
    if not isinstance(context.get("package_version"), str):
        raise ArtifactExportError("Checkpoint package_version must be a string.")
    for name in ("resolved_config", "resources", "observation", "model"):
        if not isinstance(context.get(name), Mapping):
            raise ArtifactExportError(f"Checkpoint {name} must be an object.")
    return ArtifactManifest(
        schema_version=1,
        solver=context["solver"],
        package_version=context["package_version"],
        created_utc="checkpoint",
        resolved_config=deepcopy(dict(context["resolved_config"])),
        resources=deepcopy(dict(context["resources"])),
        observation=deepcopy(dict(context["observation"])),
        model=deepcopy(dict(context["model"])),
        weights_sha256="0" * 64,
    )


def load_validated_save_state(
    save_state_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ValidatedSaveState:
    """Safely load, validate, and reconstruct a P3S file for evaluation."""

    path = Path(save_state_path).expanduser().resolve()
    if path.suffix != ".p3s":
        raise ArtifactExportError("PROM3THEUS save states must use the .p3s extension.")
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
            mmap=True,
        )
    except (
        FileNotFoundError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise ArtifactExportError(f"Could not safely load P3S file: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise ArtifactExportError("P3S file must contain a mapping.")
    context = checkpoint.get(P3S_CONTEXT_KEY)
    if not isinstance(context, Mapping):
        raise ArtifactExportError(
            f"P3S file is missing {P3S_CONTEXT_KEY!r} metadata."
        )
    manifest = _save_state_manifest(context)
    root = path.parent
    config = _validated_config(manifest, root)
    resources = _validated_resources(manifest, config)
    raw_observation = context["observation"]
    observation_fields = {
        "source_signature",
        "spec",
        "raster_names",
        "validation_raster_index",
        "times",
        "bounds",
    }
    if (
        not isinstance(raw_observation, Mapping)
        or set(raw_observation) != observation_fields
    ):
        raise ArtifactExportError("P3S observation context fields are invalid.")
    source_signature = raw_observation["source_signature"]
    if (
        not isinstance(source_signature, str)
        or len(source_signature) != 64
        or any(character not in "0123456789abcdef" for character in source_signature)
    ):
        raise ArtifactExportError("P3S observation source_signature is invalid.")
    spec = _parse_observation_spec(raw_observation["spec"])
    if config.observation.type != spec.observation_type:
        raise ArtifactExportError(
            "P3S resolved observation type differs from its observation spec."
        )
    if config.instrument.type != spec.instrument_type:
        raise ArtifactExportError(
            "P3S resolved instrument type differs from its observation spec."
        )
    model_config = _validate_model_contract(manifest, config, spec)
    expected_fields = {
        P3S_CONTEXT_KEY,
        "parameters",
        "epoch",
        "global_step",
    }
    if set(checkpoint) != expected_fields:
        raise ArtifactExportError(
            "P3S fields do not match the compact format contract; expected only "
            f"{sorted(expected_fields)}."
        )
    names = raw_observation["raster_names"]
    validation_index = raw_observation["validation_raster_index"]
    times = raw_observation["times"]
    bounds = raw_observation["bounds"]
    valid_names = (
        isinstance(names, list)
        and bool(names)
        and all(isinstance(name, str) and bool(name) for name in names)
        and len(set(names)) == len(names)
    )
    valid_time_records = (
        valid_names and isinstance(times, list) and len(times) == len(names)
    )
    if valid_time_records:
        valid_time_records = all(
            isinstance(item, Mapping)
            and set(item) == {"values", "scale"}
            and isinstance(item["values"], list)
            and bool(item["values"])
            and all(
                isinstance(value, str) and bool(value) for value in item["values"]
            )
            and item["scale"] in {"tai", "utc"}
            for item in times
        )
    bound_fields = {
        "surface_longitude_center_rad",
        "surface_longitude_offset_rad",
        "surface_latitude_rad",
        "time_hours",
        "solar_radius_m",
    }
    valid_bounds = isinstance(bounds, Mapping) and set(bounds) == bound_fields
    if valid_bounds:
        scalar_bounds = (
            bounds["surface_longitude_center_rad"],
            bounds["solar_radius_m"],
        )
        range_bounds = (
            bounds["surface_longitude_offset_rad"],
            bounds["surface_latitude_rad"],
            bounds["time_hours"],
        )
        valid_bounds = (
            all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                for value in scalar_bounds
            )
            and float(bounds["solar_radius_m"]) > 0
            and all(
                isinstance(value, list)
                and len(value) == 2
                and all(
                    not isinstance(item, bool)
                    and isinstance(item, (int, float))
                    and math.isfinite(item)
                    for item in value
                )
                for value in range_bounds
            )
        )
    if (
        not valid_names
        or type(validation_index) is not int
        or validation_index < 0
        or validation_index >= len(names)
        or not valid_time_records
        or not valid_bounds
    ):
        raise ArtifactExportError("P3S observation reference is invalid.")
    observation = ObservationReference(
        source_signature=source_signature,
        spec=spec,
        raster_names=tuple(names),
        validation_raster_index=validation_index,
        times=tuple(deepcopy(dict(item)) for item in times),
        bounds=deepcopy(dict(bounds)),
    )
    parameters = checkpoint.get("parameters")
    if not isinstance(parameters, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in parameters.items()
    ):
        raise ArtifactExportError("P3S parameters must contain only tensors.")
    nonfinite = [
        name
        for name, value in parameters.items()
        if (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise ArtifactExportError(
            "P3S parameters contain non-finite tensors: "
            + ", ".join(nonfinite[:12])
        )
    try:
        module = LTEInversionModule(**deepcopy(model_config))
        expected_parameters = dict(module.named_parameters())
        if set(parameters) != set(expected_parameters):
            missing = sorted(set(expected_parameters) - set(parameters))
            unexpected = sorted(set(parameters) - set(expected_parameters))
            raise ValueError(
                f"parameter names differ; missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            for name, parameter in expected_parameters.items():
                stored = parameters[name]
                if stored.shape != parameter.shape or stored.dtype != parameter.dtype:
                    raise ValueError(
                        f"parameter {name!r} has incompatible shape or dtype."
                    )
                parameter.copy_(stored.to(device=parameter.device))
        module.set_save_state_context(context)
        module.eval()
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactExportError(
            f"P3S could not reconstruct LTEInversionModule exactly: {error}"
        ) from error
    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    if type(epoch) is not int or type(global_step) is not int:
        raise ArtifactExportError("P3S epoch and global_step must be integers.")
    return ValidatedSaveState(
        path=path,
        resolved_config=config,
        observation=observation,
        resources=resources,
        module=module,
        epoch=epoch,
        global_step=global_step,
    )


__all__ = [
    "ObservationReference",
    "P3S_FORMAT",
    "P3S_VERSION",
    "ValidatedSaveState",
    "load_validated_save_state",
    "p3s_context",
]
