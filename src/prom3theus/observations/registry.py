"""Closed public registry for observation adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any

from prom3theus.core import sha256_file_set

from .contracts import ObservationSpec
from .contracts import velocity_synthesis_contract
from .data import StoredObservationDataModule
from .store import ObservationStore, observation_store_signature


@dataclass(frozen=True, slots=True)
class ObservationAdapter:
    """Immutable lazy bridge from one observation discriminator to its adapter."""

    name: str
    build: Callable[[Mapping[str, Any]], Any]
    describe: Callable[[Any], ObservationSpec]
    source_files: Callable[[Mapping[str, Any]], tuple[Path, ...]]


def _build_hinode(config: Mapping[str, Any]) -> Any:
    from prom3theus.instruments.hinode_sp.observation import _build_data

    return _build_data(config)


def _describe_hinode(data: Any) -> ObservationSpec:
    from prom3theus.instruments.hinode_sp.observation import _describe_data

    return _describe_data(data)


def _hinode_source_files(config: Mapping[str, Any]) -> tuple[Path, ...]:
    from prom3theus.instruments.hinode_sp.data import resolve_single_raster_files

    return tuple(resolve_single_raster_files(directory=config["directory"]))


def _build_hmi(config: Mapping[str, Any]) -> Any:
    from prom3theus.instruments.hmi.observation import _build_data

    return _build_data(config)


def _describe_hmi(data: Any) -> ObservationSpec:
    from prom3theus.instruments.hmi.observation import _describe_data

    return _describe_data(data)


def _hmi_source_files(config: Mapping[str, Any]) -> tuple[Path, ...]:
    from prom3theus.instruments.hmi.acquisition import resolve_acquisition_groups
    from prom3theus.instruments.hmi.response import (
        MANIFEST_NAME,
        load_response_manifest,
    )

    groups = resolve_acquisition_groups(directory=config["directory"])
    calibration = config["calibration"]
    response_root = Path(calibration["transmission_profile_directory"]).resolve()
    # Validation hashes every response archive against this manifest. The
    # aggregate source signature only needs the validated manifest itself,
    # avoiding a second read of potentially large calibration arrays.
    load_response_manifest(response_root)
    return (
        *(path for _, paths in groups for path in paths),
        response_root / MANIFEST_NAME,
    )


_ADAPTERS: Mapping[str, ObservationAdapter] = MappingProxyType(
    {
        "hinode_sp": ObservationAdapter(
            "hinode_sp", _build_hinode, _describe_hinode, _hinode_source_files
        ),
        "hmi_stokes": ObservationAdapter(
            "hmi_stokes", _build_hmi, _describe_hmi, _hmi_source_files
        ),
    }
)


def get_observation_adapter(name: str) -> ObservationAdapter:
    """Return one exact supported observation discriminator."""

    try:
        return _ADAPTERS[name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Unknown observation type {name!r}; expected one of {sorted(_ADAPTERS)}."
        ) from error


def build_observation_data(
    config: Mapping[str, Any],
) -> tuple[Any, ObservationSpec, ObservationAdapter]:
    """Build, prepare, and describe one configured observation."""
    resolved = dict(config)
    try:
        observation_type = resolved.pop("type")
    except KeyError as error:
        raise ValueError("observation.type is required.") from error
    adapter = get_observation_adapter(str(observation_type))
    data = adapter.build(resolved)
    data.setup("fit")
    return data, adapter.describe(data), adapter


def describe_observation_data(
    data, config: Mapping[str, Any]
) -> tuple[ObservationSpec, ObservationAdapter]:
    resolved = dict(config)
    try:
        observation_type = resolved.pop("type")
    except KeyError as error:
        raise ValueError("observation.type is required.") from error
    adapter = get_observation_adapter(str(observation_type))
    for attribute in ("raster", "run_metadata"):
        if not hasattr(data, attribute):
            raise TypeError(
                f"Observation adapter {adapter.name!r} returned data without {attribute!r}."
            )
    return adapter.describe(data), adapter


def _spec_from_metadata(metadata: Mapping[str, Any]) -> ObservationSpec:
    expected = {
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
    if not isinstance(metadata, Mapping) or set(metadata) != expected:
        raise ValueError(
            "Stored observation metadata must match the exact current contract."
        )
    support = metadata["line_support_angstrom"]
    spec = ObservationSpec(
        observation_id=str(metadata["observation_id"]),
        observation_type=str(metadata["observation_type"]),
        instrument_type=str(metadata["instrument_type"]),
        wavelength_angstrom=metadata["wavelength_angstrom"],
        continuum_indices=tuple(metadata["continuum_indices"]),
        radiance_scale_w_m3_sr=float(metadata["radiance_scale_w_m3_sr"]),
        velocity_synthesis_mode=str(metadata["velocity_synthesis_mode"]),
        required_line_ids=tuple(metadata["required_line_ids"]),
        line_support_angstrom=None if support is None else tuple(support),
        instrument_options=dict(metadata["instrument_options"]),
    )
    if metadata["velocity_synthesis"] != velocity_synthesis_contract(
        spec.velocity_synthesis_mode
    ):
        raise ValueError("Stored observation velocity metadata is inconsistent.")
    return spec


def _stored_data_module(
    rasters,
    raster_names,
    store_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    store_path: Path,
) -> StoredObservationDataModule:
    expected_metadata = {
        "adapter",
        "observation",
        "source_files_sha256",
        "validation_raster_index",
        "fits_consistency",
    }
    if set(store_metadata) != expected_metadata:
        raise ValueError(
            "Observation-store session metadata must match the exact current contract."
        )
    loader = dict(config["loader"])
    data = StoredObservationDataModule(
        rasters,
        raster_names,
        validation_raster_index=int(store_metadata["validation_raster_index"]),
        store_metadata=store_metadata,
        batch_size=int(loader["batch_size"]),
        validation_batch_size=int(loader["validation_batch_size"]),
        validation_stride=int(loader["validation_stride"]),
        num_workers=int(loader["workers"]),
        pin_memory=bool(loader["pin_memory"]),
    )
    data.observation_store_path = store_path
    data.setup("fit")
    return data


def load_or_prepare_observation(
    config: Mapping[str, Any],
    resource_metadata: Mapping[str, Any],
    cache_directory: str | Path,
    *,
    rebuild: bool = False,
) -> tuple[StoredObservationDataModule, ObservationSpec, ObservationAdapter]:
    """Load a memory-mapped sequence or prepare and atomically store it.

    The cache contains no data-module pickle.  A small generic data module is
    reconstructed from canonical raster arrays on every process.
    """
    config = dict(config)
    try:
        observation_type = str(config["type"])
    except KeyError as error:
        raise ValueError("observation.type is required.") from error
    adapter = get_observation_adapter(observation_type)
    source_files_sha256 = sha256_file_set(adapter.source_files(config))
    signature = observation_store_signature(
        config,
        resource_metadata,
        source_files_sha256=source_files_sha256,
    )
    cache_root = Path(cache_directory).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    store_path = cache_root / f"{adapter.name}-{signature}.observation"
    if rebuild and store_path.exists():
        shutil.rmtree(store_path)

    if not store_path.exists():
        prepared, spec, prepared_adapter = build_observation_data(config)
        if prepared_adapter.name != adapter.name:
            raise RuntimeError(
                "Prepared observation adapter does not match its discriminator."
            )
        session_metadata = {
            "adapter": adapter.name,
            "observation": spec.metadata(),
            "source_files_sha256": source_files_sha256,
            "validation_raster_index": int(prepared.validation_raster_index),
            "fits_consistency": dict(prepared.fits_consistency_metadata or {}),
        }
        try:
            ObservationStore.save_sequence(
                store_path,
                prepared.rasters,
                raster_names=prepared.raster_names,
                source_signature=signature,
                metadata=session_metadata,
            )
        except FileExistsError:
            # Another rank/process completed the same immutable store first.
            pass

    rasters, names, store_metadata = ObservationStore.load_sequence(
        store_path,
        expected_signature=signature,
    )
    if store_metadata["adapter"] != adapter.name:
        raise ValueError("Observation store adapter does not match the requested type.")
    spec = _spec_from_metadata(store_metadata["observation"])
    data = _stored_data_module(rasters, names, store_metadata, config, store_path)
    return data, spec, adapter


__all__ = [
    "ObservationAdapter",
    "build_observation_data",
    "describe_observation_data",
    "get_observation_adapter",
    "load_or_prepare_observation",
]
