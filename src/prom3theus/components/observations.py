"""Load observation streams and derive their shared scene bounds."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any


from prom3theus.config.joint_schema import AIAObservationConfig, ObservationStreamConfig
from prom3theus.core.parallel import parallel_map
from prom3theus.instruments.aia_euv.observation import load_prepared_aia_observation
from prom3theus.observations import (
    ObservationDescriptor,
    ObservationKind,
    ObservationStore,
    PreparedObservationStream,
    load_or_prepare_observation,
)
from prom3theus.resources import LEGACY_LTE_RESOURCE_SET_ID

from prom3theus.application.joint_contracts import LoadedJointStream


def _required_resource_sets(config):
    requested = [LEGACY_LTE_RESOURCE_SET_ID]
    for stream in config.streams:
        requested.extend(_PROVIDERS[stream.observation.type].resources(stream))
    return tuple(dict.fromkeys(requested))


def _load_image_stream(
    stream: ObservationStreamConfig,
    resources: Mapping[str, Mapping[str, Any]],
    *,
    time_bounds_tai=None,
) -> LoadedJointStream:
    observation = stream.observation
    if not isinstance(observation, AIAObservationConfig):
        raise TypeError("Expected an AIA image observation stream.")
    set_id = stream.data_term.synthesis.response_resource.split(":", 1)[0]
    try:
        convention_id = resources[set_id]["scientific_contract"][
            "calibration_convention_id"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Verified AIA resources lack their calibration convention identity."
        ) from error
    loaded = load_prepared_aia_observation(
        observation.to_dict(),
        expected_calibration_convention_id=str(convention_id),
        time_bounds_tai=time_bounds_tai,
    )
    prepared_source = loaded.stream
    prepared = PreparedObservationStream(
        name=stream.id,
        descriptor=prepared_source.descriptor,
        data_module=prepared_source.data_module,
        store_path=prepared_source.store_path,
        source_signature=prepared_source.source_signature,
    )
    statistics = loaded.store_metadata["objective_statistics"]
    return LoadedJointStream(
        prepared=prepared,
        specification=loaded.spec,
        rasters=tuple(loaded.data_module.rasters),
        store_metadata=loaded.store_metadata,
        setup_metadata={
            "asinh_scales": loaded.asinh_scales,
            "intensity_scales": loaded.intensity_scales,
            "intensity_scale_provenance": {
                "source": "selected_time_window_rasters",
                "algorithm": "max(abs(valid_intensity)) per channel; 1 if all zero",
                "values_by_channel_angstrom": {
                    str(channel): scale
                    for channel, scale in zip(
                        loaded.spec.channels_angstrom,
                        loaded.intensity_scales,
                        strict=True,
                    )
                },
                "unit": loaded.spec.intensity_unit,
                "normalized_intensity_unit": "dimensionless",
                "frozen_for_future_optimization": True,
            },
            "asinh_scale_provenance": {
                "source": "prepared_store.objective_statistics",
                "algorithm": statistics["asinh_scale_algorithm"],
                "values_by_channel_angstrom": dict(
                    statistics["asinh_scale_by_channel_dn_s_pixel"]
                ),
                "frozen_for_future_optimization": True,
            },
        },
    )


def _load_stokes_stream(
    stream: ObservationStreamConfig,
    resources: Mapping[str, Mapping[str, Any]],
    work_directory: Path,
    *,
    rebuild_observations: bool,
) -> LoadedJointStream:
    data_module, specification, _ = load_or_prepare_observation(
        stream.observation.to_dict(),
        resources[LEGACY_LTE_RESOURCE_SET_ID],
        work_directory / "observation-cache" / stream.id,
        rebuild=rebuild_observations,
    )
    data_module.setup("validate")
    store_path = Path(data_module.observation_store_path).expanduser().resolve()
    manifest = ObservationStore.manifest(store_path)
    descriptor = ObservationDescriptor.from_spec(
        specification,
        observation_kind=ObservationKind.STOKES,
        required_resource_sets=(LEGACY_LTE_RESOURCE_SET_ID,),
    )
    prepared = PreparedObservationStream(
        name=stream.id,
        descriptor=descriptor,
        data_module=data_module,
        store_path=store_path,
        source_signature=manifest["source_signature"],
    )
    return LoadedJointStream(
        prepared=prepared,
        specification=specification,
        rasters=tuple(data_module.rasters),
        store_metadata=getattr(data_module, "store_metadata", {}),
    )


def _select_hmi_time_window(stream, bounds):
    """Resolve the time window to HMI's existing acquisition-index selection."""
    from astropy.time import Time
    from prom3theus.instruments.hmi.acquisition import (
        resolve_acquisition_groups,
        read_acquisition_header,
    )

    observation = stream.observation
    groups = resolve_acquisition_groups(directory=observation.directory)
    headers = parallel_map(
        read_acquisition_header,
        [paths[0] for _, paths in groups],
        description="HMI time selection",
    )
    selection = observation.selection
    indices = tuple(
        index
        for index, header in enumerate(headers)
        if (
            selection.acquisition_indices is None
            or index in selection.acquisition_indices
        )
        and bounds[0]
        <= float(Time(header["date"], scale="tai").to_value("unix_tai"))
        < bounds[1]
    )
    if not indices:
        raise ValueError(
            f"No HMI acquisitions in the configured time window for {stream.id}."
        )
    validation = (
        selection.validation_raster
        if selection.validation_raster in indices
        else indices[0]
    )
    return replace(
        stream,
        observation=replace(
            observation,
            selection=replace(
                selection,
                acquisition_indices=indices,
                validation_raster=validation,
            ),
        ),
    )


@dataclass(frozen=True)
class ObservationProvider:
    load: object
    resources: object


def _load_aia(stream, resources, work_directory, *, rebuild_observations, bounds):
    return _load_image_stream(stream, resources, time_bounds_tai=bounds)


def _load_hmi(stream, resources, work_directory, *, rebuild_observations, bounds):
    if bounds is not None:
        stream = _select_hmi_time_window(stream, bounds)
    return _load_stokes_stream(
        stream, resources, work_directory, rebuild_observations=rebuild_observations
    )


def _load_hinode(stream, resources, work_directory, *, rebuild_observations, bounds):
    if bounds is not None:
        raise ValueError(
            "Hinode time-window selection requires a scan-time adapter; select its raster explicitly."
        )
    return _load_stokes_stream(
        stream, resources, work_directory, rebuild_observations=rebuild_observations
    )


_PROVIDERS = {
    "hmi_stokes": ObservationProvider(
        _load_hmi, lambda stream: (LEGACY_LTE_RESOURCE_SET_ID,)
    ),
    "hinode_sp": ObservationProvider(
        _load_hinode,
        lambda stream: (LEGACY_LTE_RESOURCE_SET_ID,),
    ),
    "aia_euv": ObservationProvider(
        _load_aia,
        lambda stream: (stream.data_term.synthesis.response_resource.split(":", 1)[0],),
    ),
}


def register_observation_provider(name, provider):
    if name in _PROVIDERS:
        raise ValueError(f"Observation provider already registered: {name}")
    if not isinstance(provider, ObservationProvider):
        raise TypeError("Expected ObservationProvider")
    _PROVIDERS[name] = provider


def _default_stream_loader(
    stream, resources, work_directory, *, rebuild_observations, time_window=None
):
    bounds = None
    if time_window is not None:
        from astropy.time import Time

        bounds = tuple(
            float(Time(datetime.fromisoformat(value)).tai.to_value("unix_tai"))
            for value in (time_window.start, time_window.end)
        )
    provider = _PROVIDERS[stream.observation.type]
    return provider.load(
        stream,
        resources,
        work_directory,
        rebuild_observations=rebuild_observations,
        bounds=bounds,
    )
