"""Strict adapter for prepared AIA image-observation stores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import torch

from prom3theus.observations import (
    ImageObservationSpec,
    ImageObservationStore,
    ObservationDescriptor,
    ObservationKind,
    PreparedObservationStream,
    StoredImageDataModule,
)


AIA_INTENSITY_UNIT = "DN s^-1 pixel^-1"
AIA_ASINH_SCALE_ALGORITHM = "median(abs(valid_intensity))"
_AIA_CHANNELS_ANGSTROM = (171, 193, 211)
_AIA_DEGRADATION_REFERENCE_EPOCH = "2010-03-24T00:00:00Z"
_AIA_DEGRADATION_OPERATION = (
    "divide count rate by the degradation factor",
    "divide count rate and uncertainty by the degradation factor",  # Existing stores.
)
_AIA_PREPARATION_FORMAT = "prom3theus.aia_euv.preparation"
_AIA_PREPARATION_VERSION = 1
_AIA_RAY_DIRECTION = "observer_to_sun"
_AIA_SURFACE_FRAME = "HeliographicCarrington Cartesian"
_AIA_SURFACE_POSITION = "photospheric_surface_intersection"
_AIA_TIME_REPRESENTATION = "unix_tai"


@dataclass(frozen=True, slots=True)
class PreparedAIAObservation:
    """Typed result of loading one immutable AIA image store."""

    stream: PreparedObservationStream
    spec: ImageObservationSpec
    asinh_scales: tuple[float, ...]
    intensity_scales: tuple[float, ...]
    store_metadata: Mapping[str, Any]

    @property
    def data_module(self) -> StoredImageDataModule:
        return self.stream.data_module


def robust_asinh_scales(
    rasters: Sequence,
    channels_angstrom: Sequence[int],
) -> dict[str, float]:
    """Derive deterministic non-learnable scales from all valid prepared data."""

    scales: dict[str, float] = {}
    for channel in channels_angstrom:
        selected = [raster for raster in rasters if raster.channel_angstrom == channel]
        if not selected:
            raise ValueError(f"No AIA raster is available for channel {channel}.")
        import numpy as np
        import tempfile
        from pathlib import Path
        count = sum(int(raster.valid_mask[row:row + 256].sum())
                    for raster in selected for row in range(0, raster.intensity.shape[0], 256))
        if count == 0:
            raise ValueError(f"No valid intensities for channel {channel}")
        # Exact lower median, matching torch.median, with bounded owned buffers.
        with tempfile.TemporaryDirectory(prefix="aia-statistics-") as directory:
            values = np.lib.format.open_memmap(Path(directory) / "values.npy", mode="w+",
                                               shape=(count,), dtype=np.float64)
            offset = 0
            for raster in selected:
                for row in range(0, raster.intensity.shape[0], 256):
                    mask = raster.valid_mask[row:row + 256]
                    part = raster.intensity[row:row + 256][mask].double().abs().numpy()
                    values[offset:offset + len(part)] = part
                    offset += len(part)
            middle = (count - 1) // 2
            values.partition(middle)
            value = float(values[middle])
            values._mmap.close()
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"AIA channel {channel} has no positive robust scale.")
        scales[str(int(channel))] = value
    return scales


def aia_objective_statistics(
    rasters: Sequence,
    channels_angstrom: Sequence[int],
) -> dict[str, Any]:
    """Return the exact store metadata block used by the image objective."""

    return {
        "asinh_scale_algorithm": AIA_ASINH_SCALE_ALGORITHM,
        "asinh_scale_by_channel_dn_s_pixel": robust_asinh_scales(
            rasters, channels_angstrom
        ),
    }


def channel_intensity_scales(
    rasters: Sequence,
    channels_angstrom: Sequence[int],
) -> dict[int, float]:
    """Fixed maximum absolute valid intensity per channel; never clip pixels."""

    scales = {}
    for channel in channels_angstrom:
        maxima = [
            float(raster.intensity[row:row + 256][raster.valid_mask[row:row + 256]].abs().max())
            for raster in rasters if raster.channel_angstrom == channel
            for row in range(0, raster.intensity.shape[0], 256)
            if raster.valid_mask[row:row + 256].any()
        ]
        if not maxima:
            raise ValueError(f"No AIA raster is available for channel {channel}.")
        maximum = max(maxima)
        scales[int(channel)] = maximum if maximum > 0.0 else 1.0
    return scales


def _spec_from_metadata(value: Mapping[str, Any]) -> ImageObservationSpec:
    expected = {
        "observation_id",
        "observation_type",
        "instrument_type",
        "observation_kind",
        "intensity_unit",
        "channels_angstrom",
        "exposure_groups",
        "calibration_convention",
        "geometry_convention",
        "required_resource_sets",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Stored AIA observation metadata has an invalid schema.")
    if value["observation_kind"] != ObservationKind.IMAGE.value:
        raise ValueError("Stored AIA observation must declare observation_kind=image.")
    return ImageObservationSpec(
        observation_id=str(value["observation_id"]),
        observation_type=str(value["observation_type"]),
        instrument_type=str(value["instrument_type"]),
        intensity_unit=str(value["intensity_unit"]),
        channels_angstrom=tuple(value["channels_angstrom"]),
        exposure_groups=tuple(value["exposure_groups"]),
        calibration_convention=dict(value["calibration_convention"]),
        geometry_convention=dict(value["geometry_convention"]),
        required_resource_sets=tuple(value["required_resource_sets"]),
    )


def _exact_mapping(value: Any, fields: set[str], description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{description} has an invalid schema.")
    return value


def _sha256_convention_id(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{description} must be a lowercase sha256:<digest> ID.")
    return value


def _source_sha256(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "Prepared AIA source_files_sha256 must be a lowercase SHA-256 digest."
        )
    return value


def _positive_number(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be finite and positive.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{description} must be finite and positive.")
    return result


def _validate_spec_conventions(
    spec: ImageObservationSpec, *, expected_calibration_convention_id: str
) -> None:
    calibration = _exact_mapping(
        spec.calibration_convention,
        {
            "calibration_convention_id",
            "degradation_correction",
            "measurement_semantics",
            "sensitivity_convention",
        },
        "Prepared AIA observation calibration convention",
    )
    if calibration["calibration_convention_id"] != expected_calibration_convention_id:
        raise ValueError(
            "Prepared AIA degradation/calibration convention does not match the "
            "temperature-response resource."
        )
    degradation = _exact_mapping(
        calibration["degradation_correction"],
        {"operation", "reference_epoch", "total_applications_per_raster"},
        "Prepared AIA observation degradation convention",
    )
    if (
        degradation["operation"] not in _AIA_DEGRADATION_OPERATION
        or degradation["reference_epoch"] != _AIA_DEGRADATION_REFERENCE_EPOCH
        or type(degradation["total_applications_per_raster"]) is not int
        or degradation["total_applications_per_raster"] != 1
    ):
        raise ValueError(
            "Prepared AIA observation must declare the exact once-only "
            "degradation convention."
        )
    if (
        calibration["measurement_semantics"] != "per_native_pixel"
        or calibration["sensitivity_convention"] != "reference_epoch"
    ):
        raise ValueError("Prepared AIA observation calibration semantics are invalid.")

    geometry = _exact_mapping(
        spec.geometry_convention,
        {"ray_direction", "surface_position", "surface_frame", "time_representation"},
        "Prepared AIA observation geometry convention",
    )
    if (
        geometry["ray_direction"] != _AIA_RAY_DIRECTION
        or geometry["surface_position"] != _AIA_SURFACE_POSITION
        or geometry["surface_frame"] != _AIA_SURFACE_FRAME
        or geometry["time_representation"] != _AIA_TIME_REPRESENTATION
    ):
        raise ValueError(
            "Prepared AIA observation geometry/time convention is invalid."
        )


def _validate_correction_record(
    record: Any,
    *,
    expected_identity: tuple[str, int] | None = None,
) -> tuple[tuple[str, int], dict[str, Any]]:
    record = _exact_mapping(
        record,
        {
            "exposure_group",
            "channel_angstrom",
            "degradation_factor",
            "input_correction_applications",
            "correction_performed_by_preparation",
            "total_correction_applications",
        },
        "Prepared AIA degradation record",
    )
    group = record["exposure_group"]
    channel = record["channel_angstrom"]
    if not isinstance(group, str) or not group or group != group.strip():
        raise ValueError(
            "Prepared AIA degradation record has an invalid exposure group."
        )
    if type(channel) is not int or channel <= 0:
        raise ValueError("Prepared AIA degradation record has an invalid channel.")
    identity = (group, channel)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Prepared AIA degradation record does not match its raster.")
    factor = _positive_number(
        record["degradation_factor"], "Prepared AIA degradation factor"
    )
    applications = record["input_correction_applications"]
    performed = record["correction_performed_by_preparation"]
    total = record["total_correction_applications"]
    if type(applications) is not int or applications not in (0, 1):
        raise ValueError(
            "Prepared AIA degradation input application count must be 0 or 1."
        )
    if type(performed) is not bool or performed is not (applications == 0):
        raise ValueError(
            "Prepared AIA degradation performed flag is inconsistent with its "
            "input application count."
        )
    if type(total) is not int or total != 1:
        raise ValueError(
            "Prepared AIA degradation correction must total exactly one application."
        )
    return identity, {
        "degradation_factor": factor,
        "input_correction_applications": applications,
        "correction_performed_by_preparation": performed,
        "total_correction_applications": total,
    }


def _validate_raster_preparation_metadata(
    raster,
    correction_record: Mapping[str, Any],
    *,
    expected_calibration_convention_id: str,
) -> None:
    metadata = _exact_mapping(
        raster.metadata,
        {
            "intensity_unit",
            "observation_time",
            "ray_geometry",
            "calibration",
            "mask",
            "provenance",
        },
        "Prepared AIA raster metadata",
    )
    if metadata["intensity_unit"] != AIA_INTENSITY_UNIT:
        raise ValueError("Prepared AIA raster intensity unit is invalid.")
    time = _exact_mapping(
        metadata["observation_time"],
        {"representation", "absolute_tai_seconds"},
        "Prepared AIA raster observation time",
    )
    if (
        time["representation"] != _AIA_TIME_REPRESENTATION
        or isinstance(time["absolute_tai_seconds"], bool)
        or not math.isfinite(float(time["absolute_tai_seconds"]))
        or float(time["absolute_tai_seconds"]) != raster.absolute_tai_seconds
    ):
        raise ValueError(
            "Prepared AIA raster time does not match its exact unix_tai record."
        )

    geometry = _exact_mapping(
        metadata["ray_geometry"],
        {"solar_radius_m", "frame", "surface_position", "ray_direction"},
        "Prepared AIA raster ray geometry",
    )
    radius = _positive_number(
        geometry["solar_radius_m"], "Prepared AIA raster solar radius"
    )
    if (
        geometry["frame"] != _AIA_SURFACE_FRAME
        or geometry["surface_position"] != _AIA_SURFACE_POSITION
        or geometry["ray_direction"] != _AIA_RAY_DIRECTION
    ):
        raise ValueError("Prepared AIA raster geometry convention is invalid.")
    for row in range(0, raster.intensity.shape[0], 256):
        mask = raster.valid_mask[row:row + 256]
        surface_radius = torch.linalg.vector_norm(
            raster.surface_position_m[row:row + 256][mask].to(torch.float64), dim=-1
        )
        if not torch.allclose(surface_radius, torch.full_like(surface_radius, radius),
                              rtol=2.0e-5, atol=1.0):
            raise ValueError("Prepared AIA raster geometry metadata is inconsistent.")

    calibration = _exact_mapping(
        metadata["calibration"],
        {
            "calibration_convention_id",
            "degradation_factor",
            "input_correction_applications",
            "correction_performed_by_preparation",
            "total_correction_applications",
            "operation",
            "reference_epoch",
        },
        "Prepared AIA raster calibration",
    )
    if (
        calibration["calibration_convention_id"] != expected_calibration_convention_id
        or calibration["operation"] not in _AIA_DEGRADATION_OPERATION
        or calibration["reference_epoch"] != _AIA_DEGRADATION_REFERENCE_EPOCH
    ):
        raise ValueError("Prepared AIA raster calibration convention is invalid.")
    for field in (
        "input_correction_applications",
        "correction_performed_by_preparation",
        "total_correction_applications",
    ):
        if calibration[field] != correction_record[field] or type(
            calibration[field]
        ) is not type(correction_record[field]):
            raise ValueError(
                "Prepared AIA raster calibration does not match its degradation record."
            )
    if (
        _positive_number(
            calibration["degradation_factor"],
            "Prepared AIA raster degradation factor",
        )
        != correction_record["degradation_factor"]
    ):
        raise ValueError(
            "Prepared AIA raster degradation factor does not match its record."
        )

    mask = _exact_mapping(
        metadata["mask"],
        {"definition", "on_disk_pixel_count", "valid_on_disk_pixel_count"},
        "Prepared AIA raster mask",
    )
    valid_count = sum(int(raster.valid_mask[row:row + 256].sum())
                      for row in range(0, raster.intensity.shape[0], 256))
    if (
        mask["definition"] != "upstream_valid_mask AND on_disk_mask"
        or type(mask["on_disk_pixel_count"]) is not int
        or type(mask["valid_on_disk_pixel_count"]) is not int
        or mask["valid_on_disk_pixel_count"] != valid_count
        or mask["on_disk_pixel_count"] < valid_count
        or mask["on_disk_pixel_count"] > raster.valid_mask.numel()
    ):
        raise ValueError("Prepared AIA raster mask provenance is inconsistent.")
    if not isinstance(metadata["provenance"], Mapping):
        raise ValueError("Prepared AIA raster provenance must be a mapping.")


def _validate_preparation_metadata(
    value: Any,
    *,
    spec: ImageObservationSpec,
    rasters: Sequence,
    expected_calibration_convention_id: str,
) -> None:
    preparation = _exact_mapping(
        value,
        {
            "format",
            "version",
            "channels_angstrom",
            "exposure_groups",
            "output_intensity_unit",
            "calibration_convention_id",
            "time_representation",
            "surface_frame",
            "ray_direction",
            "degradation_correction",
            "preparation_dependencies",
        },
        "Prepared AIA preparation metadata",
    )
    if (
        preparation["format"] != _AIA_PREPARATION_FORMAT
        or type(preparation["version"]) is not int
        or preparation["version"] != _AIA_PREPARATION_VERSION
    ):
        raise ValueError("Prepared AIA preparation format/version is unsupported.")
    if preparation["channels_angstrom"] != list(spec.channels_angstrom):
        raise ValueError("Prepared AIA preparation channels do not match the store.")
    if preparation["exposure_groups"] != list(spec.exposure_groups):
        raise ValueError(
            "Prepared AIA preparation exposure groups do not match the store."
        )
    if preparation["output_intensity_unit"] != AIA_INTENSITY_UNIT:
        raise ValueError("Prepared AIA preparation output unit is invalid.")
    if preparation["calibration_convention_id"] != expected_calibration_convention_id:
        raise ValueError(
            "Prepared AIA preparation calibration convention does not match the "
            "temperature-response resource."
        )
    if (
        preparation["time_representation"] != _AIA_TIME_REPRESENTATION
        or preparation["surface_frame"] != _AIA_SURFACE_FRAME
        or preparation["ray_direction"] != _AIA_RAY_DIRECTION
    ):
        raise ValueError(
            "Prepared AIA preparation geometry/time convention is invalid."
        )
    dependencies = preparation["preparation_dependencies"]
    if not isinstance(dependencies, Mapping) or any(
        not isinstance(key, str) for key in dependencies
    ):
        raise ValueError("Prepared AIA preparation dependencies must be a mapping.")

    degradation = _exact_mapping(
        preparation["degradation_correction"],
        {
            "operation",
            "reference_epoch",
            "total_applications_per_raster",
            "records",
        },
        "Prepared AIA degradation metadata",
    )
    if (
        degradation["operation"] not in _AIA_DEGRADATION_OPERATION
        or degradation["reference_epoch"] != _AIA_DEGRADATION_REFERENCE_EPOCH
        or type(degradation["total_applications_per_raster"]) is not int
        or degradation["total_applications_per_raster"] != 1
        or not isinstance(degradation["records"], list)
    ):
        raise ValueError(
            "Prepared AIA degradation metadata must declare exactly one correction."
        )
    records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in degradation["records"]:
        identity, normalized = _validate_correction_record(record)
        if identity in records:
            raise ValueError("Prepared AIA degradation records contain duplicates.")
        records[identity] = normalized

    expected_identities = {
        (group, channel)
        for group in spec.exposure_groups
        for channel in spec.channels_angstrom
    }
    raster_identities = {
        (raster.exposure_group, raster.channel_angstrom) for raster in rasters
    }
    if set(records) != expected_identities or raster_identities != expected_identities:
        raise ValueError(
            "Prepared AIA degradation records must cover every configured "
            "exposure-group/channel pair exactly."
        )
    if len(rasters) != len(raster_identities):
        raise ValueError("Prepared AIA rasters contain duplicate identities.")
    for raster in rasters:
        identity = (raster.exposure_group, raster.channel_angstrom)
        _validate_raster_preparation_metadata(
            raster,
            records[identity],
            expected_calibration_convention_id=expected_calibration_convention_id,
        )


def load_prepared_aia_observation(
    config: Mapping[str, Any],
    *,
    expected_calibration_convention_id: str,
    time_bounds_tai: tuple[float, float] | None = None,
) -> PreparedAIAObservation:
    """Load one verified AIA store and bind only loader/runtime selections."""

    expected_calibration_convention_id = _sha256_convention_id(
        expected_calibration_convention_id,
        "expected_calibration_convention_id",
    )
    raw = dict(config)
    expected = {"type", "directory", "channels_angstrom", "selection", "loader"}
    if set(raw) != expected:
        raise TypeError(
            "AIA observation config has "
            f"missing={sorted(expected - set(raw))}, unknown={sorted(set(raw) - expected)}."
        )
    if raw["type"] != "aia_euv":
        raise ValueError("The AIA adapter requires observation.type=aia_euv.")
    root = Path(raw["directory"]).expanduser().resolve()
    manifest = ImageObservationStore.manifest(root)
    rasters, names, metadata = ImageObservationStore.load_sequence(root)
    # Existing stores remain usable, but their noise arrays are no longer inputs.
    from dataclasses import fields
    from prom3theus.observations.loading import _restore_raster, _plain
    rasters = [_restore_raster(type(raster), {
        field.name: None if field.name == "uncertainty" else _plain(getattr(raster, field.name))
        for field in fields(raster)
    }) for raster in rasters]
    metadata_fields = {
        "adapter",
        "observation",
        "source_files_sha256",
        "preparation",
        "objective_statistics",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != metadata_fields:
        raise ValueError("AIA store session metadata has an invalid schema.")
    if metadata["adapter"] != "aia_euv":
        raise ValueError("AIA store adapter identity does not match aia_euv.")
    _source_sha256(metadata["source_files_sha256"])
    spec = _spec_from_metadata(metadata["observation"])
    configured_channels = tuple(int(value) for value in raw["channels_angstrom"])
    if (
        spec.observation_type != "aia_euv"
        or spec.channels_angstrom != configured_channels
    ):
        raise ValueError("Configured AIA channels do not match the prepared store.")
    if spec.intensity_unit != AIA_INTENSITY_UNIT:
        raise ValueError(f"Prepared AIA intensity unit must be {AIA_INTENSITY_UNIT!r}.")
    if "aia_euv_v1" not in spec.required_resource_sets:
        raise ValueError("Prepared AIA data must require resource set aia_euv_v1.")
    _validate_spec_conventions(
        spec,
        expected_calibration_convention_id=expected_calibration_convention_id,
    )
    _validate_preparation_metadata(
        metadata["preparation"],
        spec=spec,
        rasters=rasters,
        expected_calibration_convention_id=expected_calibration_convention_id,
    )

    statistics = metadata["objective_statistics"]
    if not isinstance(statistics, Mapping) or set(statistics) != {
        "asinh_scale_algorithm",
        "asinh_scale_by_channel_dn_s_pixel",
    }:
        raise ValueError("Prepared AIA objective statistics have an invalid schema.")
    legacy_statistics = statistics["asinh_scale_algorithm"] == "max(median(abs(valid_intensity)),median(valid_sigma))"
    if statistics["asinh_scale_algorithm"] != AIA_ASINH_SCALE_ALGORITHM and not legacy_statistics:
        raise ValueError("Prepared AIA asinh-scale algorithm is unsupported.")
    scale_mapping = statistics["asinh_scale_by_channel_dn_s_pixel"]
    if not isinstance(scale_mapping, Mapping) or set(scale_mapping) != {
        str(channel) for channel in configured_channels
    }:
        raise ValueError("Prepared AIA asinh scales must cover every channel exactly.")
    asinh_scales = tuple(
        float(scale_mapping[str(channel)]) for channel in configured_channels
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in asinh_scales):
        raise ValueError("Prepared AIA asinh scales must be finite and positive.")
    expected_scales = robust_asinh_scales(rasters, configured_channels)
    if legacy_statistics:
        asinh_scales = tuple(expected_scales[str(channel)] for channel in configured_channels)
    if any(
        not math.isclose(
            asinh_scales[index],
            expected_scales[str(channel)],
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
        for index, channel in enumerate(configured_channels)
    ):
        raise ValueError(
            "Prepared AIA asinh scales are inconsistent with the stored rasters."
        )

    selection = dict(raw["selection"])
    loader = dict(raw["loader"])
    if set(selection) != {"validation_exposure_group"}:
        raise TypeError("AIA selection must contain validation_exposure_group only.")
    if set(loader) != {"batch_size", "validation_batch_size", "workers", "pin_memory"}:
        raise TypeError("AIA loader options do not match the current exact contract.")
    validation_group = str(selection["validation_exposure_group"])
    if time_bounds_tai is not None:
        start, end = time_bounds_tai
        # Keep a group only if every channel's actual exposure time is in range.
        groups = tuple(group for group in spec.exposure_groups if all(
            start <= raster.absolute_tai_seconds < end
            for raster in rasters if raster.exposure_group == group
        ))
        if not groups:
            raise ValueError("No complete AIA exposure groups in the configured time window.")
        selected = [(raster, name) for raster, name in zip(rasters, names)
                    if raster.exposure_group in groups]
        rasters, names = map(list, zip(*selected))
        spec = replace(spec, exposure_groups=groups)
        if validation_group not in groups:
            validation_group = groups[0]
    intensity_scale_mapping = channel_intensity_scales(rasters, configured_channels)
    data = StoredImageDataModule(
        rasters,
        names,
        spec,
        validation_exposure_group=validation_group,
        intensity_scales=intensity_scale_mapping,
        store_metadata=metadata,
        batch_size=int(loader["batch_size"]),
        validation_batch_size=int(loader["validation_batch_size"]),
        num_workers=int(loader["workers"]),
        pin_memory=bool(loader["pin_memory"]),
    )
    data.observation_store_path = root
    data.setup("fit")
    descriptor = ObservationDescriptor.from_spec(spec)
    stream = PreparedObservationStream(
        name=spec.observation_id,
        descriptor=descriptor,
        data_module=data,
        store_path=root,
        source_signature=manifest["source_signature"],
    )
    return PreparedAIAObservation(
        stream=stream,
        spec=spec,
        asinh_scales=asinh_scales,
        intensity_scales=tuple(intensity_scale_mapping[channel] for channel in configured_channels),
        store_metadata=dict(metadata),
    )


__all__ = [
    "AIA_ASINH_SCALE_ALGORITHM",
    "AIA_INTENSITY_UNIT",
    "PreparedAIAObservation",
    "aia_objective_statistics",
    "channel_intensity_scales",
    "load_prepared_aia_observation",
    "robust_asinh_scales",
]
