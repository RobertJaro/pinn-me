"""Offline preparation boundary for strict AIA image-observation stores.

This module intentionally does not import SunPy or aiapy.  Acquisition, map
construction, calibration, and registration can be supplied by callers as
dependency-injected callables; the final scientific boundary is a small typed
set of tensors whose units and conventions can be validated before an
immutable :class:`~prom3theus.observations.ImageObservationStore` is published.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import numpy as np
from prom3theus.core.parallel import parallel_map
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

import torch

from prom3theus.observations import (
    ImageObservationRaster,
    ImageObservationSpec,
    ImageObservationStore,
    image_observation_store_signature,
)
from prom3theus.resources import AIA_RESOURCE_SET_ID, validate_resource_set

from .observation import AIA_INTENSITY_UNIT, AIA_ASINH_SCALE_ALGORITHM


AIA_CHANNELS_ANGSTROM = (171, 193, 211)
AIA_DEGRADATION_REFERENCE_EPOCH = "2010-03-24T00:00:00Z"
AIA_PREPARATION_FORMAT = "prom3theus.aia_euv.preparation"
AIA_PREPARATION_VERSION = 1
AIA_SURFACE_FRAME = "HeliographicCarrington Cartesian"
AIA_RAY_DIRECTION_CONVENTION = "observer_to_sun"
AIA_TIME_REPRESENTATION = "unix_tai"
AIA_DEGRADATION_OPERATION = (
    "divide count rate by the degradation factor"
)


_T = TypeVar("_T")
_M = TypeVar("_M")


class AIAMapBackend(Protocol[_T, _M]):
    """Callable converting one source product into an instrument map."""

    def __call__(self, source_product: _T) -> _M: ...


class AIAPreparationBackend(Protocol[_M]):
    """Callable converting maps into fully calibrated/registered rasters."""

    def __call__(
        self, maps: Sequence[_M]
    ) -> Sequence["RegisteredAIAChannelRaster"]: ...


def _metadata(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys.")
    return MappingProxyType(dict(value))


def _sha256_convention_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be a lowercase sha256:<digest> identifier.")
    return value


def _source_sha256(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("source_files_sha256 must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True, slots=True)
class RegisteredAIAChannelRaster:
    """One independently calibrated and registered AIA channel raster.

    ``valid_mask`` represents upstream data-quality validity and
    ``on_disk_mask`` represents the WCS-derived photospheric intersection.
    Their conjunction becomes the training mask.  The degradation factor is
    channel/time specific.  Inputs may be either uncorrected (application count
    zero) or already corrected once; publication always produces exactly-one
    correction provenance and never applies the factor twice.
    """

    intensity: torch.Tensor
    valid_mask: torch.Tensor
    on_disk_mask: torch.Tensor
    ray_direction: torch.Tensor
    surface_position_m: torch.Tensor
    absolute_tai_seconds: float
    exposure_group: str
    channel_angstrom: int
    solar_radius_m: float
    degradation_factor: float
    degradation_correction_applications: int
    calibration_convention_id: str
    intensity_unit: str
    time_representation: str
    surface_frame: str
    ray_direction_convention: str
    raster_name: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: torch.Tensor | None = None  # Legacy caller input; not used by AIA.

    def __post_init__(self) -> None:
        if type(self.channel_angstrom) is not int or self.channel_angstrom <= 0:
            raise ValueError("channel_angstrom must be a positive integer.")
        if not isinstance(self.exposure_group, str) or not self.exposure_group:
            raise ValueError("exposure_group must be a non-empty string.")
        if self.exposure_group != self.exposure_group.strip():
            raise ValueError("exposure_group cannot have surrounding whitespace.")
        if self.raster_name is not None and (
            not isinstance(self.raster_name, str) or not self.raster_name.strip()
        ):
            raise ValueError("raster_name must be None or a non-empty string.")
        if self.intensity_unit != AIA_INTENSITY_UNIT:
            raise ValueError(
                f"AIA input intensity_unit must be {AIA_INTENSITY_UNIT!r}."
            )
        if self.time_representation != AIA_TIME_REPRESENTATION:
            raise ValueError("AIA observation times must use unix_tai seconds.")
        if self.surface_frame != AIA_SURFACE_FRAME:
            raise ValueError(
                "AIA surface positions must use HeliographicCarrington Cartesian."
            )
        if self.ray_direction_convention != AIA_RAY_DIRECTION_CONVENTION:
            raise ValueError("AIA rays must use observer_to_sun orientation.")
        _sha256_convention_id(
            self.calibration_convention_id, "calibration_convention_id"
        )
        time = float(self.absolute_tai_seconds)
        radius = float(self.solar_radius_m)
        factor = float(self.degradation_factor)
        if not math.isfinite(time):
            raise ValueError("absolute_tai_seconds must be finite unix_tai seconds.")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("solar_radius_m must be finite and positive.")
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("degradation_factor must be finite and positive.")
        if type(self.degradation_correction_applications) is not int or (
            self.degradation_correction_applications not in (0, 1)
        ):
            raise ValueError(
                "degradation_correction_applications must be 0 or 1; values "
                "above one indicate double correction."
            )
        object.__setattr__(self, "absolute_tai_seconds", time)
        object.__setattr__(self, "solar_radius_m", radius)
        object.__setattr__(self, "degradation_factor", factor)
        object.__setattr__(self, "provenance", _metadata(self.provenance, "provenance"))


def _verified_response_contract() -> tuple[str, tuple[int, ...]]:
    metadata = validate_resource_set(AIA_RESOURCE_SET_ID)
    scientific = metadata.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RuntimeError("The verified AIA resource has no scientific contract.")
    convention_id = _sha256_convention_id(
        scientific.get("calibration_convention_id"),
        "AIA response calibration_convention_id",
    )
    try:
        channels = tuple(int(value) for value in scientific["channels"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("The verified AIA resource has invalid channels.") from error
    if channels != AIA_CHANNELS_ANGSTROM:
        raise RuntimeError(
            "The verified AIA response does not contain the required 171/193/211 set."
        )
    return convention_id, channels


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{name} must use a real floating-point dtype.")
    return tensor.detach().cpu().clone()


def _as_mask(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype is not torch.bool:
        raise ValueError(f"{name} must be boolean.")
    return tensor.detach().cpu().clone()


def _materialize_raster(
    source: RegisteredAIAChannelRaster,
    *,
    expected_calibration_convention_id: str,
) -> ImageObservationRaster:
    if not isinstance(source, RegisteredAIAChannelRaster):
        raise TypeError(
            "AIA preparation backends must return RegisteredAIAChannelRaster values."
        )
    if source.channel_angstrom not in AIA_CHANNELS_ANGSTROM:
        raise ValueError(
            f"Unsupported AIA channel {source.channel_angstrom}; expected "
            f"{AIA_CHANNELS_ANGSTROM}."
        )
    if source.calibration_convention_id != expected_calibration_convention_id:
        raise ValueError(
            "AIA input calibration convention does not match the verified "
            "temperature response."
        )

    intensity = _as_tensor(source.intensity, "intensity")
    ray = _as_tensor(source.ray_direction, "ray_direction")
    surface = _as_tensor(source.surface_position_m, "surface_position_m")
    valid = _as_mask(source.valid_mask, "valid_mask")
    on_disk = _as_mask(source.on_disk_mask, "on_disk_mask")
    if intensity.ndim != 2 or not intensity.numel():
        raise ValueError("intensity must be a non-empty [height, width] raster.")
    spatial_shape = tuple(intensity.shape)
    expected_shapes = {
        "valid_mask": spatial_shape,
        "on_disk_mask": spatial_shape,
        "ray_direction": (*spatial_shape, 3),
        "surface_position_m": (*spatial_shape, 3),
    }
    for name, tensor in (
        ("valid_mask", valid),
        ("on_disk_mask", on_disk),
        ("ray_direction", ray),
        ("surface_position_m", surface),
    ):
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"{name} must have shape {expected_shapes[name]}; "
                f"got {tuple(tensor.shape)}."
            )
    if not torch.isfinite(ray).all() or not torch.isfinite(surface).all():
        raise ValueError("AIA ray and surface geometry must be finite.")
    ray_norm = torch.linalg.vector_norm(ray, dim=-1)
    if not torch.allclose(ray_norm, torch.ones_like(ray_norm), rtol=0.0, atol=2.0e-5):
        raise ValueError("AIA observer-to-Sun ray vectors must be unit length.")
    surface_radius = torch.linalg.vector_norm(surface, dim=-1)
    if not torch.isfinite(surface_radius).all() or torch.any(surface_radius <= 0.0):
        raise ValueError("AIA surface positions must be finite non-zero vectors.")
    if not torch.any(on_disk):
        raise ValueError("Each AIA channel raster requires at least one on-disk pixel.")
    on_disk_radius = surface_radius[on_disk].to(torch.float64)
    if not torch.allclose(
        on_disk_radius,
        torch.full_like(on_disk_radius, source.solar_radius_m),
        rtol=2.0e-5,
        atol=1.0,
    ):
        raise ValueError(
            "On-disk AIA surface positions must lie on the declared photosphere."
        )
    surface_unit = surface / surface_radius[..., None]
    mu = (surface_unit * -ray).sum(dim=-1)
    if (
        not torch.isfinite(mu[on_disk]).all()
        or torch.any(mu[on_disk] <= 0.0)
        or torch.any(mu[on_disk] > 1.0 + 2.0e-5)
    ):
        raise ValueError(
            "On-disk AIA geometry is inconsistent with observer_to_sun rays."
        )

    output_valid = valid & on_disk
    if not torch.any(output_valid):
        raise ValueError(
            "Each AIA channel raster requires at least one valid on-disk pixel."
        )
    if not torch.isfinite(intensity[output_valid]).all():
        raise ValueError("AIA intensity must be finite at valid on-disk pixels.")

    performed = source.degradation_correction_applications == 0
    if performed:
        intensity = intensity / source.degradation_factor
    total_applications = source.degradation_correction_applications + int(performed)
    if total_applications != 1:  # defensive: constructor already excludes this state
        raise ValueError("AIA degradation correction must be applied exactly once.")

    metadata = {
        "intensity_unit": AIA_INTENSITY_UNIT,
        "observation_time": {
            "representation": AIA_TIME_REPRESENTATION,
            "absolute_tai_seconds": source.absolute_tai_seconds,
        },
        "ray_geometry": {
            "solar_radius_m": source.solar_radius_m,
            "frame": AIA_SURFACE_FRAME,
            "surface_position": "photospheric_surface_intersection",
            "ray_direction": AIA_RAY_DIRECTION_CONVENTION,
        },
        "calibration": {
            "calibration_convention_id": expected_calibration_convention_id,
            "degradation_factor": source.degradation_factor,
            "input_correction_applications": (
                source.degradation_correction_applications
            ),
            "correction_performed_by_preparation": performed,
            "total_correction_applications": total_applications,
            "operation": AIA_DEGRADATION_OPERATION,
            "reference_epoch": AIA_DEGRADATION_REFERENCE_EPOCH,
        },
        "mask": {
            "definition": "upstream_valid_mask AND on_disk_mask",
            "on_disk_pixel_count": int(on_disk.sum()),
            "valid_on_disk_pixel_count": int(output_valid.sum()),
        },
        "provenance": dict(source.provenance),
    }
    return ImageObservationRaster(
        intensity=intensity,
        ray_direction=ray,
        surface_position_m=surface,
        valid_mask=output_valid,
        absolute_tai_seconds=source.absolute_tai_seconds,
        channel_angstrom=source.channel_angstrom,
        exposure_group=source.exposure_group,
        metadata=metadata,
    )


def _resolve_inputs(
    *,
    rasters: Sequence[RegisteredAIAChannelRaster] | None,
    source_products: Sequence[_T] | None,
    map_backend: Callable[[_T], _M] | None,
    preparation_backend: Callable[
        [Sequence[_M | _T]], Sequence[RegisteredAIAChannelRaster]
    ]
    | None,
) -> tuple[RegisteredAIAChannelRaster, ...]:
    direct = rasters is not None
    injected = source_products is not None or preparation_backend is not None
    if direct == injected:
        raise TypeError(
            "Provide exactly one AIA input route: rasters, or source_products "
            "with preparation_backend."
        )
    if direct:
        if map_backend is not None:
            raise TypeError("map_backend cannot be used with direct rasters.")
        resolved = tuple(rasters or ())
    else:
        if source_products is None or preparation_backend is None:
            raise TypeError(
                "source_products and preparation_backend are both required for "
                "the injected AIA preparation route."
            )
        sources = tuple(source_products)
        if not sources:
            raise ValueError("source_products must be non-empty.")
        maps: Sequence[Any] = (
            tuple(map_backend(source) for source in sources)
            if map_backend is not None
            else sources
        )
        resolved = tuple(preparation_backend(maps))
    if not resolved:
        raise ValueError("AIA preparation requires at least one raster.")
    if any(not isinstance(item, RegisteredAIAChannelRaster) for item in resolved):
        raise TypeError(
            "AIA preparation backends must return RegisteredAIAChannelRaster values."
        )
    return resolved


def _complete_groups(
    rasters: Sequence[RegisteredAIAChannelRaster],
) -> tuple[str, ...]:
    groups = tuple(dict.fromkeys(raster.exposure_group for raster in rasters))
    identities = [
        (raster.exposure_group, raster.channel_angstrom) for raster in rasters
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "AIA preparation requires one raster per exposure group and channel."
        )
    expected = {
        (group, channel) for group in groups for channel in AIA_CHANNELS_ANGSTROM
    }
    actual = set(identities)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Every AIA exposure group must contain exactly channels 171/193/211; "
            f"missing={missing}, unexpected={unexpected}."
        )
    return groups


@dataclass
class _WrittenRaster:
    exposure_group: str
    channel_angstrom: int
    absolute_tai_seconds: float
    degradation_factor: float
    degradation_correction_applications: int
    raster_name: str
    spatial_shape: tuple
    entry: dict


def _written_statistics(directory, rasters, channels):
    """Exact global medians using disk workspace, not all images in RAM."""
    from tqdm.auto import tqdm

    scales = {}
    for channel in tqdm(channels, desc="AIA normalization", unit="channel"):
        entries = [r.entry for r in rasters if r.channel_angstrom == channel]
        def load(entry, field):
            return np.load(directory / entry["arrays"][field]["file"], mmap_mode="r")
        count = sum(int(np.count_nonzero(load(entry, "valid_mask"))) for entry in entries)
        if not count:
            raise ValueError(f"AIA channel {channel} has no valid pixels.")
        scratch = directory / f".statistics-{channel}.npy"
        values = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float64, shape=(count,))
        offset = 0
        for entry in entries:
            valid = load(entry, "valid_mask")
            samples = load(entry, "intensity")[valid]
            values[offset:offset + len(samples)] = np.abs(samples)
            offset += len(samples)
        # torch.median uses the lower middle sample for even-length inputs.
        middle = (count - 1) // 2
        values.partition(middle)
        scale = float(values[middle])
        del values
        scratch.unlink()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"AIA channel {channel} has no positive robust scale.")
        scales[str(channel)] = scale
    return {"asinh_scale_algorithm": AIA_ASINH_SCALE_ALGORITHM,
            "asinh_scale_by_channel_dn_s_pixel": scales}


def prepare_aia_observation_store(
    output_directory: str | Path,
    *,
    source_files_sha256: str | Callable[[], str],
    raster_backend: Callable | None = None,
    preparation_dependencies: Mapping[str, Any],
    rasters: Sequence[RegisteredAIAChannelRaster] | None = None,
    source_products: Sequence[_T] | None = None,
    map_backend: Callable[[_T], _M] | None = None,
    preparation_backend: Callable[
        [Sequence[_M | _T]], Sequence[RegisteredAIAChannelRaster]
    ]
    | None = None,
    observation_id: str = "aia_euv",
) -> Path:
    """Write images directly to the final directory, then write the manifest.

    Direct callers pass ``rasters``.  An acquisition/preparation integration can
    instead pass ``source_products`` and a ``preparation_backend``; an optional
    ``map_backend`` is applied independently to each source first.  Those
    callables are the only place optional SunPy/aiapy dependencies need to be
    imported.
    """

    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Image-observation store already exists: {output}.")
    if not callable(source_files_sha256):
        _source_sha256(source_files_sha256)
    if not isinstance(preparation_dependencies, Mapping):
        raise TypeError("preparation_dependencies must be a mapping.")
    if not isinstance(observation_id, str) or not observation_id.strip():
        raise ValueError("observation_id must be a non-empty string.")
    if observation_id != observation_id.strip():
        raise ValueError("observation_id cannot have surrounding whitespace.")

    if raster_backend is not None:
        if source_products is None or rasters is not None or preparation_backend is not None or map_backend is not None:
            raise TypeError("raster_backend requires only source_products.")
        jobs = tuple(source_products)
    else:
        jobs = _resolve_inputs(
            rasters=rasters, source_products=source_products, map_backend=map_backend,
            preparation_backend=preparation_backend,
        )
    convention_id, resource_channels = _verified_response_contract()
    output.mkdir(parents=True, exist_ok=True)

    def prepare_and_write(item):
        index, source = item
        raster = raster_backend(source) if raster_backend is not None else source
        prepared = _materialize_raster(raster, expected_calibration_convention_id=convention_id)
        name = raster.raster_name or f"{raster.exposure_group}-aia-{raster.channel_angstrom}"
        entry = ImageObservationStore.write_raster(output, index, name, prepared)
        return _WrittenRaster(
            raster.exposure_group, raster.channel_angstrom, raster.absolute_tai_seconds,
            raster.degradation_factor, raster.degradation_correction_applications,
            name, prepared.spatial_shape, entry,
        )

    written = parallel_map(prepare_and_write, enumerate(jobs), description="AIA read / prepare / write")
    if not written:
        raise ValueError("AIA preparation requires at least one raster.")
    source_digest = _source_sha256(source_files_sha256() if callable(source_files_sha256)
                                  else source_files_sha256)
    groups = _complete_groups(written)
    group_index = {group: index for index, group in enumerate(groups)}
    ordered_inputs = sorted(written, key=lambda raster: (
        group_index[raster.exposure_group], AIA_CHANNELS_ANGSTROM.index(raster.channel_angstrom),
    ))
    raster_names = tuple(raster.raster_name for raster in ordered_inputs)
    if len(set(raster_names)) != len(raster_names):
        raise ValueError("AIA raster names must be unique.")

    spec = ImageObservationSpec(
        observation_id=observation_id,
        observation_type="aia_euv",
        instrument_type="aia_temperature_response",
        intensity_unit=AIA_INTENSITY_UNIT,
        channels_angstrom=resource_channels,
        exposure_groups=groups,
        calibration_convention={
            "calibration_convention_id": convention_id,
            "degradation_correction": {
                "operation": AIA_DEGRADATION_OPERATION,
                "reference_epoch": AIA_DEGRADATION_REFERENCE_EPOCH,
                "total_applications_per_raster": 1,
            },
            "measurement_semantics": "per_native_pixel",
            "sensitivity_convention": "reference_epoch",
        },
        geometry_convention={
            "ray_direction": AIA_RAY_DIRECTION_CONVENTION,
            "surface_position": "photospheric_surface_intersection",
            "surface_frame": AIA_SURFACE_FRAME,
            "time_representation": AIA_TIME_REPRESENTATION,
        },
        required_resource_sets=(AIA_RESOURCE_SET_ID,),
    )
    correction_records = [
        {
            "exposure_group": raster.exposure_group,
            "channel_angstrom": raster.channel_angstrom,
            "degradation_factor": raster.degradation_factor,
            "input_correction_applications": (
                raster.degradation_correction_applications
            ),
            "correction_performed_by_preparation": (
                raster.degradation_correction_applications == 0
            ),
            "total_correction_applications": 1,
        }
        for raster in ordered_inputs
    ]
    preparation_metadata = {
        "format": AIA_PREPARATION_FORMAT,
        "version": AIA_PREPARATION_VERSION,
        "channels_angstrom": list(resource_channels),
        "exposure_groups": list(groups),
        "output_intensity_unit": AIA_INTENSITY_UNIT,
        "calibration_convention_id": convention_id,
        "time_representation": AIA_TIME_REPRESENTATION,
        "surface_frame": AIA_SURFACE_FRAME,
        "ray_direction": AIA_RAY_DIRECTION_CONVENTION,
        "degradation_correction": {
            "operation": AIA_DEGRADATION_OPERATION,
            "reference_epoch": AIA_DEGRADATION_REFERENCE_EPOCH,
            "total_applications_per_raster": 1,
            "records": correction_records,
        },
        "preparation_dependencies": dict(preparation_dependencies),
    }
    signature_configuration = {
        "adapter": "aia_euv",
        "observation": spec.metadata(),
        "preparation": preparation_metadata,
        "rasters": [
            {
                "name": name,
                "channel_angstrom": raster.channel_angstrom,
                "exposure_group": raster.exposure_group,
                "absolute_tai_seconds": raster.absolute_tai_seconds,
                "spatial_shape": list(raster.spatial_shape),
            }
            for name, raster in zip(raster_names, ordered_inputs, strict=True)
        ],
    }
    source_signature = image_observation_store_signature(
        signature_configuration,
        preparation_dependencies,
        source_files_sha256=source_digest,
    )
    session_metadata = {
        "adapter": "aia_euv",
        "observation": spec.metadata(),
        "source_files_sha256": source_digest,
        "preparation": preparation_metadata,
        "objective_statistics": _written_statistics(output, ordered_inputs, resource_channels),
    }
    return ImageObservationStore.write_manifest(
        output, [raster.entry for raster in ordered_inputs],
        source_signature=source_signature,
        metadata=session_metadata,
    )


__all__ = [
    "AIA_CHANNELS_ANGSTROM",
    "AIA_DEGRADATION_OPERATION",
    "AIA_DEGRADATION_REFERENCE_EPOCH",
    "AIA_PREPARATION_FORMAT",
    "AIA_PREPARATION_VERSION",
    "AIA_RAY_DIRECTION_CONVENTION",
    "AIA_SURFACE_FRAME",
    "AIA_TIME_REPRESENTATION",
    "AIAMapBackend",
    "AIAPreparationBackend",
    "RegisteredAIAChannelRaster",
    "prepare_aia_observation_store",
]
