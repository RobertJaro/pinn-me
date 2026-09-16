"""Typed contracts for scalar image observations.

The image boundary is deliberately parallel to, rather than a relaxation of,
the spectropolarimetric :mod:`prom3theus.observations.contracts` boundary.  One
raster represents one channel at one exposure time on its native pixel grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, TypedDict

import torch


ABSOLUTE_TAI_SECONDS = "absolute_tai_seconds"
CHANNEL_ANGSTROM = "channel_angstrom"
CHANNEL_INDEX = "channel_index"
EXPOSURE_GROUP = "exposure_group"
IMAGE_INDEX = "image_index"
INTENSITY = "intensity"
PIXEL_INDEX = "pixel_index"
RAY_DIRECTION = "ray_direction"
SURFACE_POSITION_M = "surface_position_m"
UNCERTAINTY = "uncertainty"
VALID_MASK = "valid_mask"


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _metadata_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys.")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ImageObservationSpec:
    """Scientific description shared by a sequence of scalar image rasters.

    ``channels_angstrom`` is an ordered vocabulary.  Runtime batches carry both
    the physical wavelength and its stable integer index in this vocabulary.
    ``exposure_groups`` is likewise an ordered collection of opaque, stable IDs;
    all configured channels must occur exactly once in every group.
    """

    observation_id: str
    observation_type: str
    instrument_type: str
    intensity_unit: str
    channels_angstrom: tuple[int, ...]
    exposure_groups: tuple[str, ...]
    calibration_convention: Mapping[str, Any] = field(default_factory=dict)
    geometry_convention: Mapping[str, Any] = field(default_factory=dict)
    required_resource_sets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "observation_type",
            "instrument_type",
            "intensity_unit",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))

        channels = tuple(self.channels_angstrom)
        if not channels or any(type(channel) is not int for channel in channels):
            raise TypeError("channels_angstrom must contain positive integers.")
        if any(channel <= 0 for channel in channels):
            raise ValueError("channels_angstrom must contain positive integers.")
        if any(right <= left for left, right in zip(channels, channels[1:])):
            raise ValueError(
                "channels_angstrom must be unique and strictly increasing."
            )

        groups = tuple(
            _identifier(group, "exposure group") for group in self.exposure_groups
        )
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("exposure_groups must be non-empty and unique.")
        resources = tuple(
            _identifier(resource, "required resource set")
            for resource in self.required_resource_sets
        )
        if len(set(resources)) != len(resources):
            raise ValueError("required_resource_sets must be unique.")

        object.__setattr__(self, "channels_angstrom", channels)
        object.__setattr__(self, "exposure_groups", groups)
        object.__setattr__(self, "required_resource_sets", resources)
        object.__setattr__(
            self,
            "calibration_convention",
            _metadata_mapping(self.calibration_convention, "calibration_convention"),
        )
        object.__setattr__(
            self,
            "geometry_convention",
            _metadata_mapping(self.geometry_convention, "geometry_convention"),
        )

    @property
    def observation_kind(self) -> str:
        return "image"

    def metadata(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "instrument_type": self.instrument_type,
            "observation_kind": self.observation_kind,
            "intensity_unit": self.intensity_unit,
            "channels_angstrom": list(self.channels_angstrom),
            "exposure_groups": list(self.exposure_groups),
            "calibration_convention": dict(self.calibration_convention),
            "geometry_convention": dict(self.geometry_convention),
            "required_resource_sets": list(self.required_resource_sets),
        }


@dataclass(frozen=True, slots=True)
class ImageObservationRaster:
    """One calibrated scalar image at one channel and exposure time.

    ``ray_direction`` points from the observer toward the photospheric surface,
    matching the established Stokes observation convention.  Coronal synthesis
    consequently follows ``-ray_direction`` from ``surface_position_m`` toward
    the observer.  Absolute time is seconds on the TAI scale since
    1970-01-01 00:00:00 TAI (Astropy's ``unix_tai`` representation).
    """

    intensity: torch.Tensor
    ray_direction: torch.Tensor
    surface_position_m: torch.Tensor
    valid_mask: torch.Tensor
    absolute_tai_seconds: float
    channel_angstrom: int
    exposure_group: str
    metadata: Mapping[str, Any]
    uncertainty: torch.Tensor | None = None

    _bulk_catalog: Any = field(default=None, init=False, repr=False, compare=False)

    def __reduce__(self):
        from .loading import raster_reduce

        return raster_reduce(self)

    def __post_init__(self) -> None:
        tensors = {
            "intensity": torch.as_tensor(self.intensity),
            "ray_direction": torch.as_tensor(self.ray_direction),
            "surface_position_m": torch.as_tensor(self.surface_position_m),
            "valid_mask": torch.as_tensor(self.valid_mask),
        }
        intensity = tensors["intensity"]
        if intensity.ndim != 2 or not intensity.numel():
            raise ValueError("intensity must be a non-empty [height, width] array.")
        spatial = tuple(intensity.shape)
        expected = {
            "ray_direction": (*spatial, 3),
            "surface_position_m": (*spatial, 3),
            "valid_mask": spatial,
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}; got {tuple(tensors[name].shape)}."
                )
        if tensors["valid_mask"].dtype is not torch.bool:
            raise ValueError("valid_mask must be boolean.")
        valid = tensors["valid_mask"]
        if not torch.any(valid):
            raise ValueError("An image observation requires at least one valid pixel.")
        for name in ("intensity", "ray_direction", "surface_position_m"):
            if not tensors[name].is_floating_point() or tensors[name].is_complex():
                raise ValueError(f"{name} must use a real floating-point dtype.")
        for name in ("ray_direction", "surface_position_m"):
            if not torch.isfinite(tensors[name]).all():
                raise ValueError(f"{name} must contain only finite values.")
        if not torch.isfinite(intensity[valid]).all():
            raise ValueError("intensity must be finite at every valid pixel.")
        if self.uncertainty is not None:
            uncertainty = torch.as_tensor(self.uncertainty)
            if uncertainty.shape != intensity.shape or not uncertainty.is_floating_point():
                raise ValueError("uncertainty must be a floating-point image matching intensity.")
            if not torch.isfinite(uncertainty[valid]).all() or torch.any(uncertainty[valid] <= 0):
                raise ValueError("uncertainty must be finite and strictly positive at every valid pixel.")
            tensors["uncertainty"] = uncertainty

        ray_norm = torch.linalg.vector_norm(tensors["ray_direction"], dim=-1)
        if not torch.allclose(
            ray_norm, torch.ones_like(ray_norm), rtol=0.0, atol=2.0e-5
        ):
            raise ValueError("ray_direction must contain unit vectors.")
        surface_radius = torch.linalg.vector_norm(tensors["surface_position_m"], dim=-1)
        if not torch.isfinite(surface_radius).all() or torch.any(surface_radius <= 0):
            raise ValueError("surface_position_m must contain finite non-zero vectors.")
        surface_unit = tensors["surface_position_m"] / surface_radius[..., None]
        mu = (surface_unit * -tensors["ray_direction"]).sum(dim=-1)
        if (
            not torch.isfinite(mu[valid]).all()
            or torch.any(mu[valid] <= 0)
            or torch.any(mu[valid] > 1.0 + 2.0e-5)
        ):
            raise ValueError(
                "Every valid image pixel must be on disk with observer-to-Sun ray orientation."
            )

        metadata = _metadata_mapping(self.metadata, "metadata")
        try:
            solar_radius_m = float(metadata["ray_geometry"]["solar_radius_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Image metadata must declare ray_geometry.solar_radius_m."
            ) from error
        if not math.isfinite(solar_radius_m) or solar_radius_m <= 0:
            raise ValueError("ray_geometry.solar_radius_m must be finite and positive.")
        if not torch.allclose(
            surface_radius[valid].to(torch.float64),
            torch.full_like(surface_radius[valid].to(torch.float64), solar_radius_m),
            rtol=2.0e-5,
            atol=1.0,
        ):
            raise ValueError(
                "Image surface positions do not lie on the declared solar radius."
            )

        time = float(self.absolute_tai_seconds)
        if not math.isfinite(time):
            raise ValueError("absolute_tai_seconds must be finite.")
        if type(self.channel_angstrom) is not int or self.channel_angstrom <= 0:
            raise ValueError("channel_angstrom must be a positive integer.")
        group = _identifier(self.exposure_group, "exposure_group")

        for name, value in tensors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "absolute_tai_seconds", time)
        object.__setattr__(self, "exposure_group", group)
        object.__setattr__(self, "metadata", metadata)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.intensity.shape[0]), int(self.intensity.shape[1])

    @property
    def mu(self) -> torch.Tensor:
        from .arrays import materialize_array
        position = materialize_array(self.surface_position_m)
        surface = position / torch.linalg.vector_norm(position, dim=-1, keepdim=True)
        return (surface * -materialize_array(self.ray_direction)).sum(dim=-1)


class ImageObservationSample(TypedDict, total=False):
    intensity: torch.Tensor
    uncertainty: torch.Tensor
    ray_direction: torch.Tensor
    surface_position_m: torch.Tensor
    absolute_tai_seconds: torch.Tensor
    channel_angstrom: torch.Tensor
    channel_index: torch.Tensor
    image_index: torch.Tensor
    pixel_index: torch.Tensor


class ImageObservationBatch(ImageObservationSample, total=False):
    """Batched scalar-image samples."""


__all__ = [
    "ABSOLUTE_TAI_SECONDS",
    "CHANNEL_ANGSTROM",
    "CHANNEL_INDEX",
    "EXPOSURE_GROUP",
    "IMAGE_INDEX",
    "INTENSITY",
    "ImageObservationBatch",
    "ImageObservationRaster",
    "ImageObservationSample",
    "ImageObservationSpec",
    "PIXEL_INDEX",
    "RAY_DIRECTION",
    "SURFACE_POSITION_M",
    "UNCERTAINTY",
    "VALID_MASK",
]
