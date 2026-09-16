"""Instrument-neutral physical scene contract for multi-stream observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

import torch

from prom3theus.rt.geometry import position_to_chart_height, validate_scene_basis

from .image_contracts import ImageObservationRaster


def geometry_chunks(raster, maximum_pixels=262144):
    """Owned, contiguous native row slabs for bounded scene reductions."""
    rows = max(1, maximum_pixels // raster.valid_mask.shape[1])
    for start in range(0, raster.valid_mask.shape[0], rows):
        yield (
            raster.surface_position_m[start:start + rows].clone(),
            raster.ray_direction[start:start + rows].clone(),
            raster.valid_mask[start:start + rows].clone(),
            raster.coordinates[start:start + rows].clone()
            if hasattr(raster, "coordinates") else None,
        )


def _pair(value, name: str, *, positive: bool = False) -> tuple[float, float]:
    try:
        pair = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain two finite numbers.") from error
    if len(pair) != 2 or not all(math.isfinite(item) for item in pair):
        raise ValueError(f"{name} must contain two finite numbers.")
    if positive and any(item <= 0 for item in pair):
        raise ValueError(f"{name} must contain two positive numbers.")
    return pair


@dataclass(frozen=True, slots=True)
class SceneContract:
    """One Cartesian solar scene and neural-coordinate affine.

    Physical positions are Carrington Cartesian metres.  ``scene_basis`` is a
    right-handed row basis whose final row points along the centre of the local
    gnomonic chart.  Absolute times use Astropy's ``unix_tai`` convention.
    """

    scene_basis: torch.Tensor
    solar_radius_m: float
    reference_time_tai_seconds: float
    spatial_coordinate_center_mm: tuple[float, float]
    spatial_coordinate_scale_mm: tuple[float, float]
    time_coordinate_center_hours: float
    time_coordinate_scale_hours: float
    height_bounds_m: tuple[float, float]

    def __post_init__(self) -> None:
        basis = validate_scene_basis(torch.as_tensor(self.scene_basis)).detach().clone()
        radius = float(self.solar_radius_m)
        reference_time = float(self.reference_time_tai_seconds)
        time_center = float(self.time_coordinate_center_hours)
        time_scale = float(self.time_coordinate_scale_hours)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("solar_radius_m must be finite and positive.")
        if not math.isfinite(reference_time):
            raise ValueError("reference_time_tai_seconds must be finite.")
        if not math.isfinite(time_center):
            raise ValueError("time_coordinate_center_hours must be finite.")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_coordinate_scale_hours must be finite and positive.")
        spatial_center = _pair(
            self.spatial_coordinate_center_mm, "spatial_coordinate_center_mm"
        )
        spatial_scale = _pair(
            self.spatial_coordinate_scale_mm,
            "spatial_coordinate_scale_mm",
            positive=True,
        )
        height_bounds = _pair(self.height_bounds_m, "height_bounds_m")
        if not height_bounds[0] < height_bounds[1]:
            raise ValueError("height_bounds_m must be strictly increasing.")
        if height_bounds[0] <= -radius:
            raise ValueError("The inner scene radius must remain positive.")

        object.__setattr__(self, "scene_basis", basis)
        object.__setattr__(self, "solar_radius_m", radius)
        object.__setattr__(self, "reference_time_tai_seconds", reference_time)
        object.__setattr__(self, "spatial_coordinate_center_mm", spatial_center)
        object.__setattr__(self, "spatial_coordinate_scale_mm", spatial_scale)
        object.__setattr__(self, "time_coordinate_center_hours", time_center)
        object.__setattr__(self, "time_coordinate_scale_hours", time_scale)
        object.__setattr__(self, "height_bounds_m", height_bounds)

    def transform(
        self,
        position_m: torch.Tensor,
        absolute_tai_seconds: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map physical positions/times to chart-time coordinates and height."""

        position = torch.as_tensor(position_m)
        if not position.is_floating_point():
            position = position.to(torch.get_default_dtype())
        if position.shape[-1] != 3 or not torch.isfinite(position).all():
            raise ValueError("position_m must end in finite Cartesian three-vectors.")
        chart, height = position_to_chart_height(
            position, self.scene_basis.to(position), self.solar_radius_m
        )
        absolute = torch.as_tensor(
            absolute_tai_seconds, dtype=torch.float64, device=position.device
        )
        try:
            absolute = torch.broadcast_to(absolute, position.shape[:-1])
        except RuntimeError as error:
            raise ValueError(
                "absolute_tai_seconds must be scalar or broadcast to position_m leading dimensions."
            ) from error
        if not torch.isfinite(absolute).all():
            raise ValueError("absolute_tai_seconds must be finite.")
        relative_hours = ((absolute - self.reference_time_tai_seconds) / 3600.0).to(
            position
        )
        return torch.cat((chart, relative_hours[..., None]), dim=-1), height

    @property
    def atmosphere_coordinate_metadata(self) -> Mapping[str, Any]:
        """Return constructor-ready coordinate fields for ``NeuralAtmosphere``."""

        return MappingProxyType(
            {
                "spatial_coordinate_center_mm": list(self.spatial_coordinate_center_mm),
                "spatial_coordinate_scale_mm": list(self.spatial_coordinate_scale_mm),
                "time_coordinate_center_hours": self.time_coordinate_center_hours,
                "time_coordinate_scale_hours": self.time_coordinate_scale_hours,
                "scene_geometry_config": {
                    "solar_radius_m": self.solar_radius_m,
                    "scene_basis": self.scene_basis.detach().cpu().tolist(),
                },
                "time_reference": {
                    "representation": "unix_tai",
                    "absolute_tai_seconds": self.reference_time_tai_seconds,
                    "relative_unit": "hour",
                },
                "height_bounds_m": list(self.height_bounds_m),
            }
        )

    def _outer_support_distance(
        self, surface_position_m: torch.Tensor, ray_direction: torch.Tensor
    ) -> torch.Tensor:
        """Distance from the photosphere toward the observer to the outer shell."""

        surface = surface_position_m.to(torch.float64)
        outward = -ray_direction.to(torch.float64)
        outer_radius = self.solar_radius_m + self.height_bounds_m[1]
        projection = (surface * outward).sum(dim=-1)
        discriminant = (
            projection.square() + outer_radius**2 - surface.square().sum(dim=-1)
        )
        if torch.any(discriminant < 0):
            raise ValueError("At least one image ray misses the outer scene shell.")
        distance = -projection + torch.sqrt(discriminant.clamp_min(0.0))
        if not torch.isfinite(distance).all() or torch.any(distance <= 0):
            raise ValueError(
                "Image ray outer-shell distances must be finite and positive."
            )
        return distance

    def image_ray_outer_endpoints(self, raster: ImageObservationRaster) -> torch.Tensor:
        """Return native-ray intersections with the configured outer shell."""

        if not isinstance(raster, ImageObservationRaster):
            raise TypeError("raster must be an ImageObservationRaster.")
        from .arrays import materialize_array
        position = materialize_array(raster.surface_position_m)
        direction = materialize_array(raster.ray_direction)
        distance = self._outer_support_distance(position, direction)
        return position.to(torch.float64) - distance[..., None] * direction.to(torch.float64)

    def validate_image_rasters(
        self, rasters: Sequence[ImageObservationRaster]
    ) -> dict[str, Any]:
        """Validate complete on-disk ray support and summarize its scene bounds."""

        rasters = tuple(rasters)
        if not rasters:
            raise ValueError("Scene validation requires at least one image raster.")
        if self.height_bounds_m[0] > 0 or self.height_bounds_m[1] <= 0:
            raise ValueError(
                "On-disk image rays require scene height_bounds_m to include zero "
                "and extend above the photosphere."
            )
        chart_min = torch.full((2,), math.inf, dtype=torch.float64)
        chart_max = torch.full((2,), -math.inf, dtype=torch.float64)
        time_min, time_max = math.inf, -math.inf
        distance_min, distance_max = math.inf, -math.inf
        valid_ray_count = 0
        for raster in rasters:
            for positions, rays, mask, _ in geometry_chunks(raster):
                distance_all = self._outer_support_distance(positions, rays)
                if not mask.any():
                    continue
                surface = positions[mask].to(torch.float64)
                ray = rays[mask].to(torch.float64)
                surface_radius = torch.linalg.vector_norm(surface, dim=-1)
                if not torch.allclose(
                    surface_radius,
                    torch.full_like(surface_radius, self.solar_radius_m),
                    rtol=2.0e-5,
                    atol=1.0,
                ):
                    raise ValueError(
                        "Image surface anchors do not match the SceneContract solar radius."
                    )
                distance = distance_all[mask]
                endpoint = surface - distance[..., None] * ray
                absolute = torch.full(
                    (surface.shape[0],),
                    raster.absolute_tai_seconds,
                    dtype=torch.float64,
                )
                start_coordinates, start_height = self.transform(surface, absolute)
                end_coordinates, end_height = self.transform(endpoint, absolute)
                if not torch.allclose(
                    start_height,
                    torch.zeros_like(start_height),
                    rtol=0.0,
                    atol=max(1.0, self.solar_radius_m * 2.0e-5),
                ):
                    raise ValueError("Image ray support must start at the photosphere.")
                if not torch.allclose(
                    end_height,
                    torch.full_like(end_height, self.height_bounds_m[1]),
                    rtol=2.0e-8,
                    atol=2.0,
                ):
                    raise ValueError(
                        "Image rays do not terminate on the outer scene shell."
                    )
                charts = torch.cat((start_coordinates[:, :2], end_coordinates[:, :2]))
                chart_min = torch.minimum(chart_min, charts.amin(dim=0))
                chart_max = torch.maximum(chart_max, charts.amax(dim=0))
                relative_time = float(start_coordinates[0, 2])
                time_min = min(time_min, relative_time)
                time_max = max(time_max, relative_time)
                distance_min = min(distance_min, float(distance.amin()))
                distance_max = max(distance_max, float(distance.amax()))
                valid_ray_count += int(surface.shape[0])
        return {
            "raster_count": len(rasters),
            "valid_ray_count": valid_ray_count,
            "chart_x_mm": [float(chart_min[0]), float(chart_max[0])],
            "chart_y_mm": [float(chart_min[1]), float(chart_max[1])],
            "time_hours": [time_min, time_max],
            "height_bounds_m": list(self.height_bounds_m),
            "outer_support_distance_m": [distance_min, distance_max],
            "ray_direction": "observer_to_sun; integration uses the negative direction",
        }


__all__ = ["SceneContract"]


def stokes_reference_time_tai_seconds(metadata):
    """Decode the single absolute time origin of a canonical Stokes raster."""
    from astropy.time import Time

    coordinates = metadata["coordinates"]
    origin = coordinates.get("time_origin", metadata.get("ref_time"))
    scale = str(coordinates.get("time_scale", "utc")).lower()
    if origin is None or scale not in {"tai", "utc"}:
        raise ValueError("Stokes metadata requires an absolute UTC or TAI time origin")
    return float(Time(origin, scale=scale).tai.to_value("unix_tai"))


def rebase_stokes_raster(raster, scene, *, output_path=None, chunk_pixels=262144):
    """Express a canonical solar-Cartesian raster in the shared chart/time frame."""
    from copy import deepcopy
    from dataclasses import replace
    from astropy.time import Time
    from .contracts import ObservationRaster

    if not isinstance(raster, ObservationRaster):
        raise TypeError("Expected a canonical Stokes raster")
    metadata = raster.metadata
    coordinate = metadata["coordinates"]
    scale = str(coordinate.get("time_scale", "utc")).lower()
    old_time = stokes_reference_time_tai_seconds(metadata)
    radius = float(metadata["ray_geometry"]["solar_radius_m"])
    if not math.isclose(radius, scene.solar_radius_m, rel_tol=2e-7, abs_tol=100.0):
        raise ValueError("Stokes streams must use the same physical solar radius")
    old_basis = torch.as_tensor(
        metadata["ray_geometry"]["scene_basis_rows"], dtype=torch.float64
    )
    if (
        torch.equal(old_basis, scene.scene_basis.cpu().double())
        and old_time == scene.reference_time_tai_seconds
    ):
        return raster
    from .arrays import read_slice
    mapped = None
    if output_path is None:
        coordinates = torch.empty(raster.coordinates.shape, dtype=raster.coordinates.dtype)
    else:
        import numpy as np
        from pathlib import Path
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=torch.empty((), dtype=raster.coordinates.dtype).numpy().dtype,
                                          shape=raster.coordinates.shape)
        coordinates = torch.from_numpy(mapped)
    rows = max(1, chunk_pixels // raster.spatial_shape[1])
    try:
        for start in range(0, raster.spatial_shape[0], rows):
            section = slice(start, start + rows)
            valid = read_slice(raster.valid_mask, section)
            chunk = read_slice(raster.coordinates, section)
            if valid.any():
                absolute = old_time + chunk[..., 2][valid].double() * 3600.0
                positions = read_slice(raster.surface_position_m, section)[valid].double()
                transformed, _ = scene.transform(positions, absolute)
                chunk[valid] = transformed.to(chunk)
            coordinates[section] = chunk
        if mapped is not None:
            mapped.flush()
    finally:
        if mapped is not None:
            del coordinates
            mapped._mmap.close()
    if mapped is not None:
        from .loading import load_array_tensor
        coordinates = load_array_tensor(path)
    updated = deepcopy(dict(metadata))
    reference = Time(
        scene.reference_time_tai_seconds, format="unix_tai", scale="tai"
    ).isot
    updated["ref_time"] = reference
    updated["coordinates"]["time_origin"] = reference
    updated["coordinates"]["time_scale"] = "tai"
    updated["coordinates"]["network_affine"] = {
        **coordinate["network_affine"],
        "center_mm": list(scene.spatial_coordinate_center_mm),
        "scale_mm": list(scene.spatial_coordinate_scale_mm),
    }
    updated["coordinates"]["time_affine"] = {
        **coordinate.get("time_affine", {}),
        "center_hours": scene.time_coordinate_center_hours,
        "scale_hours": scene.time_coordinate_scale_hours,
    }
    updated["ray_geometry"]["scene_basis_rows"] = scene.scene_basis.tolist()
    if "times" in updated:
        updated["times"] = [
            str(value) for value in Time(updated["times"], scale=scale).tai.isot
        ]
    from dataclasses import fields
    from .loading import _restore_raster, _plain
    return _restore_raster(type(raster), {
        field.name: _plain(coordinates if field.name == "coordinates" else
                           updated if field.name == "metadata" else getattr(raster, field.name))
        for field in fields(raster)
    })
