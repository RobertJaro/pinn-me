"""Load observation streams and derive their shared scene bounds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from prom3theus.config.joint_schema import JointInversionConfig
from prom3theus.observations import ObservationKind, SceneContract
from prom3theus.rt.geometry import direction_to_chart_mm

from .joint_contracts import LoadedJointStream
from prom3theus.observations.scene import stokes_reference_time_tai_seconds, geometry_chunks


def _scene_from_stokes(
    config: JointInversionConfig, loaded: LoadedJointStream
) -> SceneContract:
    metadata = loaded.rasters[0].metadata
    try:
        coordinate_metadata = metadata["coordinates"]
        spatial = coordinate_metadata["network_affine"]
        ray = metadata["ray_geometry"]
        scene_basis = ray["scene_basis_rows"]
        solar_radius_m = float(ray["solar_radius_m"])
        spatial_center = tuple(map(float, spatial["center_mm"]))
        spatial_scale = tuple(map(float, spatial["scale_mm"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Reference Stokes stream lacks a valid coordinate/scene contract."
        ) from error
    temporal = coordinate_metadata.get("time_affine", {})
    time_center = float(temporal.get("center_hours", 0.0))
    time_scale = float(temporal.get("scale_hours", 1.0))
    geometry = config.atmosphere.geometry
    return SceneContract(
        scene_basis=torch.as_tensor(scene_basis, dtype=torch.float64),
        solar_radius_m=solar_radius_m,
        reference_time_tai_seconds=stokes_reference_time_tai_seconds(metadata),
        spatial_coordinate_center_mm=spatial_center,
        spatial_coordinate_scale_mm=spatial_scale,
        time_coordinate_center_hours=time_center,
        time_coordinate_scale_hours=time_scale,
        height_bounds_m=(
            geometry.inner_height_megameter * 1.0e6,
            geometry.outer_height_megameter * 1.0e6,
        ),
    )


def _scene_from_image(
    config: JointInversionConfig, loaded: LoadedJointStream
) -> SceneContract:
    first = loaded.rasters[0]
    try:
        ray = first.metadata["ray_geometry"]
        scene_basis = torch.as_tensor(ray["scene_basis_rows"], dtype=torch.float64)
        solar_radius_m = float(ray["solar_radius_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "An image reference stream must declare ray_geometry.scene_basis_rows "
            "and solar_radius_m."
        ) from error
    minimum = torch.full((2,), math.inf, dtype=torch.float64)
    maximum = torch.full((2,), -math.inf, dtype=torch.float64)
    times = []
    reference_time = float(first.absolute_tai_seconds)
    for raster in loaded.rasters:
        for positions, _, valid, _ in geometry_chunks(raster):
            if not valid.any():
                continue
            chart = direction_to_chart_mm(positions[valid].to(torch.float64), scene_basis, solar_radius_m)
            minimum = torch.minimum(minimum, chart.amin(dim=0))
            maximum = torch.maximum(maximum, chart.amax(dim=0))
        times.append((float(raster.absolute_tai_seconds) - reference_time) / 3600.0)
    center = 0.5 * (minimum + maximum)
    scale = torch.maximum(0.5 * (maximum - minimum), torch.full_like(center, 1.0e-6))
    time_min, time_max = min(times), max(times)
    geometry = config.atmosphere.geometry
    return SceneContract(
        scene_basis=scene_basis,
        solar_radius_m=solar_radius_m,
        reference_time_tai_seconds=reference_time,
        spatial_coordinate_center_mm=tuple(center.tolist()),
        spatial_coordinate_scale_mm=tuple(scale.tolist()),
        time_coordinate_center_hours=0.5 * (time_min + time_max),
        time_coordinate_scale_hours=max(0.5 * (time_max - time_min), 1.0 / 3600.0),
        height_bounds_m=(
            geometry.inner_height_megameter * 1.0e6,
            geometry.outer_height_megameter * 1.0e6,
        ),
    )


def _default_scene_builder(
    config: JointInversionConfig,
    streams: Mapping[str, LoadedJointStream],
) -> SceneContract:
    reference = streams[config.scene.reference_stream]
    if reference.scene_contract is not None:
        scene = reference.scene_contract
    elif reference.prepared.descriptor.observation_kind is ObservationKind.STOKES:
        scene = _scene_from_stokes(config, reference)
    else:
        scene = _scene_from_image(config, reference)
    for loaded in streams.values():
        if loaded.prepared.descriptor.observation_kind is ObservationKind.IMAGE:
            scene.validate_image_rasters(loaded.rasters)
    return scene


def _surface_time_chunks(
    streams: Mapping[str, LoadedJointStream],
    scene: SceneContract,
    *,
    chunk_size: int = 262_144,
):
    for loaded in streams.values():
        if getattr(loaded, "sampling_support", None) is not None:
            yield from loaded.sampling_support(scene, chunk_size)
            continue
        image = loaded.prepared.descriptor.observation_kind is ObservationKind.IMAGE
        for raster in loaded.rasters:
            for positions, rays, valid, coordinates in geometry_chunks(raster, chunk_size):
                if image:
                    distance = scene._outer_support_distance(positions, rays)
                    endpoints = positions.to(torch.float64) - distance[..., None] * rays.to(torch.float64)
                if not valid.any():
                    continue
                position = positions[valid].to(torch.float64)
                if image:
                    position = torch.cat((position, endpoints[valid]))
                    time = (float(raster.absolute_tai_seconds) - scene.reference_time_tai_seconds) / 3600.0
                    relative_time = torch.full((len(position),), time, dtype=torch.float64)
                else:
                    relative_time = coordinates[..., 2][valid].to(torch.float64)
                yield position, relative_time


def _observation_times_hours(
    streams: Mapping[str, LoadedJointStream], scene: SceneContract,
) -> list[float]:
    """Distinct valid observation times in the shared scene's physical hours.

    Read contiguous geometry chunks, including provider sampling support, so
    masks, stream selection, and scene-time rebasing match the domain bounds.
    No pixelwise random reads or observation payload materialization is needed.
    """
    values = set()
    for _, time in _surface_time_chunks(streams, scene):
        time = time.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(time).all():
            raise ValueError("Potential observation times must be finite")
        values.update(torch.unique(time).tolist())
    if not values:
        raise ValueError("Potential boundaries require valid Stokes observation times")
    return sorted(values)


def _maximum_image_ray_angular_span(
    streams: Mapping[str, LoadedJointStream], scene: SceneContract
) -> float:
    maximum = 0.0
    for loaded in streams.values():
        if loaded.prepared.descriptor.observation_kind is not ObservationKind.IMAGE:
            continue
        for raster in loaded.rasters:
            for positions, rays, mask, _ in geometry_chunks(raster):
                distance = scene._outer_support_distance(positions, rays)
                if not mask.any():
                    continue
                surface = positions[mask].to(torch.float64)
                endpoint = surface - distance[mask][..., None] * rays[mask].to(torch.float64)
                surface = surface / torch.linalg.vector_norm(surface, dim=-1, keepdim=True)
                endpoint = endpoint / torch.linalg.vector_norm(endpoint, dim=-1, keepdim=True)
                separation = torch.acos((surface * endpoint).sum(dim=-1).clamp(-1.0, 1.0))
                maximum = max(maximum, float(separation.amax()))
    return maximum


def _joint_observation_bounds(
    streams: Mapping[str, LoadedJointStream], scene: SceneContract
) -> dict[str, Any]:
    sine_sum = cosine_sum = 0.0
    count = 0
    latitude_min, latitude_max = math.inf, -math.inf
    time_min, time_max = math.inf, -math.inf
    for position, time in _surface_time_chunks(streams, scene):
        radius = torch.linalg.vector_norm(position, dim=-1)
        outer_radius = scene.solar_radius_m + scene.height_bounds_m[1]
        tolerance = max(2.0, outer_radius * 2.0e-5)
        if torch.any(radius < scene.solar_radius_m - tolerance) or torch.any(
            radius > outer_radius + tolerance
        ):
            raise ValueError(
                "Joint observation support lies outside the shared solar shell."
            )
        longitude = torch.atan2(position[:, 1], position[:, 0])
        latitude = torch.atan2(
            position[:, 2], torch.linalg.vector_norm(position[:, :2], dim=-1)
        )
        sine_sum += float(torch.sin(longitude).sum())
        cosine_sum += float(torch.cos(longitude).sum())
        count += int(position.shape[0])
        latitude_min = min(latitude_min, float(latitude.amin()))
        latitude_max = max(latitude_max, float(latitude.amax()))
        time_min = min(time_min, float(time.amin()))
        time_max = max(time_max, float(time.amax()))
    if count == 0:
        raise ValueError("Joint sampling requires at least one valid surface point.")
    center = math.atan2(sine_sum, cosine_sum)
    longitude_min, longitude_max = math.inf, -math.inf
    for position, _ in _surface_time_chunks(streams, scene):
        longitude = torch.atan2(position[:, 1], position[:, 0])
        offset = torch.atan2(
            torch.sin(longitude - center), torch.cos(longitude - center)
        )
        longitude_min = min(longitude_min, float(offset.amin()))
        longitude_max = max(longitude_max, float(offset.amax()))
    # A straight native ray traces a minor great-circle arc after radial
    # normalization.  Expanding the endpoint envelope by the largest arc span
    # conservatively includes any interior longitude/latitude extremum and a
    # small floating-point margin.
    ray_span = _maximum_image_ray_angular_span(streams, scene)
    angular_epsilon = 1.0e-7
    latitude_min = max(-0.5 * math.pi, latitude_min - ray_span - angular_epsilon)
    latitude_max = min(0.5 * math.pi, latitude_max + ray_span + angular_epsilon)
    pole_distance = max(
        1.0e-3,
        math.cos(
            min(
                0.5 * math.pi - 1.0e-6,
                max(abs(latitude_min), abs(latitude_max)),
            )
        ),
    )
    longitude_margin = min(math.pi, ray_span / pole_distance + angular_epsilon)
    longitude_min -= longitude_margin
    longitude_max += longitude_margin
    if longitude_max - longitude_min > 2.0 * math.pi:
        longitude_min, longitude_max = -math.pi, math.pi
    return {
        "surface_longitude_center_rad": center,
        "surface_longitude_offset_rad": [longitude_min, longitude_max],
        "surface_latitude_rad": [latitude_min, latitude_max],
        "time_hours": [time_min, time_max],
        "solar_radius_m": scene.solar_radius_m,
        "angular_support": "surface anchors plus complete AIA near-side ray arcs",
        "maximum_image_ray_angular_span_rad": ray_span,
    }
