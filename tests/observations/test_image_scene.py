"""Physical scene transforms and complete image-ray support validation."""

from __future__ import annotations

import pytest
import torch

from prom3theus.observations import ImageObservationRaster, SceneContract


SOLAR_RADIUS_M = 6.957e8


def _scene(**changes) -> SceneContract:
    values = {
        "scene_basis": torch.eye(3, dtype=torch.float64),
        "solar_radius_m": SOLAR_RADIUS_M,
        "reference_time_tai_seconds": 1_700_000_000.0,
        "spatial_coordinate_center_mm": (1.0, -2.0),
        "spatial_coordinate_scale_mm": (10.0, 20.0),
        "time_coordinate_center_hours": 0.25,
        "time_coordinate_scale_hours": 2.0,
        "height_bounds_m": (-1.0e5, 5.0e7),
    }
    values.update(changes)
    return SceneContract(**values)


def _raster(time=1_700_001_800.0) -> ImageObservationRaster:
    x = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float64) * SOLAR_RADIUS_M
    z = torch.sqrt(SOLAR_RADIUS_M**2 - x.square())
    surface = torch.stack((x, torch.zeros_like(x), z), dim=-1)[None]
    ray = torch.zeros_like(surface)
    ray[..., 2] = -1.0
    return ImageObservationRaster(
        intensity=torch.ones(1, 3),
        uncertainty=torch.full((1, 3), 0.1),
        ray_direction=ray,
        surface_position_m=surface,
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
        absolute_tai_seconds=time,
        channel_angstrom=171,
        exposure_group="group-0",
        metadata={
            "intensity_unit": "DN s-1 pixel-1",
            "ray_geometry": {"solar_radius_m": SOLAR_RADIUS_M},
        },
    )


def test_scene_transform_maps_physical_position_and_absolute_time():
    scene = _scene()
    position = torch.tensor(
        [[0.0, 0.0, SOLAR_RADIUS_M + 1.0e6]], dtype=torch.float64
    )
    coordinates, height = scene.transform(position, 1_700_001_800.0)
    torch.testing.assert_close(
        coordinates, torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)
    )
    torch.testing.assert_close(height, torch.tensor([1.0e6], dtype=torch.float64))

    metadata = scene.atmosphere_coordinate_metadata
    assert metadata["spatial_coordinate_center_mm"] == [1.0, -2.0]
    assert metadata["time_coordinate_scale_hours"] == 2.0
    assert metadata["scene_geometry_config"]["scene_basis"] == torch.eye(3).tolist()
    assert metadata["time_reference"]["representation"] == "unix_tai"


def test_scene_validates_full_image_ray_support_to_outer_shell():
    scene = _scene()
    summary = scene.validate_image_rasters([_raster()])
    assert summary["raster_count"] == 1
    assert summary["valid_ray_count"] == 3
    assert summary["height_bounds_m"] == [-1.0e5, 5.0e7]
    assert summary["time_hours"] == [0.5, 0.5]
    assert summary["outer_support_distance_m"][0] >= 5.0e7
    assert summary["chart_x_mm"][0] < 0 < summary["chart_x_mm"][1]


def test_scene_rejects_incompatible_radius_and_noncoronal_height_domain():
    raster = _raster()
    with pytest.raises(ValueError, match="SceneContract solar radius"):
        _scene(solar_radius_m=SOLAR_RADIUS_M + 1.0e6).validate_image_rasters(
            [raster]
        )
    with pytest.raises(ValueError, match="include zero"):
        _scene(height_bounds_m=(1.0, 5.0e7)).validate_image_rasters([raster])


def test_scene_transform_rejects_points_outside_chart_hemisphere():
    scene = _scene()
    far_side = torch.tensor([[0.0, 0.0, -SOLAR_RADIUS_M]])
    with pytest.raises(ValueError, match="visible hemisphere"):
        scene.transform(far_side, scene.reference_time_tai_seconds)
