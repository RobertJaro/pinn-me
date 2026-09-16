"""Independent Stokes observations share physical geometry and absolute time."""
import math
from dataclasses import replace
import pytest
import torch
from astropy.time import Time
from prom3theus.observations import ObservationRaster, SceneContract
from prom3theus.observations.scene import rebase_stokes_raster
from prom3theus.rt.geometry import direction_to_chart_mm


def test_rebase_changes_only_chart_and_time_coordinates():
    radius = 6.957e8
    origin = Time("2024-01-01T00:00:00", scale="tai")
    position = torch.tensor(
        [[[0.1 * radius, 0, math.sqrt(0.99) * radius]]], dtype=torch.float64
    )
    chart = direction_to_chart_mm(position, torch.eye(3, dtype=torch.float64), radius)
    coordinates = torch.cat(
        (chart, torch.full((1, 1, 1), 2.0, dtype=torch.float64)), dim=-1
    )
    raster = ObservationRaster(
        stokes=torch.ones(1, 1, 4, 2),
        wavelength_angstrom=torch.tensor([6301.0, 6302.0]),
        coordinates=coordinates,
        ray_direction=torch.tensor([[[0.0, 0.0, -1.0]]]),
        surface_position_m=position,
        stokes_basis=torch.eye(3).reshape(1, 1, 3, 3),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        metadata={
            "ref_time": origin.isot,
            "times": [origin.isot],
            "coordinates": {
                "time_scale": "tai",
                "network_affine": {"center_mm": [0.0, 0.0], "scale_mm": [100.0, 100.0]},
            },
            "ray_geometry": {
                "solar_radius_m": radius,
                "scene_basis_rows": torch.eye(3).tolist(),
            },
        },
    )
    angle = 0.2
    basis = torch.tensor(
        [
            [math.cos(angle), math.sin(angle), 0.0],
            [-math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    scene = SceneContract(
        basis,
        radius,
        float(origin.to_value("unix_tai")) + 3600,
        (0.0, 0.0),
        (100.0, 100.0),
        0.0,
        1.0,
        (-1e5, 1e7),
    )
    rebased = rebase_stokes_raster(raster, scene)
    expected = direction_to_chart_mm(position, basis, radius)
    torch.testing.assert_close(rebased.coordinates[..., :2], expected)
    torch.testing.assert_close(
        rebased.coordinates[..., 2], torch.ones(1, 1, dtype=torch.float64)
    )
    torch.testing.assert_close(raster.coordinates, coordinates)
    assert rebased.stokes.data_ptr() == raster.stokes.data_ptr()
    torch.testing.assert_close(rebased.surface_position_m, raster.surface_position_m)
    torch.testing.assert_close(rebased.ray_direction, raster.ray_direction)
    assert rebase_stokes_raster(rebased, scene) is rebased

    with pytest.raises(ValueError, match="same physical solar radius"):
        rebase_stokes_raster(rebased, replace(scene, solar_radius_m=radius * 1.01))


def test_explicit_time_origin_does_not_require_duplicate_reference_time():
    from prom3theus.observations.scene import stokes_reference_time_tai_seconds

    metadata = {
        "coordinates": {"time_origin": "2024-01-01T00:00:00", "time_scale": "UTC"}
    }
    expected = float(Time("2024-01-01T00:00:00", scale="utc").tai.to_value("unix_tai"))
    assert stokes_reference_time_tai_seconds(metadata) == expected
    with pytest.raises(ValueError, match="time origin"):
        stokes_reference_time_tai_seconds({"coordinates": {}})
