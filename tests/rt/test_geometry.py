import math

import pytest
import torch

from prom3theus.rt.geometry import (
    chart_height_to_position_m,
    chart_to_direction,
    direction_to_chart_mm,
    intersect_sphere_near_side,
    intersect_sphere_near_side_from_local_point,
    project_vectors_to_stokes,
    validate_scene_basis,
)


def test_gnomonic_chart_round_trip_preserves_direction():
    basis = torch.eye(3, dtype=torch.float32)
    chart = torch.tensor(((0.0, 0.0), (50.0, -20.0)))
    direction = chart_to_direction(chart, basis, 695_700_000.0)
    torch.testing.assert_close(
        direction_to_chart_mm(direction, basis, 695_700_000.0),
        chart,
        rtol=2e-6,
        atol=2e-5,
    )


def test_scene_basis_and_stokes_projection_are_explicit():
    basis = validate_scene_basis(torch.eye(3))
    position = chart_height_to_position_m(
        torch.tensor([[12.0, -4.0]]), torch.tensor([250.0]), basis, 695_700_000.0
    )
    assert torch.isfinite(position).all()
    vector = torch.tensor([[[1.0, 2.0, 3.0]]])
    stokes_basis = torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(
        project_vectors_to_stokes(vector, stokes_basis),
        torch.tensor([[[2.0, -1.0, 3.0]]]),
    )


def test_near_side_intersections_work_in_float32_solar_radius_units():
    observer = torch.tensor(((0.0, 0.0, 10.0),))
    ray = torch.tensor(((0.0, 0.0, -1.0),))
    distances = intersect_sphere_near_side(
        observer[:, None], ray[:, None], torch.tensor(((3.0, 2.0),))
    )
    torch.testing.assert_close(distances, torch.tensor(((7.0, 8.0),)))
    local = torch.tensor([[0.0, 0.0, 1.0]])
    offset = intersect_sphere_near_side_from_local_point(
        local, ray, torch.tensor([1.001])
    )
    assert offset.dtype == torch.float32 and torch.isfinite(offset).all()


def test_near_tangent_local_intersection_remains_finite():
    impact = 0.9999
    ray = torch.tensor([[impact, 0.0, -math.sqrt(1.0 - impact**2)]])
    local = torch.tensor([[0.0, 0.0, 1.0]])
    offset = intersect_sphere_near_side_from_local_point(
        local, ray, torch.tensor([1.0])
    )
    assert torch.isfinite(offset).all()


def test_geometry_preserves_supported_precision_and_rejects_negative_radii():
    basis = validate_scene_basis(torch.eye(3, dtype=torch.float64))
    assert basis.dtype == torch.float64

    with pytest.raises(ValueError, match="strictly positive"):
        intersect_sphere_near_side(
            torch.tensor([[0.0, 0.0, 10.0]]),
            torch.tensor([[0.0, 0.0, -1.0]]),
            -2.0,
        )
    with pytest.raises(ValueError, match="solar_radius_m"):
        chart_to_direction(torch.zeros(1, 2), torch.eye(3), 0.0)
