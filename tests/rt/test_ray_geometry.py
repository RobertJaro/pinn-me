import math

import pytest
import torch

from prom3theus.rt.geometry import chart_to_direction, position_to_chart_height
from prom3theus.rt.ray_geometry import (
    canonical_basis, rotate_stokes_basis, sample_shell_characteristic, shell_characteristic,
)


def _trace(ray, *, coords=None, heights=(-1500., 2000.), bounds=None):
    ray = torch.tensor(ray, dtype=torch.float64)
    if coords is None:
        coords = torch.zeros_like(ray)
    return shell_characteristic(
        coords, ray, scene_basis=torch.eye(3, dtype=torch.float64),
        solar_radius_m=695700000., height_bounds_m=heights, chart_bounds_megameter=bounds,
    )


def test_vertical_and_oblique_geometry_reach_true_lower_sphere():
    ray = [[0., 0., -1.], [.6, 0., -.8]]
    result = _trace(ray)
    radius = 695700000.
    torch.testing.assert_close(torch.linalg.vector_norm(result["outgoing_position_m"], dim=-1) - radius, torch.full((2,), 2000., dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(result["inflow_position_m"], dim=-1) - radius, torch.full((2,), -1500., dtype=torch.float64), atol=1e-6, rtol=0)
    displacement = result["outgoing_position_m"] - result["inflow_position_m"]
    torch.testing.assert_close(displacement, result["path_length_m"][:, None] * result["direction"], atol=1e-7, rtol=1e-10)
    assert abs(result["path_length_m"][0] - 3500.) < 1e-8
    assert result["path_length_m"][1] > result["path_length_m"][0]


def test_tangent_is_rejected_instead_of_becoming_thermal_inflow():
    with pytest.raises(ValueError, match="true lower"):
        _trace([[math.sqrt(1 - .001**2), 0., -.001]])


def test_lateral_boundary_requires_a_real_halo():
    bounds = torch.tensor([[-.0001, -.01], [.0001, .01]])
    with pytest.raises(ValueError, match="lateral"):
        _trace([[.6, 0., -.8]], bounds=bounds)
    _trace([[.6, 0., -.8]], bounds=torch.tensor([[-1., -1.], [1., 1.]]))


def test_basis_rotation_matches_spin_two_and_preserves_circular_sign():
    n = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    source = canonical_basis(n)
    angle = .37
    target = canonical_basis(n, torch.tensor([math.cos(angle), math.sin(angle), 0.], dtype=torch.float64))
    stokes = torch.tensor([[[2.], [1.], [0.], [.3]]], dtype=torch.float64)
    rotated = rotate_stokes_basis(stokes, source, target)
    torch.testing.assert_close(rotated, stokes.new_tensor([[[2.], [math.cos(2*angle)], [-math.sin(2*angle)], [.3]]]))
    torch.testing.assert_close(rotate_stokes_basis(rotated, target, source), stokes)
    torch.testing.assert_close(torch.linalg.det(source), torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="singularity"):
        canonical_basis(torch.tensor([[1., 0., 0.]]))


def test_submetre_solar_shell_is_not_lost_to_large_radius_subtraction():
    result = _trace([[0., 0., -1.]], heights=(-.01, .02))
    assert result["path_length_m"].item() == pytest.approx(.03, abs=1e-12)


def _sample_local_shell(dtype, heights, *, requires_grad=False):
    coordinates = torch.tensor([[100., -30., 1.], [0., 0., 0.]], dtype=dtype,
                               requires_grad=requires_grad)
    ray = torch.tensor([[.8, 0., -.6], [.6, 0., -.8]], dtype=dtype,
                       requires_grad=requires_grad)
    bounds = torch.tensor(heights, dtype=dtype, requires_grad=requires_grad)
    basis = torch.eye(3, dtype=dtype)
    geometry = shell_characteristic(coordinates, ray, scene_basis=basis,
        solar_radius_m=695700000., height_bounds_m=bounds)
    distance = geometry["path_length_m"][:, None] * torch.linspace(0, 1, 33, dtype=dtype)
    samples = sample_shell_characteristic(geometry, distance,
        solar_radius_m=695700000.)
    return geometry, samples, (coordinates, ray, bounds)


@pytest.mark.parametrize("bounds,atol", [
    ((-.01, .02), 5e-9),
    ((-1500., 2000.), 5e-4),
    ((-1e6, 1.5e6), .5),
])
def test_float32_shell_sampling_preserves_local_height_against_float64(bounds, atol):
    geometry, (chart, height, position), _ = _sample_local_shell(torch.float32, bounds)
    reference, (chart64, height64, position64), _ = _sample_local_shell(torch.float64, bounds)
    assert all(value.dtype == torch.float32 for value in geometry.values())
    assert chart.dtype == height.dtype == position.dtype == torch.float32
    torch.testing.assert_close(height.double(), height64, rtol=0, atol=atol)
    torch.testing.assert_close(height[:, [0, -1]], height.new_tensor(bounds[::-1]).expand(2, 2),
                               rtol=0, atol=atol)
    torch.testing.assert_close(chart.double(), chart64, rtol=2e-7, atol=2e-5)
    torch.testing.assert_close(geometry["path_length_m"].double(), reference["path_length_m"],
                               rtol=2e-7, atol=1e-8)
    assert (height[:, 1:] < height[:, :-1]).all()
    # The optional double path still agrees with independent global geometry.
    expected_chart, expected_height = position_to_chart_height(
        position64, torch.eye(3, dtype=torch.float64), 695700000.)
    torch.testing.assert_close(height64, expected_height, rtol=0, atol=3e-7)
    torch.testing.assert_close(chart64, expected_chart, rtol=1e-12, atol=1e-12)


def test_float32_local_sampling_gradients_match_float64():
    gradients = []
    for dtype in (torch.float32, torch.float64):
        _, (chart, height, _), inputs = _sample_local_shell(
            dtype, (-1e6, 1.5e6), requires_grad=True)
        loss = chart[:, 1:-1].square().mean() + (height[:, 1:-1] / 1e6).square().mean()
        gradients.append(torch.autograd.grad(loss, inputs))
    for actual, expected in zip(*gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_height_endpoints_remain_shell_boundaries_with_rotated_scene(dtype):
    angle = .63
    basis = torch.tensor([[math.cos(angle), 0., -math.sin(angle)], [0., 1., 0.],
                          [math.sin(angle), 0., math.cos(angle)]], dtype=dtype)
    coordinates = torch.tensor([[37., -21., 0.]], dtype=dtype)
    ray = -basis[2].expand_as(coordinates)
    bounds = torch.tensor([-1500., 2000.], dtype=dtype, requires_grad=True)
    geometry = shell_characteristic(coordinates, ray, scene_basis=basis,
        solar_radius_m=695700000., height_bounds_m=bounds)
    distance = geometry["path_length_m"][:, None] * torch.tensor([0., 1.], dtype=dtype)
    _, height, _ = sample_shell_characteristic(geometry, distance,
        solar_radius_m=695700000.)
    torch.testing.assert_close(height[0], bounds.flip(0), rtol=1e-6, atol=1e-7)
    derivative = torch.autograd.grad(height.sum(), bounds)[0]
    torch.testing.assert_close(derivative, torch.ones_like(bounds), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("angle", [.17, .63, 1.13])
def test_float32_chart_reference_avoids_solar_scale_rotation_roundoff(angle):
    basis = torch.tensor([[math.cos(angle), 0., -math.sin(angle)], [0., 1., 0.],
                          [math.sin(angle), 0., math.cos(angle)]])
    coordinates = torch.tensor([[.001, 0., 0.]])
    geometry = shell_characteristic(coordinates, -basis[2].expand_as(coordinates),
        scene_basis=basis, solar_radius_m=695700000., height_bounds_m=(-1500., 2000.))
    chart, height, _ = sample_shell_characteristic(geometry,
        -geometry["outgoing_offset_m"][:, None], solar_radius_m=695700000.)
    # At the nominal reference sphere, even kilometre-scale chart coordinates
    # remain accurate without a global Cartesian round trip (25 m error here).
    torch.testing.assert_close(chart[:, 0], coordinates[:, :2], rtol=2e-7, atol=1e-10)
    torch.testing.assert_close(height, torch.zeros_like(height), rtol=0, atol=0)


def test_float32_radial_ray_on_lateral_boundary_tolerates_projection_roundoff():
    angle = .63
    basis = torch.tensor([[math.cos(angle), 0., -math.sin(angle)], [0., 1., 0.],
                          [math.sin(angle), 0., math.cos(angle)]])
    coordinates = torch.tensor([[.001, 0., 0.]])
    ray = -chart_to_direction(coordinates[:, :2], basis, 695700000.)
    shell_characteristic(coordinates, ray, scene_basis=basis, solar_radius_m=695700000.,
        height_bounds_m=(-1e6, 1.5e6),
        chart_bounds_megameter=torch.tensor([[-.001, -1.], [.001, 1.]]))
    # The roundoff allowance must not hide a physically misplaced boundary.
    with pytest.raises(ValueError, match="lateral"):
        shell_characteristic(coordinates, ray, scene_basis=basis, solar_radius_m=695700000.,
            height_bounds_m=(-1e6, 1.5e6),
            chart_bounds_megameter=torch.tensor([[-.001, -1.], [.0009, 1.]]))
