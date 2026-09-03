import pytest
import torch

from prom3theus.rt import (
    GeometricHeightPath,
    OpticalDepthPath,
    PolarizedFormalSolver,
    RayDistancePath,
    scalar_formal_solution,
)


def _constant_problem():
    dtype = torch.float64
    grid = torch.tensor([-2.0, 0.0], dtype=dtype)
    matrix_value = torch.tensor(
        [
            [1.4, 0.12, -0.04, 0.09],
            [0.12, 1.4, 0.03, 0.02],
            [-0.04, -0.03, 1.4, 0.05],
            [0.09, -0.02, -0.05, 1.4],
        ],
        dtype=dtype,
    )
    matrix = matrix_value.expand(2, 3, 4, 4).clone()
    source_value = torch.tensor([0.8, 0.0, 0.0, 0.0], dtype=dtype)
    source = source_value.expand(2, 3, 4).clone()
    bottom = torch.tensor(
        [[1.1, 0.02, -0.01, 0.03], [0.9, -0.03, 0.02, -0.01], [1.0, 0.01, 0.04, 0.02]],
        dtype=dtype,
    )
    return grid, matrix_value, matrix, source_value, source, bottom


def test_optical_depth_path_matches_constant_matrix_solution():
    grid, matrix_value, matrix, source_value, source, bottom = _constant_problem()
    mu = 0.73
    actual = PolarizedFormalSolver()(
        matrix,
        source,
        grid,
        path=OpticalDepthPath(mu),
        bottom_boundary=bottom,
    )
    delta_tau = (10.0 ** grid[1] - 10.0 ** grid[0]) / mu
    attenuation = torch.matrix_exp(-matrix_value * delta_tau)
    expected = torch.stack(
        [source_value + attenuation @ (ray - source_value) for ray in bottom]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_geometric_and_ray_paths_are_equivalent_when_units_match():
    dtype = torch.float64
    grid = torch.tensor([-2.0, -1.0, 0.0], dtype=dtype)
    tau = torch.pow(torch.tensor(10.0, dtype=dtype), grid)
    extinction = torch.full((2, 3), 2.5, dtype=dtype)
    height = -(tau - tau[0]) / extinction[:, :1]
    matrix = 1.4 * torch.eye(4, dtype=dtype).expand(2, 3, 2, 4, 4).clone()
    source = torch.zeros(2, 3, 2, 4, dtype=dtype)
    source[..., 0] = torch.tensor((0.6, 0.8, 1.0), dtype=dtype)[None, :, None]
    solver = PolarizedFormalSolver()

    optical = solver(matrix, source, grid, path=OpticalDepthPath(0.71))
    geometric = solver(
        matrix,
        source,
        grid,
        path=GeometricHeightPath(height, 0.71),
        reference_extinction_m1=extinction,
    )
    ray_distance = 100.0 - height / 0.71
    ray = solver(
        matrix * extinction[..., :, None, None, None],
        source,
        grid,
        path=RayDistancePath(ray_distance),
    )
    torch.testing.assert_close(geometric, optical, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(ray, optical, rtol=2e-12, atol=2e-12)


def test_path_contract_rejects_cross_mode_inputs():
    grid, _, matrix, _, source, _ = _constant_problem()
    with pytest.raises(ValueError, match="only valid for GeometricHeightPath"):
        PolarizedFormalSolver()(
            matrix,
            source,
            grid,
            path=OpticalDepthPath(),
            reference_extinction_m1=torch.ones(2),
        )
    with pytest.raises(ValueError, match="requires reference_extinction"):
        PolarizedFormalSolver()(
            matrix,
            source,
            grid,
            path=GeometricHeightPath(torch.tensor([1.0, 0.0])),
        )


def test_physical_paths_keep_reversed_intervals_signed_and_differentiable():
    grid, _, matrix, _, source, bottom = _constant_problem()
    solver = PolarizedFormalSolver()
    extinction = torch.ones(2, dtype=matrix.dtype)

    reversed_height = torch.tensor(
        [0.0, 1.0], dtype=matrix.dtype, requires_grad=True
    )
    geometric = solver(
        matrix,
        source,
        grid,
        path=GeometricHeightPath(reversed_height),
        bottom_boundary=bottom,
        reference_extinction_m1=extinction,
    )
    reversed_distance = torch.tensor(
        [1.0, 0.0], dtype=matrix.dtype, requires_grad=True
    )
    ray = solver(
        matrix,
        source,
        grid,
        path=RayDistancePath(reversed_distance),
        bottom_boundary=bottom,
    )

    assert torch.isfinite(geometric).all()
    assert torch.isfinite(ray).all()
    torch.testing.assert_close(geometric, ray)
    (geometric.square().mean() + ray.square().mean()).backward()
    assert torch.isfinite(reversed_height.grad).all()
    assert torch.isfinite(reversed_distance.grad).all()


def test_physical_paths_accept_zero_width_intervals():
    grid, _, matrix, _, source, bottom = _constant_problem()
    solver = PolarizedFormalSolver()
    distance = torch.tensor([2.0, 2.0], dtype=matrix.dtype, requires_grad=True)

    actual = solver(
        matrix,
        source,
        grid,
        path=RayDistancePath(distance),
        bottom_boundary=bottom,
    )

    torch.testing.assert_close(actual, bottom)
    actual.square().mean().backward()
    assert torch.isfinite(distance.grad).all()


def test_formal_solvers_reject_nonfinite_boundaries_and_negative_opacity():
    grid, _, matrix, _, source, bottom = _constant_problem()
    bottom[0, 0] = torch.nan
    with pytest.raises(ValueError, match="bottom_boundary"):
        PolarizedFormalSolver()(
            matrix,
            source,
            grid,
            path=OpticalDepthPath(),
            bottom_boundary=bottom,
        )

    opacity = torch.ones(2, 3, dtype=torch.float64)
    opacity[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        scalar_formal_solution(
            opacity,
            torch.ones_like(opacity),
            grid,
            path=OpticalDepthPath(),
        )


def test_scalar_solution_requires_explicit_optical_path():
    grid = torch.tensor([-3.0, -1.4, 0.0], dtype=torch.float64)
    opacity = torch.full((3, 2), 1.7, dtype=torch.float64)
    source = torch.full_like(opacity, 0.35)
    actual = scalar_formal_solution(
        opacity,
        source,
        grid,
        path=OpticalDepthPath(0.73),
        bottom_boundary=torch.full((2,), 1.2, dtype=torch.float64),
    )
    optical_path = 1.7 * (10.0 ** grid[-1] - 10.0 ** grid[0]) / 0.73
    expected = 0.35 + (1.2 - 0.35) * torch.exp(-optical_path)
    torch.testing.assert_close(actual, expected.expand(2), rtol=1e-12, atol=1e-12)


def test_optical_depth_path_rejects_unrepresentable_tau_values():
    opacity = torch.ones(2, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="cannot be represented"):
        scalar_formal_solution(
            opacity,
            torch.ones_like(opacity),
            torch.tensor([-1000.0, -999.0], dtype=torch.float64),
            path=OpticalDepthPath(),
        )
