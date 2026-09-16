import numpy as np
import pytest
import torch

from prom3theus.rt.spherical_potential import (
    source_cell_quadrature,
    spherical_neumann_kernel,
    spherical_potential_matrix,
    spherical_source_cells,
)


def test_constant_radial_source_and_near_surface_limit():
    cells = spherical_source_cells((-np.pi, np.pi), (-np.pi / 2, np.pi / 2), 16)
    unit = np.array([0.8, 0.48, 0.36])
    radii = np.array([1.000001, 1.001, 1.1, 2.0])
    points = radii[:, None] * unit
    matrix = spherical_potential_matrix(cells, points)
    result = matrix @ torch.ones(len(cells))
    # The retained l=0 mode decays radially as 1/r^2, rather than a vertical
    # Cartesian mean mode. Also checks integration only metres above the Sun.
    expected = torch.tensor(unit[None] / radii[:, None] ** 2, dtype=result.dtype)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)


def test_dipole_reconstruction_and_source_resolution_convergence():
    points = np.array([[0.8, 0.48, 0.36], [0.0, 0.0, 1.0]]) * 1.1
    radius = np.linalg.norm(points, axis=-1)[:, None]
    expected = (3 * points[:, 2:3] * points / radius**2 - [0, 0, 1]) / radius**3
    errors = []
    for size in [32, 64, 256]:
        cells = spherical_source_cells((-np.pi, np.pi), (-np.pi / 2, np.pi / 2), size)
        directions, weights = source_cell_quadrature(cells, 4)
        br = (2 * directions[:, :, 2] * weights).sum(1)
        matrix = spherical_potential_matrix(cells, points)
        result = (matrix @ torch.tensor(br, dtype=matrix.dtype)).numpy()
        errors.append(np.max(np.abs(result - expected)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.001


def test_local_br_is_recovered_from_above_without_burial():
    cells = spherical_source_cells((-0.12, 0.12), (-0.12, 0.12), 8)
    # Query inside one source cell, away from discontinuities at cell edges.
    cell = cells[19]
    longitude, mu = cell[:2].mean(), cell[2:].mean()
    unit = np.array(
        [
            np.sqrt(1 - mu**2) * np.cos(longitude),
            np.sqrt(1 - mu**2) * np.sin(longitude),
            mu,
        ]
    )
    source = torch.arange(64, dtype=torch.float32) - 30
    heights = [1e-3, 1e-5, 1e-7]
    result = (
        spherical_potential_matrix(cells, np.array([unit * (1 + h) for h in heights]))
        @ source
    )
    br = result.double() @ torch.tensor(unit)
    errors = (br - source[19]).abs()
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.001


def test_kernel_is_curl_free_and_divergence_free():
    source = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    point = torch.tensor([1.1, 0.05, -0.04], dtype=torch.float64, requires_grad=True)

    def field(x):
        return spherical_neumann_kernel(x[None], source)[0].sum(0)

    jacobian = torch.autograd.functional.jacobian(field, point)
    torch.testing.assert_close(jacobian, jacobian.T, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        jacobian.trace(), torch.zeros((), dtype=point.dtype), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("radius", [0.9, 1.0])
def test_interior_and_surface_extrapolation_are_rejected(radius):
    cells = spherical_source_cells((-0.1, 0.1), (-0.1, 0.1), 4)
    with pytest.raises(ValueError, match="strictly above"):
        spherical_potential_matrix(cells, [[radius, 0, 0]])
    with pytest.raises(ValueError, match="strictly above"):
        spherical_neumann_kernel(
            torch.tensor([[radius, 0, 0]]), torch.tensor([[1.0, 0, 0]])
        )


def test_photospheric_limit_radial_identity_monopole_and_dipole_convergence():
    from prom3theus.rt.spherical_potential import spherical_photosphere_matrix

    errors = []
    for n in (8, 16):
        cells = spherical_source_cells((-np.pi, np.pi), (-np.pi / 2, np.pi / 2), n)
        unit = torch.tensor(
            source_cell_quadrature(cells, 1)[0][:, 0], dtype=torch.float32
        )
        operator = spherical_photosphere_matrix(cells)
        radial = (operator * unit[:, :, None]).sum(1)
        torch.testing.assert_close(radial, torch.eye(len(cells)), atol=3e-7, rtol=3e-7)
        torch.testing.assert_close(
            operator @ torch.ones(len(cells)), unit, atol=1e-4, rtol=1e-4
        )
        nodes, weights = source_cell_quadrature(cells, 4)
        moment = np.array([0.4, -0.7, 0.6])
        br = torch.tensor((2 * (nodes @ moment) * weights).sum(-1), dtype=torch.float32)
        result = operator @ br
        m = torch.tensor(moment, dtype=torch.float32)
        expected = 3 * (unit @ m)[:, None] * unit - m
        errors.append((result - expected).square().mean().sqrt().item())
    assert errors[1] < 0.065 and errors[1] < 0.6 * errors[0]


def test_photospheric_tangent_limit_stable_under_numerical_height_refinement():
    from prom3theus.rt.spherical_potential import spherical_photosphere_matrix

    cells = spherical_source_cells((-0.1, 0.1), (-0.4, -0.2), 4)
    coarse = spherical_photosphere_matrix(cells, relative_height=1e-4)
    fine = spherical_photosphere_matrix(cells, relative_height=5e-5)
    # Compare all source weights, not only a specially cancelling source map.
    torch.testing.assert_close(coarse, fine, atol=2e-6, rtol=2e-5)
