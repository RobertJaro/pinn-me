"""Independent coordinate and unit checks for the spherical potential pipeline."""

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from prom3theus.config.schema import PotentialBoundaryConfig, PotentialPhotosphereConfig
from prom3theus.inversion.data_terms.potential_boundary import (
    ProgressivePotentialBoundary,
)
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.rt.geometry import project_vectors_to_stokes
from prom3theus.rt.spherical_potential import (
    source_cell_quadrature,
    spherical_neumann_kernel,
    spherical_potential_matrix,
    spherical_source_cells,
)


def test_kernel_rotates_as_global_cartesian_vector_under_general_rotation():
    torch.manual_seed(41)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    source = torch.randn(17, 3, dtype=torch.float64)
    source /= source.norm(dim=-1, keepdim=True)
    points = torch.randn(5, 3, dtype=torch.float64)
    points *= 1.3 / points.norm(dim=-1, keepdim=True)
    original = spherical_neumann_kernel(points, source)
    rotated = spherical_neumann_kernel(points @ rotation.T, source @ rotation.T)
    torch.testing.assert_close(rotated, original @ rotation.T, atol=1e-12, rtol=1e-12)


def test_tilted_dipole_all_components_and_observer_projection():
    moment = np.array([0.4, -0.7, 0.6])
    points = np.array([[1.4, 0.3, -0.2], [-0.7, 1.3, 0.4], [0.3, -0.5, 1.6]])
    radius = np.linalg.norm(points, axis=-1, keepdims=True)
    direction = points / radius
    expected = (3 * (direction @ moment)[:, None] * direction - moment) / radius**3
    cells = spherical_source_cells((-np.pi, np.pi), (-np.pi / 2, np.pi / 2), 128)
    source, weights = source_cell_quadrature(cells, 4)
    br = (2 * (source @ moment) * weights).sum(-1)
    result = spherical_potential_matrix(cells, points) @ torch.tensor(
        br, dtype=torch.float32
    )
    torch.testing.assert_close(
        result, torch.tensor(expected, dtype=result.dtype), atol=8e-4, rtol=3e-3
    )
    # An independently selected observer basis must see the same rotated dipole.
    basis = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    projected = project_vectors_to_stokes(result, basis)
    torch.testing.assert_close(
        projected,
        torch.tensor(expected[:, [2, 0, 1]], dtype=result.dtype),
        atol=8e-4,
        rtol=3e-3,
    )


class CartesianSource(nn.Module):
    def __init__(self, vector):
        super().__init__()
        self.vector = nn.Parameter(vector.float())

    def evaluate_position_rsun(self, positions, time):
        return {"magnetic_field": self.vector.expand_as(positions) * (1 + time)}


def make_term(domain, scene):
    options = PotentialBoundaryConfig(
        enabled=True,
        start_step=0,
        ramp_steps=0,
        grid_size=4,
        source_supersampling=2,
        top_grid_size=2,
        side_horizontal_points=2,
        side_height_points=2,
        photosphere=PotentialPhotosphereConfig(
            enabled=True,
            start_step=0,
            ramp_steps=0,
            batch_size=4,
        ),
    ).to_dict()
    return ProgressivePotentialBoundary(
        options, domain, domain, scene, observation_times_hours=[0.0, 1.0]
    )


def test_source_radial_projection_and_full_pipeline_longitude_rotation_and_units():
    domain = SphericalShellDomain(
        2.9, (-0.1, 0.1), (-0.4, -0.2), (0.0, 1.0), (0.0, 30.0), 696e6
    )
    vector = torch.tensor([80.0, -35.0, 51.0], dtype=torch.float64)
    angle = 0.7  # Rotated footprint crosses the conventional +/-pi seam.
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    term = make_term(domain, torch.eye(3))
    rotated = make_term(
        replace(domain, longitude_center_rad=domain.longitude_center_rad + angle),
        rotation,
    )
    for item, value in ((term, vector), (rotated, rotation @ vector)):
        item.prepare(CartesianSource(value))
        nodes = item.source_directions.reshape(-1, 4, 3)
        expected_br = ((nodes @ value) * item.source_weights).sum(-1)
        torch.testing.assert_close(
            item.source_reference[0].double(),
            expected_br,
            atol=1e-5,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            item.source_positions.norm(dim=-1),
            torch.full_like(item.source_positions[:, 0], item.radius),
        )
    torch.testing.assert_close(
        rotated.positions, term.positions @ rotation.T, atol=1e-6, rtol=1e-12
    )
    torch.testing.assert_close(
        rotated.targets.double(),
        term.targets.double() @ rotation.T,
        atol=2e-5,
        rtol=2e-5,
    )
    # Changing the scene basis alone must not rotate a global Cartesian solution.
    other_scene = make_term(domain, rotation)
    other_scene.prepare(CartesianSource(vector))
    torch.testing.assert_close(other_scene.targets, term.targets, atol=0, rtol=0)
    # Same dimensionless geometry at twice the physical radius: same gauss field,
    # four times the physical magnetic flux. No missing or extra radius factor.
    scaled = make_term(
        replace(
            domain,
            solar_radius_m=2 * domain.solar_radius_m,
            height_bounds_Mm=(0.0, 60.0),
        ),
        torch.eye(3),
    )
    scaled.prepare(CartesianSource(vector))
    torch.testing.assert_close(scaled.targets, term.targets, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(scaled.source_flux, 4 * term.source_flux)
