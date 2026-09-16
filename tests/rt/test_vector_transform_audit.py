"""Independent checks spanning chart inputs, Cartesian fields, and derivatives."""

import pytest
import torch
from torch import nn

from prom3theus.core import (
    cartesian_to_spherical,
    project_cartesian_to_spherical,
    project_spherical_to_observer,
)
from prom3theus.instruments.base import MagneticAzimuthConvention
from prom3theus.inversion.constraints.magnetofluid import (
    MagnetofluidConstraints,
    _curl,
    _divergence,
)
from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.rt.geometry import project_vectors_to_stokes


class AnalyticCartesianHead(nn.Module):
    """Reconstruct physical points independently from normalized chart inputs."""

    def __init__(self, basis):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("basis", basis)
        self.register_buffer(
            "matrix",
            torch.tensor(
                [[2.0, 3.0, -1.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]],
                dtype=torch.float64,
            ),
        )

    def forward(self, inputs):
        # Tests use a common spatial scale of 10 Mm, centre [2,-3] Mm,
        # and time centre 5 h with scale 2 h.
        xy = inputs[:, :2] * 10 + inputs.new_tensor([2.0, -3.0])
        local = torch.cat((xy / 696, torch.ones_like(xy[:, :1])), -1)
        direction = (local / local.norm(dim=-1, keepdim=True)) @ self.basis
        position = direction * (1 + inputs[:, 2:3] * 10 / 696)
        time = inputs[:, 3:4] * 2 + 5
        magnetic = self.amplitude * (
            position @ self.matrix.T * 100 + time * inputs.new_tensor([1.0, -2.0, 3.0])
        )
        zeros = inputs[:, :1] * 0
        return torch.cat((zeros.expand(-1, 4), magnetic / 100, zeros.expand(-1, 2)), -1)


@pytest.fixture
def model():
    angle = torch.tensor(0.63, dtype=torch.float64)
    c, s = angle.cos(), angle.sin()
    basis = torch.stack(
        (
            torch.stack((c, s, c * 0)),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.stack((s, -c, c * 0)),
        )
    )
    model = StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(50.0, -0.1),
        time_dependent=True,
        uniform_spatial_scaling=True,
        height_input_scale_m=1e7,
        spatial_coordinate_center_mm=(2.0, -3.0),
        time_coordinate_center_hours=5.0,
        time_coordinate_scale_hours=2.0,
        scene_geometry_config={"solar_radius_m": 696e6, "scene_basis": basis},
        model_config={
            "type": "mlp",
            "dim": 8,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    ).double()
    model.network = AnalyticCartesianHead(basis)
    return model


def test_rotated_chart_yields_global_cartesian_field_and_physical_derivatives(model):
    coords = torch.tensor([[22.0, 7.0, 7.0], [-13.0, -8.0, 4.0]], dtype=torch.float64)
    height = torch.tensor([0.0, 3e6], dtype=torch.float64)
    positions = model.position_from_coords_height(coords, height)
    by_chart = model.evaluate_chart_height_points(coords, height)["magnetic_field"]
    expected = positions / 696e6 @ model.network.matrix.T * 100 + coords[
        :, 2:
    ] * coords.new_tensor([1.0, -2.0, 3.0])
    torch.testing.assert_close(by_chart, expected)
    torch.testing.assert_close(
        model.evaluate_position_points(positions, coords[:, 2])["magnetic_field"],
        expected,
    )
    physics = MagnetofluidConstraints(vector_basis_matches_spatial_coordinates=True)
    state = physics.build_state(
        model,
        positions,
        coords[:, 2],
        {"magnetic_divergence"},
        create_graph=True,
        height_group_shape=(1, 2),
    )
    jac = state.derivative("magnetic")
    torch.testing.assert_close(
        jac, (model.network.matrix * 100 / 696e6).expand(2, -1, -1)
    )
    torch.testing.assert_close(
        state.time_derivative("magnetic"),
        coords.new_tensor([1.0, -2.0, 3.0]).expand(2, -1) / 3600,
    )
    torch.testing.assert_close(_divergence(jac), coords.new_full((2,), 1600 / 696e6))
    torch.testing.assert_close(
        _curl(jac), coords.new_tensor([-14.0, -8.0, -7.0]).expand(2, -1) * 100 / 696e6
    )
    jac.square().sum().backward()
    assert model.network.amplitude.grad.abs() > 0


def test_decoder_has_no_component_clipping_or_sign_restriction(model):
    raw = torch.zeros(4, 9, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        raw[:, 4:7] = torch.tensor(
            [
                [-100.0, 0.0, 100.0],
                [0.0, -50.0, 50.0],
                [50.0, 100.0, -50.0],
                [0.0, 0.0, 0.0],
            ]
        )
    b = model._decode_raw(raw, geometric_height_m=torch.zeros(4, dtype=torch.float64))[
        "magnetic_field"
    ]
    torch.testing.assert_close(b, raw[:, 4:7] * 100)
    b.sum().backward()
    torch.testing.assert_close(
        raw.grad[:, 4:7], torch.full((4, 3), 100.0, dtype=torch.float64)
    )


def test_observer_export_and_synthesis_projection_at_multiple_depths():
    torch.manual_seed(12)
    basis, _ = torch.linalg.qr(torch.randn(4, 3, 3, dtype=torch.float64))
    vectors = torch.randn(4, 7, 3, dtype=torch.float64) * 1000
    positions = torch.randn(4, 7, 3, dtype=torch.float64)
    direct = project_vectors_to_stokes(vectors, basis)
    angles = cartesian_to_spherical(positions)
    exported = project_spherical_to_observer(
        project_cartesian_to_spherical(vectors, angles), angles, basis
    )
    torch.testing.assert_close(exported, direct)
    torch.testing.assert_close(direct.norm(dim=-1), vectors.norm(dim=-1))
    synthesis = MagneticAzimuthConvention(name="hmi", offset_deg=90).to_synthesis_frame(
        direct
    )
    torch.testing.assert_close(
        synthesis, torch.stack((-direct[..., 1], direct[..., 0], direct[..., 2]), -1)
    )
    torch.testing.assert_close(synthesis.norm(dim=-1), vectors.norm(dim=-1))


def test_ray_synthesis_and_point_evaluation_use_same_global_vector(model):
    coords = torch.tensor([[22.0, 7.0, 7.0], [-13.0, -8.0, 4.0]], dtype=torch.float64)
    surface = model.position_from_coords_height(
        coords, torch.zeros(2, dtype=torch.float64)
    )
    outward = surface / surface.norm(dim=-1, keepdim=True)
    tangent = model.scene_basis[0].expand_as(outward)
    tangent = tangent - (tangent * outward).sum(-1, keepdim=True) * outward
    # Inclined rays are needed to expose errors hidden by radial-only tracing.
    ray = -outward + 0.2 * tangent
    ray /= ray.norm(dim=-1, keepdim=True)
    sampled, trace = model.trace_rays(
        coords, ray, torch.tensor([2.5e6, 1e6, 0.0, -1e5], dtype=torch.float64)
    )
    expected = trace.position_m / 696e6 @ model.network.matrix.T * 100 + coords[
        :, None, 2:
    ] * coords.new_tensor([1.0, -2.0, 3.0])
    torch.testing.assert_close(sampled.magnetic_field, expected)
    pointwise = model.evaluate_position_rsun(
        trace.position_m / 696e6, coords[:, 2:3].expand(2, 4)
    )["magnetic_field"]
    torch.testing.assert_close(sampled.magnetic_field, pointwise)
