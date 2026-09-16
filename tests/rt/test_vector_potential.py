"""Checks for the Cartesian vector-potential magnetic representation."""

import torch
from torch import nn

from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints
from prom3theus.rt import StratifiedAtmosphereModel


class LinearVectorPotentialHead(nn.Module):
    """Return ``A_z = alpha * X`` in physical Carrington coordinates."""

    def __init__(
        self, solar_radius_m: float, height_scale_m: float, magnetic_scale_gauss: float
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(2.0e-6, dtype=torch.float64))
        self.solar_radius_m = solar_radius_m
        self.height_scale_m = height_scale_m
        self.vector_potential_scale = magnetic_scale_gauss * height_scale_m

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        chart_xy_m = inputs[:, :2] * self.height_scale_m
        local = torch.cat(
            (
                chart_xy_m / self.solar_radius_m,
                torch.ones_like(chart_xy_m[:, :1]),
            ),
            dim=-1,
        )
        direction = local / torch.linalg.vector_norm(local, dim=-1, keepdim=True)
        position_m = direction * (
            self.solar_radius_m + inputs[:, 2:3] * self.height_scale_m
        )
        zero = inputs[:, :1] * 0.0
        vector_potential = torch.cat(
            (zero, zero, self.alpha * position_m[:, 0:1]), dim=-1
        )
        return torch.cat(
            (
                zero.expand(-1, 4),
                vector_potential / self.vector_potential_scale,
                zero.expand(-1, 2),
            ),
            dim=-1,
        )


def _model() -> StratifiedAtmosphereModel:
    model = StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(5.0, -0.1),
        magnetic_representation="vector_potential",
        magnetic_scale_gauss=100.0,
        height_input_scale_m=1.0e7,
        uniform_spatial_scaling=True,
        scene_geometry_config={
            "solar_radius_m": 696.0e6,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 8,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    ).double()
    model.network = LinearVectorPotentialHead(
        model.solar_radius_value_m,
        model.height_input_scale_m,
        model.magnetic_scale_gauss,
    )
    return model


def test_vector_potential_decodes_a_and_physical_curl_b():
    model = _model()
    coords = torch.tensor(
        [[12.0, -8.0, 0.0], [-20.0, 15.0, 1.5]], dtype=torch.float64
    )
    heights = torch.tensor([0.0, 2.0e6], dtype=torch.float64)
    positions = model.position_from_coords_height(coords, heights)
    fields = model.evaluate_chart_height_points(coords, heights)
    alpha = model.network.alpha.detach()

    expected_a = torch.stack(
        (
            torch.zeros_like(positions[:, 0]),
            torch.zeros_like(positions[:, 0]),
            alpha * positions[:, 0],
        ),
        dim=-1,
    )
    expected_b = torch.stack(
        (
            torch.zeros_like(alpha).expand_as(positions[:, 0]),
            -alpha.expand_as(positions[:, 0]),
            torch.zeros_like(positions[:, 0]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(fields["vector_potential"], expected_a)
    torch.testing.assert_close(fields["magnetic_field"], expected_b, rtol=1e-9, atol=1e-12)
    assert model.output_names[4:7] == ("a_x", "a_y", "a_z")
    assert model.reference_metadata()["magnetic_representation"] == "vector_potential"

    fields["magnetic_field"].square().mean().backward()
    assert model.network.alpha.grad is not None
    assert model.network.alpha.grad.abs() > 0


def test_vector_potential_divergence_is_exactly_zero_without_a_b_jacobian():
    model = _model()
    coords = torch.tensor(
        [[5.0, -3.0, 0.0], [-9.0, 4.0, 0.0]], dtype=torch.float64
    )
    heights = torch.tensor([0.0, 1.0e6], dtype=torch.float64)
    positions = model.position_from_coords_height(coords, heights)
    constraints = MagnetofluidConstraints(
        {"magnetic_divergence": {"enabled": True}},
        vector_basis_matches_spatial_coordinates=True,
    )
    result = constraints.volume(
        model,
        positions,
        coords[:, 2],
        height_group_shape=(1, 2),
        return_state=True,
    )

    assert result.losses["magnetic_divergence"].item() == 0.0
    assert result.state.primitive_derivatives.slices == {}
    result.losses["magnetic_divergence"].backward()
    assert model.network.alpha.grad is not None
    assert model.network.alpha.grad.abs() == 0.0


def test_normalized_vector_potential_matches_the_physical_adapter():
    model = _model()
    positions = model.position_from_coords_height(
        torch.tensor([[12.0, -8.0, 0.0], [-20.0, 15.0, 1.5]], dtype=torch.float64),
        torch.tensor([0.0, 2.0e6], dtype=torch.float64),
    )
    physical = model.evaluate_position_points(positions)
    normalized = model.evaluate_position_rsun_normalized(
        positions / model.solar_radius_m,
    )
    for name, scale in (
        ("temperature", model.temperature_scale_k),
        ("gas_pressure", model.gas_pressure_scale_pa),
        ("microturbulence", model.microturbulence_scale_m_per_s),
        ("velocity_field", model.velocity_scale_m_per_s),
        ("magnetic_field", model.magnetic_scale_gauss),
        ("vector_potential", model.vector_potential_scale_gauss_m),
    ):
        torch.testing.assert_close(normalized[name], physical[name] / scale)
