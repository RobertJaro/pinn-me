"""Checks for the ``potential_delta`` magnetic representation.

``B = grad(psi) + alpha * B_delta``, where ``alpha`` is a step-scheduled
scalar (not learned): zero for ``potential_delta_cool_steps``, then a linear
ramp to one over the following ``potential_delta_ramp_steps``. At alpha=0 the
field is exactly ``grad(psi)`` -- curl-free by construction, regardless of
what the network has learned.
"""

import torch
from torch import nn

from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints
from prom3theus.rt import StratifiedAtmosphereModel


class LinearPotentialDeltaHead(nn.Module):
    """Return a constant delta field and a linear scalar potential.

    ``psi = beta * X`` (in physical Carrington coordinates), so
    ``grad(psi) = (beta, 0, 0)`` everywhere; ``B_delta`` is a constant
    Cartesian vector independent of position.
    """

    def __init__(
        self,
        solar_radius_m: float,
        height_scale_m: float,
        magnetic_scale_gauss: float,
        vector_potential_scale_gauss_m: float,
    ):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(3.0e-6, dtype=torch.float64))
        self.delta = nn.Parameter(
            torch.tensor([5.0, -7.0, 2.0], dtype=torch.float64)
        )
        self.solar_radius_m = solar_radius_m
        self.height_scale_m = height_scale_m
        self.magnetic_scale_gauss = magnetic_scale_gauss
        self.vector_potential_scale_gauss_m = vector_potential_scale_gauss_m

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        chart_xy_m = inputs[:, :2] * self.height_scale_m
        local = torch.cat(
            (chart_xy_m / self.solar_radius_m, torch.ones_like(chart_xy_m[:, :1])),
            dim=-1,
        )
        direction = local / torch.linalg.vector_norm(local, dim=-1, keepdim=True)
        position_m = direction * (
            self.solar_radius_m + inputs[:, 2:3] * self.height_scale_m
        )
        zero = inputs[:, :1] * 0.0
        b_delta = (self.delta / self.magnetic_scale_gauss).expand(
            inputs.shape[0], 3
        )
        psi = self.beta * position_m[:, 0:1] / self.vector_potential_scale_gauss_m
        return torch.cat(
            (zero.expand(-1, 4), b_delta, zero.expand(-1, 2), psi), dim=-1
        )


def _model(*, cool_steps=4_000, ramp_steps=2_000) -> StratifiedAtmosphereModel:
    model = StratifiedAtmosphereModel(
        shell_height_bounds_Mm=(5.0, -0.1),
        magnetic_representation="potential_delta",
        magnetic_potential_delta_cool_steps=cool_steps,
        magnetic_potential_delta_ramp_steps=ramp_steps,
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
    model.network = LinearPotentialDeltaHead(
        model.solar_radius_value_m,
        model.height_input_scale_m,
        model.magnetic_scale_gauss,
        model.vector_potential_scale_gauss_m,
    )
    return model


def test_output_layout_appends_psi_after_the_direct_channels():
    model = _model()
    assert model.output_names == (
        "temperature",
        "v_x",
        "v_y",
        "v_z",
        "b_x",
        "b_y",
        "b_z",
        "microturbulence",
        "gas_pressure",
        "psi",
    )


def test_alpha_is_zero_during_cool_steps_and_ramps_to_one():
    model = _model(cool_steps=4_000, ramp_steps=2_000)
    for step, expected in (
        (0, 0.0),
        (3_999, 0.0),
        (4_000, 0.0),
        (5_000, 0.5),
        (5_999, 1_999 / 2_000),
        (6_000, 1.0),
        (10_000, 1.0),
    ):
        model.set_step(step)
        assert model._potential_delta_alpha() == expected


def test_field_is_exactly_grad_psi_during_cool_steps():
    model = _model(cool_steps=4_000, ramp_steps=2_000)
    model.set_step(0)
    coords = torch.tensor(
        [[12.0, -8.0, 0.0], [-20.0, 15.0, 1.5]], dtype=torch.float64
    )
    heights = torch.tensor([0.0, 2.0e6], dtype=torch.float64)
    positions = model.position_from_coords_height(coords, heights)

    fields = model.evaluate_chart_height_points(coords, heights)

    expected = torch.stack(
        (
            model.network.beta.detach().expand_as(positions[:, 0]),
            torch.zeros_like(positions[:, 0]),
            torch.zeros_like(positions[:, 0]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        fields["magnetic_field"], expected, rtol=1e-9, atol=1e-9
    )
    assert "vector_potential" not in fields


def test_field_includes_alpha_weighted_delta_after_the_ramp():
    model = _model(cool_steps=4_000, ramp_steps=2_000)
    model.set_step(6_000)
    coords = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)
    heights = torch.tensor([1.0e6], dtype=torch.float64)
    positions = model.position_from_coords_height(coords, heights)

    fields = model.evaluate_chart_height_points(coords, heights)

    expected = torch.stack(
        (
            model.network.beta.detach().expand_as(positions[:, 0]),
            torch.zeros_like(positions[:, 0]),
            torch.zeros_like(positions[:, 0]),
        ),
        dim=-1,
    ) + model.network.delta.detach()
    torch.testing.assert_close(
        fields["magnetic_field"], expected, rtol=1e-9, atol=1e-9
    )


def test_gradients_reach_beta_and_delta_after_the_ramp():
    model = _model(cool_steps=4_000, ramp_steps=2_000)
    model.set_step(6_000)
    coords = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)
    heights = torch.tensor([1.0e6], dtype=torch.float64)

    fields = model.evaluate_chart_height_points(coords, heights)
    fields["magnetic_field"].square().sum().backward()

    assert model.network.beta.grad is not None and model.network.beta.grad.abs() > 0
    assert model.network.delta.grad is not None
    assert model.network.delta.grad.abs().sum() > 0


def test_divergence_uses_a_real_jacobian_not_a_hardcoded_zero():
    model = _model(cool_steps=0, ramp_steps=0)
    model.set_step(0)
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

    # A spatially uniform field (grad(psi) is constant, B_delta is constant)
    # has exactly zero divergence -- but through a real Jacobian, not the
    # vector_potential representation's structural short-circuit.
    assert result.state.primitive_derivatives.slices != {}
    torch.testing.assert_close(
        result.losses["magnetic_divergence"],
        result.losses["magnetic_divergence"].new_zeros(()),
        atol=1e-10,
        rtol=0,
    )


def test_normalized_potential_delta_matches_the_physical_adapter():
    model = _model()
    model.set_step(6_000)
    positions = model.position_from_coords_height(
        torch.tensor([[12.0, -8.0, 0.0], [-20.0, 15.0, 1.5]], dtype=torch.float64),
        torch.tensor([0.0, 2.0e6], dtype=torch.float64),
    )
    physical = model.evaluate_position_rsun(positions / model.solar_radius_m)
    normalized = model.evaluate_position_rsun_normalized(
        positions / model.solar_radius_m,
    )
    for name, scale in (
        ("temperature", model.temperature_scale_k),
        ("gas_pressure", model.gas_pressure_scale_pa),
        ("microturbulence", model.microturbulence_scale_m_per_s),
        ("velocity_field", model.velocity_scale_m_per_s),
        ("magnetic_field", model.magnetic_scale_gauss),
    ):
        torch.testing.assert_close(normalized[name], physical[name] / scale)
