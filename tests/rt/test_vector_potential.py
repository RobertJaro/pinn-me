"""Checks for the Cartesian vector-potential magnetic representation."""

import pytest
import torch
from torch import nn

from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints
from prom3theus.rt import StratifiedAtmosphereModel


class LinearVectorPotentialHead(nn.Module):
    """Return ``A[axis] = alpha * X`` in physical Carrington coordinates.

    ``axis=2`` is solenoidal and divergence free (``div A = 0``, ``B =
    -alpha y_hat``); ``axis=0`` is pure gauge (``div A = alpha``, ``B = 0``).
    """

    def __init__(
        self,
        solar_radius_m: float,
        height_scale_m: float,
        magnetic_scale_gauss: float,
        axis: int = 2,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(2.0e-6, dtype=torch.float64))
        self.solar_radius_m = solar_radius_m
        self.height_scale_m = height_scale_m
        self.vector_potential_scale = magnetic_scale_gauss * height_scale_m
        self.axis = axis

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
        components = [zero, zero, zero]
        components[self.axis] = self.alpha * position_m[:, 0:1]
        vector_potential = torch.cat(components, dim=-1)
        return torch.cat(
            (
                zero.expand(-1, 4),
                vector_potential / self.vector_potential_scale,
                zero.expand(-1, 2),
            ),
            dim=-1,
        )


def _model(axis: int = 2, scale_vector_potential: float = 1.0) -> StratifiedAtmosphereModel:
    model = StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(5.0, -0.1),
        magnetic_representation="vector_potential",
        magnetic_scale_gauss=100.0,
        scale_vector_potential=scale_vector_potential,
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
        axis=axis,
    )
    return model


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_vector_potential_decodes_a_and_physical_curl_b(scale):
    model = _model(scale_vector_potential=scale)
    coords = torch.tensor(
        [[12.0, -8.0, 0.0], [-20.0, 15.0, 1.5]], dtype=torch.float64
    )
    heights = torch.tensor([0.0, 2.0e6], dtype=torch.float64)
    positions = model.position_from_coords_height(coords, heights)
    fields = model.evaluate_chart_height_points(coords, heights)
    alpha = model.network.alpha.detach() * scale

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
    assert model.reference_metadata()["scale_vector_potential"] == scale
    torch.testing.assert_close(
        model._evaluate_vector_potential_only(positions / model.solar_radius_m, None),
        expected_a,
    )

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


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_normalized_vector_potential_matches_the_physical_adapter(scale):
    model = _model(scale_vector_potential=scale)
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
    torch.testing.assert_close(
        model._evaluate_vector_potential_only_normalized(
            positions / model.solar_radius_m, None,
        ),
        normalized["vector_potential"],
    )


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_vector_potential_divergence_accompanies_the_evaluated_field(scale):
    """div A is published with B because it is the same Jacobian's trace."""

    model = _model(axis=0, scale_vector_potential=scale)
    positions = model.position_from_coords_height(
        torch.tensor([[12.0, -8.0, 0.0], [-20.0, 15.0, 1.5]], dtype=torch.float64),
        torch.tensor([0.0, 2.0e6], dtype=torch.float64),
    )
    physical = model.evaluate_position_points(positions)
    normalized = model.evaluate_position_rsun_normalized(
        positions / model.solar_radius_m
    )
    alpha = model.network.alpha.detach() * scale

    expected = alpha.expand(physical["vector_potential_divergence"].shape)
    torch.testing.assert_close(
        physical["vector_potential_divergence"], expected, rtol=1e-9, atol=1e-12
    )
    torch.testing.assert_close(
        normalized["vector_potential_divergence"],
        expected / model.magnetic_scale_gauss,
        rtol=1e-9,
        atol=1e-14,
    )
    # A pure gauge field carries no magnetic field at all.
    torch.testing.assert_close(
        physical["magnetic_field"],
        torch.zeros_like(physical["magnetic_field"]),
        rtol=0.0,
        atol=1e-12,
    )
    round_trip = model.normalize_atmosphere_fields(physical)
    torch.testing.assert_close(
        round_trip["vector_potential_divergence"],
        normalized["vector_potential_divergence"],
    )


def _gauge_result(model, normalization=None, physical=False):
    coords = torch.tensor([[5.0, -3.0, 0.0], [-9.0, 4.0, 0.0]], dtype=torch.float64)
    heights = torch.tensor([0.0, 1.0e6], dtype=torch.float64)
    positions = model.position_from_coords_height(coords, heights)
    constraints = MagnetofluidConstraints(
        {"vector_potential_gauge": {"enabled": True}},
        vector_basis_matches_spatial_coordinates=True,
        normalization=normalization,
    )
    evaluate = constraints._volume_physical if physical else constraints.volume
    return evaluate(
        model,
        positions,
        coords[:, 2],
        height_group_shape=(1, 2),
        return_state=True,
        return_diagnostics=True,
    )


def test_gauge_residual_penalizes_div_a_without_a_second_jacobian():
    model = _model(axis=0)
    result = _gauge_result(model)

    # No primitive is differentiated for this equation: the divergence comes
    # from the Jacobian the magnetic decoder already built.
    assert result.state.primitive_derivatives.slices == {}
    alpha = float(model.network.alpha.detach())
    divergence_model = alpha / float(model.magnetic_scale_gauss)
    expected = divergence_model ** 2
    torch.testing.assert_close(
        result.losses["vector_potential_gauge"],
        torch.tensor(expected, dtype=torch.float64),
    )
    result.losses["vector_potential_gauge"].backward()
    assert model.network.alpha.grad is not None
    assert model.network.alpha.grad.abs() > 0


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("fixed_scale", [None, 500.0])
def test_gauge_ignores_physics_normalization(physical, detach, fixed_scale):
    model = _model(axis=0)
    baseline = _gauge_result(model)
    result = _gauge_result(model, normalization={
        "magnetic_field_floor_gauss": 75.0,
        "magnetic_field_scale_gauss": fixed_scale,
        "length_m": 3.e7,
        "detach_normalization_scale": detach,
    }, physical=physical)
    key = "vector_potential_gauge"
    torch.testing.assert_close(result.losses[key], baseline.losses[key], atol=1e-30, rtol=1e-10)
    expected_gradient = torch.autograd.grad(baseline.losses[key], model.network.alpha)[0]
    gradient = torch.autograd.grad(result.losses[key], model.network.alpha)[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-30, rtol=1e-10)
    diagnostic = result.equation_diagnostics[key]["normalized_residual"]
    torch.testing.assert_close(diagnostic.square().mean(), result.losses[key])


def test_gauge_residual_vanishes_for_a_divergence_free_vector_potential():
    model = _model(axis=2)
    result = _gauge_result(model)

    # Analytically zero; only float round-trip round-off remains, which is
    # ~40 orders of magnitude below the pure-gauge residual above.
    assert result.losses["vector_potential_gauge"].item() < 1e-30
    result.losses["vector_potential_gauge"].backward()
    assert model.network.alpha.grad is not None
    assert model.network.alpha.grad.abs() < 1e-20


def test_gauge_residual_requires_a_vector_potential_model():
    model = _model()
    model.magnetic_representation = "direct"
    with pytest.raises(ValueError, match="vector_potential_gauge"):
        _gauge_result(model)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_nonlinear_siren_curl_and_weight_gradient_match_finite_differences(scale):
    """Exercise the mixed spatial/weight derivative used by polarized fitting.

    A linear potential alone cannot expose mistakes that suppress the nonlinear
    network's magnetic gradients. Check both its physical curl and a parameter
    derivative of that curl independently, at noncentral positions.
    """
    torch.manual_seed(23)
    model = StratifiedAtmosphereModel(
        shell_height_bounds_megameter=(1.5, -0.1),
        time_dependent=True,
        magnetic_representation="vector_potential",
        magnetic_scale_gauss=100.0,
        scale_vector_potential=scale,
        height_input_scale_m=1.0e7,
        uniform_spatial_scaling=True,
        scene_geometry_config={
            "solar_radius_m": 695700000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "siren",
            "dim": 12,
            "n_layers": 3,
            "first_omega_0": 5.0,
            "hidden_omega_0": 1.0,
            "radial_weighting_config": None,
        },
    ).double()
    coords = torch.tensor([[12.0, -8.0, 0.4], [-20.0, 15.0, -0.2]], dtype=torch.float64)
    height = torch.tensor([7.0e4, 2.0e5], dtype=torch.float64)
    position = model.position_from_coords_height(coords, height) / model.solar_radius_m
    field = model.evaluate_position_rsun(position, time_hours=coords[:, 2])["magnetic_field"]

    step_m = 10.0
    columns = []
    for axis in range(3):
        displacement = torch.zeros_like(position)
        displacement[:, axis] = step_m / model.solar_radius_m
        plus = model._evaluate_vector_potential_only(position + displacement, coords[:, 2])
        minus = model._evaluate_vector_potential_only(position - displacement, coords[:, 2])
        columns.append((plus - minus) / (2 * step_m))
    jacobian = torch.stack(columns, dim=-1)
    expected = torch.stack((
        jacobian[:, 2, 1] - jacobian[:, 1, 2],
        jacobian[:, 0, 2] - jacobian[:, 2, 0],
        jacobian[:, 1, 0] - jacobian[:, 0, 1],
    ), dim=-1)
    torch.testing.assert_close(field, expected, rtol=2e-7, atol=1e-7)

    parameter = model.network.in_layer.weight
    direction = torch.randn_like(parameter)
    direction /= direction.norm()
    loss = (field / model.magnetic_scale_gauss).square().mean()
    gradient, = torch.autograd.grad(loss, parameter)
    expected_derivative = (gradient * direction).sum()
    original = parameter.detach().clone()
    step = 1e-6
    values = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                parameter.copy_(original + sign * step * direction)
                trial = model.evaluate_position_rsun(position, time_hours=coords[:, 2])["magnetic_field"]
                values.append((trial / model.magnetic_scale_gauss).square().mean())
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    finite_difference = (values[0] - values[1]) / (2 * step)
    assert expected_derivative.abs() > 1e-6 * scale ** 2
    torch.testing.assert_close(expected_derivative, finite_difference, rtol=2e-6, atol=1e-8)
