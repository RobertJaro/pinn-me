"""Current-free and outflow-only boundary conditions."""

import pytest
import torch

from prom3theus.inversion.constraints.magnetofluid import MagnetofluidConstraints


class LinearAtmosphere(torch.nn.Module):
    solar_radius_m = 10.0

    def __init__(self, velocity, slope=1.0, amplitude=1.0):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(slope))
        self.amplitude = amplitude
        self.velocity = torch.nn.Parameter(torch.tensor(velocity))

    def evaluate_position_rsun(self, position_rsun, time_hours=None):
        x = position_rsun[:, 0] * self.solar_radius_m
        zero = 0 * x
        return {
            "temperature": zero + 5000,
            "gas_pressure": zero + 1,
            "magnetic_field": self.amplitude
            * torch.stack((zero + 1, self.slope * x, zero), -1),
            "velocity_field": self.velocity.expand(len(x), -1) + zero[:, None],
        }


def evaluate(model, boundary, equations, *, position=None, floor=0.1):
    constraints = MagnetofluidConstraints(
        {f"{boundary}_boundary_{name}": {"enabled": True} for name in equations},
        normalization={
            "length_m": 2.0,
            "velocity_scale_m_per_s": 4.0,
            "magnetic_field_floor_gauss": floor,
        },
        vector_basis_matches_spatial_coordinates=True,
    )
    position = (
        torch.tensor([[10.0, 0.0, 0.0], [10.0, 1.0, 0.0]])
        if position is None
        else position
    )
    time = torch.zeros(len(position), 1)
    if boundary == "upper":
        return constraints.upper_boundary(model, position, time)
    return constraints.side_boundary(
        model, position, time, torch.tensor([[1.0, 0.0, 0.0]]).expand_as(position)
    )


@pytest.mark.parametrize("boundary", ["upper", "side"])
@pytest.mark.parametrize("speed", [-4.0, 0.0, 4.0])
def test_no_inflow_sign_and_gradient(boundary, speed):
    model = LinearAtmosphere([speed, 0.0, 0.0])
    result = evaluate(
        model, boundary, ["no_inflow"], position=torch.tensor([[10.0, 0.0, 0.0]])
    )
    loss = result.losses[f"{boundary}_boundary_no_inflow"]
    assert (loss > 0) == (speed < 0)
    torch.testing.assert_close(loss, torch.tensor((max(-speed, 0.0) / 4.0) ** 2))
    loss.backward()
    assert torch.isfinite(model.velocity.grad).all()
    assert (model.velocity.grad[0] < 0) == (speed < 0)


@pytest.mark.parametrize("boundary", ["upper", "side"])
def test_tangential_velocity_is_free(boundary):
    model = LinearAtmosphere([0.0, 100.0, 0.0])
    loss = evaluate(
        model, boundary, ["no_inflow"], position=torch.tensor([[10.0, 0.0, 0.0]])
    ).losses[f"{boundary}_boundary_no_inflow"]
    assert loss == 0


@pytest.mark.parametrize("boundary", ["upper", "side"])
def test_current_uses_boundary_mean_magnitude(boundary):
    model = LinearAtmosphere([0.0, 0.0, 0.0])
    position = torch.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    loss = evaluate(model, boundary, ["current_free"], position=position).losses[
        f"{boundary}_boundary_current_free"
    ]
    # curl(B)=(0,0,1) G/m; one shared mean, not pointwise normalization.
    mean = torch.sqrt(1 + position[:, 0].square()).mean()
    residual = 2 / torch.sqrt(mean.square() + 0.1**2)
    expected = residual.square() / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(model.slope.grad)
    # The field-strength denominator must not contribute gradients.
    expected_gradient = 2 * residual.square() / 3
    torch.testing.assert_close(model.slope.grad, expected_gradient)
    constant = LinearAtmosphere([0.0, 0.0, 0.0], slope=0.0)
    assert (
        evaluate(constant, boundary, ["current_free"]).losses[
            f"{boundary}_boundary_current_free"
        ]
        == 0
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("speed", [-100.0, 100.0])
def test_side_no_inflow_removes_uniform_horizontal_through_flow(axis, speed):
    velocity = [0.0, 0.0, 0.0]
    velocity[axis] = speed
    model = LinearAtmosphere(velocity)
    constraints = MagnetofluidConstraints(
        {
            "side_boundary_open_velocity": {"enabled": True},
            "side_boundary_no_inflow": {"enabled": True},
        },
        normalization={"velocity_scale_m_per_s": 4.0},
        vector_basis_matches_spatial_coordinates=True,
    )
    normals = torch.tensor([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    ])
    positions = torch.tensor([[0.0, 0.0, 20.0]]).expand(4, -1) + normals
    result = constraints.side_boundary(model, positions, torch.zeros(4, 1), normals)
    assert result.losses["side_boundary_open_velocity"] == 0
    loss = result.losses["side_boundary_no_inflow"]
    assert loss > 0
    loss.backward()
    # Gradient descent must reduce either sign of uniform horizontal drift.
    assert model.velocity.grad[axis] * speed > 0
    assert torch.isfinite(model.velocity.grad).all()


@pytest.mark.parametrize("filename", ["hmi_aia_dynamic.yaml", "hmi_lte_dynamic.yaml"])
def test_dynamic_hmi_configs_use_progressive_potential_and_outflow_boundaries(filename):
    from prom3theus.config import load_config
    from prom3theus.inversion.constraints.magnetofluid import BOUNDARY_EQUATIONS

    config = load_config(f"configs/{filename}")
    equations = config.physics.equations
    expected = {
        f"{boundary}_boundary_{condition}"
        for boundary in ("upper", "side")
        for condition in ("no_inflow",)
    }
    for name in BOUNDARY_EQUATIONS:
        equation = getattr(equations, name)
        assert equation.enabled == (name in expected)
        assert equation.weight == (1.0 if name in expected else 0.0)

    assert config.physics.potential_boundary.enabled
    assert config.physics.potential_boundary.weight == 1.0
