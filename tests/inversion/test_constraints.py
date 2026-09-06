import pytest
import torch

import prom3theus.inversion.constraints.magnetofluid as magnetofluid
from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.core import CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
from prom3theus.inversion.constraints.magnetofluid import (
    MagnetofluidConstraints,
    PhysicsNormalization,
    PhysicsState,
    PrimitiveDerivatives,
    SIDE_BOUNDARY_EQUATIONS,
    UPPER_BOUNDARY_EQUATIONS,
    UPPER_VOLUME_EQUATIONS,
    _curl,
)


class _Opacity:
    @staticmethod
    def mass_density(temperature, gas_pressure):
        return gas_pressure / (temperature * 3.0e7)

    @staticmethod
    def volume_extinction_at_5000(temperature, gas_pressure):
        return gas_pressure * temperature.new_tensor(1.0e-10)


def _model():
    return StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 9),
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 10,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )


def _time_model():
    return StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 9),
        time_dependent=True,
        time_coordinate_center_hours=12.0,
        time_coordinate_scale_hours=12.0,
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 10,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )


def _volume(module, model, opacity, coords, height, *, position_m=None, **kwargs):
    if position_m is None:
        position_m = model.position_from_coords_height(coords, height)
    if not hasattr(model, "thermodynamic_eos"):
        model.thermodynamic_eos = opacity
    return module.volume(
        model,
        position_m,
        coords[:, 2:3],
        height_group_shape=kwargs.pop(
            "height_group_shape", (int(position_m.shape[0]), 1)
        ),
        **kwargs,
    )


def test_curl_uses_vector_component_by_spatial_coordinate_jacobian_order():
    jacobian = torch.zeros(1, 3, 3)
    jacobian[0, 2, 1] = 2.0  # dBz/dy
    jacobian[0, 1, 2] = -3.0  # dBy/dz
    jacobian[0, 0, 2] = 5.0  # dBx/dz
    jacobian[0, 2, 0] = 7.0  # dBz/dx
    jacobian[0, 1, 0] = 11.0  # dBy/dx
    jacobian[0, 0, 1] = 13.0  # dBx/dy
    assert torch.equal(_curl(jacobian), torch.tensor([[5.0, -2.0, -2.0]]))


def test_only_current_equations_and_fixed_weights_are_accepted():
    with pytest.raises(KeyError, match="Unknown magnetofluid"):
        MagnetofluidConstraints({"unsupported": {"enabled": True}})
    with pytest.raises(TypeError, match="fixed numeric weight"):
        MagnetofluidConstraints(
            {"momentum": {"enabled": True, "weight": {"start": 0, "end": 1}}},
            gravity_m_per_s2=275.0,
            vector_basis_matches_spatial_coordinates=True,
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        MagnetofluidConstraints({"magnetic_divergence": 0.5})
    with pytest.raises(TypeError, match="enabled must be boolean"):
        MagnetofluidConstraints(
            {"magnetic_divergence": {"enabled": "true", "weight": 0.5}},
            vector_basis_matches_spatial_coordinates=True,
        )
    with pytest.raises(TypeError, match="vector_basis_matches"):
        MagnetofluidConstraints(
            {"magnetic_divergence": {"enabled": True, "weight": 0.5}},
            vector_basis_matches_spatial_coordinates="cartesian",
        )
    module = MagnetofluidConstraints(
        {"magnetic_divergence": {"enabled": True, "weight": 0.5}},
        vector_basis_matches_spatial_coordinates=True,
    )
    weights = module.loss_weights
    assert weights["magnetic_divergence"] == 0.5
    weights["magnetic_divergence"] = 9.0
    assert module.loss_weights["magnetic_divergence"] == 0.5
    with pytest.raises(ValueError, match="mutually exclusive"):
        MagnetofluidConstraints(
            {
                "hydrostatic_equilibrium": {"enabled": True},
                "momentum": {"enabled": True},
            },
            gravity_m_per_s2=275.0,
            vector_basis_matches_spatial_coordinates=True,
        )


def test_zero_weight_equations_are_not_active():
    module = MagnetofluidConstraints(
        {
            "magnetic_divergence": {"enabled": True, "weight": 0.0},
            "upper_boundary_gas_pressure_prior": {
                "enabled": True,
                "weight": 0.0,
            },
        },
    )

    assert not module.any_active
    assert not module.volume_active
    assert not module.boundary_active
    assert not module.is_active("magnetic_divergence")


def test_magnetic_divergence_uses_cartesian_float32_state_and_backpropagates():
    model = _model()
    coords = torch.tensor([[-0.2, 0.3, 0.0], [0.4, -0.1, 0.0]])
    q = torch.tensor([-4.0, 0.0])
    module = MagnetofluidConstraints(
        {
            "magnetic_divergence": {"enabled": True},
        },
        vector_basis_matches_spatial_coordinates=True,
    )
    height = model.depth_to_height(q)
    result = _volume(module, model, _Opacity(), coords, height, return_state=True)
    assert torch.isfinite(result.losses["magnetic_divergence"])
    assert result.state.position_m.dtype == torch.float32
    result.losses["magnetic_divergence"].backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_magnetic_normalization_floor_bounds_near_null_gradients():
    torch.manual_seed(7)
    model = _model()
    with torch.no_grad():
        model.network.out_layer.weight[4:7].mul_(1.0e-8)
        model.network.out_layer.bias[4:7].mul_(1.0e-8)
    coords = torch.tensor([[-0.2, 0.3, 0.0], [0.4, -0.1, 0.0]])
    height = model.depth_to_height(torch.tensor([-4.0, 0.0]))
    constraints = MagnetofluidConstraints(
        {"magnetic_divergence": {"enabled": True}},
        vector_basis_matches_spatial_coordinates=True,
        normalization={"magnetic_field_floor_gauss": 1.0},
    )

    loss = _volume(
        constraints,
        model,
        _Opacity(),
        coords,
        height,
    ).losses["magnetic_divergence"]
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]

    assert torch.isfinite(loss)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert max(float(gradient.abs().max()) for gradient in gradients) < 1.0e3


def test_hydrostatic_equilibrium_uses_only_pressure_density_and_gravity():
    model = _model()
    coords = torch.tensor([[-0.2, 0.3, 0.0], [0.4, -0.1, 0.0]])
    height = model.depth_to_height(torch.tensor([-4.0, 0.0]))
    module = MagnetofluidConstraints(
        {"hydrostatic_equilibrium": {"enabled": True, "weight": 1.0e-5}},
        gravity_m_per_s2=275.0,
        vector_basis_matches_spatial_coordinates=True,
    )

    result = _volume(module, model, _Opacity(), coords, height)

    loss = result.losses["hydrostatic_equilibrium"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_magnetohydrostatic_equilibrium_combines_pressure_gravity_and_lorentz_force():
    model = _model()
    coords = torch.tensor([[-0.2, 0.3, 0.0], [0.4, -0.1, 0.0]])
    height = model.depth_to_height(torch.tensor([-4.0, 0.0]))
    module = MagnetofluidConstraints(
        {
            "magnetohydrostatic_equilibrium": {
                "enabled": True,
                "weight": 1.0e-3,
            }
        },
        gravity_m_per_s2=275.0,
        vector_basis_matches_spatial_coordinates=True,
    )

    result = _volume(module, model, _Opacity(), coords, height, return_state=True)

    assert torch.isfinite(result.losses["magnetohydrostatic_equilibrium"])
    assert "pressure" in result.state.primitive_derivatives.slices
    assert "magnetic" in result.state.primitive_derivatives.slices
    assert "velocity" not in result.state.primitive_derivatives.slices
    result.losses["magnetohydrostatic_equilibrium"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_upper_boundary_pressure_prior_matches_reference_initially_and_backpropagates():
    model = _model()
    coords = torch.tensor([[0.0, 0.0, 0.0], [10.0, -5.0, 0.0]])
    module = MagnetofluidConstraints(
        {"upper_boundary_gas_pressure_prior": {"enabled": True, "weight": 1e-4}},
    )
    height = coords.new_full(
        (coords.shape[0],), model.shell_height_bounds_Mm[0] * 1.0e6
    )
    position_m = model.position_from_coords_height(coords, height)
    result = module.upper_boundary_gas_pressure_prior(
        model,
        position_m,
        coords[:, 2:3],
    )
    loss = result.losses["upper_boundary_gas_pressure_prior"]
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0.0)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_normalization_accepts_physical_time_scale_and_rejects_unknown_keys():
    assert PhysicsNormalization().configuration() == {
        "length_m": 1.0e6,
        "time_s": 3_600.0,
        "magnetic_field_floor_gauss": 1.0,
        "velocity_scale_m_per_s": 1_000.0,
    }
    assert PhysicsNormalization.from_config({"time_s": 60.0}).time_s == 60.0
    assert PhysicsNormalization().transport_time_s == pytest.approx(
        1.0 / (1.0 / 3_600.0 + 1_000.0 / 1.0e6)
    )
    with pytest.raises(TypeError, match="Unknown physics normalization"):
        PhysicsNormalization.from_config({"cadence_s": 1000.0})
    with pytest.raises(TypeError, match="must be numeric"):
        PhysicsNormalization(time_s=True)
    with pytest.raises(ValueError, match="finite and positive"):
        PhysicsNormalization(magnetic_field_floor_gauss=0.0)


def test_height_normalization_uses_an_independent_mean_for_every_height():
    state = PhysicsState(
        position_m=torch.zeros(4, 3),
        gas_pressure=torch.ones(4),
        mass_density=torch.ones(4),
        velocity_field_m_per_s=torch.zeros(4, 3),
        magnetic_field_gauss=torch.zeros(4, 3),
        primitive_derivatives=PrimitiveDerivatives(torch.empty(4, 0, 4), {}),
        radial_unit=torch.zeros(4, 3),
        height_group_shape=(2, 2),
    )
    residual = torch.tensor([2.0, 2.0, 20.0, 20.0])
    normalization_quantity = torch.tensor([1.0, 3.0, 10.0, 30.0])

    normalized = state.normalize_height_groups(
        residual,
        normalization_quantity,
        dimensional_factor=1.0,
    )

    torch.testing.assert_close(normalized, torch.ones(2, 2))


def test_transport_normalization_uses_q_star_times_combined_rate():
    state = PhysicsState(
        position_m=torch.zeros(4, 3),
        gas_pressure=torch.ones(4),
        mass_density=torch.ones(4),
        velocity_field_m_per_s=torch.zeros(4, 3),
        magnetic_field_gauss=torch.zeros(4, 3),
        primitive_derivatives=PrimitiveDerivatives(torch.empty(4, 0, 4), {}),
        radial_unit=torch.zeros(4, 3),
        height_group_shape=(2, 2),
    )
    normalization = PhysicsNormalization(
        length_m=2.0,
        time_s=4.0,
        velocity_scale_m_per_s=1.0,
    )
    q_star = torch.tensor([2.0, 2.0, 5.0, 5.0])
    residual = q_star * normalization.transport_rate_per_s

    normalized = state.normalize_height_groups(
        residual,
        q_star,
        dimensional_factor=normalization.transport_time_s,
    )

    torch.testing.assert_close(normalized, torch.ones(2, 2))


def test_height_normalization_detaches_its_scale():
    state = PhysicsState(
        position_m=torch.zeros(4, 3),
        gas_pressure=torch.ones(4),
        mass_density=torch.ones(4),
        velocity_field_m_per_s=torch.zeros(4, 3),
        magnetic_field_gauss=torch.zeros(4, 3),
        primitive_derivatives=PrimitiveDerivatives(torch.empty(4, 0, 4), {}),
        radial_unit=torch.zeros(4, 3),
        height_group_shape=(2, 2),
    )
    residual_amplitude = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
    scale_amplitude = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
    shape = torch.tensor([1.0, 3.0, 10.0, 30.0], dtype=torch.float64)

    normalized = state.normalize_height_groups(
        residual_amplitude * shape,
        scale_amplitude * shape,
        dimensional_factor=1.0,
    )
    loss = normalized.square().mean()
    residual_gradient, scale_gradient = torch.autograd.grad(
        loss,
        (residual_amplitude, scale_amplitude),
        allow_unused=True,
    )

    torch.testing.assert_close(loss, torch.tensor(1.25, dtype=torch.float64))
    assert torch.isfinite(residual_gradient)
    assert scale_gradient is None


def test_relative_scale_is_detached_when_residual_uses_the_same_field():
    state = PhysicsState(
        position_m=torch.zeros(4, 3),
        gas_pressure=torch.ones(4),
        mass_density=torch.ones(4),
        velocity_field_m_per_s=torch.zeros(4, 3),
        magnetic_field_gauss=torch.zeros(4, 3),
        primitive_derivatives=PrimitiveDerivatives(torch.empty(4, 0, 4), {}),
        radial_unit=torch.zeros(4, 3),
        height_group_shape=(2, 2),
    )
    amplitude = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
    shape = torch.tensor([1.0, 3.0, 10.0, 30.0], dtype=torch.float64)

    normalized = state.normalize_height_groups(
        amplitude * shape,
        amplitude.abs() * shape,
        dimensional_factor=1.0,
    )
    loss = normalized.square().mean()
    gradient = torch.autograd.grad(loss, amplitude)[0]

    torch.testing.assert_close(loss, torch.tensor(1.25, dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.tensor(2_500.0, dtype=torch.float64))


def test_height_group_robust_loss_averages_components_and_is_quadratic_near_zero():
    state = PhysicsState(
        position_m=torch.zeros(1, 3),
        gas_pressure=torch.ones(1),
        mass_density=torch.ones(1),
        velocity_field_m_per_s=torch.zeros(1, 3),
        magnetic_field_gauss=torch.zeros(1, 3),
        primitive_derivatives=PrimitiveDerivatives(torch.empty(1, 0, 4), {}),
        radial_unit=torch.zeros(1, 3),
        height_group_shape=(1, 1),
    )
    epsilon = torch.tensor(1.0e-3, dtype=torch.float64)
    grouped_residual = torch.stack(
        (epsilon, torch.zeros_like(epsilon), torch.zeros_like(epsilon))
    ).reshape(1, 1, 3)

    loss = state.height_group_robust_loss(grouped_residual)

    torch.testing.assert_close(
        loss,
        epsilon.square() / 3.0,
        rtol=1.0e-6,
        atol=0.0,
    )


def test_active_volume_equation_requires_explicit_height_groups():
    model = _model()
    constraints = MagnetofluidConstraints(
        {"magnetic_divergence": {"enabled": True}},
        vector_basis_matches_spatial_coordinates=True,
    )
    coords = torch.tensor([[0.0, 0.0, 0.0]])
    position_m = model.position_from_coords_height(coords, torch.tensor([0.0]))

    with pytest.raises(TypeError, match="height_group_shape"):
        constraints.volume(model, position_m, coords[:, 2:3])


def test_gravity_must_be_finite_positive_and_numeric_at_physics_boundary():
    for invalid in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="finite and positive"):
            MagnetofluidConstraints(gravity_m_per_s2=invalid)
    with pytest.raises(TypeError, match="must be numeric or null"):
        MagnetofluidConstraints(gravity_m_per_s2=True)


def test_all_dynamic_equations_share_one_base_output_jacobian(monkeypatch):
    model = _time_model()
    coords = torch.tensor([[-0.2, 0.3, 1.0], [0.4, -0.1, 20.0]])
    height = model.depth_to_height(torch.tensor([-4.0, 0.0]))
    module = MagnetofluidConstraints(
        {
            "momentum": {"enabled": True},
            "induction": {"enabled": True},
            "continuity": {"enabled": True},
            "adiabatic_pressure": {"enabled": True},
        },
        gravity_m_per_s2=275.0,
        vector_basis_matches_spatial_coordinates=True,
        normalization={"length_m": 1.0e6, "time_s": 3_600.0},
    )
    calls = 0
    original = magnetofluid._pointwise_output_jacobian

    def counted_jacobian(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(magnetofluid, "_pointwise_output_jacobian", counted_jacobian)
    result = _volume(module, model, _Opacity(), coords, height, return_state=True)
    assert calls == 1
    assert torch.isfinite(result.losses["momentum"])
    assert torch.isfinite(result.losses["induction"])
    assert torch.isfinite(result.losses["continuity"])
    assert torch.isfinite(result.losses["adiabatic_pressure"])
    assert result.state.time_derivative("magnetic").shape == (2, 3)
    assert result.state.time_derivative("log_density").shape == (2, 1)
    assert set(result.state.primitive_derivatives.slices) == {
        "pressure",
        "log_pressure",
        "magnetic",
        "velocity",
        "density",
        "log_density",
    }
    assert result.state.derivative("velocity_cross_magnetic").shape == (2, 3, 3)
    assert result.state.derivative("mass_flux").shape == (2, 3, 3)
    (
        result.losses["momentum"]
        + result.losses["induction"]
        + result.losses["continuity"]
        + result.losses["adiabatic_pressure"]
    ).backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_temporal_equations_reject_static_atmosphere():
    module = MagnetofluidConstraints(
        {"induction": {"enabled": True}},
        vector_basis_matches_spatial_coordinates=True,
    )
    model = _model()
    with pytest.raises(ValueError, match="time_dependent=true"):
        _volume(
            module,
            model,
            _Opacity(),
            torch.tensor([[0.0, 0.0, 0.0]]),
            model.depth_to_height(torch.tensor([-2.0])),
        )


def test_adiabatic_pressure_matches_uniform_three_dimensional_expansion():
    class ExactAtmosphere(torch.nn.Module):
        solar_radius_m = 10.0
        time_dependent = True

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def evaluate_position_rsun(self, position_rsun, time_hours=None):
            rate = position_rsun.new_tensor(2.0e-4)
            gamma = position_rsun.new_tensor(5.0 / 3.0)
            time_s = time_hours[..., 0] * 3_600.0
            position_m = position_rsun * self.solar_radius_m
            velocity = rate * position_m
            pressure = torch.exp(-3.0 * gamma * rate * time_s) + 0.0 * self.dummy
            connected_zero = 0.0 * pressure
            magnetic = torch.stack(
                (connected_zero, connected_zero, connected_zero), dim=-1
            )
            return {
                "temperature": torch.full_like(pressure, 5_000.0),
                "gas_pressure": pressure,
                "magnetic_field": magnetic,
                "velocity_field": velocity,
            }

    model = ExactAtmosphere()
    constraints = MagnetofluidConstraints(
        {"adiabatic_pressure": {"enabled": True}},
        adiabatic_index=5.0 / 3.0,
        vector_basis_matches_spatial_coordinates=True,
    )
    position_m = torch.tensor([[5.0, 1.0, 1.0], [8.0, 1.0, 1.0]])
    result = constraints.volume(
        model,
        position_m,
        torch.tensor([[0.2], [0.6]]),
        height_group_shape=(1, 2),
    )

    torch.testing.assert_close(
        result.losses["adiabatic_pressure"],
        torch.zeros(()),
        atol=4.0e-12,
        rtol=0.0,
    )


def test_adiabatic_index_requires_a_physical_numeric_value():
    with pytest.raises(TypeError, match="adiabatic_index must be numeric"):
        MagnetofluidConstraints(adiabatic_index=True)
    with pytest.raises(ValueError, match="greater than one"):
        MagnetofluidConstraints(adiabatic_index=1.0)


def test_momentum_matches_exact_rotating_frame_balance():
    class ConstantDensityOpacity:
        @staticmethod
        def mass_density(temperature, gas_pressure):
            return torch.ones_like(gas_pressure)

    class ExactAtmosphere(torch.nn.Module):
        solar_radius_m = 7.0e8
        time_dependent = True

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def evaluate_position_rsun(self, position_rsun, time_hours=None):
            position_m = position_rsun * self.solar_radius_m
            x, y, z = position_m.unbind(dim=-1)
            time_s = time_hours[..., 0] * 3_600.0
            rate = x.new_tensor(1.0e-6)
            time_acceleration = x.new_tensor(1.0e-3)
            gravity = x.new_tensor(2.0)
            omega = x.new_tensor(CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S)
            velocity = torch.stack(
                (
                    rate * x + time_acceleration * time_s,
                    -rate * y + 0.0 * time_s,
                    0.0 * z + 0.0 * time_s,
                ),
                dim=-1,
            )
            radius = torch.linalg.vector_norm(position_m, dim=-1)
            acceleration_potential = (
                (time_acceleration + rate * time_acceleration * time_s) * x
                + 0.5 * (rate.square() - omega.square()) * (x.square() + y.square())
                + 2.0 * omega * rate * x * y
                + 2.0 * omega * time_acceleration * time_s * y
                - gravity * self.solar_radius_m**2 / radius
            )
            pressure = x.new_tensor(5.0e9) - acceleration_potential + 0.0 * self.dummy
            connected_zero = 0.0 * (x + y + z)
            magnetic = torch.stack(
                (torch.ones_like(x), connected_zero, connected_zero), dim=-1
            )
            return {
                "temperature": torch.full_like(x, 5_000.0),
                "gas_pressure": pressure,
                "magnetic_field": magnetic,
                "velocity_field": velocity,
            }

    model = ExactAtmosphere()
    module = MagnetofluidConstraints(
        {"momentum": {"enabled": True}},
        gravity_m_per_s2=2.0,
        vector_basis_matches_spatial_coordinates=True,
    )
    coords = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.6]])
    position_m = torch.tensor([[6.8e8, 1.0e8, 5.0e7], [6.7e8, -1.2e8, 8.0e7]])
    result = _volume(
        module,
        model,
        ConstantDensityOpacity(),
        coords,
        torch.zeros(2),
        position_m=position_m,
        return_state=True,
    )
    torch.testing.assert_close(
        result.losses["momentum"], torch.zeros(()), atol=4.0e-12, rtol=0.0
    )
    assert result.state.time_derivative("velocity").shape == (2, 3)


def test_induction_and_continuity_match_an_exact_expanding_flow_solution():
    class ExactAtmosphere(torch.nn.Module):
        solar_radius_m = 10.0
        time_dependent = True

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def evaluate_position_rsun(self, position_rsun, time_hours=None):
            rate = position_rsun.new_tensor(2.0e-4)
            time_s = time_hours[..., 0] * 3_600.0
            factor = torch.exp(-rate * time_s) + 0.0 * self.dummy
            x_m = position_rsun[:, 0] * self.solar_radius_m
            velocity = torch.stack(
                (rate * x_m, torch.zeros_like(x_m), torch.zeros_like(x_m)), dim=-1
            )
            magnetic = torch.stack(
                (torch.zeros_like(factor), factor, torch.zeros_like(factor)), dim=-1
            )
            temperature = torch.full_like(factor, 5_000.0)
            pressure = factor * temperature * 3.0e7
            return {
                "temperature": temperature,
                "gas_pressure": pressure,
                "magnetic_field": magnetic,
                "velocity_field": velocity,
            }

    model = ExactAtmosphere()
    module = MagnetofluidConstraints(
        {"induction": {"enabled": True}, "continuity": {"enabled": True}},
        vector_basis_matches_spatial_coordinates=True,
    )
    coords = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.6]])
    position_m = torch.tensor([[5.0, 1.0, 1.0], [8.0, 1.0, 1.0]])
    result = _volume(
        module,
        model,
        _Opacity(),
        coords,
        torch.zeros(2),
        position_m=position_m,
        return_state=True,
    )
    torch.testing.assert_close(
        result.losses["induction"], torch.zeros(()), atol=4.0e-12, rtol=0.0
    )
    torch.testing.assert_close(
        result.losses["continuity"], torch.zeros(()), atol=4.0e-12, rtol=0.0
    )


def test_upper_domain_and_boundary_losses_backpropagate():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 7),
        shell_height_bounds_Mm=(3.0, -0.1),
        line_formation_height_bounds_Mm=(1.5, -0.1),
        upper_atmosphere_config={
            "type": "hydrostatic_corona",
            "transition_region_top_megameter": 2.5,
            "coronal_temperature_k": 1.0e6,
            "reference_grid_points": 128,
        },
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 10,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )
    constraints = MagnetofluidConstraints(
        {
            "upper_boundary_open_velocity": {"enabled": True, "weight": 1e-5},
            "upper_domain_microturbulence_prior": {
                "enabled": True,
                "weight": 1e-5,
            },
            "upper_domain_temperature_prior": {
                "enabled": True,
                "weight": 1e-5,
            },
            "upper_boundary_current_free": {"enabled": True, "weight": 1e-3},
            "upper_boundary_gas_pressure_prior": {
                "enabled": True,
                "weight": 1e-4,
            },
        },
        vector_basis_matches_spatial_coordinates=True,
    )
    coords = torch.tensor([[0.0, 0.0, 0.0], [0.05, -0.03, 0.0]])
    upper_height = torch.tensor([2.0e6, 2.5e6])
    upper_position = model.position_from_coords_height(coords, upper_height)
    upper = constraints.upper_domain(
        model,
        upper_position,
        coords[:, 2:3],
        height_group_shape=(2, 1),
    )
    torch.testing.assert_close(
        upper.losses["upper_domain_microturbulence_prior"],
        torch.zeros(()),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        upper.losses["upper_domain_temperature_prior"],
        torch.zeros(()),
        rtol=0.0,
        atol=1.0e-12,
    )
    top_position = model.position_from_coords_height(coords, torch.full((2,), 3.0e6))
    boundary = constraints.upper_boundary(
        model,
        top_position,
        coords[:, 2:3],
    )

    active_losses = {
        **{name: upper.losses[name] for name in UPPER_VOLUME_EQUATIONS},
        **{name: boundary.losses[name] for name in UPPER_BOUNDARY_EQUATIONS},
    }
    assert all(torch.isfinite(value) for value in active_losses.values())
    sum(active_losses.values()).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_radial_magnetic_energy_gradient_is_one_sided_on_height_mean():
    class RadialEnergyAtmosphere(torch.nn.Module):
        solar_radius_m = 10.0

        def __init__(self, profile: str):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.tensor(2.0))
            self.profile = profile

        def evaluate_position_rsun(self, position_rsun, time_hours=None):
            del time_hours
            x = position_rsun[:, 0] * self.solar_radius_m
            zero = 0.0 * x
            if self.profile == "increasing":
                radial_field = self.amplitude * x
            elif self.profile == "decreasing":
                radial_field = self.amplitude * (x.new_tensor(3.0) - x)
            elif self.profile == "constant":
                radial_field = self.amplitude + zero
            elif self.profile == "redistributing":
                radial_field = self.amplitude * (x - x.new_tensor(0.9))
            else:
                raise ValueError(f"Unknown magnetic profile {self.profile!r}.")
            return {
                "temperature": torch.full_like(x, 5_000.0),
                "gas_pressure": torch.ones_like(x),
                "magnetic_field": torch.stack(
                    (radial_field, zero, zero),
                    dim=-1,
                ),
                "velocity_field": torch.stack((zero, zero, zero), dim=-1),
            }

    constraints = MagnetofluidConstraints(
        {
            "radial_magnetic_energy_gradient": {
                "enabled": True,
                "weight": 1e-5,
            }
        },
        vector_basis_matches_spatial_coordinates=True,
        normalization={"length_m": 1.0},
    )
    diagonal = 2.0**-0.5
    position_m = torch.tensor([[1.0, 0.0, 0.0], [diagonal, diagonal, 0.0]])
    time_hours = torch.zeros(2, 1)

    increasing = RadialEnergyAtmosphere("increasing")
    increasing_result = constraints.volume(
        increasing,
        position_m,
        time_hours,
        height_group_shape=(1, 2),
    )
    increasing_loss = increasing_result.losses["radial_magnetic_energy_gradient"]
    # The two angular points share one radius. mean(d(B^2)/dr) = 6 G^2/m,
    # while the detached height scale is sqrt(mean([4, 2])^2 + (1 G)^4)
    # = sqrt(10) G^2.
    torch.testing.assert_close(increasing_loss, torch.tensor(3.6))
    increasing_loss.backward()
    assert increasing.amplitude.grad > 0.0

    # The redistributing profile increases in the first angular column but its
    # layer-mean energy decreases. Only the layer mean is constrained.
    for profile in ("constant", "decreasing", "redistributing"):
        unconstrained = RadialEnergyAtmosphere(profile)
        unconstrained_result = constraints.volume(
            unconstrained,
            position_m,
            time_hours,
            height_group_shape=(1, 2),
        )
        torch.testing.assert_close(
            unconstrained_result.losses["radial_magnetic_energy_gradient"],
            torch.zeros(()),
        )


def test_upper_boundary_open_velocity_penalizes_normal_velocity_gradient():
    class RadialLinearVelocityAtmosphere(torch.nn.Module):
        solar_radius_m = 10.0

        def __init__(self, radial_slope_per_s: float):
            super().__init__()
            self.radial_slope_per_s = torch.nn.Parameter(
                torch.tensor(radial_slope_per_s)
            )

        def evaluate_position_rsun(self, position_rsun, time_hours=None):
            del time_hours
            position_m = position_rsun * self.solar_radius_m
            constant_velocity = position_m.new_tensor([1.0, -2.0, 3.0])
            velocity = constant_velocity + self.radial_slope_per_s * position_m
            scalar = position_m[:, 0]
            zero = 0.0 * scalar
            return {
                "temperature": torch.full_like(scalar, 5_000.0),
                "gas_pressure": torch.ones_like(scalar),
                "magnetic_field": torch.stack((zero, zero, zero), dim=-1),
                "velocity_field": velocity,
            }

    constraints = MagnetofluidConstraints(
        {"upper_boundary_open_velocity": {"enabled": True}},
        normalization={"length_m": 2.0, "velocity_scale_m_per_s": 4.0},
        vector_basis_matches_spatial_coordinates=True,
    )
    position_m = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )
    time_hours = torch.zeros(3, 1)

    constant = RadialLinearVelocityAtmosphere(0.0)
    constant_result = constraints.upper_boundary(
        constant,
        position_m,
        time_hours,
    )
    torch.testing.assert_close(
        constant_result.losses["upper_boundary_open_velocity"],
        torch.zeros(()),
    )

    varying = RadialLinearVelocityAtmosphere(2.0)
    varying_result = constraints.upper_boundary(
        varying,
        position_m,
        time_hours,
    )
    loss = varying_result.losses["upper_boundary_open_velocity"]
    assert loss > 0.0
    loss.backward()
    assert torch.isfinite(varying.radial_slope_per_s.grad)
    assert varying.radial_slope_per_s.grad > 0.0


def test_side_boundary_applies_open_velocity_and_current_free_losses():
    class LinearSideAtmosphere(torch.nn.Module):
        solar_radius_m = 10.0

        def __init__(self):
            super().__init__()
            self.velocity_slope = torch.nn.Parameter(torch.tensor(2.0))
            self.magnetic_slope = torch.nn.Parameter(torch.tensor(3.0))

        def evaluate_position_rsun(self, position_rsun, time_hours=None):
            del time_hours
            position_m = position_rsun * self.solar_radius_m
            x = position_m[:, 0]
            zero = 0.0 * x
            return {
                "temperature": torch.full_like(x, 5_000.0),
                "gas_pressure": torch.ones_like(x),
                "magnetic_field": torch.stack(
                    (zero, self.magnetic_slope * x, zero), dim=-1
                ),
                "velocity_field": self.velocity_slope * position_m,
            }

    model = LinearSideAtmosphere()
    constraints = MagnetofluidConstraints(
        {
            "side_boundary_open_velocity": {"enabled": True},
            "side_boundary_current_free": {"enabled": True},
        },
        normalization={
            "length_m": 2.0,
            "velocity_scale_m_per_s": 4.0,
            "magnetic_field_floor_gauss": 1.0,
        },
        vector_basis_matches_spatial_coordinates=True,
    )
    position_m = torch.tensor(
        [[10.0, 1.0, 0.0], [10.0, -1.0, 0.0], [10.0, 0.0, 1.0], [10.0, 0.0, -1.0]]
    )
    time_hours = torch.zeros(4, 1)
    normal = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )

    result = constraints.side_boundary(
        model,
        position_m,
        time_hours,
        normal,
        height_group_shape=(1, 4),
    )

    assert set(result.losses) == set(SIDE_BOUNDARY_EQUATIONS)
    assert result.losses["side_boundary_open_velocity"] > 0.0
    assert result.losses["side_boundary_current_free"] > 0.0
    sum(result.losses.values()).backward()
    assert torch.isfinite(model.velocity_slope.grad)
    assert torch.isfinite(model.magnetic_slope.grad)
