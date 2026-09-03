import pytest
import torch

import prom3theus.inversion.constraints.magnetofluid as magnetofluid
from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.core import CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S
from prom3theus.inversion.constraints.magnetofluid import (
    MagnetofluidConstraints,
    PhysicsNormalization,
    _curl,
)


class _Opacity:
    @staticmethod
    def reference_mass_density(temperature, gas_pressure):
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
    return module.volume(
        model,
        opacity,
        position_m,
        coords[:, 2:3],
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
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-12, rtol=0.0)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_normalization_accepts_physical_time_scale_and_rejects_unknown_keys():
    assert PhysicsNormalization().configuration() == {
        "length_m": 1.0e6,
        "time_s": 3_600.0,
    }
    assert PhysicsNormalization.from_config({"time_s": 60.0}).time_s == 60.0
    with pytest.raises(TypeError, match="Unknown physics normalization"):
        PhysicsNormalization.from_config({"cadence_s": 1000.0})
    with pytest.raises(TypeError, match="must be numeric"):
        PhysicsNormalization(time_s=True)


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
    assert result.state.time_derivative("magnetic").shape == (2, 3)
    assert result.state.time_derivative("density").shape == (2, 1)
    assert set(result.state.primitive_derivatives.slices) == {
        "pressure",
        "magnetic",
        "velocity",
        "density",
    }
    assert result.state.derivative("velocity_cross_magnetic").shape == (2, 3, 3)
    assert result.state.derivative("mass_flux").shape == (2, 3, 3)
    (
        result.losses["momentum"]
        + result.losses["induction"]
        + result.losses["continuity"]
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


def test_momentum_matches_exact_rotating_frame_balance():
    class ConstantDensityOpacity:
        @staticmethod
        def reference_mass_density(temperature, gas_pressure):
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
                + gravity * radius
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
