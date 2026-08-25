import pytest
import torch

from pme.lte.atmosphere import StratifiedAtmosphere, StratifiedAtmosphereModel


def _model(depth=9):
    return StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, depth),
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "dim": 10,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )


def test_stratified_atmosphere_validates_shapes_and_depth_order():
    q = torch.linspace(-5.0, 1.0, 7)
    scalar = torch.ones(2, 7)
    vector = torch.zeros(2, 7, 3)
    atmosphere = StratifiedAtmosphere(q, scalar * 6000, vector, scalar, vector)
    assert atmosphere.depth == 7
    assert atmosphere.v_los.shape == (2, 7)
    with pytest.raises(ValueError, match="increase|top.*bottom|depth"):
        StratifiedAtmosphere(q.flip(0), scalar, vector, scalar, vector)


def test_model_starts_at_falc_reference_and_predicts_log_perturbations():
    model = _model(13)
    coords = torch.tensor([[0.0, 0.0, 0.0]])
    atmosphere = model(coords)
    height = model.depth_to_height(model.log_tau500)
    reference = model.reference_atmosphere.logs_at_height(height)
    torch.testing.assert_close(atmosphere.temperature[0], reference[0].exp())
    torch.testing.assert_close(atmosphere.gas_pressure[0], reference[1].exp())
    torch.testing.assert_close(atmosphere.microturbulence[0], reference[2].exp())
    assert height[torch.argmin(model.log_tau500.abs())].abs() < 1.0

    with torch.no_grad():
        model.network.out_layer.bias[0] = 0.4
        model.network.out_layer.bias[8] = -0.2
    perturbed = model(coords)
    assert torch.all(perturbed.temperature > 0)
    assert not torch.allclose(perturbed.temperature, atmosphere.temperature)
    assert not torch.allclose(perturbed.gas_pressure, atmosphere.gas_pressure)


def test_configured_shell_bounds_rescale_depth_mapping_around_solar_surface():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 25),
        shell_height_bounds_Mm=(1.5, -0.1),
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "dim": 10,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    height = model.depth_to_height(model.log_tau500)
    torch.testing.assert_close(height[0], torch.tensor(1.5e6))
    torch.testing.assert_close(height[-1], torch.tensor(-0.1e6))
    torch.testing.assert_close(height[20], torch.tensor(0.0), atol=1.0, rtol=0.0)
    assert torch.all(height[1:] < height[:-1])
    assert torch.all(model.depth_metric_from_log_tau(model.log_tau500) > 0)

    below_reference = model.reference_atmosphere.logs_at_height(
        torch.tensor((-68_815.437, -100_000.0))
    )
    assert below_reference[1][1] > below_reference[1][0]


def test_model_is_continuous_differentiable_xyz_field_in_float32():
    model = _model()
    coords = torch.tensor([[0.0, -0.2, 0.1], [0.0, 0.3, -0.1]])
    height = torch.tensor([[-3.0e4], [4.0e5]], requires_grad=True)
    fields = model.evaluate_at_height(coords, height)
    loss = fields["temperature"].mean() + fields["magnetic_field"].square().mean()
    loss.backward()
    assert height.grad is not None and torch.isfinite(height.grad).all()
    assert all(value.dtype == torch.float32 for value in fields.values())


def test_trace_rays_uses_ordered_local_solar_radius_intersections():
    model = _model(17)
    coords = torch.tensor([[0.0, 20.0, -10.0]])
    origin = torch.tensor([[0.0, 0.0, 1.5e11]])
    surface = model.position_from_coords_height(coords, torch.zeros(1))
    direction = surface - origin
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    atmosphere, trace = model.trace_rays(coords, origin, direction, model.log_tau500)
    expected_height = model.depth_to_height(model.log_tau500)
    torch.testing.assert_close(trace.geometric_height_m[0], expected_height, atol=128.0, rtol=0)
    assert torch.all(trace.distance_m[..., 1:] > trace.distance_m[..., :-1])
    assert trace.position_m.dtype == torch.float32
    assert atmosphere.temperature.shape == (1, 17)


def test_model_rejects_nonsmooth_activation_and_legacy_options():
    with pytest.raises(ValueError, match="smooth activation"):
        StratifiedAtmosphereModel(
            torch.linspace(-5.0, 1.0, 5),
            scene_geometry_config={"solar_radius_m": 695_700_000.0, "scene_basis": torch.eye(3)},
            model_config={"activation": "relu"},
        )
    with pytest.raises(TypeError):
        StratifiedAtmosphereModel(
            torch.linspace(-5.0, 1.0, 5),
            coordinate_mode="log_tau",
            scene_geometry_config={"solar_radius_m": 695_700_000.0, "scene_basis": torch.eye(3)},
        )
