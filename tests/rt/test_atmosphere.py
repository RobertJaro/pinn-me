import pytest
import torch

from prom3theus.rt import StratifiedAtmosphere, StratifiedAtmosphereModel


def _model(depth: int = 9) -> StratifiedAtmosphereModel:
    return StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, depth),
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


def test_atmosphere_model_starts_at_radial_reference():
    model = _model(13)
    atmosphere = model(torch.tensor([[0.0, 0.0, 0.0]]))
    height = model.depth_to_height(model.log_tau500)
    reference = model.reference_atmosphere.logs_at_height(height)
    torch.testing.assert_close(atmosphere.temperature[0], reference[0].exp())
    torch.testing.assert_close(atmosphere.gas_pressure[0], reference[1].exp())
    torch.testing.assert_close(atmosphere.microturbulence[0], reference[2].exp())
    assert atmosphere.velocity_field.shape == (1, 13, 3)
    assert atmosphere.magnetic_field.shape == (1, 13, 3)


def test_atmosphere_is_a_differentiable_continuous_field():
    model = _model()
    coordinates = torch.tensor([[-0.2, 0.1, 0.0], [0.3, -0.1, 0.0]])
    height = torch.tensor([[-3.0e4], [4.0e5]], requires_grad=True)
    fields = model.evaluate_at_height(coordinates, height)
    (fields["temperature"].mean() + fields["magnetic_field"].square().mean()).backward()
    assert height.grad is not None and torch.isfinite(height.grad).all()


@pytest.mark.parametrize("activation", ["relu", "swish"])
def test_atmosphere_rejects_noncanonical_or_nonsmooth_activations(activation):
    with pytest.raises(ValueError, match="smooth activation"):
        StratifiedAtmosphereModel(
            torch.linspace(-5.0, 1.0, 5),
            scene_geometry_config={
                "solar_radius_m": 695_700_000.0,
                "scene_basis": torch.eye(3),
            },
            model_config={
                "type": "mlp",
                "dim": 10,
                "n_layers": 2,
                "activation": activation,
                "encoding_config": {"type": "identity"},
            },
        )


def test_stratified_atmosphere_enforces_depth_and_dtype_contracts():
    fields = {
        "temperature": torch.ones(1, 2),
        "velocity_field": torch.zeros(1, 2, 3),
        "microturbulence": torch.ones(1, 2),
        "magnetic_field": torch.zeros(1, 2, 3),
        "gas_pressure": torch.ones(1, 2),
    }
    with pytest.raises(ValueError, match="strictly increase"):
        StratifiedAtmosphere(log_tau500=torch.tensor([0.0, -1.0]), **fields)
    with pytest.raises(TypeError, match="same dtype"):
        StratifiedAtmosphere(
            log_tau500=torch.tensor([-1.0, 0.0]),
            **{**fields, "gas_pressure": torch.ones(1, 2, dtype=torch.float64)},
        )


def test_atmosphere_model_rejects_invalid_evaluation_grids_and_rays():
    model = _model()
    coordinates = torch.tensor([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="strictly increase"):
        model(coordinates, torch.tensor([-4.0, -3.0, -3.5]))
    with pytest.raises(ValueError, match="represented depth interval"):
        model(coordinates, torch.tensor([-5.1, 0.0]))
    with pytest.raises(ValueError, match="finite non-zero"):
        model.trace_rays(
            coordinates,
            torch.zeros(1, 3),
            torch.tensor([-4.0, 0.0]),
        )
