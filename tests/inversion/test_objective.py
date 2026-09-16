import math

import pytest
import torch

from prom3theus.inversion.objective import StokesObjective


def test_asinh_polarization_linear_intensity_values_and_gradients():
    prediction = torch.tensor([[[0.8], [0.0], [-0.01], [0.1]]], requires_grad=True)
    target = torch.tensor([[[0.9], [0.002], [0.01], [-0.05]]])
    objective = StokesObjective(type="asinh_mse", asinh_scale=1e-3)
    loss = objective(prediction, target)
    expected_residual = torch.cat(
        (
            prediction[:, :1] - target[:, :1],
            (
                torch.asinh(prediction[:, 1:] / 0.001)
                - torch.asinh(target[:, 1:] / 0.001)
            )
            / math.asinh(1000),
        ),
        dim=1,
    )
    torch.testing.assert_close(loss, expected_residual.square())
    (gradient,) = torch.autograd.grad(loss.sum(), prediction)
    derivative = torch.cat(
        (
            torch.ones_like(prediction[:, :1]),
            1 / ((prediction[:, 1:].square() + 1e-6).sqrt() * math.asinh(1000)),
        ),
        dim=1,
    )
    torch.testing.assert_close(gradient, 2 * expected_residual * derivative)
    assert torch.isfinite(gradient).all()
    assert gradient[0, 1, 0] < 0  # Nonzero corrective gradient at zero Q.
    endpoints = torch.tensor([-1.0, 0.0, 1.0]).expand(2, 4, 3)
    torch.testing.assert_close(objective.transform(endpoints), endpoints)
    restored = StokesObjective(**objective.configuration())
    restored.load_state_dict(objective.state_dict())
    torch.testing.assert_close(restored(prediction, target), loss)


def test_omitted_sigmas_mean_unscaled_mse():
    prediction, target = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    torch.testing.assert_close(
        StokesObjective()(prediction, target), (prediction - target).square()
    )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True])
def test_asinh_scale_must_be_positive_finite(scale):
    with pytest.raises(ValueError):
        StokesObjective(type="asinh_mse", asinh_scale=scale)


def test_dynamic_config_objective_constructs_and_roundtrips():
    from prom3theus.config import load_config

    config = load_config("configs/hmi_aia_dynamic.yaml")
    options = config.streams[0].data_term.objective.to_dict()
    options.pop("stokes_weights")
    options.pop("qu_warmup_steps")
    objective = StokesObjective(**options)
    restored = StokesObjective(**objective.configuration())
    prediction, target = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    torch.testing.assert_close(
        restored(prediction, target), objective(prediction, target)
    )


@pytest.mark.parametrize("amplitude", [0.01, 1.0, 100.0])
def test_stokes_objective_applies_unscaled_squared_error(amplitude):
    prediction = torch.zeros(2, 4, 3)
    target = torch.ones_like(prediction) * amplitude
    result = StokesObjective()(prediction, target)
    expected = torch.full_like(result, amplitude**2)
    torch.testing.assert_close(result, expected)


def test_stokes_objective_configuration_is_canonical():
    objective = StokesObjective()
    assert objective.configuration() == {
        "type": "mse",
    }


def test_stokes_objective_rejects_non_stokes_shape():
    with pytest.raises(ValueError, match="four components"):
        StokesObjective()(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 4, 5)])
def test_training_and_joint_stokes_reductions_match_values_and_gradients(shape):
    from types import SimpleNamespace

    from prom3theus.inversion.data_terms.stokes import StokesObservationTerm

    torch.manual_seed(42)
    prediction = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction)
    spectral = torch.tensor([1.0, 0.0, 2.0, 0.5, 0.0], dtype=torch.float64)
    components = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    objective = StokesObjective()
    harness = SimpleNamespace(
        _qu_weight_factor=lambda: 1.0,
        stokes_loss=objective,
        objective=objective,
        wavelength_weights=spectral,
        stokes_weights=components,
    )
    joint_components, joint_loss = StokesObservationTerm._loss(
        harness, prediction, target
    )
    # Independent reference through PyTorch's MSE loss and explicit sample means.
    residual = prediction - target
    elementwise = torch.nn.functional.mse_loss(
        residual, torch.zeros_like(residual), reduction="none"
    )
    expected = (elementwise * spectral).sum(-1).reshape(-1, 4).mean(0) / spectral.sum()
    expected_loss = (expected * components).sum()
    torch.testing.assert_close(joint_components, expected)
    torch.testing.assert_close(joint_loss, expected_loss)
    gradients = [
        torch.autograd.grad(loss, prediction, retain_graph=True)[0]
        for loss in (joint_loss, expected_loss)
    ]
    torch.testing.assert_close(gradients[0], gradients[1])
    assert torch.count_nonzero(gradients[0][..., [1, 4]]) == 0
