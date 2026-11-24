import math

import pytest
import torch

from prom3theus.inversion.objective import StokesObjective


@pytest.mark.parametrize("normalize", [False, True])
def test_linear_stokes_scales_and_observed_normalization_values_and_gradients(normalize):
    prediction = torch.tensor([[[.8, .9], [.01, -.02], [.03, 0.], [-.1, .2]]], requires_grad=True)
    target = prediction.detach().clone() * .5
    target[..., 0, 0] = .6
    objective = StokesObjective(stokes_scales={"I": 1., "Q": .1, "U": .1, "V": .1},
        continuum_normalization=normalize, continuum_indices=[0, 1])
    scales = prediction.new_tensor([1., .1, .1, .1])[:, None]
    observed_continuum = torch.quantile(target[..., :1, :], .9, dim=-1, keepdim=True) if normalize else 1.
    expected = ((prediction - target) / observed_continuum / scales).square()
    actual = objective(prediction, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), prediction, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), prediction)[0])
    restored = StokesObjective(**objective.configuration())
    restored.load_state_dict(objective.state_dict(), strict=True)
    torch.testing.assert_close(restored(prediction, target), actual)
    assert "stokes_scales" not in objective.state_dict()  # Configuration controls resume.


def test_linear_scaling_is_not_an_asinh_stretch():
    objective = StokesObjective(stokes_scales=[1., .1, .1, .1])
    profiles = torch.tensor([[[-1., 0., 1.]]]).expand(2, 4, 3)
    torch.testing.assert_close(objective.transform(profiles)[:, 0], profiles[:, 0])
    torch.testing.assert_close(objective.transform(profiles)[:, 1:], 10 * profiles[:, 1:])


@pytest.mark.parametrize("scales", [[1., 0., .1, .1], [1., -.1, .1, .1],
    [1., float("nan"), .1, .1], [1., float("inf"), .1, .1], [1., .1, .1], {"Q": .1}])
def test_invalid_linear_stokes_scales(scales):
    with pytest.raises(ValueError, match="stokes_scales"):
        StokesObjective(stokes_scales=scales)


def test_linear_scales_cannot_be_combined_with_asinh():
    with pytest.raises(ValueError, match="either"):
        StokesObjective(type="asinh_mse", stokes_scales=[1., .1, .1, .1])


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

    config = load_config("configs/sdo/hmi_aia_dynamic.yaml")
    options = config.streams[0].data_term.objective.to_dict()
    options.pop("stokes_weights")
    options.pop("qu_warmup_steps")
    objective = StokesObjective(**options, continuum_indices=[0, 2])
    restored = StokesObjective(**objective.configuration())
    prediction, target = torch.rand(2, 4, 3) + .1, torch.rand(2, 4, 3) + .1
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
@pytest.mark.parametrize("normalize", [False, True])
def test_training_and_joint_stokes_reductions_match_values_and_gradients(shape, normalize):
    from types import SimpleNamespace

    from prom3theus.inversion.data_terms.stokes import StokesObservationTerm

    torch.manual_seed(42)
    prediction = (torch.rand(shape, dtype=torch.float64) + .1).requires_grad_()
    target = torch.rand_like(prediction) + .1
    spectral = torch.tensor([1.0, 0.0, 2.0, 0.5, 0.0], dtype=torch.float64)
    components = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    objective = StokesObjective(continuum_normalization=normalize, continuum_indices=[0, 4])
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
    if normalize:
        observed_continuum = torch.quantile(target[..., :1, :], .9, dim=-1, keepdim=True)
        residual = residual / observed_continuum
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
    # The observed denominator cannot couple excluded prediction wavelengths.
    assert torch.count_nonzero(gradients[0][..., [1, 4]]) == 0


def test_continuum_normalization_equalizes_fractional_errors_and_preserves_brightness():
    brightness = torch.tensor([1., .5, .2], dtype=torch.float64)
    target = torch.zeros(3, 4, 3, dtype=torch.float64)
    target[:, 0] = brightness[:, None]
    target[:, 3] = .05 * brightness[:, None]
    prediction = target.clone()
    prediction[:, 0] *= .9
    prediction[:, 3] = 0
    prediction.requires_grad_()
    objective = StokesObjective(continuum_normalization=True, continuum_indices=[0, 2])
    loss = objective(prediction, target)
    torch.testing.assert_close(loss[:, 0], torch.full((3, 3), .01, dtype=loss.dtype))
    torch.testing.assert_close(loss[:, 3], torch.full((3, 3), .0025, dtype=loss.dtype))
    expected = ((prediction - target) / brightness[:, None, None]).square()
    torch.testing.assert_close(loss, expected)
    gradient, = torch.autograd.grad(loss.sum(), prediction)
    assert torch.isfinite(gradient).all()
    assert (gradient[:, 0] < 0).all()  # Increase the underpredicted intensity.
    gain = torch.tensor(1.6, dtype=target.dtype, requires_grad=True)
    gain_loss = objective(target * gain, target).sum()
    assert gain_loss > 0
    assert torch.autograd.grad(gain_loss, gain)[0] > 0  # Correct overbrightness.
    restored = StokesObjective(**objective.configuration())
    restored.load_state_dict(objective.state_dict(), strict=True)
    torch.testing.assert_close(restored(prediction, target), loss)


def test_continuum_uses_observed_indices_mean_with_floor_and_detached_denominator():
    target = torch.tensor([[[.01, .9, .03], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]], requires_grad=True)
    prediction = torch.ones_like(target, requires_grad=True)
    objective = StokesObjective(continuum_normalization=True, continuum_floor=.1,
                               continuum_indices=[0, 2])
    torch.testing.assert_close(objective(prediction, target), ((prediction-target)/.1).square())
    # Above the floor, the selected mean is .02, not the brightest sample .9.
    objective.continuum_floor = .001
    loss = objective(prediction, target)
    torch.testing.assert_close(loss, ((prediction-target)/.02).square())
    target_gradient, = torch.autograd.grad(loss.sum(), target)
    torch.testing.assert_close(target_gradient, -2 * (prediction.detach()-target.detach())/.02**2)


@pytest.mark.parametrize("floor", [0, -1, float('nan'), float('inf'), True])
def test_invalid_continuum_floor(floor):
    with pytest.raises(ValueError, match="continuum_floor"):
        StokesObjective(continuum_floor=floor)


@pytest.mark.parametrize("indices", [None, [], [-1], [0, 0], [0.5], [True]])
def test_legacy_normalization_requires_explicit_valid_continuum_indices(indices):
    with pytest.raises(ValueError, match="continuum_indices"):
        StokesObjective(continuum_normalization=True, continuum_floor=.1, continuum_indices=indices)


@pytest.mark.parametrize("quantile", [0, -.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_continuum_quantile(quantile):
    with pytest.raises(ValueError, match="continuum_quantile"):
        StokesObjective(continuum_quantile=quantile)


def test_observed_quantile_normalization_detaches_denominator_and_handles_invalid_samples():
    objective = StokesObjective(continuum_normalization=True)
    prediction = torch.tensor([[[.2, .6, 1.], [.01, .02, .03], [0., 0., 0.], [0., 0., 0.]]], requires_grad=True)
    target = prediction.detach().clone()
    target[:, 1] *= 2
    target.requires_grad_()
    loss = objective(prediction, target)
    target_gradient, = torch.autograd.grad(loss.sum(), target)
    continuum = torch.quantile(target.detach()[:, 0], .9, dim=-1)[:, None, None]
    torch.testing.assert_close(target_gradient, -2 * (prediction.detach() - target.detach()) / continuum.square())
    gradient, = torch.autograd.grad(objective(prediction, target)[:, 1].sum(), prediction)
    assert gradient[:, 0].abs().sum() == 0  # Q errors do not change the I denominator.
    assert objective.valid_sample_mask(target).all()
    for bad in (0., -1., float("nan"), float("inf")):
        invalid = target.clone()
        invalid[:, 0] = bad
        assert not objective.valid_sample_mask(invalid).any()
        assert not torch.isfinite(objective(prediction, invalid)).all()
        if math.isfinite(bad):
            # Finite nonpositive predictions still get a corrective I gradient.
            assert torch.isfinite(objective(invalid, target)).all()
        else:
            assert not torch.isfinite(objective(invalid, target)).all()
        assert objective.valid_sample_mask(target).all()  # A bad prediction never masks valid data.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_continuum_quantile_supports_mixed_precision_without_float64(dtype):
    objective = StokesObjective(continuum_normalization=True)
    profiles = torch.rand(2, 4, 7).to(dtype) + .1
    result = objective(profiles, profiles)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert not result.any()
