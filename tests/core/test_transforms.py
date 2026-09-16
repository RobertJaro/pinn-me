import pytest
import torch

from prom3theus.core.transforms import normalized_asinh


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalized_asinh_broadcast_units_sign_and_gradient(dtype):
    scale = torch.tensor([0.001, 0.02, 0.1], dtype=dtype)
    value = (
        torch.tensor([[-1.0], [0.0], [1.0]], dtype=dtype)
        .expand(3, 3)
        .clone()
        .requires_grad_()
    )
    result = normalized_asinh(value, scale)
    assert result.dtype == dtype
    torch.testing.assert_close(result, value)
    (gradient,) = torch.autograd.grad(result.sum(), value)
    expected = 1 / ((value.square() + scale.square()).sqrt() * torch.asinh(1 / scale))
    torch.testing.assert_close(gradient, expected)


def test_stokes_and_aia_compare_identical_normalized_values():
    from prom3theus.inversion.data_terms.aia_objective import AsinhMSEImageObjective
    from prom3theus.inversion.objective import StokesObjective

    prediction = torch.tensor([0.001, 0.02, 0.7], requires_grad=True)
    target = torch.tensor([-0.002, 0.1, 1.0])
    stokes = StokesObjective(type="asinh_mse", asinh_scale=0.001)
    aia = AsinhMSEImageObjective(
        (171, 193, 211), asinh_scales=(0.001,) * 3, channel_weights=(1.0,) * 3
    )
    profile = torch.cat((prediction.new_ones(1), prediction)).reshape(1, 4, 1)
    reference = torch.cat((target.new_ones(1), target)).reshape(1, 4, 1)
    stokes_loss = stokes(profile, reference)[:, 1:].mean()
    aia_loss, _, residual = aia(prediction, target, torch.arange(3))
    torch.testing.assert_close(
        residual, (stokes.transform(profile) - stokes.transform(reference))[0, 1:, 0]
    )
    torch.testing.assert_close(aia_loss, stokes_loss)
    torch.testing.assert_close(
        torch.autograd.grad(aia_loss, prediction, retain_graph=True)[0],
        torch.autograd.grad(stokes_loss, prediction)[0],
    )
