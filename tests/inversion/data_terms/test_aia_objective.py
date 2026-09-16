from __future__ import annotations

import pytest
import torch

from prom3theus.inversion.data_terms.aia_objective import AsinhMSEImageObjective


def _objective():
    return AsinhMSEImageObjective(
        (171, 193, 211),
        asinh_scales=(2.0, 3.0, 4.0),
        channel_weights=(1.0, 2.0, 1.0),
    )


@pytest.mark.parametrize("amplitude", [1.0, 1.0e8])
def test_asinh_mse_accepts_negative_targets_and_backpropagates(amplitude):
    prediction = torch.tensor((amplitude, 3.0, 2.0), requires_grad=True)
    target = torch.tensor((-0.2, 4.0, 1.0))
    loss, components, residual = _objective()(prediction, target, torch.arange(3))
    assert set(components) == {"171", "193", "211"}
    assert torch.isfinite(loss)
    assert torch.isfinite(residual).all()
    expected_residual = torch.asinh(
        prediction / torch.tensor([2.0, 3.0, 4.0])
    ) - torch.asinh(target / torch.tensor([2.0, 3.0, 4.0]))
    expected_residual = expected_residual / torch.asinh(
        1 / torch.tensor([2.0, 3.0, 4.0])
    )
    expected_loss = torch.nn.functional.mse_loss(
        expected_residual, torch.zeros_like(expected_residual), reduction="none"
    )
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(
        loss, (expected_loss * torch.tensor([0.25, 0.5, 0.25])).sum()
    )
    reference_loss = (expected_loss * torch.tensor([0.25, 0.5, 0.25])).sum()
    actual_gradient = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(reference_loss, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all()


def test_masked_samples_contribute_exactly_zero():
    prediction = torch.tensor((1.0, 1000.0, 2.0, 3.0))
    target = torch.tensor((1.0, -1000.0, 1.0, 3.0))
    indices = torch.tensor((0, 0, 1, 2))
    valid = torch.tensor((True, False, True, True))
    masked, _, _ = _objective()(prediction, target, indices, valid)
    reference, _, _ = _objective()(
        prediction[[0, 2, 3]], target[[0, 2, 3]], torch.arange(3)
    )
    assert torch.allclose(masked, reference)


def test_missing_channel_policy_is_strict_by_default_and_optional_for_split_batches():
    prediction = torch.tensor((2.0, 3.0))
    target = torch.tensor((1.0, 1.5))
    channel_index = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="193 Angstrom"):
        _objective()(prediction, target, channel_index)

    loss, components, residual = _objective()(
        prediction,
        target,
        channel_index,
        require_all_channels=False,
    )
    assert set(components) == {"171"}
    torch.testing.assert_close(loss, components["171"])
    assert torch.isfinite(loss)
    assert torch.isfinite(residual).all()


def test_zero_weight_channel_remains_renderable_in_split_diagnostic_batch():
    objective = AsinhMSEImageObjective(
        (171, 193, 211),
        asinh_scales=(2.0, 3.0, 4.0),
        channel_weights=(0.0, 1.0, 1.0),
    )

    loss, components, _ = objective(
        torch.tensor((2.0,)),
        torch.tensor((1.0,)),
        torch.tensor((0,)),
        require_all_channels=False,
    )

    assert set(components) == {"171"}
    torch.testing.assert_close(loss, torch.zeros(()))


def test_bulk_channel_reduction_matches_masked_reference_gradient():
    objective = _objective()
    indices = torch.arange(105).remainder(3).reshape(15, 7)
    prediction = torch.linspace(0.1, 30.0, 105).reshape(15, 7).requires_grad_()
    target = prediction.detach() * 0.7
    valid = torch.arange(105).reshape(15, 7).remainder(5) != 0
    loss, _, residual = objective(prediction, target, indices, valid)
    reference = sum(
        objective.channel_weights[i].to(prediction)
        * residual[valid & (indices == i)].square().mean()
        for i in range(3)
    )
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(
        torch.autograd.grad(loss, prediction, retain_graph=True)[0],
        torch.autograd.grad(reference, prediction)[0],
    )
