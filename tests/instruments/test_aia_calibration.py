from __future__ import annotations

import torch

from prom3theus.instruments.aia_euv.calibration import (
    AIAChannelCalibration,
    helmert_contrast,
)


def test_helmert_contrasts_are_orthonormal_and_zero_sum():
    matrix = helmert_contrast(3)
    assert torch.allclose(matrix.T @ matrix, torch.eye(2, dtype=matrix.dtype))
    assert torch.allclose(matrix.sum(dim=0), torch.zeros(2, dtype=matrix.dtype))


def test_aia_gains_start_at_one_and_are_unbounded_identifiable():
    model = AIAChannelCalibration((171, 193, 211))
    assert torch.equal(model.gains, torch.ones(3))
    assert torch.equal(model.prior_loss(), torch.zeros(()))
    with torch.no_grad():
        model.common_log_gain.fill_(2.0)
        model.relative_log_gain.copy_(torch.tensor((1.0, -0.5)))
    relative = model.contrast_matrix @ model.relative_log_gain
    assert torch.allclose(relative.sum(), torch.zeros(()), atol=1e-6)
    assert torch.all(model.gains > 0)
    assert model.prior_loss() > 0
    model.prior_loss().backward()
    assert torch.isfinite(model.common_log_gain.grad)
    assert torch.isfinite(model.relative_log_gain.grad).all()


def test_channel_gain_only_changes_selected_samples():
    model = AIAChannelCalibration((171, 193, 211))
    with torch.no_grad():
        model.relative_log_gain[0] = 0.4
    values = torch.ones(3)
    adjusted = model(values, torch.arange(3))
    assert torch.allclose(adjusted, model.gains)


def test_relative_gaussian_prior_sums_independent_helmert_coordinates():
    model = AIAChannelCalibration(
        (171, 193, 211),
        absolute_prior_fraction=0.25,
        relative_prior_fraction=0.15,
    )
    with torch.no_grad():
        model.relative_log_gain.fill_(model.relative_prior_log_sigma)

    # Two independent one-sigma Helmert coordinates contribute 1/2 each.
    torch.testing.assert_close(model.prior_loss(), torch.tensor(1.0))
