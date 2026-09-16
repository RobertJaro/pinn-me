"""Acceptance checks must not mistake a collapsed magnetic fit for success."""

from pathlib import Path
import runpy

import numpy as np
import pytest


recipe = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/hmi/run_local.py")
)
fit_report = recipe["fit_report"]


def _profiles():
    coordinate = np.linspace(-1, 1, 6)
    profiles = np.stack((1 - 0.5 * np.exp(-coordinate**2 / 0.2),
                         0.01 * np.cos(coordinate * 3),
                         0.01 * np.sin(coordinate * 3),
                         0.05 * coordinate * np.exp(-coordinate**2)))
    return np.broadcast_to(profiles, (8, 4, 6)).copy()


def test_zero_magnetic_prediction_cannot_pass_with_perfect_intensity():
    target = _profiles()
    zero = target.copy()
    zero[:, 1:] = 0
    result = fit_report(np.zeros_like(target), zero, target)
    assert result["checks"]["weighted_loss_halved"]
    assert not result["checks"]["stokes_rms_within_limits"]
    assert not result["passed"]


def test_recovered_polarized_profiles_pass_and_document_sample_role():
    target = _profiles()
    result = fit_report(np.zeros_like(target), target, target)
    assert result["passed"]
    assert result["components"]["V"]["final_rms"] == 0
    assert "eligible for training" in result["sample_role"]


def test_unpolarized_sample_is_insufficient_for_coronal_baseline():
    target = _profiles()
    target[:, 1:] = 0
    assert not fit_report(np.zeros_like(target), target, target)["passed"]


def test_nonfinite_and_mismatched_predictions_are_rejected():
    target = _profiles()
    with pytest.raises(ValueError, match="matching shapes"):
        fit_report(target, target[:1], target)
    invalid = target.copy()
    invalid[0, 3, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        fit_report(target, invalid, target)


def test_representative_lattice_covers_crop_without_primary_sample_overlap():
    indices = recipe["representative_pixel_indices"]((256, 256))
    assert indices.shape == (1024, 2)
    assert len(np.unique(indices, axis=0)) == 1024
    for axis in (0, 1):
        coordinates, counts = np.unique(indices[:, axis], return_counts=True)
        np.testing.assert_array_equal(coordinates, np.arange(4, 256, 8))
        np.testing.assert_array_equal(counts, np.full(32, 32))
    primary = {(row, column) for row in range(0, 256, 16) for column in range(0, 256, 16)}
    assert not primary.intersection(map(tuple, indices))
    with pytest.raises(ValueError, match="256 x 256"):
        recipe["representative_pixel_indices"]((128, 256))


def test_stokes_sample_prediction_uses_eval_mode_and_restores_training_mode():
    import torch

    class Term(torch.nn.Module):
        def predict(self, batch):
            assert not self.training
            assert not torch.is_grad_enabled()
            return batch["stokes"]

    model = torch.nn.Module()
    model.terms = torch.nn.ModuleDict({"stokes": Term()})
    target = torch.arange(8 * 4 * 6).reshape(8, 4, 6).float()
    for training in (True, False):
        model.train(training)
        prediction = recipe["predict_stokes_sample"](
            model, "stokes", {"stokes": target}, batch_size=3,
        )
        np.testing.assert_array_equal(prediction, target.numpy())
        assert model.training is training
        assert model.terms["stokes"].training is training
