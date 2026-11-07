"""Stokes likelihood metrics distinguish raw errors from loss reweighting."""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from prom3theus.inversion.data_terms.stokes import StokesObservationTerm
from prom3theus.inversion.objective import StokesObjective


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("warmup", [0, 10])
@pytest.mark.parametrize("fit_velocity", [True, False])
def test_stokes_raw_normalization_metrics(training, warmup, fit_velocity):
    term = StokesObservationTerm.__new__(StokesObservationTerm)
    nn.Module.__init__(term)
    term.instrument_type = "spectrograph"
    term.objective = StokesObjective(continuum_normalization=True, continuum_indices=[0, 2])
    term.stokes_weights = torch.ones(4)
    term.wavelength_weights = torch.ones(3)
    term.qu_warmup_steps, term.step = warmup, 0
    term.maximum_velocity_correction = torch.tensor(10000.)
    term.instrument_line_of_sight_velocity_correction_normalized = nn.Parameter(
        torch.tensor(0.), requires_grad=fit_velocity)
    target = torch.ones(2, 4, 3) * torch.tensor([.05, .5])[:, None, None]
    predicted = target * .9
    calls = []

    def predict(batch):
        calls.append(batch)
        return predicted, predicted

    term._prediction_and_stokes = predict
    term.synthesis_window = SimpleNamespace(admissibility_weight=.01)
    term.train(training)
    result = term.evaluate_batch({
        "coordinates": torch.zeros(2, 3), "ray_direction": None,
        "stokes_basis": None, "observer_los_velocity_m_per_s": None,
        "stokes": target,
    })
    assert len(calls) == 1
    for name in ("I", "Q", "U", "V"):
        torch.testing.assert_close(
            result.metrics[f"stokes_{name}_raw_residual_rms"],
            (predicted - target).square().mean().sqrt(),
        )
        torch.testing.assert_close(result.component_losses[name], torch.tensor(.01))
    expected = {f"stokes_{name}_raw_residual_rms" for name in "IQUV"}
    expected.add("stokes_outgoing_admissibility_max")
    if warmup:
        expected.add("qu_weight_factor")
        assert result.metrics["qu_weight_factor"] == (0 if training else 1)
    if fit_velocity:
        expected.add("line_of_sight_velocity_correction_m_per_s")
    assert set(result.metrics) == expected

    from prom3theus.inversion.joint import JointEvaluation
    evaluation = JointEvaluation(result.likelihood_loss, {"stokes": result}, {}, {}, {})
    train, valid = evaluation.training_metrics(), evaluation.validation_metrics()
    assert "streams.stokes.stokes_outgoing_admissibility_max" not in train
    assert "streams.stokes.stokes_outgoing_admissibility_max" in valid
    assert "streams.stokes.line_of_sight_velocity_correction_m_per_s" not in valid
    assert ("streams.stokes.line_of_sight_velocity_correction_m_per_s" in train) == fit_velocity
    assert ("streams.stokes.qu_weight_factor" in train) == bool(warmup)



def test_training_scoring_reuses_transforms_and_keeps_diagnostics_optional(monkeypatch):
    term = StokesObservationTerm.__new__(StokesObservationTerm)
    nn.Module.__init__(term)
    term.objective = StokesObjective(continuum_normalization=True, continuum_indices=[0, 2])
    term.stokes_weights = torch.tensor([1., 10., 10., 2.])
    term.wavelength_weights = torch.ones(3)
    term.qu_warmup_steps = term.step = 0
    term.maximum_velocity_correction = torch.tensor(10000.)
    term.instrument_line_of_sight_velocity_correction_normalized = nn.Parameter(torch.tensor(0.))
    prediction = torch.full((2, 4, 3), .3, requires_grad=True)
    target = torch.full_like(prediction, .5)
    term._prediction_and_stokes = lambda batch: (prediction, prediction)
    calls = []
    original = term.objective.observed_continuum
    monkeypatch.setattr(term.objective, 'observed_continuum', lambda target: (calls.append(1), original(target))[1])
    full = term.evaluate_batch({'stokes': target})
    assert len(calls) == 1
    assert full.metrics and all(not value.requires_grad for value in full.metrics.values())
    full_gradient = torch.autograd.grad(full.likelihood_loss, prediction)[0]
    fast = term.evaluate_training_batch({'stokes': target})
    assert len(calls) == 2  # One continuum evaluation per scoring call.
    assert fast.metrics == {}
    assert fast.diagnostics['prediction'] is prediction
    torch.testing.assert_close(full.likelihood_loss, fast.likelihood_loss)
    torch.testing.assert_close(full_gradient, torch.autograd.grad(fast.likelihood_loss, prediction)[0])


def test_centered_float32_queries_preserve_submetre_velocity_shifts_and_gradient():
    from prom3theus.observations import resolve_velocity_synthesis_mode
    from prom3theus.rt.wavelength import CenteredWavelength
    from prom3theus.core import SPEED_OF_LIGHT
    term = StokesObservationTerm.__new__(StokesObservationTerm)
    nn.Module.__init__(term)
    term.velocity_synthesis_mode = resolve_velocity_synthesis_mode('carrington_registered_relative')
    term.maximum_velocity_correction = torch.tensor(10000.)
    term.instrument_line_of_sight_velocity_correction_normalized = nn.Parameter(torch.tensor(0.))
    term.spectral_origin_angstrom = 6303.
    exact_offsets = torch.tensor([-.725123456, .123456789], dtype=torch.float64)
    term.synthesis_wavelength_vacuum_offset_angstrom = exact_offsets
    speeds = torch.tensor([0., .1, 1., 10., 1000., 10000.])
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Float32QueryAudit(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            assert not any(isinstance(value, torch.Tensor) and value.dtype == torch.float64
                           for value in tree_flatten(result)[0]), func
            return result

    with Float32QueryAudit():
        query = term.spectral_queries({'coordinates': torch.zeros(len(speeds), 3),
            'removed_solar_los_velocity_m_per_s': speeds})
    assert isinstance(query, CenteredWavelength)
    assert query.offset_angstrom.dtype == torch.float32
    reference = ((6303. + exact_offsets) * torch.sqrt(
        (1 - speeds.double()[:, None] / SPEED_OF_LIGHT) /
        (1 + speeds.double()[:, None] / SPEED_OF_LIGHT)) - 6303.)
    torch.testing.assert_close(query.offset_angstrom.double(), reference, rtol=0, atol=8e-8)
    gradient = torch.autograd.grad(query.offset_angstrom[0, 0],
        term.instrument_line_of_sight_velocity_correction_normalized)[0]
    expected = -(6303. + exact_offsets[0]) / SPEED_OF_LIGHT * 10000.
    torch.testing.assert_close(gradient.double(), expected, rtol=2e-7, atol=0)
