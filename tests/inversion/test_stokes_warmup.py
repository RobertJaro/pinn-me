import pytest
import torch
from torch import nn

from prom3theus.config.schema import PotentialBoundaryConfig, PotentialPhotosphereConfig
from prom3theus.inversion.data_terms.base import DataTermBatchResult, SharedTermResult
from prom3theus.inversion.data_terms.potential_boundary import (
    ProgressivePotentialBoundary,
)
from prom3theus.inversion.data_terms.stokes import StokesObservationTerm
from prom3theus.inversion.joint import JointForwardModel
from prom3theus.inversion.objective import STOKES_COMPONENTS, StokesObjective


class StokesProbe(StokesObservationTerm):
    def __init__(self, warmup=8000):
        nn.Module.__init__(self)
        self.qu_warmup_steps = warmup
        self.step = 0
        self.prediction = nn.Parameter(torch.ones(1, 4, 2))
        self.register_buffer("stokes_weights", torch.tensor([1.0, 1.0, 1.0, 10.0]) / 13)
        self.register_buffer("wavelength_weights", torch.ones(2))
        self.objective = StokesObjective()

    def evaluate_batch(self, batch):
        component, loss = self._loss(self.prediction, torch.zeros_like(self.prediction))
        return DataTermBatchResult(
            likelihood_loss=loss,
            component_losses=dict(zip(STOKES_COMPONENTS, component)),
            metrics={
                "qu_weight_factor": self.prediction.new_tensor(self._qu_weight_factor())
            },
            sample_count=1,
            diagnostics={},
        )


class PotentialProbe(ProgressivePotentialBoundary):
    def __init__(self, enabled=True):
        nn.Module.__init__(self)
        self.step = 0
        self.options = PotentialBoundaryConfig(
            enabled=True,
            start_step=500,
            photosphere=PotentialPhotosphereConfig(enabled=enabled),
        ).to_dict()

    def prepare(self, model):
        pass

    def evaluate(self, model):
        return SharedTermResult(torch.tensor(0.0), {}, {})


def model_and_terms(enabled=True, warmup=8000):
    stokes, potential = StokesProbe(warmup), PotentialProbe(enabled)
    model = JointForwardModel(
        nn.Linear(1, 1), {"stokes": stokes}, {"stokes": 1.0}, {"potential": potential}
    )
    return model, stokes, potential


@pytest.mark.parametrize("step", [0, 499, 500, 750, 1000, 7999, 8000, 9000])
def test_qu_gradients_switch_on_at_warmup_end(step):
    model, stokes, _ = model_and_terms()
    stokes.set_step(step)
    result = model.evaluate_batches({"stokes": {}})
    result.loss.backward()
    grad = stokes.prediction.grad
    factor = 0 if step < 8000 else 1
    torch.testing.assert_close(grad[:, 1:3], torch.full_like(grad[:, 1:3], factor / 13))
    torch.testing.assert_close(grad[:, 0], torch.full_like(grad[:, 0], 1 / 13))
    torch.testing.assert_close(grad[:, 3], torch.full_like(grad[:, 3], 10 / 13))
    assert result.streams["stokes"].component_losses["Q"] > 0
    assert result.training_metrics()["streams.stokes.qu_weight_factor"] == factor


@pytest.mark.parametrize("warmup,validation", [(0, False), (8000, True)])
def test_disabled_warmup_and_validation_keep_full_stokes(warmup, validation):
    model, stokes, _ = model_and_terms(warmup=warmup)
    stokes.set_step(1000)
    if validation:
        model.eval()
    result = model.evaluate_batches({"stokes": {}})
    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    assert result.streams["stokes"].metrics["qu_weight_factor"] == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_warmup_is_independent_of_potential_schedule(enabled):
    model, stokes, potential = model_and_terms(enabled=enabled, warmup=4000)
    potential.options["photosphere"]["end_step"] = 12000
    potential.set_step(13000)
    for step, factor in [(3999, 0), (4000, 1)]:
        stokes.set_step(step)
        result = model.evaluate_batches({"stokes": {}})
        assert result.streams["stokes"].metrics["qu_weight_factor"] == factor
    del model.shared_objectives["potential"]
    stokes.set_step(0)
    assert (
        model.evaluate_batches({"stokes": {}})
        .streams["stokes"]
        .metrics["qu_weight_factor"]
        == 0
    )


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_warmup_requires_nonnegative_integer(value):
    from dataclasses import replace

    from prom3theus.config import load_config

    config = load_config("configs/hmi_aia_dynamic.yaml")
    objective = config.streams[0].data_term.objective
    assert objective.qu_warmup_steps == 0
    with pytest.raises(ValueError, match="qu_warmup_steps"):
        replace(objective, qu_warmup_steps=value)


def test_training_lifecycle_sets_observation_and_shared_steps():
    from types import SimpleNamespace

    from prom3theus.training.joint import JointInversionModule

    model, stokes, potential = model_and_terms()
    for step in (7999, 8000):
        JointInversionModule.set_objective_step(
            SimpleNamespace(model=model, global_step=step)
        )
        assert stokes.step == potential.step == step
        assert stokes._qu_weight_factor() == (0 if step < 8000 else 1)


@pytest.mark.parametrize("name", ["hmi_aia_mhs", "hmi_aia_dynamic", "hmi_lte_dynamic"])
def test_presets_default_to_no_warmup(name):
    from prom3theus.config import load_config
    config = load_config(f"configs/{name}.yaml")
    assert config.streams[0].data_term.objective.qu_warmup_steps == 0
