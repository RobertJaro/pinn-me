from __future__ import annotations

from collections.abc import Mapping

import torch

from prom3theus.inversion.data_terms import (
    DataTermBatchResult,
    ObservationDataTerm,
    SharedObjectiveTerm,
    SharedTermResult,
)
from prom3theus.inversion.joint import JointForwardModel


class _Term(ObservationDataTerm):
    observation_kind = "test"

    def __init__(self, likelihood: float, nuisance: float):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(likelihood))
        self.prior = torch.nn.Parameter(torch.tensor(nuisance))
        self.nuisance_calls = 0

    def evaluate_batch(self, batch: Mapping) -> DataTermBatchResult:
        loss = self.value * batch["scale"]
        return DataTermBatchResult(
            likelihood_loss=loss,
            component_losses={"data": loss},
            metrics={"mean": self.value},
            sample_count=1,
        )

    def nuisance_losses(self):
        self.nuisance_calls += 1
        return {"prior": self.prior.square()}


class _Shared(SharedObjectiveTerm):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def evaluate(self, atmosphere_model):
        self.calls += 1
        value = atmosphere_model.weight.square().mean()
        return SharedTermResult(value, {"weight": value}, {})


def test_zero_weight_stream_renders_without_likelihood_or_prior_gradients():
    atmosphere = torch.nn.Linear(1, 1)
    left = _Term(2.0, 3.0)
    right = _Term(5.0, 7.0)
    shared = _Shared()
    model = JointForwardModel(
        atmosphere,
        {"left": left, "right": right},
        {"left": 0.25, "right": 0.0},
        {"regularizer": shared},
    )

    result = model.evaluate_batches(
        {"left": {"scale": torch.tensor(2.0)}, "right": {"scale": torch.tensor(1.0)}}
    )

    expected = 1.0 + 9.0 + atmosphere.weight.square().mean()
    assert torch.allclose(result.loss, expected)
    assert left.nuisance_calls == 1
    assert right.nuisance_calls == 0
    assert result.streams["right"].likelihood_loss.item() == 5.0
    assert not result.streams["right"].likelihood_loss.requires_grad
    result.loss.backward()
    assert right.value.grad is None
    assert right.prior.grad is None
    assert left.value.grad is not None
    assert left.prior.grad is not None
    assert "streams.right.data_loss" not in result.training_metrics()
    assert shared.calls == 1
    assert set(result.scalar_metrics()) >= {
        "loss",
        "streams.left.likelihood",
        "streams.right.likelihood",
        "nuisance.left.prior",
        "shared.regularizer.loss",
    }


def test_positive_weight_enables_likelihood_and_calibration_priors():
    term = _Term(2.0, 3.0)
    model = JointForwardModel(torch.nn.Linear(1, 1), {"aia": term}, {"aia": 0.1})
    result = model.evaluate_batches({"aia": {"scale": torch.tensor(2.0)}})
    torch.testing.assert_close(result.loss, torch.tensor(9.4))
    result.loss.backward()
    torch.testing.assert_close(term.value.grad, torch.tensor(0.2))
    torch.testing.assert_close(term.prior.grad, torch.tensor(6.0))


def test_joint_model_requires_exact_stream_batch_mapping():
    model = JointForwardModel(
        torch.nn.Linear(1, 1), {"only": _Term(1.0, 0.0)}, {"only": 1.0}
    )
    try:
        model.evaluate_batches({})
    except KeyError as error:
        assert "match configured streams" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Missing stream batch was accepted.")


def test_stream_weight_schedule_prioritizes_early_data_fit_then_returns_to_base():
    term = _Term(2.0, 0.0)
    model = JointForwardModel(
        torch.nn.Linear(1, 1),
        {"stokes": term},
        {"stokes": 1.0},
        weight_schedules={
            "stokes": {
                "initial_weight": 5.0,
                "initial_steps": 2,
                "ramp_steps": 4,
            }
        },
    )

    for step, expected_weight in ((0, 5.0), (2, 5.0), (4, 3.0), (6, 1.0), (10, 1.0)):
        model.set_step(step)
        result = model.evaluate_batches({"stokes": {"scale": torch.tensor(1.0)}})
        assert model.effective_stream_weight("stokes") == expected_weight
        torch.testing.assert_close(
            result.weighted_likelihoods["stokes"],
            torch.tensor(2.0 * expected_weight),
        )
        torch.testing.assert_close(
            result.training_metrics()["streams.stokes.weight"],
            torch.tensor(expected_weight),
        )
