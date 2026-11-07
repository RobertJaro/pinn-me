"""Metrics retain stream identity even when modalities repeat."""
import torch
import pytest
from prom3theus.inversion.joint import JointForwardModel
from prom3theus.inversion.data_terms.base import (
    ObservationDataTerm,
    DataTermBatchResult,
)


class Term(ObservationDataTerm):
    observation_kind = "stokes"

    def evaluate_batch(self, batch):
        return DataTermBatchResult(batch["loss"], {"I": batch["loss"]}, {}, 1)


def test_scientific_contract_ignores_loader_policy_and_schedule_owns_batch_size():
    from prom3theus.training.joint import _scientific_contract

    def contract(workers, pin_memory, batch_size=2):
        return {"configuration": {"streams": [{"observation": {"loader": {
            "workers": workers, "pin_memory": pin_memory, "batch_size": batch_size,
        }}}]}}

    old = contract(0, True)
    assert _scientific_contract(old) == _scientific_contract(contract(1, False))
    assert _scientific_contract(old) == _scientific_contract(contract(1, False, 4))
    assert old["configuration"]["streams"][0]["observation"]["loader"]["pin_memory"]


def test_loss_balancer_uses_training_reference_stream():
    from prom3theus.config.joint_schema import JointTrainingConfig, LossBalanceConfig
    from prom3theus.training.joint import JointInversionModule

    model = JointForwardModel(
        torch.nn.Linear(1, 1),
        {"stokes": Term(), "aia": Term()},
        {"stokes": 1.0, "aia": 1.0},
    )
    settings = JointTrainingConfig(
        max_steps=1,
        learning_rate=0.01,
        final_learning_rate=0.001,
        reference_stream="stokes",
        loss_balance=LossBalanceConfig(enabled=True),
    )
    module = JointInversionModule(model, settings, {})

    assert module.loss_balancer is not None
    assert module.loss_balancer.reference_stream == "stokes"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_gradient_checks_preserve_parameter_specific_errors(invalid):
    from prom3theus.training.joint import JointInversionModule
    from prom3theus.config.joint_schema import JointTrainingConfig

    layer = torch.nn.Linear(3, 2)
    model = JointForwardModel(layer, {"data": Term()}, {"data": 1.0})
    module = JointInversionModule(
        model,
        JointTrainingConfig(max_steps=1, learning_rate=0.01, final_learning_rate=0.001),
        {},
    )
    module.on_after_backward()  # Parameters without gradients are valid.
    layer(torch.ones(1, 3)).sum().backward()
    module.on_after_backward()
    layer.bias.grad[1] = invalid
    with pytest.raises(FloatingPointError, match="atmosphere_model.bias"):
        module.on_after_backward()


def test_same_modality_metrics_keep_both_streams():
    model = JointForwardModel(
        torch.nn.Linear(1, 1),
        {"hmi": Term(), "hinode": Term()},
        {"hmi": 1.0, "hinode": 1.0},
    )
    result = model.evaluate_batches(
        {"hmi": {"loss": torch.tensor(2.0)}, "hinode": {"loss": torch.tensor(3.0)}}
    )
    metrics = result.scalar_metrics()
    assert metrics["streams.hmi.likelihood"] == 2
    assert metrics["streams.hinode.likelihood"] == 3
    assert metrics["loss"] == 5


def test_training_and_validation_have_distinct_minimal_summaries():
    from prom3theus.inversion.joint import JointEvaluation
    from prom3theus.inversion.data_terms.base import SharedTermResult

    evaluation = JointEvaluation(
        loss=torch.tensor(10.0),
        streams={
            "aia": DataTermBatchResult(
                torch.tensor(2.0),
                {"171": torch.tensor(1.0)},
                {"calibration": torch.tensor(1.0), "asinh_rmse": torch.tensor(0.5)},
                1,
            )
        },
        weighted_likelihoods={"aia": torch.tensor(0.2)},
        nuisance_losses={
            "aia.absolute": torch.tensor(0.1),
            "aia.relative": torch.tensor(0.2),
        },
        shared_terms={
            "physics": SharedTermResult(
                torch.tensor(3.0),
                {"divergence": torch.tensor(2.0)},
                {"raw_divergence": torch.tensor(20.0)},
            )
        },
    )
    summary = evaluation.training_metrics()
    assert set(summary) == {
        "loss",
        "streams.aia.data_loss",
        "shared.physics.divergence",
        "nuisance.aia.absolute",
        "nuisance.aia.relative",
    }
    torch.testing.assert_close(summary["streams.aia.data_loss"], torch.tensor(0.2))
    validation = evaluation.validation_metrics()
    assert set(validation) == {"streams.aia.data_loss", "streams.aia.asinh_rmse"}
    assert validation["streams.aia.data_loss"] == 2.0
    details = evaluation.scalar_metrics()
    assert "streams.aia.weighted_likelihood" in details
    assert "shared.physics.metrics.raw_divergence" in details


def test_training_logs_only_progress_summary(monkeypatch):
    from prom3theus.training.joint import JointInversionModule
    from prom3theus.config.joint_schema import JointTrainingConfig

    model = JointForwardModel(torch.nn.Linear(1, 1), {"hmi": Term()}, {"hmi": 1.0})
    module = JointInversionModule(
        model,
        JointTrainingConfig(max_steps=1, learning_rate=0.01, final_learning_rate=0.001),
        {},
    )
    logged = {}
    monkeypatch.setattr(
        module, "log", lambda key, value, **kwargs: logged.update({key: value})
    )
    module.training_step({"hmi": {"loss": torch.tensor(2.0, requires_grad=True)}}, 0)
    assert set(logged) == {"train.loss", "train.streams.hmi.data_loss", "train.streams.hmi.component_mse.I"}


def test_zero_weight_stream_is_not_logged_but_active_zero_loss_is():
    model = JointForwardModel(
        torch.nn.Linear(1, 1),
        {"active": Term(), "disabled": Term()},
        {"active": 2.0, "disabled": 0.0},
    )
    result = model.evaluate_batches(
        {
            "active": {"loss": torch.tensor(0.0)},
            "disabled": {"loss": torch.tensor(99.0)},
        }
    )
    assert set(result.training_metrics()) == {"loss", "streams.active.data_loss", "streams.active.component_mse.I"}
    assert "streams.disabled.data_loss" not in result.validation_metrics()
    model.eval()
    validation = model.evaluate_batches(
        {"active": {"loss": torch.tensor(0.0)}, "disabled": {"loss": torch.tensor(99.0)}}
    )
    assert "streams.disabled.data_loss" in validation.validation_metrics()


def test_inactive_shared_term_does_not_create_a_training_series():
    from dataclasses import replace
    from prom3theus.inversion.data_terms.base import SharedTermResult

    model = JointForwardModel(torch.nn.Linear(1, 1), {"data": Term()}, {"data": 1.0})
    result = model.evaluate_batches({"data": {"loss": torch.tensor(1.0)}})
    result = replace(
        result,
        shared_terms={
            "disabled": SharedTermResult(torch.tensor(0.0), {}, {}, active=False)
        },
    )
    assert set(result.training_metrics()) == {"loss", "streams.data.data_loss", "streams.data.component_mse.I"}


@pytest.mark.parametrize("clip_norm", [0.1, None])
@pytest.mark.parametrize("multiple_groups", [False, True])
def test_optimizer_logs_only_shared_learning_rate(monkeypatch, clip_norm, multiple_groups):
    from prom3theus.config.joint_schema import JointTrainingConfig
    from prom3theus.training.joint import JointInversionModule

    layer = torch.nn.Linear(2, 1, bias=True)
    model = JointForwardModel(layer, {"data": Term()}, {"data": 1.0})
    module = JointInversionModule(
        model,
        JointTrainingConfig(
            max_steps=1, learning_rate=0.01, final_learning_rate=0.001,
            gradient_clip_norm=clip_norm,
        ),
        {},
    )
    parameters = ([{"params": [layer.weight]}, {"params": [layer.bias]}]
                  if multiple_groups else module.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.003)
    layer.weight.grad = torch.tensor([[3., 0.]])
    layer.bias.grad = torch.tensor([4.])
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.update({name: value}))
    module.on_before_optimizer_step(optimizer)

    assert logged == {"train.optimizer.learning_rate": .003}
    # Removing metrics must not alter gradients; Lightning owns clipping.
    torch.testing.assert_close(layer.weight.grad, torch.tensor([[3., 0.]]))
    torch.testing.assert_close(layer.bias.grad, torch.tensor([4.]))


def test_training_omits_diagnostics_and_metric_collectives_between_log_steps(monkeypatch):
    from types import SimpleNamespace
    from prom3theus.config.joint_schema import JointTrainingConfig
    from prom3theus.training.joint import JointInversionModule
    import prom3theus.training.joint as training

    class AuditedTerm(Term):
        def __init__(self):
            super().__init__()
            self.calls = []

        def evaluate_batch(self, batch):
            self.calls.append('diagnostics')
            return super().evaluate_batch(batch)

        def evaluate_training_batch(self, batch):
            self.calls.append('score')
            return super().evaluate_batch(batch)

    term = AuditedTerm()
    model = JointForwardModel(torch.nn.Linear(1, 1), {'stokes': term}, {'stokes': 1.})
    module = JointInversionModule(model, JointTrainingConfig(
        max_steps=20, learning_rate=.01, final_learning_rate=.001, log_every_n_steps=10), {})
    module._trainer = SimpleNamespace(global_step=1)
    logs, reductions = [], []
    monkeypatch.setattr(module, 'log', lambda *args, **kw: logs.append(args))
    monkeypatch.setattr(training, 'reduce_training_metrics', lambda metrics, *args, **kw: (reductions.append(1), metrics)[1])
    batch = {'stokes': {'loss': torch.tensor(2., requires_grad=True)}}
    module.training_step(batch, 1)
    assert term.calls == ['score'] and not logs and not reductions
    module._trainer.global_step = 9
    module.training_step(batch, 9)
    assert term.calls == ['score', 'diagnostics'] and logs and reductions == [1]
    model.evaluate_batches(batch)
    assert term.calls[-1] == 'diagnostics'  # Public evaluation stays complete.
