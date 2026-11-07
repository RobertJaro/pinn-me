"""Adaptive weights require explicit inclusion, including after resume."""

from types import SimpleNamespace

import pytest
import torch

from prom3theus.config.joint_schema import JointTrainingConfig, LossBalanceConfig
from prom3theus.inversion.data_terms.base import ObservationDataTerm, SharedTermResult
from prom3theus.inversion.joint import JointForwardModel
from prom3theus.training.joint import GradientLossBalancer, JointInversionModule


class ReferenceTerm(ObservationDataTerm):
    observation_kind = "stokes"

    def evaluate_batch(self, batch):
        raise NotImplementedError


@pytest.mark.parametrize("included", [(), ("shared.physics.momentum",), ("streams.aia",)])
def test_only_included_objectives_adapt_with_unlisted_reference(included):
    atmosphere = torch.nn.Linear(1, 1, bias=False)
    model = JointForwardModel(atmosphere, {"stokes": ReferenceTerm()}, {"stokes": 1.0})
    loss = atmosphere.weight.sum()
    evaluation = SimpleNamespace(
        weighted_likelihoods={"aia": 3 * loss, "stokes": loss},
        shared_terms={
            "physics": SharedTermResult(
                5 * loss,
                {"momentum": 2 * loss, "upper_boundary_no_inflow": 3 * loss},
                {},
            ),
        },
    )
    balancer = GradientLossBalancer(
        LossBalanceConfig(enabled=True, start_step=0, included_objectives=included),
        reference_stream="stokes",
    )
    metrics = balancer.update(model, evaluation, 0)
    assert set(model.adaptive_multiplier_state()) == set(included)
    assert model.adaptive_multiplier("streams.stokes") == 1.0
    if included:
        assert model.adaptive_multiplier(included[0]) < 1.0
        assert metrics["loss_balance.reference_gradient_norm"] == 1.0
    else:
        assert metrics == {}


@pytest.mark.parametrize("included", [(), ("shared.physics.momentum", "streams.stokes")])
def test_resume_resets_unlisted_and_reference_multipliers(included):
    model = JointForwardModel(
        torch.nn.Linear(1, 1), {"stokes": ReferenceTerm()}, {"stokes": 1.0},
    )
    module = JointInversionModule(
        model,
        JointTrainingConfig(
            max_steps=1, learning_rate=0.01, final_learning_rate=0.001,
            reference_stream="stokes",
            loss_balance=LossBalanceConfig(enabled=True, included_objectives=included),
        ),
        {"contract": {}},
    )
    saved = {
        "shared.physics.momentum": 10.0,
        "shared.physics.upper_boundary_no_inflow": 80.0,
        "streams.stokes": 5.0,
        "streams.aia": 2.0,
    }
    module.on_load_checkpoint({
        "formal_transfer_format_version": 1,
        "prom3theus_joint": {"contract": {}},
        "loss_balance": {"multipliers": saved, "ema_ratios": dict.fromkeys(saved, 0.2)},
    })
    for key, value in saved.items():
        expected = value if key in included and key != "streams.stokes" else 1.0
        assert model.adaptive_multiplier(key) == expected
    assert set(module.loss_balancer.state_dict()["ema_ratios"]) == (
        set(included) - {"streams.stokes"}
    )
