from copy import deepcopy

import pytest
import torch

from prom3theus.artifacts.checkpoint import _mse_evaluation_context
from prom3theus.inversion.objective import StokesObjective, weighted_stokes_loss


@pytest.mark.parametrize(
    "sigmas", [None, {"i": 0.01, "q": 0.001, "u": 0.001, "v": 0.001}]
)
def test_old_sigma_objective_migrates_without_changing_loss_or_gradient(sigmas):
    weights = {"i": 1.0, "q": 2.0, "u": 3.0, "v": 4.0}
    objective = {"type": "mse", "stokes_sigmas": sigmas, "stokes_weights": weights}
    context = {
        "configuration": {
            "streams": [{"data_term": {"type": "lte_stokes", "objective": objective}}]
        },
        "terms": {
            "hmi": {
                "type": "lte_stokes",
                "options": {
                    "objective_config": {
                        "type": "mse",
                        "stokes_sigmas": None
                        if sigmas is None
                        else {k.upper(): v for k, v in sigmas.items()},
                    },
                    "weight_config": {k.upper(): v for k, v in weights.items()},
                },
            }
        },
    }
    original = deepcopy(context)
    converted = _mse_evaluation_context(context)
    assert context == original
    assert _mse_evaluation_context(converted) == converted
    options = converted["terms"]["hmi"]["options"]
    assert "stokes_sigmas" not in options["objective_config"]
    prediction = torch.randn(3, 4, 5, requires_grad=True)
    target = torch.randn_like(prediction)
    direct = torch.tensor(list(options["weight_config"].values()))
    _, actual = weighted_stokes_loss(
        StokesObjective(**options["objective_config"]),
        prediction,
        target,
        wavelength_weights=torch.ones(5),
        stokes_weights=direct,
    )
    old_weights = torch.tensor(list(weights.values()))
    old_sigmas = (
        torch.ones(4) if sigmas is None else torch.tensor(list(sigmas.values()))
    )
    expected = (
        ((prediction - target) / old_sigmas[None, :, None]).square().mean((0, 2))
        * old_weights
        / old_weights.sum()
    ).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, prediction, retain_graph=True)[0],
        torch.autograd.grad(expected, prediction)[0],
    )
