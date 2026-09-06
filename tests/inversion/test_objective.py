import pytest
import torch

from prom3theus.inversion.objective import StokesObjective


SIGMAS = {"I": 1.0, "Q": 2.0, "U": 4.0, "V": 8.0}


def test_stokes_objective_standardizes_residuals_and_applies_huber_penalty():
    prediction = torch.zeros(2, 4, 3)
    target = torch.ones_like(prediction)
    result = StokesObjective(stokes_sigmas=SIGMAS, huber_delta=0.5)(prediction, target)
    standardized = torch.tensor([1.0, 0.5, 0.25, 0.125]).reshape(1, 4, 1)
    expected = torch.where(
        standardized <= 0.5,
        0.5 * standardized.square(),
        0.5 * (standardized - 0.25),
    ).expand_as(result)
    torch.testing.assert_close(result, expected)


def test_stokes_objective_configuration_is_canonical():
    objective = StokesObjective(stokes_sigmas=SIGMAS, huber_delta=1.5)
    assert objective.configuration() == {
        "type": "huber",
        "stokes_sigmas": SIGMAS,
        "huber_delta": 1.5,
    }


def test_stokes_objective_rejects_non_stokes_shape():
    with pytest.raises(ValueError, match="four components"):
        StokesObjective(stokes_sigmas=SIGMAS)(
            torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)
        )


@pytest.mark.parametrize(
    "sigmas",
    [
        {"I": 1.0, "Q": 1.0, "U": 1.0},
        {"I": 1.0, "Q": 1.0, "U": 1.0, "V": 0.0},
    ],
)
def test_stokes_objective_rejects_invalid_sigmas(sigmas):
    with pytest.raises((TypeError, ValueError)):
        StokesObjective(stokes_sigmas=sigmas)
