import pytest
import torch

from prom3theus.inversion.objective import StokesObjective


class Identity(torch.nn.Module):
    def forward(self, value):
        return value


def test_stokes_objective_is_elementwise_squared_error():
    prediction = torch.zeros(2, 4, 3)
    target = torch.ones_like(prediction)
    result = StokesObjective()(prediction, target, Identity())
    assert torch.equal(result, torch.ones_like(result))


def test_stokes_objective_rejects_non_stokes_shape():
    with pytest.raises(ValueError, match="four components"):
        StokesObjective()(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), Identity())
