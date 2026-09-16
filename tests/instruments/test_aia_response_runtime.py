from __future__ import annotations

import json

import pytest
import torch

from prom3theus.instruments.aia_euv import AIAResponseTable
from prom3theus.resources import (
    AIA_TEMPERATURE_RESPONSE_RESOURCE,
    resolve_resource_reference,
)


def test_aia_response_reproduces_nodes_and_selected_channel_order():
    operator = AIAResponseTable(channels=(211, "A171"))
    table = json.loads(
        resolve_resource_reference(AIA_TEMPERATURE_RESPONSE_RESOURCE).read_text(
            encoding="utf-8"
        )
    )
    node = 43
    temperature = torch.tensor(10.0 ** table["axes"]["log10_temperature_k"][node], dtype=torch.float64)

    actual = operator(temperature)

    assert operator.channels == ("211", "171")
    assert actual.shape == (2,)
    assert actual.tolist() == pytest.approx(
        [table["response"][2][node], table["response"][0][node]],
        rel=1.0e-13,
    )
    assert operator.state_dict() == {}


def test_aia_response_zero_continuation_does_not_clip_temperature():
    operator = AIAResponseTable(channels=(171,))
    temperature = torch.tensor(
        [1.0e3, 10.0**5.925, 1.0e10],
        dtype=torch.float64,
        requires_grad=True,
    )

    response = operator(temperature)
    response.sum().backward()

    assert response.shape == (3, 1)
    assert response[0, 0].item() == 0.0
    assert response[1, 0].item() > 0.0
    assert response[2, 0].item() == 0.0
    assert temperature.detach().tolist() == [1.0e3, 10.0**5.925, 1.0e10]
    assert temperature.grad is not None
    assert torch.isfinite(temperature.grad).all()
    assert temperature.grad[0].item() == 0.0
    assert temperature.grad[1].item() != 0.0
    assert temperature.grad[2].item() == 0.0


def test_aia_response_tapers_continuously_to_zero_at_both_support_edges():
    operator = AIAResponseTable(channels=(171,)).to(torch.float64)
    log_temperature = torch.tensor(
        [4.0, 4.025, 4.05, 8.95, 8.975, 9.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    response = operator(torch.pow(10.0, log_temperature))[:, 0]
    gradients = torch.autograd.grad(response.sum(), log_temperature)[0]

    assert response[0].item() == 0.0
    assert response[-1].item() == 0.0
    assert response[1].item() > 0.0
    assert response[-2].item() > 0.0
    assert response[2].item() > response[1].item()
    assert torch.isfinite(gradients).all()
    torch.testing.assert_close(gradients[[0, -1]], torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize(
    "temperature",
    [torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(float("inf"))],
)
def test_aia_response_rejects_nonphysical_temperature(temperature):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        AIAResponseTable()(temperature)


def test_aia_response_rejects_missing_or_duplicate_channels():
    with pytest.raises(KeyError, match="no channels"):
        AIAResponseTable(channels=(304,))
    with pytest.raises(ValueError, match="unique"):
        AIAResponseTable(channels=(171, "A171"))
