import torch

from prom3theus.application.batches import _collect_loader


def test_validation_does_not_read_past_requested_limit():
    def batches():
        yield {"data": torch.arange(8).reshape(4, 2)}
        raise AssertionError("Read another batch after the requested samples were collected")

    result = _collect_loader(batches(), 3)
    assert result["data"].shape == (3, 2)
