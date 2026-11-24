import pytest
import torch
from torch.utils.data import Subset

from prom3theus.application.batches import _collect_loader, deterministic_validation_batch
from prom3theus.observations.data import ObservationDataModule


def test_validation_does_not_read_past_requested_limit():
    def batches():
        yield {"data": torch.arange(8).reshape(4, 2)}
        raise AssertionError("Read another batch after the requested samples were collected")

    result = _collect_loader(batches(), 3)
    assert result["data"].shape == (3, 2)


@pytest.mark.parametrize("stride", [1, 3])
def test_bounded_stokes_validation_covers_the_lattice(stride):
    data = ObservationDataModule(num_workers=0, validation_batch_size=3)
    data.dataset = [{"pixel": i} for i in range(60)]
    lattice = range(0, len(data.dataset), stride)
    data.validation_dataset = Subset(data.dataset, lattice) if stride > 1 else data.dataset
    count = 7
    sample = deterministic_validation_batch(data, max_samples=count)["pixel"].tolist()
    for i, pixel in enumerate(sample):
        assert pixel in lattice[i * len(lattice) // count : (i + 1) * len(lattice) // count]
    assert deterministic_validation_batch(data, max_samples=count)["pixel"].tolist() == sample
    assert deterministic_validation_batch(data, max_samples=60)["pixel"].tolist() == list(lattice)
