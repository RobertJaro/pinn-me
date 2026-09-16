import os

import torch
from torch.utils.data import Dataset

from prom3theus.config.schema import DataLoaderConfig
from prom3theus.config.joint_schema import ImageDataLoaderConfig
from prom3theus.observations.loading import buffered_dataloader


class WorkerDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {"index": index, "pid": os.getpid()}


def test_workers_prefetch_and_persist_without_pinned_memory():
    loader = buffered_dataloader(
        WorkerDataset(), batch_size=2, num_workers=1, pin_memory=False,
    )
    assert loader.num_workers == 1
    assert loader.prefetch_factor == 2
    assert loader.persistent_workers
    assert not loader.pin_memory
    first = list(loader)
    second = list(loader)
    assert torch.cat([batch["index"] for batch in first]).tolist() == list(range(12))
    pids = {int(pid) for batch in first for pid in batch["pid"]}
    assert os.getpid() not in pids
    assert pids == {int(pid) for batch in second for pid in batch["pid"]}
    assert all(not batch["index"].is_pinned() for batch in first)


def test_loader_settings_preserve_explicit_worker_and_pinning_choices():
    for cls in (DataLoaderConfig, ImageDataLoaderConfig):
        config = cls(batch_size=2, validation_batch_size=2, workers=0, pin_memory=True)
        assert config.workers == 0
        assert config.pin_memory is True
