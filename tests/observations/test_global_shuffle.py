"""Regression coverage for global mixing and tensor-slice-only training."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset

from prom3theus.observations.bulk import BulkReader, concatenate
from prom3theus.observations.dataset import ObservationPixelDataset
from prom3theus.observations.loading import buffered_dataloader
from prom3theus.observations.tensor_loader import PersistentTensorLoader
from prom3theus.training.streams import StreamBatchScheduler


def dataset(time, count=4096):
    ids = torch.arange(count).reshape(1, count) + time * 10000
    return ObservationPixelDataset(
        SimpleNamespace(
            spatial_shape=(1, count),
            valid_mask=torch.ones(1, count, dtype=torch.bool),
            coordinates=torch.stack(
                (
                    ids.float(),
                    ids.float() * 2,
                    torch.full_like(ids, float(time), dtype=torch.float32),
                ),
                -1,
            ),
            ray_direction=torch.ones(1, count, 3),
            stokes_basis=torch.eye(3).expand(1, count, 3, 3),
            stokes=ids[..., None, None].expand(1, count, 4, 6).float(),
            auxiliary={
                "instrument_response:spectral_weights": ids[..., None, None]
                .expand(1, count, 6, 2)
                .float()
            },
        ),
        include_pixel_index=True,
    )


def assert_aligned(batch):
    ids = batch["coordinates"][:, 0]
    torch.testing.assert_close(batch["stokes"][:, 0, 0], ids)
    torch.testing.assert_close(
        batch["instrument_response"]["spectral_weights"][:, 0, 0], ids
    )
    torch.testing.assert_close(batch["pixel_index"][:, 1], ids.long() % 10000)


def test_first_batches_mix_all_times_and_positions_without_pixel_fetch(monkeypatch):
    sources = [dataset(time) for time in range(3)]
    for source in sources:
        monkeypatch.setattr(
            source, "pixels_at", lambda *args: pytest.fail("individual pixel lookup")
        )
    calls = []
    original = BulkReader.read

    def read(self, source, start, stop):
        if source in sources:
            calls.append((sources.index(source), start, stop))
        return original(self, source, start, stop)

    monkeypatch.setattr(BulkReader, "read", read)
    loader = buffered_dataloader(ConcatDataset(sources), batch_size=256, shuffle=True)
    assert isinstance(loader, PersistentTensorLoader)
    before = torch.get_rng_state()
    try:
        batches = list(loader)
        for batch in batches:
            assert batch["coordinates"][:, 2].unique().tolist() == [0.0, 1.0, 2.0]
            assert_aligned(batch)
        actual = concatenate(batches)["coordinates"][:, 0].sort().values
        expected = (
            torch.cat(
                [source.raster.coordinates[..., 0].flatten() for source in sources]
            )
            .sort()
            .values
        )
        torch.testing.assert_close(actual, expected)
        for index, source in enumerate(sources):
            spans = [(start, stop) for pool, start, stop in calls if pool == index]
            assert spans[0][0] == 0 and spans[-1][1] == len(source)
            assert all(a[1] == b[0] for a, b in pairwise(spans))
        assert torch.equal(before, torch.get_rng_state())
    finally:
        loader.close()


def test_training_only_slices_prepared_tensors_and_randomizes_batch_order(monkeypatch):
    loader = PersistentTensorLoader(dataset(0, 205), 7)
    loader.prepare()
    assert loader.dataset is None
    assert loader.channels == ()
    assert all(
        not isinstance(value, torch.Tensor)
        for value in vars(loader._stores[0]).values()
    )
    monkeypatch.setattr(
        BulkReader, "read", lambda *args: pytest.fail("disk read after preparation")
    )
    monkeypatch.setattr(
        torch.Tensor,
        "index_select",
        lambda *args: pytest.fail("pixel gather during training"),
    )
    try:
        first = list(loader)
        second = list(loader)
        for batch in first + second:
            assert_aligned(batch)
        a = [tuple(batch["coordinates"][:, 0].tolist()) for batch in first]
        b = [tuple(batch["coordinates"][:, 0].tolist()) for batch in second]
        assert a != b and set(a) == set(b)
        assert sum(map(len, a)) == 205  # The partial final chunk is not discarded.
        assert sum(len(chunk) == 2 for chunk in a) == 1
    finally:
        loader.close()


def test_resume_preserves_next_batch_across_budgets_and_prefetch():
    source = ConcatDataset([dataset(0, 64), dataset(1, 64)])
    first = PersistentTensorLoader(source, 7, seed=31, stream_id="hmi")
    scheduler = StreamBatchScheduler({"hmi": first}, "hmi", explicit_commit=True)
    iterator = iter(scheduler)
    for _ in range(5):
        next(iterator)
        scheduler.commit()
    state = scheduler.state_dict()
    expected = next(iterator)["hmi"]  # Lookahead must not move the saved cursor.
    restored = PersistentTensorLoader(
        source,
        7,
        seed=31,
        stream_id="hmi",
        block_bytes=20000,
        cache_bytes=40000,
        prefetch_batches=1,
    )
    resumed = StreamBatchScheduler({"hmi": restored}, "hmi", explicit_commit=True)
    try:
        resumed.load_state_dict(state)
        actual = next(iter(resumed))["hmi"]
        torch.testing.assert_close(actual["stokes"], expected["stokes"])
        torch.testing.assert_close(actual["coordinates"], expected["coordinates"])
        changed = PersistentTensorLoader(source, 7, seed=32, stream_id="hmi")
        with pytest.raises(ValueError, match="schedule differs"):
            StreamBatchScheduler({"hmi": changed}, "hmi").load_state_dict(state)
    finally:
        scheduler.close()
        resumed.close()
        first.close()
        restored.close()


def test_channels_always_present_and_each_pool_exhausts_before_repeating():
    pools = [dataset(0, 2), dataset(1, 6), dataset(2, 15)]
    loader = PersistentTensorLoader(ConcatDataset(pools), 7, channels=pools)
    try:
        batches = list(loader.iter_from(0, 40))
        for batch in batches:
            assert len(batch["stokes"]) <= 7
            assert batch["coordinates"][:, 2].unique().tolist() == [0.0, 1.0, 2.0]
            assert_aligned(batch)
        values = concatenate(batches)
        for time, pool in enumerate(pools):
            selected = values["coordinates"][values["coordinates"][:, 2] == time, 0]
            for start in range(0, len(selected) - len(pool) + 1, len(pool)):
                torch.testing.assert_close(
                    selected[start : start + len(pool)].sort().values,
                    torch.arange(len(pool)).float() + time * 10000,
                )
    finally:
        loader.close()


def test_sequential_checkpoint_requires_explicit_order_migration():
    from prom3theus.observations.bulk import SequentialBulkLoader

    source = dataset(0, 20)
    old = StreamBatchScheduler(
        {"hmi": SequentialBulkLoader(source, 7)}, "hmi"
    ).state_dict()
    new = StreamBatchScheduler({"hmi": PersistentTensorLoader(source, 7)}, "hmi")
    with pytest.raises(ValueError, match="schedule differs"):
        new.load_state_dict(old)
    new.load_state_dict(old, migrate=True)
    assert new.order_migrated and new.completed == 0


def test_persistent_training_pool_reused_without_preparation_reads(
    tmp_path, monkeypatch
):
    source = dataset(0, 101)
    first = PersistentTensorLoader(source, 7, seed=19)
    first.tensor_cache_directory = tmp_path
    first.tensor_cache_identity = {"source": "fixture-v1"}
    first.prepare()
    expected = first._stores[0].read(0, 101)
    files_before = {
        str(path.relative_to(tmp_path)): (
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    first.close()
    monkeypatch.setattr(
        BulkReader, "read", lambda *args: pytest.fail("rebuilding cached tensors")
    )
    from prom3theus.observations.tensor_dataset import TensorDiskDataset

    monkeypatch.setattr(
        TensorDiskDataset,
        "from_batches",
        lambda *args, **kwargs: pytest.fail("rewrote existing tensor files"),
    )
    second = PersistentTensorLoader(source, 13, seed=19)
    second.tensor_cache_directory = tmp_path
    second.tensor_cache_identity = {"source": "fixture-v1"}
    try:
        second.prepare()
        torch.testing.assert_close(second._stores[0].read(0, 101), expected)
        assert second.dataset is None
        list(second.iter_from(0, 2))
        files_after = {
            str(path.relative_to(tmp_path)): (
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in tmp_path.rglob("*")
            if path.is_file()
        }
        assert files_after == files_before
    finally:
        second.close()


def test_resume_with_changed_batch_size_restarts_chunks_but_preserves_data_checks():
    source = dataset(0, 101)
    old_loader = PersistentTensorLoader(source, 7, seed=19)
    new_loader = PersistentTensorLoader(source, 13, seed=19)
    old = StreamBatchScheduler({"hmi": old_loader}, "hmi")
    new = StreamBatchScheduler({"hmi": new_loader}, "hmi")
    try:
        state = old.state_dict()
        state["completed"] = 10
        new.load_state_dict(state)
        assert new.order_migrated and new.completed == 0
        new_loader.seed = 20
        with pytest.raises(ValueError, match="schedule differs"):
            new.load_state_dict(state)
    finally:
        old.close()
        new.close()
        old_loader.close()
        new_loader.close()


def test_exactly_one_sample_shuffle_and_none_when_reusing_files(tmp_path, monkeypatch):
    calls = []
    original = torch.randperm

    def record(size, *args, **kwargs):
        calls.append(size)
        return original(size, *args, **kwargs)

    monkeypatch.setattr(torch, "randperm", record)
    source = dataset(0, 101)
    for run in range(2):
        loader = PersistentTensorLoader(source, 7, seed=19)
        loader.tensor_cache_directory = tmp_path
        loader.tensor_cache_identity = {"source": "same"}
        try:
            list(loader)
            list(loader)
        finally:
            loader.close()
        # One 101-sample shuffle at first preparation; only 15-chunk
        # permutations during training and on reload.
        assert calls.count(101) == 1
        assert calls.count(15) == (run + 1) * 2
        assert set(calls) == {101, 15}


def test_training_preparation_finishes_before_batch_iteration(tmp_path, monkeypatch):
    from prom3theus.application.data import _prepare_training_tensors

    loader = PersistentTensorLoader(dataset(0, 101), 7)
    loader.tensor_cache_directory = tmp_path
    loader.tensor_cache_identity = {"source": "fixture"}
    try:
        _prepare_training_tensors({"hmi": loader})
        assert loader.dataset is None
        assert loader._stores is not None
        assert list(tmp_path.rglob("manifest.json"))
        monkeypatch.setattr(BulkReader, "read", lambda *args: pytest.fail("preparation during training"))
        assert len(list(loader)) == 15
    finally:
        loader.close()
