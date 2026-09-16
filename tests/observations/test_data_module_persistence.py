from types import MappingProxyType

import pytest
import torch

from prom3theus.observations.persistence import (
    JointDataModule,
    load_data_module,
    restore_or_create_data_module,
    save_data_module,
)
from prom3theus.observations.tensor_dataset import TensorDiskDataset
from prom3theus.observations.arrays import ArrayRef


def test_restore_keeps_only_paths_and_reads_on_demand(tmp_path, monkeypatch):
    values = torch.arange(1200).reshape(100, 12)
    module = JointDataModule(
        {"values": values, "same": values}, MappingProxyType({"x": 1})
    )
    path = tmp_path / "data_module.pt"
    save_data_module(module, path)
    from prom3theus.observations import loading

    original = loading.load_array_tensor
    monkeypatch.setattr(
        loading, "load_array_tensor", lambda *a, **k: pytest.fail("eager payload read")
    )
    restored = load_data_module(path)
    assert isinstance(restored.streams["values"], ArrayRef)
    assert restored.streams["values"] == restored.streams["same"]
    assert not any(
        isinstance(v, torch.Tensor) for v in vars(restored.streams["values"]).values()
    )
    assert restored.scene["x"] == 1
    monkeypatch.setattr(loading, "load_array_tensor", original)
    torch.testing.assert_close(restored.streams["values"][2:5], values[2:5])
    assert len(list(tmp_path.glob("data-arrays-*/*.npy"))) == 1


def test_rank_zero_reuses_and_worker_never_builds(tmp_path, monkeypatch):
    path = tmp_path / "data_module.pt"
    calls = []

    def build():
        calls.append(1)
        return JointDataModule({"value": torch.ones(4)}, None)

    monkeypatch.setenv("RANK", "0")
    restore_or_create_data_module(path, build)
    restore_or_create_data_module(path, build)
    monkeypatch.setenv("RANK", "1")
    restored = restore_or_create_data_module(path, build)
    assert calls == [1]
    torch.testing.assert_close(restored.streams["value"][:], torch.ones(4))


def test_worker_timeout_does_not_prepare(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    with pytest.raises(TimeoutError):
        restore_or_create_data_module(
            tmp_path / "missing.pt", lambda: pytest.fail("worker prepared"), timeout=0
        )


def test_prepared_training_loader_restores_without_preparation(tmp_path):
    from prom3theus.observations.tensor_loader import PersistentTensorLoader

    store = TensorDiskDataset.create({"x": torch.arange(10)}, tmp_path / "pool", 3)
    # A prepared loader holds no source dataset or active producer.
    loader = PersistentTensorLoader.__new__(PersistentTensorLoader)
    loader.__dict__.update(
        _stores=[store],
        _counts=[10],
        batch_size=3,
        seed=0,
        stream_id="x",
        _source_contract={},
        dataset=None,
        channels=(),
        _active=[],
        _iteration_start=0,
        pin_memory=False,
        max_batch_bytes=10000,
        _temporary_directory=None,
    )
    path = tmp_path / "data_module.pt"
    save_data_module(JointDataModule({}, None, {"x": loader}), path)
    restored = load_data_module(path).train_dataloader()["x"]
    values = torch.cat([b["x"] for b in restored._batches(0, 4)])
    torch.testing.assert_close(values.sort().values, torch.arange(10))


def test_native_image_dataset_roundtrip_preserves_bulk_and_sparse_reads(tmp_path):
    from types import SimpleNamespace
    from prom3theus.observations.image_dataset import ImagePixelDataset
    from prom3theus.observations.bulk import BulkReader

    dataset = ImagePixelDataset(
        SimpleNamespace(
            intensity=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            uncertainty=torch.ones(3, 4),
            ray_direction=torch.ones(3, 4, 3),
            surface_position_m=torch.ones(3, 4, 3),
            valid_mask=torch.ones(3, 4, dtype=torch.bool),
            spatial_shape=(3, 4),
            absolute_tai_seconds=10.0,
            channel_angstrom=171,
        ),
        image_index=0,
        channel_index=0,
        intensity_scale=2.0,
    )
    expected = BulkReader().read(dataset, 0, len(dataset))
    path = tmp_path / "data_module.pt"
    save_data_module(JointDataModule({"image": dataset}, None), path)
    restored = load_data_module(path).streams["image"]
    assert isinstance(restored.raster.intensity, ArrayRef)
    torch.testing.assert_close(BulkReader().read(restored, 0, len(restored)), expected)
    torch.testing.assert_close(restored[3], dataset[3])


def test_failed_save_keeps_previous_module(tmp_path):
    path = tmp_path / "data_module.pt"
    save_data_module(JointDataModule({}, "original"), path)
    with pytest.raises((AttributeError, TypeError)):
        save_data_module(JointDataModule({}, lambda: None), path)
    assert load_data_module(path).scene == "original"
    assert len(list(tmp_path.glob("data-arrays-*"))) == 1


def test_unchanged_canonical_arrays_are_referenced_and_run_root_can_move(tmp_path):
    import shutil
    import numpy as np
    from prom3theus.observations.loading import load_array_tensor

    root = tmp_path / "run"
    root.mkdir()
    np.save(root / "canonical.npy", np.arange(30, dtype=np.float32))
    values = load_array_tensor(root / "canonical.npy")
    save_data_module(JointDataModule({"values": values}, None), root / "module.pt")
    assert not list(root.glob("data-arrays-*/*.npy"))
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    restored = load_data_module(moved / "module.pt")
    assert restored.streams["values"].path == str(moved / "canonical.npy")
    torch.testing.assert_close(restored.streams["values"][:], torch.arange(30).float())


def test_changed_or_missing_array_fails_without_rebuilding(tmp_path):
    path = tmp_path / "module.pt"
    save_data_module(JointDataModule({"values": torch.arange(8)}, None), path)
    ref = load_data_module(path).streams["values"]
    from pathlib import Path

    Path(ref.path).write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Prepared array changed"):
        restore_or_create_data_module(path, lambda: pytest.fail("unexpected rebuild"))
    Path(ref.path).unlink()
    with pytest.raises(FileNotFoundError):
        load_data_module(path)


@pytest.mark.parametrize("temporary_pools", [False, True])
def test_finalize_adds_training_pools_without_rebuilding_native_data(
    tmp_path, temporary_pools
):
    from types import SimpleNamespace
    from prom3theus.observations.image_dataset import ImagePixelDataset
    from prom3theus.observations.tensor_loader import PersistentTensorLoader

    dataset = ImagePixelDataset(
        SimpleNamespace(
            intensity=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            uncertainty=None,
            ray_direction=torch.ones(3, 4, 3),
            surface_position_m=torch.ones(3, 4, 3),
            valid_mask=torch.ones(3, 4, dtype=torch.bool),
            spatial_shape=(3, 4),
            absolute_tai_seconds=10.0,
            channel_angstrom=171,
        ),
        image_index=0,
        channel_index=0,
    )
    path = tmp_path / "module.pt"
    save_data_module(JointDataModule({"dataset": dataset}, None), path)

    def finalize(module):
        if module.train_loaders is not None:
            return False
        loader = PersistentTensorLoader(module.streams["dataset"], 4)
        if not temporary_pools:
            loader.tensor_cache_directory = tmp_path / "training"
            loader.tensor_cache_identity = {"source": "fixture"}
        loader.prepare()
        module.train_loaders = {"a": loader}
        return True

    def no_build():
        pytest.fail("reingested sources")

    module = restore_or_create_data_module(path, no_build, finalize=finalize)
    generation = module.generation
    assert isinstance(module.streams["dataset"].raster.intensity, ArrayRef)
    first = module.train_dataloader(batch_sizes={"a": 5}, prefetch_batches=1)["a"]
    batches = list(first)
    first.close()
    assert sorted(torch.cat([x["intensity"] for x in batches]).tolist()) == list(
        range(12)
    )
    assert max(len(x["intensity"]) for x in batches) == 5
    # Close never invalidates the prepared descriptor.
    torch.testing.assert_close(list(first.iter_from(0, len(first))), batches)
    assert (
        restore_or_create_data_module(path, no_build, finalize=finalize).generation
        == generation
    )


def test_concurrent_rank_zero_calls_publish_only_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep

    calls = []

    def build():
        calls.append(1)
        sleep(0.05)
        return JointDataModule({}, None)

    def load(_):
        return restore_or_create_data_module(tmp_path / "module.pt", build).generation

    with ThreadPoolExecutor(2) as executor:
        generations = list(executor.map(load, range(2)))
    assert len(set(generations)) == 1
    assert len(calls) == 1


def test_worker_first_rebuild_waits_for_new_generation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import current_thread
    from prom3theus.observations import persistence

    path = tmp_path / "module.pt"
    save_data_module(JointDataModule({}, "old"), path)
    monkeypatch.setenv("PROM3THEUS_RUN_ID", "unique-rebuild-test")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        persistence,
        "global_rank",
        lambda: 0 if current_thread().name == "MainThread" else 1,
    )
    with ThreadPoolExecutor(1) as executor:
        worker = executor.submit(
            restore_or_create_data_module,
            path,
            lambda: pytest.fail("worker built"),
            rebuild=True,
            timeout=10,
        )
        rank_zero = restore_or_create_data_module(
            path, lambda: JointDataModule({}, "new"), rebuild=True
        )
        result = worker.result(timeout=10)
    assert result.scene == "new"
    assert result.generation == rank_zero.generation


def test_scalar_and_transposed_payload_roundtrip(tmp_path):
    path = tmp_path / "module.pt"
    scalar = torch.tensor(3.0)
    transposed = torch.arange(12).reshape(3, 4).T
    save_data_module(
        JointDataModule({"scalar": scalar, "matrix": transposed}, None), path
    )
    module = load_data_module(path)
    torch.testing.assert_close(module.streams["scalar"].read(), scalar)
    torch.testing.assert_close(module.streams["matrix"].read(), transposed)


def test_transient_provider_batches_do_not_accumulate_payloads(tmp_path):
    import gc
    import weakref
    from prom3theus.observations.snapshot import SnapshotWriter, materialize

    writer = SnapshotWriter(tmp_path)
    references, encoded = [], []
    for index in range(5):
        batch = torch.full((100, 4), float(index))
        references.append(weakref.ref(batch))
        encoded.append(writer.encode_batch({"values": batch}))
    del batch
    gc.collect()
    assert all(ref() is None for ref in references)
    for index, batch in enumerate(encoded):
        torch.testing.assert_close(
            materialize(batch)["values"], torch.full((100, 4), float(index))
        )
