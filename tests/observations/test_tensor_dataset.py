import gc
import weakref

import pytest
import torch

from prom3theus.observations.tensor_dataset import TensorDiskDataset


def test_joint_shuffle_disk_reload_and_batch_resize(tmp_path):
    ids = torch.arange(103)
    ref = weakref.ref(ids)
    original = ids.clone()
    dataset = TensorDiskDataset.create(
        {"id": ids, "nested": {"value": ids[:, None] * 2}},
        tmp_path / "tensors",
        16,
        seed=3,
    )
    del ids
    gc.collect()
    assert ref() is None
    assert not any(isinstance(v, torch.Tensor) for v in vars(dataset).values())
    batches = list(dataset)
    order = torch.cat([b["id"] for b in batches])
    assert not torch.equal(order, original)
    torch.testing.assert_close(order.sort().values, original)
    assert len(batches[-1]["id"]) == 7
    for batch in batches:
        torch.testing.assert_close(batch["nested"]["value"][:, 0], batch["id"] * 2)
    reopened = TensorDiskDataset(tmp_path / "tensors", 7)
    torch.testing.assert_close(torch.cat([b["id"] for b in reopened]), order)
    # Editing a returned batch cannot change the persistent data.
    dataset[0]["id"].zero_()
    torch.testing.assert_close(reopened.read(0, 103)["id"], order)


def test_invalid_shapes_never_publish_partial_store(tmp_path):
    with pytest.raises(ValueError, match="disagree"):
        TensorDiskDataset.create(
            {"a": torch.ones(3), "b": torch.ones(2)}, tmp_path / "bad", 2
        )
    assert not (tmp_path / "bad").exists()
    assert list(tmp_path.iterdir()) == []


def test_serialized_dataset_contains_only_metadata_and_initializes_reusable_lazy_maps(
    tmp_path, monkeypatch
):
    import io
    import pickle

    import numpy as np

    dataset = TensorDiskDataset.create(
        {"a": torch.arange(1000), "nested": {"b": torch.ones(1000, 3)}},
        tmp_path / "serialized",
        13,
    )

    class MetadataOnlyPickler(pickle.Pickler):
        def persistent_id(self, value):
            assert not isinstance(value, (torch.Tensor, np.ndarray))

    buffer = io.BytesIO()
    MetadataOnlyPickler(buffer).dump(dataset)
    assert len(buffer.getvalue()) < 2048
    # A fresh process/session initializes read-only mappings, without copying data.
    from prom3theus.observations.arrays import process_reader
    from unittest.mock import patch

    process_reader().close()
    with patch.object(np, "load", wraps=np.load) as opened:
        restored = pickle.loads(buffer.getvalue())
        assert opened.call_count == 2
        assert all(
            call.kwargs == {"mmap_mode": "r", "allow_pickle": False}
            for call in opened.call_args_list
        )
        reopened = TensorDiskDataset(tmp_path / "serialized", 7)
        for _ in range(3):
            restored[0]
            reopened.read(0, 7)
        assert opened.call_count == 2
    torch.testing.assert_close(restored[0], dataset[0])
    assert reopened.batch_size == 7
    assert set(vars(restored)) == {
        "directory",
        "sample_count",
        "batch_size",
        "files",
        "arrays",
    }


@pytest.mark.parametrize("batch_index, expected_count", [(0, 64), (70, 64), (156, 17)])
def test_batch_reads_readonly_maps_and_copies_only_the_selected_slice(
    tmp_path, monkeypatch, batch_index, expected_count
):
    import numpy as np

    dataset = TensorDiskDataset.create(
        {"id": torch.arange(10001), "nested": {"values": torch.ones(10001, 4)}},
        tmp_path / "chunks",
        64,
    )
    from prom3theus.observations.arrays import ReadSession

    original = np.array
    copied_shapes = []

    def bounded_copy(value, *args, **kwargs):
        assert not value.flags.writeable
        assert value.shape[0] == expected_count
        assert kwargs["copy"] is True
        copied_shapes.append(value.shape)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "array", bounded_copy)
    with ReadSession() as reader:
        start = batch_index * dataset.batch_size
        stop = min(start + dataset.batch_size, dataset.sample_count)
        batch = dataset.read_into(start, stop, reader=reader)
        assert copied_shapes == [(expected_count,), (expected_count, 4)]
        assert reader.mapping_count == 2
        dataset.read_into(start, stop, reader=reader)
        assert reader.mapping_count == 2
    assert batch["id"].shape == (expected_count,)
    assert batch["id"].untyped_storage().nbytes() == expected_count * 8
    assert (
        batch["nested"]["values"].untyped_storage().nbytes() == expected_count * 4 * 4
    )


def test_read_closes_mapping_if_batch_copy_fails(tmp_path, monkeypatch):

    dataset = TensorDiskDataset.create({"x": torch.ones(100)}, tmp_path / "failure", 8)
    from prom3theus.observations.arrays import ReadSession

    with ReadSession() as reader:
        monkeypatch.setattr(
            torch,
            "from_numpy",
            lambda *args: (_ for _ in ()).throw(RuntimeError("copy failed")),
        )
        with pytest.raises(RuntimeError, match="copy failed"):
            dataset.read_into(0, 8, reader=reader)
        assert all(entry[2] == 0 for entry in reader._maps.values())
    assert not reader._maps


def test_dataset_initialization_opens_lazy_maps_and_reads_never_reopen(
    tmp_path, monkeypatch
):
    import builtins
    import numpy as np
    from unittest.mock import patch
    from prom3theus.observations.arrays import process_reader, configure_reader

    TensorDiskDataset.create({"x": torch.arange(20)}, tmp_path / "initialization", 4)
    process_reader().close()
    with patch.object(np, "load", wraps=np.load) as opened:
        dataset = TensorDiskDataset(tmp_path / "initialization", 4)
        assert opened.call_count == 1
        assert opened.call_args.kwargs["mmap_mode"] == "r"
        entry = process_reader()._maps[dataset.arrays[0]]
        mapping = entry[1]
        assert isinstance(entry[0], np.memmap)
        configure_reader(workers=3)
        with monkeypatch.context() as context:
            context.setattr(
                builtins,
                "open",
                lambda *a, **k: pytest.fail("opened a file during read"),
            )
            context.setattr(
                np, "load", lambda *a, **k: pytest.fail("remapped during read")
            )
            for index in range(len(dataset)):
                assert len(dataset[index]["x"]) == 4
                assert not mapping.closed
        assert process_reader()._maps[dataset.arrays[0]][1] is mapping
