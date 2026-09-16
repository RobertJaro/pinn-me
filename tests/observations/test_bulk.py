from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset, Subset

from prom3theus.observations.image_dataset import (
    ImagePixelDataset,
    collate_image_samples,
)
from prom3theus.observations.bulk import BulkReader, SequentialBulkLoader, concatenate


def image(channel=0, shape=(7, 5), mask=None):
    return ImagePixelDataset(
        SimpleNamespace(
            intensity=torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(
                shape
            ),
            uncertainty=torch.full(shape, 2.0),
            ray_direction=torch.ones(*shape, 3),
            surface_position_m=torch.ones(*shape, 3) * 2,
            valid_mask=torch.ones(shape, dtype=torch.bool) if mask is None else mask,
            spatial_shape=shape,
            absolute_tai_seconds=10.0 + channel,
            channel_angstrom=171 + channel,
        ),
        image_index=channel,
        channel_index=channel,
        intensity_scale=2,
    )


def assert_batch(left, right):
    assert set(left) == set(right)
    for key in left:
        if isinstance(left[key], dict):
            assert_batch(left[key], right[key])
        else:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_bulk_matches_samples_without_pixel_fetch_or_permutation(monkeypatch):
    mask = torch.ones(7, 5, dtype=torch.bool)
    mask[1:3] = False
    mask[::2, 2] = False
    dataset = image(mask=mask)
    expected = collate_image_samples([dataset[i] for i in range(len(dataset))])
    monkeypatch.setattr(
        ImagePixelDataset, "__getitem__", lambda *a: pytest.fail("per-pixel read")
    )
    monkeypatch.setattr(torch, "randperm", lambda *a, **k: pytest.fail("permutation"))
    loader = SequentialBulkLoader(dataset, 6, block_bytes=400, cache_bytes=800)
    first, second = list(loader), list(loader)
    assert [len(b["intensity"]) for b in first] == [6, 6, 6, 4]
    assert_batch(concatenate(first), expected)
    for a, b in zip(first, second):
        assert_batch(a, b)
    assert dataset._pixel_indices is None


def test_resident_blocks_reused_and_budget_enforced():
    dataset = image(shape=(20, 5))
    reader = BulkReader(block_bytes=800, cache_bytes=1600)
    result = [reader.read(dataset, i, i + 1) for i in range(len(dataset))]
    assert reader.read_count <= len(dataset) // 5
    assert reader.resident_bytes <= 1600
    assert_batch(
        concatenate(result),
        collate_image_samples([dataset[i] for i in range(len(dataset))]),
    )
    with pytest.raises(ValueError, match="row exceeds"):
        BulkReader(block_bytes=1, cache_bytes=1).read(dataset, 0, 1)


def test_sparse_diagnostics_read_slabs_and_keep_native_ids():
    dataset = image()
    indices = [0, 4, 12, 20, 34]
    loader = SequentialBulkLoader(
        Subset(dataset, indices), 2, block_bytes=500, cache_bytes=1000
    )
    assert_batch(
        concatenate(list(loader)), collate_image_samples([dataset[i] for i in indices])
    )
    with pytest.raises(ValueError, match="increasing"):
        list(SequentialBulkLoader(Subset(dataset, [3, 1]), 2))


def test_bulk_owns_payload_and_does_not_modify_maps(tmp_path):
    import numpy as np

    path = tmp_path / "intensity.npy"
    np.save(path, np.arange(35, dtype=np.float32).reshape(7, 5))
    dataset = image()
    dataset.raster.intensity = torch.from_numpy(np.load(path, mmap_mode="c"))
    reader = BulkReader()
    batch = reader.read(dataset, 0, 5)
    dataset.raster.intensity[:] = -999
    assert batch["intensity"].tolist() == [0, 0.5, 1, 1.5, 2]
    assert np.load(path)[0].tolist() == [0, 1, 2, 3, 4]



def test_reader_failure_reaches_consumer_and_closes():
    loader = SequentialBulkLoader(image(), 10, max_batch_bytes=1)
    with pytest.raises(ValueError, match="max_batch_bytes"):
        list(loader)
    assert not any(item.thread.is_alive() for item in loader._active)


def test_validation_in_slabs_preserves_invalid_geometry_checks():
    from prom3theus.observations.catalog import validated_raster
    from prom3theus.observations.image_contracts import ImageObservationRaster

    shape = (5, 3)
    radius = 6.957e8
    ray = torch.zeros(*shape, 3)
    ray[..., 2] = -1
    position = -ray * radius
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[-1, 0] = True
    values = dict(
        intensity=torch.ones(shape),
        ray_direction=ray,
        surface_position_m=position,
        valid_mask=mask,
        absolute_tai_seconds=1.0,
        channel_angstrom=171,
        exposure_group="x",
        metadata={"ray_geometry": {"solar_radius_m": radius}},
    )
    result = validated_raster(ImageObservationRaster, budget_bytes=100, **values)
    assert result.intensity is values["intensity"]
    ray[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validated_raster(ImageObservationRaster, budget_bytes=100, **values)


def test_auxiliary_partial_block_survives_reference_epochs():
    from prom3theus.training.streams import StreamBatchScheduler

    loader = SequentialBulkLoader(image(shape=(20, 5)), 1)
    scheduler = StreamBatchScheduler({"reference": [0], "image": loader}, "reference")
    try:
        for index in range(12):
            batch = list(scheduler)[0]["image"]
            assert batch["intensity"].item() == index / 2
        assert loader._reader.read_count == 1
    finally:
        scheduler.close()
        loader.close()


def test_bulk_stokes_response_and_chronological_order():
    from prom3theus.observations.dataset import (
        ObservationPixelDataset,
        ObservationResponseCollator,
    )

    def stokes(time):
        source = image(shape=(2, 3)).raster
        del source.absolute_tai_seconds
        source.coordinates = torch.full((2, 3, 3), time)
        source.stokes_basis = torch.eye(3).expand(2, 3, 3, 3)
        source.stokes = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4)
        source.auxiliary = {"instrument_response:weights": torch.ones(2, 3, 4, 2)}
        return ObservationPixelDataset(source, include_surface_position=True)

    early, late = stokes(1.0), stokes(2.0)
    loader = SequentialBulkLoader(ConcatDataset([late, early]), 5)
    batches = list(loader)
    assert [len(b["stokes"]) for b in batches] == [5, 5, 2]
    expected = ObservationResponseCollator()(
        [d[i] for d in (early, late) for i in range(len(d))]
    )
    assert_batch(concatenate(batches), expected)


def test_npy_mapping_is_copy_on_write_and_rejects_noncontiguous_store(tmp_path):
    import numpy as np
    from prom3theus.observations.loading import load_array_tensor

    path = tmp_path / "array.npy"
    np.save(path, np.arange(12, dtype=np.float64).reshape(3, 4))
    tensor = load_array_tensor(path)
    assert tensor.dtype == torch.float64
    tensor.zero_()
    assert np.load(path).sum() == 66
    np.save(path, np.asfortranarray(np.ones((3, 4))))
    with pytest.raises(ValueError, match="C-order"):
        load_array_tensor(path)


def test_unordered_explicit_pixel_selection_fails_without_hanging():
    dataset = image()
    dataset.pixel_indices = torch.tensor([[3, 0], [0, 1]])
    with pytest.raises(ValueError, match="native order"):
        list(SequentialBulkLoader(dataset, 2))


def test_dense_batches_slice_precompacted_storage(monkeypatch):
    dataset = image(shape=(64, 8))
    reader = BulkReader(block_bytes=1024 * 1024, cache_bytes=2 * 1024 * 1024)
    calls = []
    original = dataset.finish_bulk

    def finish(batch, pixels):
        calls.append(len(pixels))
        return original(batch, pixels)

    monkeypatch.setattr(dataset, "finish_bulk", finish)
    monkeypatch.setattr(
        dataset, "pixels_at", lambda *a: pytest.fail("rebuilt pixel table")
    )
    first = reader.read(dataset, 0, 8)
    second = reader.read(dataset, 8, 16)
    assert calls == [512]
    assert reader.read_count == 1
    assert (
        first["intensity"].untyped_storage().data_ptr()
        == second["intensity"].untyped_storage().data_ptr()
    )

