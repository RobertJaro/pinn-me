"""Reproduce the data-loading audit; synthetic warm-cache CPU results only.

PYTHONPATH=src conda run -n nf2 python docs/assets/data-loading-audit/probe.py
"""
import contextlib
import io
import json
import os
import platform
import statistics
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from prom3theus.observations import bulk, loading
from prom3theus.observations.image_dataset import ImagePixelDataset
from prom3theus.observations.persistence import (
    JointDataModule, save_data_module, load_data_module, restore_or_create_data_module,
)
from prom3theus.observations.tensor_dataset import TensorDiskDataset
from prom3theus.observations.tensor_loader import PersistentTensorLoader


def image():
    shape = (32, 32)
    return ImagePixelDataset(SimpleNamespace(
        intensity=torch.arange(1024, dtype=torch.float32).reshape(shape),
        uncertainty=torch.ones(shape), ray_direction=torch.ones(*shape, 3),
        surface_position_m=torch.ones(*shape, 3), valid_mask=torch.ones(shape, dtype=torch.bool),
        spatial_shape=shape, absolute_tai_seconds=10., channel_angstrom=171,
    ), image_index=0, channel_index=0)


def run(root):
    torch.set_num_threads(1)
    out = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "platform": platform.platform(),
           "cuda_available": torch.cuda.is_available(), "torch_threads": 1},
           "scope": "Synthetic local files, warm OS cache; no GPU or production storage claims."}
    native = root / "native.npy"
    np.save(native, np.arange(16384, dtype=np.float32))
    path = root / "data_module.pt"
    save_data_module(JointDataModule({"x": loading.load_array_tensor(native)}, None), path)
    restored = load_data_module(path)
    snapshots = list(root.glob("data-arrays-*/*.npy"))
    out["duplicated_existing_payload"] = {"native_bytes": native.stat().st_size,
         "snapshot_array_bytes": sum(p.stat().st_size for p in snapshots),
         "new_array_files": len(snapshots)}
    x = restored.streams["x"]
    with patch.object(loading, "load_array_tensor", wraps=loading.load_array_tensor) as opened:
        metadata_only = (x.shape, x.dtype, x.numel())
        before = opened.call_count
        for _ in range(100):
            _ = x[:64].clone()
        out["disk_tensor_mapping_calls"] = {"metadata_only": before,
                                             "100_slice_copies": opened.call_count - before}
    before = x[:4].clone()
    x.add_(10)
    out["disk_tensor_inplace_update_lost"] = torch.equal(x[:4], before)
    copied = Path(x.path)
    copied.unlink()
    loaded_missing = load_data_module(path)
    try:
        loaded_missing.streams["x"][:1]
    except FileNotFoundError:
        out["missing_payload"] = "restore succeeded; first slice raised FileNotFoundError"
    save_data_module(JointDataModule({}, "old_generation"), path)
    with patch.dict(os.environ, {"RANK": "1"}):
        out["worker_rebuild_without_process_group"] = restore_or_create_data_module(
            path, lambda: None, rebuild=True, timeout=0).scene

    native_dataset_path = root / "native_dataset.pt"
    save_data_module(JointDataModule({"image": image()}, None), native_dataset_path)
    native_dataset = load_data_module(native_dataset_path).streams["image"]
    reader = bulk.BulkReader()
    reader.read(native_dataset, 0, 64)
    with patch.object(loading, "load_array_tensor", wraps=loading.load_array_tensor) as opened:
        for _ in range(100):
            reader.read(native_dataset, 0, 64)
        out["cached_native_slab"] = {"bulk_payload_reads": reader.read_count,
            "mapping_calls_for_100_cache_hits": opened.call_count}

    loader = PersistentTensorLoader(image(), 64)
    loader.tensor_cache_directory = root / "train"
    loader.tensor_cache_identity = {"id": "audit"}
    with contextlib.redirect_stdout(io.StringIO()):
        loader.prepare()
    with patch.object(bulk, "_READ_SLOTS") as slots:
        _ = next(loader._batches(0, 1))
        out["training_read_semaphore_entries"] = slots.__enter__.call_count
    loader_path = root / "loader.pt"
    save_data_module(JointDataModule({}, None, {"image": loader}), loader_path)
    a = load_data_module(loader_path).train_loaders["image"]
    b = load_data_module(loader_path).train_loaders["image"]
    out["two_restored_loaders_identical_first_batch"] = torch.equal(
        next(a._batches(0, 1))["intensity"], next(b._batches(0, 1))["intensity"])
    loader.close()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            loader.prepare()
    except Exception as error:
        out["close_then_prepare"] = f"{type(error).__name__}: {error}"

    values = {f"field_{i}": torch.arange(65536 * 16, dtype=torch.float32).reshape(65536, 16) + i
              for i in range(6)}
    dataset = TensorDiskDataset.create(values, root / "benchmark", 256)
    del values
    result = []
    for batch_size in (64, 256, 4096):
        slices = [(i, min(i + batch_size, 65536)) for i in range(0, 65536, batch_size)]
        np.random.default_rng(123).shuffle(slices)
        # Reference prototype: process-owned mappings, same owned batch copies.
        maps = [np.load(dataset.directory / f["file"], mmap_mode="r") for f in dataset.files]
        def mapped_read(start, stop):
            return {record["keys"][0]: torch.from_numpy(np.array(array[start:stop], copy=True))
                    for record, array in zip(dataset.files, maps)}
        torch.testing.assert_close(dataset.read(*slices[0]), mapped_read(*slices[0]))
        times = {"current_seconds": [], "reused_mappings_seconds": []}
        for _ in range(3):
            for key, reader in (("current_seconds", dataset.read), ("reused_mappings_seconds", mapped_read)):
                started = perf_counter()
                for begin, end in slices:
                    batch = reader(begin, end)
                times[key].append(perf_counter() - started)
        with patch.object(np, "load", wraps=np.load) as opened:
            dataset.read(*slices[0])
            calls = opened.call_count
        result.append({"batch_size": batch_size, "batches": len(slices), "fields": 6,
                       "np_load_calls_per_current_batch": calls, **times,
                       "median_current_over_prototype": statistics.median(times["current_seconds"]) /
                           statistics.median(times["reused_mappings_seconds"])})
        for array in maps:
            array._mmap.close()
    out["warm_cache_benchmark"] = result
    return out


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="p3s-data-audit-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
