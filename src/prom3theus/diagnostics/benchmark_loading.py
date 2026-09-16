"""Compare CPU batch assembly with the former per-pixel loader (no disk/GPU claim).

Run: PYTHONPATH=src python -m prom3theus.diagnostics.benchmark_loading
"""

import argparse
import json
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from prom3theus.observations.image_dataset import ImagePixelDataset
from prom3theus.observations.bulk import SequentialBulkLoader
from prom3theus.observations.tensor_loader import PersistentTensorLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.side, args.batch_size, args.repeats) < 1:
        parser.error("All options must be positive")
    torch.set_num_threads(1)
    shape = (args.side, args.side)
    raster = SimpleNamespace(
        intensity=torch.arange(args.side**2, dtype=torch.float32).reshape(shape),
        uncertainty=torch.ones(shape),
        ray_direction=torch.ones(*shape, 3),
        surface_position_m=torch.ones(*shape, 3),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        spatial_shape=shape,
        absolute_tai_seconds=1.0,
        channel_angstrom=171,
    )
    legacy = ImagePixelDataset(
        raster,
        image_index=0,
        channel_index=0,
        pixel_indices=torch.nonzero(raster.valid_mask),
    )
    sequential = ImagePixelDataset(raster, image_index=0, channel_index=0)
    results = {}
    for name, loader in (
        (
            "scalar_sequential",
            DataLoader(legacy, batch_size=args.batch_size, num_workers=0),
        ),
        ("bulk_sequential", SequentialBulkLoader(sequential, args.batch_size)),
        (
            "global_tensor_shuffle",
            PersistentTensorLoader(sequential, args.batch_size),
        ),
    ):
        times = []
        try:
            for _ in range(args.repeats):
                start = perf_counter()
                count = sum(len(batch["intensity"]) for batch in loader)
                times.append(perf_counter() - start)
        finally:
            if hasattr(loader, "close"):
                loader.close()
        results[name] = {
            "samples": count,
            "seconds": times,
            "best_samples_per_second": count / min(times),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
