"""Equal-copy warm-cache comparison; run with PYTHONPATH=src, no GPU required."""

import json
import platform
from pathlib import Path
import statistics
import tempfile
from time import perf_counter

import numpy as np
import torch
from prom3theus.observations.arrays import ReadSession
from prom3theus.observations.tensor_dataset import TensorDiskDataset


def main(root):
    torch.set_num_threads(1)
    values = {
        f"field_{i}": torch.arange(65536 * 16).reshape(65536, 16).float() + i
        for i in range(6)
    }
    dataset = TensorDiskDataset.create(values, root / "pool", 256)
    del values

    def reopen(start, stop):
        result = {}
        for record in dataset.files:
            array = np.load(dataset.directory / record["file"], mmap_mode="r")
            try:
                result[record["keys"][0]] = torch.from_numpy(
                    np.array(array[start:stop], copy=True)
                )
            finally:
                array._mmap.close()
        return result

    results = []
    for size in (64, 256, 4096):
        slices = [(i, min(i + size, 65536)) for i in range(0, 65536, size)]
        np.random.default_rng(123).shuffle(slices)
        with ReadSession() as reader:

            def current(start, stop):
                return dataset.read_into(start, stop, reader=reader)

            torch.testing.assert_close(current(*slices[0]), reopen(*slices[0]))
            measurements = {"reopen_per_batch_seconds": [], "read_session_seconds": []}
            for _ in range(3):
                for key, function in [
                    ("reopen_per_batch_seconds", reopen),
                    ("read_session_seconds", current),
                ]:
                    started = perf_counter()
                    for start, stop in slices:
                        batch = function(start, stop)
                    measurements[key].append(perf_counter() - started)
            results.append(
                {
                    "batch_size": size,
                    **measurements,
                    "mapping_count": reader.mapping_count,
                    "median_speedup": statistics.median(
                        measurements["reopen_per_batch_seconds"]
                    )
                    / statistics.median(measurements["read_session_seconds"]),
                }
            )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
        "samples": 65536,
        "fields": 6,
        "features": 16,
        "results": results,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="data-read-benchmark-") as directory:
        print(json.dumps(main(Path(directory)), indent=2))
