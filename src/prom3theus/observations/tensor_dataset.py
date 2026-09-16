"""Persist aligned training tensors once; retain only paths and array metadata."""

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _leaves(values, prefix=()):
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError("Tensor dictionary keys must be strings")
        if isinstance(value, dict):
            yield from _leaves(value, (*prefix, key))
        elif isinstance(value, torch.Tensor) and value.ndim > 0:
            yield (*prefix, key), value
        else:
            raise ValueError("Expected a nested dictionary of sample-shaped tensors")


class TensorDiskDataset(Dataset):
    """Batch-indexed dataset retaining only file references and schema metadata."""

    def __init__(self, directory, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest["version"] != 1:
            raise ValueError("Unsupported tensor dataset version")
        self.sample_count = manifest["sample_count"]
        self.batch_size = batch_size
        self.files = manifest["files"]
        from .arrays import ArrayRef

        self.arrays = tuple(
            ArrayRef.from_path(self.directory / item["file"]) for item in self.files
        )
        for item in self.files:
            path = self.directory / item["file"]
            if path.parent != self.directory or path.stat().st_size != item["bytes"]:
                raise ValueError("Invalid or incomplete tensor dataset")
        self.initialize_reader()

    def initialize_reader(self):
        from .arrays import process_reader

        process_reader().open(self.arrays)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.initialize_reader()

    @classmethod
    def create(cls, tensors, directory, batch_size, *, seed=0):
        leaves = list(_leaves(tensors))
        if not leaves:
            raise ValueError("Tensor dictionary cannot be empty")
        return cls.from_batches(
            [tensors],
            len(leaves[0][1]),
            directory,
            batch_size,
            generator=torch.Generator().manual_seed(seed),
        )

    @classmethod
    def from_batches(cls, batches, sample_count, directory, batch_size, *, generator):
        """Build from contiguous preparation blocks without a full RAM copy."""
        if sample_count < 1:
            raise ValueError("Tensor dataset must contain samples")
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".tensor-build-", dir=directory.parent))
        arrays, records = {}, []
        order = torch.randperm(sample_count, generator=generator).numpy()
        offset = 0
        try:
            for batch in batches:
                leaves = dict(_leaves(batch))
                if not leaves:
                    raise ValueError("Empty preparation batch")
                count = len(next(iter(leaves.values())))
                if count < 1 or offset + count > sample_count:
                    raise ValueError("Invalid preparation sample count")
                if not arrays:
                    for index, (keys, tensor) in enumerate(leaves.items()):
                        array = tensor.detach().cpu().numpy()
                        name = f"tensor_{index:04d}.npy"
                        shape = (sample_count, *array.shape[1:])
                        arrays[keys] = np.lib.format.open_memmap(
                            staging / name, mode="w+", dtype=array.dtype, shape=shape
                        )
                        records.append({"keys": list(keys), "file": name})
                if set(leaves) != set(arrays):
                    raise ValueError("Preparation tensor keys changed")
                for keys, tensor in leaves.items():
                    value = tensor.detach().cpu().numpy()
                    target = arrays[keys]
                    if (
                        len(value) != count
                        or value.shape[1:] != target.shape[1:]
                        or value.dtype != target.dtype
                    ):
                        raise ValueError(
                            "Tensor shapes, dtypes, or sample counts disagree"
                        )
                    # One common global permutation; reads remain contiguous.
                    target[order[offset : offset + count]] = value
                offset += count
            if offset != sample_count:
                raise ValueError("Incomplete preparation data")
            for array in arrays.values():
                array.flush()
                array._mmap.close()
            arrays.clear()
            for record in records:
                record["bytes"] = (staging / record["file"]).stat().st_size
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "sample_count": sample_count,
                        "files": records,
                    }
                )
            )
            os.rename(staging, directory)
        finally:
            for array in arrays.values():
                array._mmap.close()
            if staging.exists():
                shutil.rmtree(staging)
        return cls(directory, batch_size)

    def __len__(self):
        return math.ceil(self.sample_count / self.batch_size)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.batch_size
        return self.read(start, min(start + self.batch_size, self.sample_count))

    def read(self, start, stop):
        if not 0 <= start <= stop <= self.sample_count:
            raise IndexError((start, stop))
        return self.read_into(start, stop)

    def read_into(self, start, stop, destination=None, *, offset=0, reader=None):
        from .arrays import process_reader

        reader = reader or process_reader()
        if not 0 <= start <= stop <= self.sample_count:
            raise IndexError((start, stop))
        result = {} if destination is None else destination
        for record, ref in zip(self.files, self.arrays, strict=True):
            node = result
            for key in record["keys"][:-1]:
                node = node.setdefault(key, {})
            key = record["keys"][-1]
            if destination is None:
                node[key] = reader.read(ref, slice(start, stop))
            else:
                reader.read_into(
                    ref, start, stop, node[key][offset : offset + stop - start]
                )
        return result

    def allocate(self, count, *, pin_memory=False):
        result = {}
        for record, ref in zip(self.files, self.arrays, strict=True):
            node = result
            for key in record["keys"][:-1]:
                node = node.setdefault(key, {})
            node[record["keys"][-1]] = torch.empty(
                (count, *ref.shape[1:]), dtype=ref.dtype, pin_memory=pin_memory
            )
        return result

    @property
    def sample_bytes(self):
        return sum(math.prod(ref.shape[1:]) * ref.element_size() for ref in self.arrays)
