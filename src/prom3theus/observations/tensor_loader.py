"""Global tensor shuffling at startup, followed only by contiguous batch slices."""

import hashlib
import json
import math
import tempfile
from pathlib import Path
from dataclasses import dataclass
from time import perf_counter

import torch

from .bulk import (
    BulkBatchLoader,
    BulkReader,
    sample_bytes,
)
from .tensor_dataset import TensorDiskDataset


class PersistentTensorLoader(BulkBatchLoader):
    """Persist one global permutation per stream/channel as training tensor files.

    Native stores are read sequentially in bounded blocks exactly once. Every
    field is scattered with the same permutation into disk-backed arrays.
    Training permutes only batch IDs, without replacement on each pool pass;
    it never gathers individual pixels. AIA pools preserve channel coverage.
    Pixel layout and batch ordering use private deterministic generators, making
    direct resume independent of model RNG, read budgets, and prefetch depth.
    """

    def __init__(self, *args, seed=0, stream_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.stream_id = stream_id
        self._stores = None
        self._iteration_start = 0
        self.tensor_cache_directory = None
        self.tensor_cache_identity = None
        self._temporary_directory = None
        self._counts = [len(pool) for pool in self._pools()]
        self._source_contract = super().contract()

    def _generator(self, purpose, pool, cycle=0):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("loader_seed must be an integer in [0, 2**63)")
        identity = f"{self.seed}:{self.stream_id}:{purpose}:{pool}:{cycle}".encode()
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % (
            2**63 - 1
        )
        return torch.Generator().manual_seed(seed)

    def _pools(self):
        return self.channels or (self.dataset,)

    def _quotas(self):
        base, extra = divmod(self.batch_size, len(self._counts))
        return [base + (index < extra) for index in range(len(self._counts))]

    def __len__(self):
        return max(
            math.ceil(count / quota)
            for count, quota in zip(self._counts, self._quotas(), strict=True)
        )

    def contract(self):
        return {
            **self._source_contract,
            "batch_size": self.batch_size,
            "shuffle": {
                "version": 2,
                "policy": "global_tensor_shuffle_random_batches",
                "seed": self.seed,
                "stream_id": self.stream_id,
            },
        }

    def prepare(self):
        if self._stores is not None:
            return
        started = perf_counter()
        identity = {
            "source": self.tensor_cache_identity,
            "seed": self.seed,
            "stream": self.stream_id,
            "version": 1,
        }
        if self.tensor_cache_directory is None:
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="training-tensors-"
            )
            root = Path(self._temporary_directory.name)
        else:
            if self.tensor_cache_identity is None:
                raise ValueError(
                    "Persistent training tensors require a source identity"
                )
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            root = Path(self.tensor_cache_directory) / key
        stores = []
        for index, pool in enumerate(self._pools()):
            pool_started = perf_counter()
            directory = root / f"pool_{index:03d}"
            if directory.exists():
                store = TensorDiskDataset(directory, self._quotas()[index])
                if store.sample_count != len(pool):
                    raise ValueError("Cached training tensor sample count differs")
                print(f"Reusing training tensors: {directory}", flush=True)
            else:
                reader = BulkReader(self.block_bytes, self.cache_bytes)
                block_samples = max(1, self.block_bytes // sample_bytes(pool))

                def blocks(pool=pool, reader=reader, block_samples=block_samples):
                    for begin in range(0, len(pool), block_samples):
                        yield reader.read(
                            pool, begin, min(begin + block_samples, len(pool))
                        )

                store = TensorDiskDataset.from_batches(
                    blocks(),
                    len(pool),
                    directory,
                    self._quotas()[index],
                    generator=self._generator("pixels", index),
                )
                del reader
                print(
                    f"Persisted training tensors: {directory} "
                    f"(pool {perf_counter() - pool_started:.2f}s; total {perf_counter() - started:.2f}s)",
                    flush=True,
                )
            stores.append(store)
        self._stores = stores
        # Training retains file paths and small schedule metadata, not rasters.
        self.dataset = None
        self.channels = ()

    def __iter__(self):
        start = self._iteration_start
        self._iteration_start += len(self)
        producer = self.iter_from(start, len(self))
        try:
            yield from producer
        finally:
            producer.close()

    def iter_indices(self, indices):
        from .bulk import _Producer

        self.close()
        producer = _Producer(self._indexed_batches(indices), self.prefetch_batches)
        self._active.append(producer)
        return producer

    def _batches(self, start, count):
        yield from self._indexed_batches(range(start, start + count))

    def _indexed_batches(self, indices):
        self.prepare()
        schedules = {}
        for number in indices:
            slices = []
            for index, (count_samples, quota, store) in enumerate(
                zip(self._counts, self._quotas(), self._stores, strict=True)
            ):
                chunks = math.ceil(count_samples / quota)
                cycle, offset = divmod(number, chunks)
                previous = schedules.get(index)
                if previous is None or previous[0] != cycle:
                    order = torch.randperm(
                        chunks, generator=self._generator("batches", index, cycle)
                    )
                    schedules[index] = (cycle, order)
                begin = int(schedules[index][1][offset]) * quota
                slices.append((store, begin, min(begin + quota, count_samples)))
            count_samples = sum(stop - begin for _, begin, stop in slices)
            if (
                sum(
                    (stop - begin) * store.sample_bytes for store, begin, stop in slices
                )
                > self.max_batch_bytes
            ):
                raise ValueError(
                    "Batch exceeds max_batch_bytes; increase the loader budget"
                )
            batch = self._stores[0].allocate(count_samples, pin_memory=self.pin_memory)
            offset = 0
            for store, begin, stop in slices:
                store.read_into(begin, stop, batch, offset=offset)
                offset += stop - begin
            yield batch

    def snapshot(self, directory=None):
        self.prepare()
        stores = self._stores
        if self._temporary_directory is not None and directory is not None:
            # A saved module must outlive an ad-hoc loader's temporary pools.
            import shutil
            import errno
            import os

            def copy(source, target):
                try:
                    os.link(source, target)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(source, target)
                return target

            stores = []
            for index, store in enumerate(self._stores):
                target = Path(directory) / f"pool_{index:03d}"
                shutil.copytree(store.directory, target, copy_function=copy)
                stores.append(TensorDiskDataset(target, store.batch_size))
        return PackedLoaderSpec(
            tuple(stores),
            tuple(self._counts),
            self._source_contract,
            self.batch_size,
            self.seed,
            self.stream_id,
        )

    def close(self):
        # Descriptors survive closing; TemporaryDirectory owns cleanup at object GC.
        super().close()


@dataclass
class PackedLoaderSpec:
    stores: tuple
    counts: tuple
    source_contract: dict
    batch_size: int
    seed: int
    stream_id: str

    def bind(
        self,
        *,
        batch_size=None,
        pin_memory=False,
        prefetch_batches=2,
        max_batch_bytes=64 * 1024**2,
        seed=None,
    ):
        loader = PersistentTensorLoader.__new__(PersistentTensorLoader)
        for store in self.stores:
            store.initialize_reader()
        loader._stores = list(self.stores)
        loader._counts = list(self.counts)
        loader._source_contract = self.source_contract
        loader.batch_size = self.batch_size if batch_size is None else batch_size
        if loader.batch_size < len(self.counts):
            raise ValueError("Training batch_size must include every channel")
        loader.seed, loader.stream_id = (
            self.seed if seed is None else seed,
            self.stream_id,
        )
        loader.pin_memory = pin_memory and torch.cuda.is_available()
        loader.prefetch_batches, loader.max_batch_bytes = (
            prefetch_batches,
            max_batch_bytes,
        )
        loader._active, loader._reader = [], None
        loader.dataset, loader.channels = None, ()
        loader._iteration_start = 0
        loader._temporary_directory = None
        return loader
