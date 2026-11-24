"""Persistent tensors supporting coverage passes and random sample batches."""

import hashlib
import json
import math
import tempfile
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import closing
from threading import local
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

    Native stores are read concurrently in contiguous blocks. Every
    field is scattered with the same permutation into disk-backed arrays.
    Coverage passes permute batch IDs without replacement. Fixed-length training
    epochs gather fresh random pixels with replacement. AIA pools preserve channel coverage.
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

    def _seed(self, purpose, pool, cycle=0):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("loader_seed must be an integer in [0, 2**63)")
        identity = f"{self.seed}:{self.stream_id}:{purpose}:{pool}:{cycle}".encode()
        return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % (
            2**63 - 1
        )

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

    def _preparation_blocks(self, pool):
        block_samples = max(1, self.block_bytes // sample_bytes(pool))
        starts = range(0, len(pool), block_samples)
        readers = local()

        def read(start):
            # Slab caches are mutable and belong to individual reader threads.
            if not hasattr(readers, "reader"):
                readers.reader = BulkReader(self.block_bytes, self.cache_bytes)
            return readers.reader.read(pool, start, min(start + block_samples, len(pool)))

        workers = min(self.num_workers, len(starts))
        if workers <= 1:
            yield from map(read, starts)
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            remaining = iter(starts)
            pending = deque(executor.submit(read, next(remaining)) for _ in range(workers))
            while pending:
                yield pending.popleft().result()
                start = next(remaining, None)
                if start is not None:
                    pending.append(executor.submit(read, start))

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
                with closing(self._preparation_blocks(pool)) as blocks:
                    store = TensorDiskDataset.from_batches(
                        blocks,
                        len(pool),
                        directory,
                        self._quotas()[index],
                        generator=torch.Generator().manual_seed(self._seed("pixels", index)),
                    )
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

    def _batch_dataset(self):
        self.prepare()
        return _TrainingBatches(self._stores, self._quotas(), self.max_batch_bytes)

    def iter_indices(self, indices):
        return self._load_batches(self._batch_dataset(), self._coverage_slices(indices), persist=True)

    def iter_random(self, indices):
        """Draw full batches from all samples, keyed by global rank/update slot."""
        dataset = self._batch_dataset()
        seeds = (
            tuple(self._seed("samples", pool, number) for pool in range(len(self._stores)))
            for number in indices
        )
        return self._load_batches(dataset, seeds, persist=True)

    def _coverage_slices(self, indices):
        schedules = {}
        for number in indices:
            slices = []
            for index, (count_samples, quota) in enumerate(
                zip(self._counts, self._quotas(), strict=True)
            ):
                chunks = math.ceil(count_samples / quota)
                cycle, offset = divmod(number, chunks)
                previous = schedules.get(index)
                if previous is None or previous[0] != cycle:
                    generator = torch.Generator().manual_seed(self._seed("batches", index, cycle))
                    order = torch.randperm(chunks, generator=generator)
                    schedules[index] = (cycle, order)
                begin = int(schedules[index][1][offset]) * quota
                slices.append(slice(begin, min(begin + quota, count_samples)))
            yield tuple(slices)

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
            self.num_workers,
        )

    def close(self):
        # Descriptors survive closing; TemporaryDirectory owns cleanup at object GC.
        super().close()


@dataclass
class _TrainingBatches:
    """Read one scheduled batch from the worker's process-local file mappings."""

    stores: list
    quotas: list
    max_batch_bytes: int

    def __getitem__(self, selections):
        counts = [
            selection.stop - selection.start if isinstance(selection, slice) else quota
            for selection, quota in zip(selections, self.quotas, strict=True)
        ]
        size = sum(count * store.sample_bytes for count, store in zip(counts, self.stores, strict=True))
        if size > self.max_batch_bytes:
            raise ValueError("Batch exceeds max_batch_bytes; increase the loader budget")
        batch = self.stores[0].allocate(sum(counts))
        offset = 0
        for store, selection, count in zip(self.stores, selections, counts, strict=True):
            if isinstance(selection, slice):
                store.read_into(selection.start, selection.stop, batch, offset=offset)
            else:
                generator = torch.Generator().manual_seed(selection)
                indices = torch.randint(store.sample_count, (count,), generator=generator)
                store.read_indices_into(indices, batch, offset=offset)
            offset += count
        return batch


@dataclass
class PackedLoaderSpec:
    stores: tuple
    counts: tuple
    source_contract: dict
    batch_size: int
    seed: int
    stream_id: str
    num_workers: int = 4

    def bind(
        self,
        *,
        batch_size=None,
        num_workers=None,
        pin_memory=False,
        prefetch_batches=2,
        max_batch_bytes=64 * 1024**2,
        seed=None,
    ):
        loader = PersistentTensorLoader.__new__(PersistentTensorLoader)
        for store in self.stores:
            store.initialize_reader()
        loader.num_workers = self.num_workers if num_workers is None else num_workers
        loader._stores = list(self.stores)
        loader._counts = list(self.counts)
        loader._source_contract = self.source_contract
        loader.batch_size = self.batch_size if batch_size is None else batch_size
        if loader.batch_size < len(self.counts):
            raise ValueError("Training batch_size must include every channel")
        loader.seed = self.seed if seed is None else seed
        loader.stream_id = self.stream_id
        loader.pin_memory = pin_memory and torch.cuda.is_available()
        loader.prefetch_batches, loader.max_batch_bytes = (
            prefetch_batches,
            max_batch_bytes,
        )
        loader._active, loader._reader = [], None
        loader._worker_loader = None
        loader._worker_settings = None
        loader.dataset, loader.channels = None, ()
        loader._iteration_start = 0
        loader._temporary_directory = None
        return loader
