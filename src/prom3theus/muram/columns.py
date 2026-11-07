"""Bounded native-column access shared by diagnostics and synthesis."""

from __future__ import annotations

from numbers import Integral

import os

import numpy as np
from torch.utils.data import DataLoader, Dataset


class ArrayColumns:
    """Column batches from an already converted in-memory atmosphere."""

    def __init__(self, quantities):
        self.quantities = quantities
        self.shape = quantities["temperature"].shape
        self.ds_m = quantities["ds_m"]

    def strided(self, step):
        if step == 1:
            return self
        return ArrayColumns({
            name: value[::step, ::step, :] if isinstance(value, np.ndarray)
            and value.ndim == 3 else value
            for name, value in self.quantities.items()
        })

    def batches(self, batch_size, fields=None):
        nx, ny, _ = self.shape
        names = fields if fields is not None else [
            name for name, value in self.quantities.items()
            if isinstance(value, np.ndarray) and value.ndim == 3
        ]
        for start in range(0, nx * ny, batch_size):
            ix, iy = np.unravel_index(np.arange(start, min(start + batch_size, nx * ny)), (nx, ny))
            yield ix, iy, {"ds_m": self.ds_m, **{
                name: self.quantities[name][ix, iy][:, None, :] for name in names
            }}


class NativeColumnDataset(Dataset):
    """Flattened horizontal pixels; each sample contains one full native column.

    Only filenames and orientation metadata are serialized to workers. Each
    process opens its own read-only maps on first use. No reshaping of the
    oriented cube into a contiguous full-volume array is needed.
    """

    def __init__(self, paths, native_shape, axis_order, scales, signs, *, y_sign, z_sign):
        self.paths = paths
        self.native_shape = tuple(native_shape)
        self.axis_order = tuple(axis_order)
        self.shape = tuple(self.native_shape[axis] for axis in self.axis_order)
        self.scales = scales
        self.signs = signs
        self.y_sign, self.z_sign = y_sign, z_sign
        self._mappings = {}
        self._pid = None

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __getstate__(self):
        # NumPy memmaps otherwise pickle their contents, not just filenames.
        return {**self.__dict__, "_mappings": {}, "_pid": None}

    def _mapped(self, name):
        if self._pid != os.getpid():
            self._mappings = {}
            self._pid = os.getpid()
        if name not in self._mappings:
            raw = np.memmap(self.paths[name], mode="r", dtype=np.float32,
                            shape=self.native_shape[::-1])
            array = raw.transpose(2, 1, 0).transpose(self.axis_order)
            if self.y_sign < 0:
                array = np.flip(array, axis=1)
            if self.z_sign < 0:
                array = np.flip(array, axis=2)
            self._mappings[name] = array
        return self._mappings[name]

    def __getitem__(self, request):
        if isinstance(request, Integral):
            native_index = output_index = int(request)
            fields = tuple(self.paths)
        else:
            native_index, output_index, fields = request
        if not 0 <= native_index < len(self):
            raise IndexError(native_index)
        ix, iy = divmod(native_index, self.shape[1])
        values = {}
        for name in fields:
            column = np.array(self._mapped(name)[ix, iy, :], dtype=np.float32, copy=True)
            np.multiply(column, np.float32(self.scales[name]), out=column)
            if self.signs[name] < 0:
                np.negative(column, out=column)
            values[name] = column
        return {"index": output_index, "fields": values}


class _ColumnBatchSampler:
    """Sequential flattened pixel batches, with per-pass field selection.

    The sampler runs in the parent process. Keeping selection in the requests
    supports field/stride selection without serializing any cube contents.
    """

    def __init__(self, shape):
        self.shape = shape
        self.batch_size = 256
        self.stride = 1
        self.fields = ()

    @property
    def selected_shape(self):
        return tuple((size + self.stride - 1) // self.stride for size in self.shape[:2])

    def __len__(self):
        nx, ny = self.selected_shape
        return (nx * ny + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        nx, ny = self.selected_shape
        for start in range(0, nx * ny, self.batch_size):
            requests = []
            for index in range(start, min(start + self.batch_size, nx * ny)):
                ix, iy = divmod(index, ny)
                native_index = ix * self.stride * self.shape[1] + iy * self.stride
                requests.append((native_index, index, self.fields))
            yield requests


class _ColumnLoader:
    """One two-worker DataLoader per snapshot; synthesis uses one source pass."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.sampler = _ColumnBatchSampler(dataset.shape)
        self.loader = None
        self.active = False

    def batches(self, batch_size, stride, fields):
        if self.active:
            raise RuntimeError("Concurrent passes over one native snapshot are not supported.")
        self.active = True
        try:
            self.sampler.batch_size = batch_size
            self.sampler.stride = stride
            self.sampler.fields = tuple(fields)
            if self.loader is None:
                self.loader = DataLoader(
                    self.dataset, batch_sampler=self.sampler, num_workers=2,
                    prefetch_factor=1, persistent_workers=True,
                    multiprocessing_context="spawn",
                )
            for batch in self.loader:
                yield batch
        finally:
            self.active = False

    def close(self):
        if self.active:
            raise RuntimeError("Cannot close an active column loader.")
        # Releasing DataLoader releases its persistent iterator and workers.
        self.loader = None


class NativeColumns:
    """Sequential column batches from a lazy memory-mapped dataset."""

    def __init__(self, dataset, ds_m, *, stride=1, owner=None):
        self.dataset = dataset
        self.ds_m = ds_m
        self.stride = stride
        self.shape = tuple((size + stride - 1) // stride for size in dataset.shape[:2]) + (dataset.shape[2],)
        self._owner = _ColumnLoader(dataset) if owner is None else owner

    def strided(self, step):
        if step == 1:
            return self
        return NativeColumns(self.dataset, self.ds_m, stride=self.stride * step, owner=self._owner)

    def batches(self, batch_size, fields=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        names = tuple(self.dataset.paths) if fields is None else tuple(fields)
        for batch in self._owner.batches(batch_size, self.stride, names):
            ix, iy = np.unravel_index(batch["index"].numpy(), self.shape[:2])
            yield ix, iy, {"ds_m": self.ds_m, **{
                name: value.numpy()[:, None, :] for name, value in batch["fields"].items()
            }}

    def close(self):
        self._owner.close()


def as_columns(quantities):
    return quantities if isinstance(quantities, (ArrayColumns, NativeColumns)) else ArrayColumns(quantities)


def snapshot_columns(snapshot):
    factory = getattr(snapshot, "synthesis_columns", None)
    return factory() if factory is not None else as_columns(snapshot.synthesis_ready_quantities())


def close_snapshot(snapshot):
    close = getattr(snapshot, "close", None)
    if close is not None:
        close()


def strided_columns(quantities, stride):
    step = 1 if stride is None else stride
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 1:
        raise ValueError("stride must be a positive integer.")
    return as_columns(quantities).strided(int(step)), int(step)


def column_batches(quantities, batch_size, *, fields=None, progress=False, description="Synthesis"):
    from tqdm.auto import tqdm

    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    source = as_columns(quantities)
    with tqdm(total=source.shape[0] * source.shape[1], desc=description,
              unit="column", disable=not progress) as bar:
        for ix, iy, columns in source.batches(batch_size, fields):
            yield ix, iy, columns
            bar.update(len(ix))
