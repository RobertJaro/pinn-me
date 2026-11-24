"""Eager native volumes with sequential column batches for synthesis."""

from __future__ import annotations

import logging
import os
from numbers import Integral
from time import perf_counter

import numpy as np
from .reporting import progress_bar, timed_stage

logger = logging.getLogger(__name__)


class ArrayColumns:
    """Column batches from an already converted in-memory atmosphere."""

    def __init__(self, quantities):
        self.quantities = quantities
        self.shape = next(value.shape for value in quantities.values()
                          if isinstance(value, np.ndarray) and value.ndim == 3)
        self.ds_m = quantities["ds_m"]

    def strided(self, step):
        if step == 1:
            return self
        return ArrayColumns({
            name: value[::step, ::step, :] if isinstance(value, np.ndarray)
            and value.ndim == 3 else value
            for name, value in self.quantities.items()
        })

    def cropped(self, depth):
        return ArrayColumns({
            name: value[..., :depth] if isinstance(value, np.ndarray) and value.ndim == 3 else value
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


class NativeColumnDataset:
    """Read-only native field mappings with unit and orientation descriptors.

    Also exposes individual columns for inspection; synthesis materializes full
    fields before iterating over in-memory column batches.
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
        return {"index": output_index, "fields": self.read_fields(ix, iy, fields)}

    def read_fields(self, ix, iy, fields):
        """Read one column or a batch of coordinate pairs in synthesis units."""
        values = {}
        for name in fields:
            column = np.array(self._mapped(name)[ix, iy, :], dtype=np.float32, copy=True)
            np.multiply(column, np.float32(self.scales[name]), out=column)
            if self.signs[name] < 0:
                np.negative(column, out=column)
            values[name] = column
        return values


class _ColumnVolume:
    """One cache of converted full cubes, shared by strided views of a snapshot."""

    def __init__(self, dataset, *, depth=None, stride=1):
        self.dataset = dataset
        self.depth = dataset.shape[2] if depth is None else depth
        self.stride = stride
        self.shape = tuple((size + stride - 1) // stride for size in dataset.shape[:2]) + (self.depth,)
        self.fields = {}

    def load(self, names, *, progress=False):
        missing = [name for name in names if name not in self.fields]
        if not missing:
            return
        gib = int(np.prod(self.shape)) * len(missing) * 4 / 2**30
        with timed_stage(logger, "Loading %d selected native fields (%.2f GiB)", len(missing), gib), progress_bar(
            missing, description="Loading cube fields", unit="field", enabled=progress,
        ) as fields:
            for name in fields:
                logger.info("Loading native field %s", name)
                selected = self.dataset._mapped(name)[::self.stride, ::self.stride, :self.depth]
                values = np.array(selected, dtype=np.float32, copy=True, order="C")
                np.multiply(values, np.float32(self.dataset.scales[name]), out=values)
                if self.dataset.signs[name] < 0:
                    np.negative(values, out=values)
                values.flags.writeable = False
                self.fields[name] = values

    def close(self):
        self.fields.clear()
        self.dataset._mappings.clear()


class NativeColumns:
    """Full native cubes loaded once, then processed in sequential column batches."""

    def __init__(self, dataset, ds_m, *, stride=1, owner=None):
        self.dataset = dataset
        self.ds_m = ds_m
        self.stride = stride
        self._owner = _ColumnVolume(dataset) if owner is None else owner
        self.shape = tuple((size + stride - 1) // stride for size in dataset.shape[:2]) + (self._owner.depth,)

    def strided(self, step):
        if step == 1:
            return self
        return NativeColumns(self.dataset, self.ds_m, stride=self.stride * step, owner=self._owner)

    def cropped(self, depth):
        if depth == self.shape[2] and self.stride == self._owner.stride:
            return self
        owner = _ColumnVolume(self.dataset, depth=depth, stride=self.stride)
        return NativeColumns(self.dataset, self.ds_m, stride=self.stride, owner=owner)

    def batches(self, batch_size, fields=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        names = tuple(self.dataset.paths) if fields is None else tuple(fields)
        self.load_fields(names)
        source = ArrayColumns({"ds_m": self.ds_m, **{name: self._owner.fields[name] for name in names}})
        yield from source.strided(self.stride // self._owner.stride).batches(batch_size, names)

    def load_fields(self, fields=None, *, progress=False):
        names = tuple(self.dataset.paths) if fields is None else tuple(fields)
        self._owner.load(names, progress=progress)

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
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    source = as_columns(quantities)
    if isinstance(source, NativeColumns):
        source.load_fields(fields, progress=progress)
    total = source.shape[0] * source.shape[1]
    logger.debug("%s: shape=%s, batch_size=%d, fields=%s", description, source.shape, batch_size, fields)
    started = finished = last_report = perf_counter()
    loaded_seconds = processing_seconds = 0.0
    completed = 0
    with progress_bar(total=total, description=description, unit="column", enabled=progress) as bar:
        try:
            for ix, iy, columns in source.batches(batch_size, fields):
                ready = perf_counter()
                loaded_seconds += ready - finished
                yield ix, iy, columns
                processing_seconds += perf_counter() - ready
                completed += len(ix)
                bar.update(len(ix))
                now = perf_counter()
                if progress and bar.disable and now - last_report >= 30.0:
                    elapsed = now - started
                    rate = completed / elapsed
                    logger.info("%s: %d/%d columns (%.1f%%), %.1f columns/s, ETA %.0f s",
                                description, completed, total, 100 * completed / total,
                                rate, (total - completed) / rate)
                    last_report = now
                finished = perf_counter()
        finally:
            logger.info("%s: processed %d/%d columns in %.1f s (loading/waiting %.1f s; processing %.1f s)",
                        description, completed, total, perf_counter() - started,
                        loaded_seconds, processing_seconds)
