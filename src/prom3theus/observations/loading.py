"""Shared worker and bounded-buffer policy for observation access."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

import torch
from torch.utils.data import DataLoader


def buffered_dataloader(dataset, *, num_workers=2, shuffle=False, **options):
    """Bulk observation loading; generic plugin datasets retain DataLoader support."""
    from .bulk import SequentialBulkLoader, supported
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")
    if supported(dataset):
        options.pop("collate_fn", None)  # Native sources already return complete contracts.
        options.pop("persistent_workers", None)
        if shuffle:
            from .tensor_loader import PersistentTensorLoader
            return PersistentTensorLoader(dataset, **options)
        return SequentialBulkLoader(dataset, **options)
    # Non-raster extension datasets retain their own contract and sampling policy.
    pin = options.pop("pin_memory", False) and torch.cuda.is_available()
    return DataLoader(dataset, num_workers=num_workers, shuffle=shuffle,
                      pin_memory=pin, **({"persistent_workers": True,
                      "prefetch_factor": 2, "multiprocessing_context": "spawn"}
                      if num_workers else {}), **options)


@dataclass
class _ReadOnlyMapping:
    values: dict


def _plain(value):
    if isinstance(value, MappingProxyType):
        return _ReadOnlyMapping({key: _plain(item) for key, item in value.items()})
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _restored(value):
    if isinstance(value, _ReadOnlyMapping):
        return MappingProxyType(_restored(value.values))
    if isinstance(value, dict):
        return {key: _restored(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_restored(item) for item in value)
    if isinstance(value, list):
        return [_restored(item) for item in value]
    return value


def _restore_raster(cls, values):
    # These are already validated live rasters or versioned local setup caches.
    # Re-running constructors would scan all image arrays in every worker and
    # defeat the setup cache's promise to reuse validated geometry and indexes.
    raster = cls.__new__(cls)
    for name, value in values.items():
        object.__setattr__(raster, name, _restored(value))
    return raster


def raster_reduce(raster):
    """Make immutable raster mappings serializable to spawned loader workers."""
    return _restore_raster, (
        type(raster),
        {field.name: _plain(getattr(raster, field.name)) for field in fields(raster)},
    )


def load_array_tensor(path, *, mmap=True, shape=None, dtype=None):
    """Map a native C-order NPY with a closed file descriptor and COW storage.

    PyTorch's private file mapping closes the descriptor after mmap; retaining
    hundreds of raster arrays therefore does not retain hundreds of open files.
    Header reads are small; payload pages are touched only by bulk copies.
    """
    import math
    import numpy as np
    from pathlib import Path

    path = Path(path)
    if not mmap:
        array = np.load(path, allow_pickle=False)
        if shape is not None and (list(array.shape) != list(shape) or array.dtype != dtype):
            raise ValueError("Observation array schema mismatch")
        return torch.from_numpy(array)
    with path.open('rb') as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY version {version}; rewrite as native numeric NPY")
        offset = handle.tell()
    if actual_dtype.hasobject or not actual_dtype.isnative or fortran:
        raise ValueError("Bulk stores require native numeric C-order NPY arrays")
    if shape is not None and (list(actual_shape) != list(shape) or actual_dtype != dtype):
        raise ValueError("Observation array schema mismatch")
    count = math.prod(actual_shape)
    size = count * actual_dtype.itemsize
    if path.stat().st_size < offset + size:
        raise ValueError("Truncated observation array")
    tensor_dtype = torch.from_numpy(np.empty(0, dtype=actual_dtype)).dtype
    if count == 0:
        return torch.empty(actual_shape, dtype=tensor_dtype)
    storage = torch.from_file(str(path), shared=False, size=offset + size, dtype=torch.uint8)
    tensor = storage[offset:].view(tensor_dtype).view(actual_shape)
    from .arrays import ArrayRef
    stat = path.stat()
    tensor._array_ref = ArrayRef(str(path.resolve()), tuple(actual_shape), actual_dtype.str,
                               offset, stat.st_size, stat.st_mtime_ns)
    tensor._array_version = -1 if torch.is_inference(tensor) else tensor._version
    return tensor
