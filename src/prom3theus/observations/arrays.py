"""Filename-only arrays and process-local, bounded read ownership."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import BoundedSemaphore, Condition

import numpy as np
import torch


@dataclass(frozen=True)
class ArrayRef:
    path: str
    shape: tuple[int, ...]
    numpy_dtype: str
    offset: int
    size_bytes: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path):
        path = Path(path).resolve()
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported NPY version: {version}")
            offset = handle.tell()
        if dtype.hasobject or not dtype.isnative or fortran:
            raise ValueError("Arrays must use native numeric C-order NPY storage")
        stat = path.stat()
        if stat.st_size != offset + math.prod(shape) * dtype.itemsize:
            raise ValueError(f"Incomplete array: {path}")
        return cls(
            str(path), tuple(shape), dtype.str, offset, stat.st_size, stat.st_mtime_ns
        )

    @property
    def dtype(self):
        return torch.from_numpy(np.empty(0, dtype=self.numpy_dtype)).dtype

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return np.dtype(self.numpy_dtype).itemsize

    def __len__(self):
        return self.shape[0]

    def validate(self):
        stat = Path(self.path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.size_bytes, self.mtime_ns):
            raise ValueError(
                f"Prepared array changed: {self.path}; explicitly rebuild data"
            )

    def read(self, index=Ellipsis, *, reader=None):
        return (reader or process_reader()).read(self, index)

    def __getitem__(self, index):
        """Array indexing is an owned disk read, never a tensor dispatch hook."""
        return self.read(index)


class ReadSession:
    """Own mappings while leased; outputs never borrow their storage."""

    def __init__(self, max_mappings=64, workers=2):
        if max_mappings < 1 or workers < 1:
            raise ValueError("Reader limits must be positive")
        self.max_mappings = max_mappings
        self._slots = BoundedSemaphore(workers)
        self._condition = Condition()
        self._maps = OrderedDict()
        self.mapping_count = self.read_count = self.read_bytes = 0

    @contextmanager
    def _lease(self, ref):
        with self._condition:
            while ref not in self._maps and len(self._maps) >= self.max_mappings:
                idle = next(
                    (key for key, value in self._maps.items() if value[2] == 0), None
                )
                if idle is None:
                    self._condition.wait()
                    continue
                _, mapping, _ = self._maps.pop(idle)
                mapping.close()
            if ref not in self._maps:
                ref.validate()
                array = np.load(ref.path, mmap_mode="r", allow_pickle=False)
                mapping = array._mmap
                if (
                    tuple(array.shape) != ref.shape
                    or array.dtype.str != ref.numpy_dtype
                    or array.offset != ref.offset
                    or not array.flags.c_contiguous
                ):
                    mapping.close()
                    raise ValueError(f"Prepared array schema changed: {ref.path}")
                self._maps[ref] = [array, mapping, 0]
                self.mapping_count += 1
            entry = self._maps[ref]
            entry[2] += 1
            self._maps.move_to_end(ref)
        try:
            yield entry[0]
        finally:
            with self._condition:
                entry[2] -= 1
                self._condition.notify_all()

    def open(self, refs):
        """Initialize lazy mappings without touching array payloads."""
        for ref in refs:
            with self._lease(ref):
                pass

    @staticmethod
    def _index(index):
        if isinstance(index, torch.Tensor):
            return index.cpu().numpy()
        if isinstance(index, tuple):
            return tuple(ReadSession._index(item) for item in index)
        return index

    def read(self, ref, index=Ellipsis):
        with self._slots, self._lease(ref) as source:
            result = torch.from_numpy(np.array(source[self._index(index)], copy=True))
        self.read_count += 1
        self.read_bytes += result.numel() * result.element_size()
        return result

    def read_tensor(self, tensor, index=Ellipsis):
        with self._slots:
            return tensor[index].clone()

    def read_into(self, ref, start, stop, destination):
        if (
            tuple(destination.shape) != (stop - start, *ref.shape[1:])
            or destination.dtype != ref.dtype
        ):
            raise ValueError("Destination does not match array slice")
        with self._slots, self._lease(ref) as source:
            np.copyto(destination.numpy(), source[start:stop])
        self.read_count += 1
        self.read_bytes += destination.numel() * destination.element_size()

    def close(self):
        with self._condition:
            while any(entry[2] for entry in self._maps.values()):
                self._condition.wait()
            for array, mapping, _ in self._maps.values():
                mapping.close()
            self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


_PROCESS_READER = None
_PROCESS_ID = None


def process_reader():
    global _PROCESS_READER, _PROCESS_ID
    if _PROCESS_READER is None or _PROCESS_ID != os.getpid():
        _PROCESS_READER = ReadSession()
        _PROCESS_ID = os.getpid()
    return _PROCESS_READER


def configure_reader(workers=2, max_mappings=64):
    global _PROCESS_READER, _PROCESS_ID
    if workers < 1 or max_mappings < 1:
        raise ValueError("Reader limits must be positive")
    if _PROCESS_READER is None or _PROCESS_ID != os.getpid():
        _PROCESS_READER = ReadSession(max_mappings=max_mappings, workers=workers)
        _PROCESS_ID = os.getpid()
    else:
        # Runtime binding happens before producers start. Preserve mappings opened
        # during dataset initialization instead of reopening them for the first batch.
        _PROCESS_READER.max_mappings = max_mappings
        _PROCESS_READER._slots = BoundedSemaphore(workers)


def read_slice(value, index=Ellipsis):
    if isinstance(value, ArrayRef):
        return value.read(index)
    return process_reader().read_tensor(value, index)


def materialize_array(value):
    """Explicit full-array read for scientific APIs that require a tensor."""
    return value.read() if isinstance(value, ArrayRef) else value


def source_reference(tensor):
    ref = getattr(tensor, "_array_ref", None)
    if (
        ref is not None
        and not torch.is_inference(tensor)
        and tensor._version == getattr(tensor, "_array_version", -1)
    ):
        return ref
    return None


def write_array(path, tensor):
    """Reuse immutable prepared files; changed COW tensors get new payloads."""
    import errno
    import shutil

    path = Path(path)
    ref = source_reference(tensor)
    if ref is not None:
        ref.validate()
        try:
            os.link(ref.path, path)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(ref.path, path)
        return ref.numpy_dtype, ref.shape
    array = tensor.detach().cpu().contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return array.dtype.str, tuple(array.shape)


def spill_raster(raster, directory):
    """Release acquisition heap storage as soon as a worker finishes it."""
    from dataclasses import fields
    import tempfile
    from .loading import _restore_raster, _plain, load_array_tensor

    root = Path(tempfile.mkdtemp(prefix="acquisition-", dir=directory))
    count = 0

    def spill(value):
        nonlocal count
        if isinstance(value, torch.Tensor):
            if value.numel() <= 64:
                return value
            path = root / f"array-{count}.npy"
            count += 1
            write_array(path, value)
            return load_array_tensor(path)
        from collections.abc import Mapping

        if isinstance(value, Mapping):
            return {key: spill(item) for key, item in value.items()}
        return value

    return _restore_raster(
        type(raster),
        {
            field.name: _plain(spill(getattr(raster, field.name)))
            for field in fields(raster)
        },
    )


def read_flat_samples(value, indices):
    """Read selected native-grid pixels without materializing the grid."""
    if isinstance(value, ArrayRef):
        return value.read((indices // value.shape[1], indices % value.shape[1]))
    return value.reshape(-1, *value.shape[2:]).index_select(0, indices)
