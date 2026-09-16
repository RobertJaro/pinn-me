"""Ordered, bounded observation reads. Payload indexing happens only in RAM."""

import math
from bisect import bisect_right
from collections import OrderedDict
from queue import Full, Queue
from threading import Event, Thread

import torch
from torch.utils.data import ConcatDataset, Subset

def configure_readers(count):
    if type(count) is not int or count < 1:
        raise ValueError("reader_workers must be positive")
    from .arrays import configure_reader
    configure_reader(workers=count)


class PixelCatalog:
    """Small row-block prefix index; no full-corpus pixel coordinates."""

    version = 3

    def __init__(self, mask, rows=256):
        rows = min(rows, max(1, 65536 // mask.shape[1]))
        self.mask, self.rows = mask, rows
        counts = torch.empty(mask.shape[0], dtype=torch.long)
        for start in range(0, mask.shape[0], rows):
            counts[start : start + rows] = mask[start : start + rows].clone().sum(dim=1)
        self.row_prefix = torch.cat((counts.new_zeros(1), counts.cumsum(0))).tolist()
        self.prefix = self.row_prefix[::rows]
        if mask.shape[0] % rows:
            self.prefix.append(self.row_prefix[-1])

    def __len__(self):
        return self.prefix[-1]

    def pixels(self, start, stop):
        parts = []
        while start < stop:
            block = bisect_right(self.prefix, start) - 1
            row = block * self.rows
            pixels = torch.nonzero(self.mask[row : row + self.rows].clone())
            pixels[:, 0] += row
            take = min(stop, self.prefix[block + 1])
            parts.append(pixels[start - self.prefix[block] : take - self.prefix[block]])
            start = take
        return torch.cat(parts) if parts else torch.empty((0, 2), dtype=torch.long)


def concatenate(parts):
    if len(parts) == 1:
        return parts[0]
    return {
        key: (
            concatenate([part[key] for part in parts])
            if isinstance(parts[0][key], dict)
            else torch.cat([part[key] for part in parts], dim=0)
        )
        for key in parts[0]
    }


def slice_batch(batch, start, stop):
    return {
        key: slice_batch(value, start, stop)
        if isinstance(value, dict)
        else value[start:stop]
        for key, value in batch.items()
    }


def gather_batch(batch, indices):
    return {
        key: gather_batch(value, indices)
        if isinstance(value, dict)
        else value.index_select(0, indices)
        for key, value in batch.items()
    }


def tensor_bytes(value):
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return value.numel() * value.element_size()


def supported(dataset):
    if isinstance(dataset, Subset):
        return supported(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return all(supported(child) for child in dataset.datasets)
    return hasattr(dataset, "bulk_fields")


def chronological_key(dataset):
    raster = dataset.raster
    if hasattr(raster, "absolute_tai_seconds"):
        time = float(raster.absolute_tai_seconds)
    else:
        time = math.inf
        for row in range(0, raster.spatial_shape[0], 256):
            valid = raster.valid_mask[row : row + 256].clone()
            if valid.any():
                coordinates = raster.coordinates[row : row + 256].clone()
                time = min(time, float(coordinates[..., 2][valid].amin()))
    return time, getattr(dataset, "sequence_name", "")


def sample_bytes(dataset):
    if isinstance(dataset, Subset):
        return sample_bytes(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return max(sample_bytes(child) for child in dataset.datasets)
    # Field slices only inspect shapes; no payload access. Include generated IDs
    # and scalar time/channel metadata conservatively in the admission bound.
    return (
        sum(
            math.prod(value.shape[2:]) * value.element_size()
            for value in dataset.bulk_fields().values()
        )
        + 64
    )


class BulkReader:
    """One byte-limited LRU of owned row slabs, shared across channel reads."""

    def __init__(self, block_bytes=32 * 1024**2, cache_bytes=128 * 1024**2):
        if block_bytes < 1 or cache_bytes < block_bytes:
            raise ValueError("Require 0 < block_bytes <= cache_bytes")
        self.block_bytes, self.cache_bytes = block_bytes, cache_bytes
        self.cache = OrderedDict()
        self.pool = 0
        self.pool_keys = {}
        self.resident_bytes = 0
        self.read_count = self.read_bytes = 0

    def _block(self, dataset, row):
        fields = dataset.bulk_fields()
        width = dataset.raster.spatial_shape[1]
        row_bytes = (
            sum(math.prod(value.shape[1:]) * value.element_size() for value in fields.values())
            + 64 * width
        )
        rows = max(1, self.block_bytes // row_bytes)
        start = row // rows * rows
        stop = min(start + rows, dataset.raster.spatial_shape[0])
        key = (id(dataset), start)
        if key in self.cache:
            self.cache.move_to_end(key)
            return start, self.cache[key][0]
        previous = self.pool_keys.get(self.pool)
        if previous in self.cache:
            _, previous_size = self.cache.pop(previous)
            self.resident_bytes -= previous_size
        self.pool_keys[self.pool] = key
        # Read every field contiguously, then compact and normalize the entire
        # resident slab once. Batches subsequently take views, not fresh gathers.
        from .arrays import read_slice
        raw = {
            name: read_slice(value, slice(start, stop))
            for name, value in fields.items()
        }
        mask = read_slice(dataset.raster.valid_mask, slice(start, stop))
        linear = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
        values = {
            name: value.reshape(-1, *value.shape[2:]).index_select(0, linear)
            for name, value in raw.items()
        }
        pixels = (
            torch.stack((linear // width + start, linear % width), dim=-1)
            if dataset.include_pixel_index
            else linear
        )
        values = dataset.finish_bulk(values, pixels)
        size = tensor_bytes(values) + linear.numel() * linear.element_size()
        if size > self.cache_bytes:
            raise ValueError(
                "A native row exceeds cache_bytes; increase the loader budget"
            )
        while self.cache and self.resident_bytes + size > self.cache_bytes:
            _, (_, old_size) = self.cache.popitem(last=False)
            self.resident_bytes -= old_size
        self.read_count += 1
        self.read_bytes += sum(
            value.numel() * value.element_size() for value in raw.values()
        )
        entry = (stop, linear, values)
        self.cache[key] = (entry, size)
        self.resident_bytes += size
        return start, entry

    def _leaf(self, dataset, indices):
        if isinstance(indices, slice) and dataset._pixel_indices is None:
            start, stop = indices.start, indices.stop
            prefix = dataset._pixel_catalog.row_prefix
            parts = []
            while start < stop:
                row = bisect_right(prefix, start) - 1
                row_start, (row_stop, _, values) = self._block(dataset, row)
                end = min(stop, prefix[row_stop])
                parts.append(
                    slice_batch(
                        values, start - prefix[row_start], end - prefix[row_start]
                    )
                )
                start = end
            return concatenate(parts)
        # Sparse diagnostics retain native identity but gather only from the
        # packed slab. This path is never used for dense training batches.
        pixels = dataset.pixels_at(indices)
        linear = pixels[:, 0] * dataset.raster.spatial_shape[1] + pixels[:, 1]
        if len(linear) > 1 and torch.any(linear[1:] <= linear[:-1]):
            raise ValueError(
                "Bulk pixel selections must follow strictly increasing native order"
            )
        pixel_rows = pixels[:, 0].contiguous()
        parts = []
        offset = 0
        while offset < len(pixels):
            row_start, (row_stop, valid_linear, values) = self._block(
                dataset, int(pixels[offset, 0])
            )
            end = int(torch.searchsorted(pixel_rows, row_stop))
            local_linear = (
                linear[offset:end] - row_start * dataset.raster.spatial_shape[1]
            )
            selected = torch.searchsorted(valid_linear, local_linear)
            parts.append(gather_batch(values, selected))
            offset = end
        return concatenate(parts)

    def read(self, dataset, start, stop):
        if not 0 <= start < stop <= len(dataset):
            raise IndexError("Invalid bulk range")
        if isinstance(dataset, ConcatDataset):
            parts = []
            while start < stop:
                index = bisect_right(dataset.cumulative_sizes, start)
                base = dataset.cumulative_sizes[index - 1] if index else 0
                end = min(stop, dataset.cumulative_sizes[index])
                parts.append(
                    self.read(dataset.datasets[index], start - base, end - base)
                )
                start = end
            return concatenate(parts)
        if isinstance(dataset, Subset):
            indices = torch.as_tensor(dataset.indices[start:stop], dtype=torch.long)
            if len(indices) > 1 and torch.any(indices[1:] <= indices[:-1]):
                raise ValueError(
                    "Bulk diagnostic selections must be strictly increasing"
                )
            if isinstance(dataset.dataset, (Subset, ConcatDataset)):
                raise TypeError("Bulk subsets must reference a native raster dataset")
            return self._leaf(dataset.dataset, indices)
        return self._leaf(dataset, slice(start, stop))


class _Producer:
    """Cancelable bounded prefetch; no live raster serialization or CUDA workers."""

    def __init__(self, source, count):
        self.source = source
        self.queue = Queue(maxsize=count)
        self.stop = Event()
        self.thread = Thread(
            target=self._run, daemon=True, name="observation-bulk-reader"
        )
        self.thread.start()

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return
            except Full:
                pass

    @torch.no_grad()
    def _run(self):
        try:
            for value in self.source:
                if self.stop.is_set():
                    break
                self._put((True, value))
        except BaseException as error:  # noqa: BLE001 - propagate producer failures to its consumer
            self._put((False, error))
        finally:
            self._put((False, StopIteration()))
            self.source.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop.is_set():
            raise StopIteration
        ok, value = self.queue.get()
        if not ok:
            self.close()
            raise value
        return value

    def close(self):
        self.stop.set()
        self.thread.join()  # Reader executes bounded slabs; never waits on consumer.


class BulkBatchLoader:
    """Shared source metadata, bounded prefetch, and batch-reader lifecycle."""

    def __init__(
        self,
        dataset,
        batch_size,
        *,
        channels=None,
        pin_memory=False,
        block_bytes=32 * 1024**2,
        cache_bytes=128 * 1024**2,
        prefetch_batches=2,
        max_batch_bytes=64 * 1024**2,
    ):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not channels and isinstance(dataset, ConcatDataset):
            dataset = ConcatDataset(sorted(dataset.datasets, key=chronological_key))
        self.dataset, self.batch_size = dataset, batch_size
        self.channels = tuple(channels or ())
        if self.channels and batch_size < len(self.channels):
            raise ValueError("batch_size must include every channel")
        if not supported(dataset) or len(dataset) < 1:
            raise ValueError("Bulk loading requires nonempty raster datasets")
        if min(block_bytes, cache_bytes, prefetch_batches, max_batch_bytes) < 1:
            raise ValueError("Bulk loader budgets must be positive")
        if block_bytes > cache_bytes:
            raise ValueError("block_bytes must not exceed cache_bytes")
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.block_bytes, self.cache_bytes = block_bytes, cache_bytes
        self.prefetch_batches, self.max_batch_bytes = prefetch_batches, max_batch_bytes
        self._active = []
        self._reader = None

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def contract(self):
        def describe(dataset):
            if isinstance(dataset, ConcatDataset):
                return [describe(child) for child in dataset.datasets]
            if isinstance(dataset, Subset):
                return {
                    "source": describe(dataset.dataset),
                    "selection": list(dataset.indices),
                }
            return {
                "name": getattr(dataset, "sequence_name", ""),
                "count": len(dataset),
                "shape": list(dataset.raster.spatial_shape),
            }

        return {
            "version": 1,
            "batch_size": self.batch_size,
            "sequences": [describe(x) for x in self.channels]
            or [describe(self.dataset)],
        }

    def iter_from(self, start, count):
        for active in self._active:
            active.close()
        producer = _Producer(self._batches(start, count), self.prefetch_batches)
        self._active = [item for item in self._active if item.thread.is_alive()]
        self._active.append(producer)
        return producer

    def __iter__(self):
        producer = self.iter_from(0, len(self))
        try:
            yield from producer
        finally:
            producer.close()

    def close(self):
        for producer in self._active:
            producer.close()
        self._active.clear()
        self._reader = None


class SequentialBulkLoader(BulkBatchLoader):
    """Sequential native reads for preparation and image diagnostics."""

    def _batches(self, start, count):
        if self.channels:
            raise ValueError("Channel-balanced training uses PersistentTensorLoader")
        estimated = min(self.batch_size, len(self.dataset)) * sample_bytes(self.dataset)
        if estimated > self.max_batch_bytes:
            raise ValueError(
                "Batch exceeds max_batch_bytes; increase the loader budget"
            )
        if self._reader is None:
            self._reader = BulkReader(self.block_bytes, self.cache_bytes)
        for number in range(start, start + count):
            begin = (number % len(self)) * self.batch_size
            batch = self._reader.read(
                self.dataset, begin, min(begin + self.batch_size, len(self.dataset))
            )
            if tensor_bytes(batch) > self.max_batch_bytes:
                raise ValueError(
                    "Batch exceeds max_batch_bytes; increase the loader budget"
                )
            if self.pin_memory:
                from torch.utils.data._utils.pin_memory import pin_memory

                batch = pin_memory(batch)
            yield batch


def grid_indices(dataset, rows, columns):
    """Resolve a bounded diagnostic grid without a full native-pixel table."""
    rows, columns = torch.as_tensor(rows), torch.as_tensor(columns)
    if not hasattr(dataset, "_pixel_catalog") or dataset._pixel_indices is not None:
        pixels = dataset.pixel_indices
        selected = torch.isin(pixels[:, 0], rows) & torch.isin(pixels[:, 1], columns)
        return torch.nonzero(selected, as_tuple=True)[0]
    catalog = dataset._pixel_catalog
    parts = []
    for block, base in enumerate(catalog.prefix[:-1]):
        start, end = (
            block * catalog.rows,
            min((block + 1) * catalog.rows, catalog.mask.shape[0]),
        )
        if not torch.any((rows >= start) & (rows < end)):
            continue
        pixels = catalog.pixels(base, catalog.prefix[block + 1])
        selected = torch.isin(pixels[:, 0], rows) & torch.isin(pixels[:, 1], columns)
        parts.append(torch.nonzero(selected, as_tuple=True)[0] + base)
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)


def read_native_samples(array, pixels, block_bytes=32 * 1024**2):
    """Read sparse diagnostic samples through ordered, owned row slabs."""
    pixels = torch.as_tensor(pixels, dtype=torch.long, device="cpu")
    row_bytes = math.prod(array.shape[1:]) * array.element_size()
    rows = max(1, block_bytes // max(1, row_bytes))
    output = torch.empty(
        (len(pixels), *array.shape[2:]), dtype=array.dtype, device=array.device
    )
    for block in torch.unique(pixels[:, 0] // rows, sorted=True):
        start = int(block) * rows
        selected = (pixels[:, 0] >= start) & (pixels[:, 0] < start + rows)
        slab = array[start : start + rows].clone(memory_format=torch.contiguous_format)
        output[selected] = slab[pixels[selected, 0] - start, pixels[selected, 1]]
    return output


def read_native_grid(array, rows, columns):
    row, column = torch.meshgrid(
        torch.as_tensor(rows), torch.as_tensor(columns), indexing="ij"
    )
    pixels = torch.stack((row.reshape(-1), column.reshape(-1)), dim=-1)
    return read_native_samples(array, pixels).reshape(
        len(rows), len(columns), *array.shape[2:]
    )
