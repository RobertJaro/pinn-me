"""Portable prepared state: importable records, array references and small constants."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .arrays import ArrayRef


@dataclass(frozen=True)
class TensorValue:
    values: object
    dtype: torch.dtype


@dataclass(frozen=True)
class ObjectRecord:
    module: str
    name: str
    state: dict


@dataclass
class MetadataSpec:
    values: dict

    def metadata(self):
        return self.values


@dataclass
class BatchSequence(Sequence):
    batches: tuple

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return materialize(self.batches[index])


@dataclass
class BatchDataModule:
    training: BatchSequence | None
    validation: tuple[BatchSequence, ...]
    metadata: dict

    def setup(self, stage=None):
        pass

    def run_metadata(self):
        return self.metadata

    def train_dataloader(self):
        if self.training is None:
            raise ValueError("Provider did not supply training batches")
        return self.training

    def val_dataloader(self):
        return list(self.validation)


@dataclass
class SamplingSupport:
    blocks: tuple

    def __call__(self, scene, chunk_size):
        for positions, times in self.blocks:
            for start in range(0, len(positions), chunk_size):
                yield (
                    materialize(positions[start : start + chunk_size]),
                    materialize(times[start : start + chunk_size]),
                )


def materialize(value):
    if isinstance(value, ArrayRef):
        return value.read()
    if isinstance(value, TensorValue):
        return torch.tensor(value.values, dtype=value.dtype)
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(materialize(item) for item in value)
    return value


class SnapshotWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.memo = {}
        self.sources = {}
        self.counter = 0

    def encode_batch(self, value):
        """Serialize a transient provider batch without retaining its payload."""
        batch_writer = SnapshotWriter(self.directory)
        batch_writer.counter = self.counter
        result = batch_writer.encode(value, payload=True)
        self.counter = batch_writer.counter
        return result

    def encode(self, value, *, payload=False):
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
                Path,
                torch.dtype,
                Enum,
                ArrayRef,
                TensorValue,
            ),
        ):
            return value
        if id(value) in self.memo:
            return self.memo[id(value)]
        if inspect.isroutine(value) or isinstance(value, type):
            raise TypeError(
                "Provider callbacks cannot be persisted; supply prepared descriptors"
            )
        self.sources[id(value)] = value
        if isinstance(value, torch.Tensor):
            if not payload and value.numel() <= 4096:
                result = TensorValue(value.detach().cpu().tolist(), value.dtype)
            else:
                from .arrays import source_reference

                result = source_reference(value)
                if result is None:
                    path = self.directory / f"array_{self.counter:06d}.npy"
                    self.counter += 1
                    np.save(
                        path,
                        value.detach().cpu().contiguous().numpy(),
                        allow_pickle=False,
                    )
                    result = ArrayRef.from_path(path)
        elif isinstance(value, Mapping):
            result = {
                key: self.encode(item, payload=payload) for key, item in value.items()
            }
        elif isinstance(value, (tuple, list)):
            result = type(value)(self.encode(item, payload=payload) for item in value)
        elif is_dataclass(value):
            raster = type(value).__name__ in (
                "ObservationRaster",
                "ImageObservationRaster",
            )
            result = ObjectRecord(
                type(value).__module__,
                type(value).__qualname__,
                {
                    field.name: self.encode(
                        getattr(value, field.name),
                        payload=raster and field.name != "wavelength_angstrom",
                    )
                    for field in fields(value)
                },
            )
        elif hasattr(value, "__dict__") and "<locals>" not in type(value).__qualname__:
            from torch.utils.data import Dataset, Subset

            payload = payload or isinstance(value, Dataset)
            state = dict(vars(value))
            if isinstance(value, Subset) and not isinstance(value.indices, ArrayRef):
                state["indices"] = torch.as_tensor(value.indices, dtype=torch.long)
            if isinstance(value, SimpleNamespace) and callable(state.get("metadata")):
                state = {key: item for key, item in state.items() if not callable(item)}
                result = ObjectRecord(
                    __name__, "MetadataSpec", {"values": self.encode(value.metadata())}
                )
                # Specifications may also expose fields used by term builders.
                result.state.update(
                    {key: self.encode(item) for key, item in state.items()}
                )
            else:
                result = ObjectRecord(
                    type(value).__module__,
                    type(value).__qualname__,
                    {
                        key: self.encode(item, payload=payload)
                        for key, item in state.items()
                    },
                )
        else:
            raise TypeError(
                f"Prepared state requires a portable descriptor, got {type(value).__qualname__}"
            )
        self.memo[id(value)] = result
        return result


def restore(value, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, TensorValue):
        result = torch.tensor(value.values, dtype=value.dtype)
    elif isinstance(value, ObjectRecord):
        cls = importlib.import_module(value.module)
        for name in value.name.split("."):
            cls = getattr(cls, name)
        result = cls.__new__(cls)
        for key, item in value.state.items():
            object.__setattr__(result, key, restore(item, memo))
        if hasattr(result, "initialize_reader"):
            result.initialize_reader()
    elif isinstance(value, dict):
        result = {key: restore(item, memo) for key, item in value.items()}
    elif isinstance(value, (tuple, list)):
        result = type(value)(restore(item, memo) for item in value)
    else:
        result = value
    memo[id(value)] = result
    return result


def array_references(value):
    seen = set()

    def walk(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, ArrayRef):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from walk(item)
        elif is_dataclass(value):
            for field in fields(value):
                yield from walk(getattr(value, field.name))

    return walk(value)
