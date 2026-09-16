"""Deterministic validation batch selection and structured batch operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


def _validation_loaders(data_module: Any) -> list[Iterable[Mapping[str, Any]]]:
    image_loaders = getattr(data_module, "validation_dataloaders", None)
    if callable(image_loaders):
        loaders = image_loaders()
        if isinstance(loaders, Mapping):
            return list(loaders.values())
    loader = data_module.val_dataloader()
    if isinstance(loader, (list, tuple)):
        if loader and isinstance(loader[0], Mapping):
            return [loader]
        return list(loader)
    return [loader]


def _batch_length(batch: Mapping[str, Any]) -> int:
    def leading_lengths(value: Any) -> set[int]:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return {int(value.shape[0])}
        if isinstance(value, Mapping):
            result: set[int] = set()
            for item in value.values():
                result.update(leading_lengths(item))
            return result
        return set()

    lengths: set[int] = set()
    for value in batch.values():
        lengths.update(leading_lengths(value))
    if not lengths:
        raise ValueError("A validation batch contains no sample-shaped tensor.")
    if len(lengths) != 1:
        raise ValueError(
            f"Validation batch fields disagree in length: {sorted(lengths)}."
        )
    return lengths.pop()


def _slice_batch(batch: Mapping[str, Any], count: int) -> dict[str, Any]:
    length = _batch_length(batch)
    count = min(int(count), length)

    def sliced(value: Any) -> Any:
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == length
        ):
            return value[:count]
        if isinstance(value, Mapping):
            return {name: sliced(item) for name, item in value.items()}
        return value

    return {name: sliced(value) for name, value in batch.items()}


def _concatenate_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("Cannot concatenate an empty validation batch sequence.")
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches[1:]):
        raise ValueError("Validation loader batches must expose identical fields.")
    lengths = [_batch_length(batch) for batch in batches]

    def combined_value(values: Sequence[Any], name: str) -> Any:
        if all(
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == length
            for value, length in zip(values, lengths, strict=True)
        ):
            return torch.cat(values, dim=0)
        if all(isinstance(value, Mapping) for value in values):
            nested_keys = set(values[0])
            if any(set(value) != nested_keys for value in values[1:]):
                raise ValueError(f"Nested batch field {name!r} has inconsistent keys.")
            return {
                key: combined_value([value[key] for value in values], f"{name}.{key}")
                for key in values[0]
            }
        first = values[0]
        if isinstance(first, torch.Tensor):
            consistent = all(
                isinstance(value, torch.Tensor) and torch.equal(value, first)
                for value in values[1:]
            )
        else:
            consistent = all(value == first for value in values[1:])
        if not consistent:
            raise ValueError(f"Non-sample batch field {name!r} is inconsistent.")
        return first

    return {
        name: combined_value([batch[name] for batch in batches], name)
        for name in batches[0]
    }


def _collect_loader(loader: Iterable[Mapping[str, Any]], limit: int) -> dict[str, Any]:
    parts = []
    remaining = int(limit)
    for batch in loader:
        if not isinstance(batch, Mapping):
            raise TypeError("Validation loaders must return mapping batches.")
        if remaining <= 0:
            break
        part = _slice_batch(batch, remaining)
        parts.append(part)
        remaining -= _batch_length(part)
        if remaining <= 0:
            break
    if not parts:
        raise ValueError("A configured validation loader produced no samples.")
    return _concatenate_batches(parts)


def deterministic_validation_batch(
    data_module: Any, *, max_samples: int
) -> dict[str, Any]:
    """Collect a stable, bounded, channel-balanced validation batch."""

    loaders = _validation_loaders(data_module)
    if max_samples < len(loaders):
        raise ValueError(
            "max_samples_per_stream must include at least one sample from every "
            "validation loader."
        )
    base, extra = divmod(max_samples, len(loaders))
    pieces = [
        _collect_loader(loader, base + (index < extra))
        for index, loader in enumerate(loaders)
    ]
    return _concatenate_batches(pieces)


def _indexed_batch(batch: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    length = _batch_length(batch)

    def indexed(value: Any) -> Any:
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == length
        ):
            return value.index_select(0, indices.to(value.device))
        if isinstance(value, Mapping):
            return {name: indexed(item) for name, item in value.items()}
        return value

    return {name: indexed(value) for name, value in batch.items()}
