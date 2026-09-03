"""Runtime tensor contract for LTE optimizer batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def validate_batch_tensors(
    batch: Mapping[str, Any],
    *,
    reference: torch.Tensor,
) -> None:
    """Require finite real tensors on the model's explicit device and dtype."""

    if not isinstance(batch, Mapping):
        raise TypeError("An LTE training batch must be a mapping.")

    finite_checks: list[tuple[str, torch.Tensor]] = []

    def validate(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.device != reference.device:
                raise ValueError(
                    f"{path} must be on {reference.device}; got {value.device}."
                )
            if value.is_complex():
                raise TypeError(f"{path} must be real-valued.")
            if value.is_floating_point():
                if value.dtype != reference.dtype:
                    raise TypeError(
                        f"{path} must use {reference.dtype}; got {value.dtype}."
                    )
                finite_checks.append((path, torch.isfinite(value).all()))
            return
        if isinstance(value, Mapping):
            for name, item in value.items():
                validate(item, f"{path}.{name}")
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                validate(item, f"{path}[{index}]")

    validate(batch, "batch")
    if (
        finite_checks
        and not torch.stack([is_finite for _, is_finite in finite_checks]).all()
    ):
        # The ordinary valid path synchronizes an accelerator only once. On
        # failure, identify the exact malformed field for an actionable error.
        for path, is_finite in finite_checks:
            if not is_finite:
                raise FloatingPointError(f"{path} contains non-finite values.")


__all__ = ["validate_batch_tensors"]
