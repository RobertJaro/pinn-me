"""Device, evaluation-state, and serialization helpers for application services."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _move_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {name: _move_value(item, device) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {value}.")
    return device


@contextmanager
def _preserve_rng():
    """Keep diagnostics from changing training's RNG state, without reseeding."""
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        yield


def state_dict_sha256(module: nn.Module) -> str:
    """Hash tensors and primitive module extra state without a checkpoint."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            digest.update(name.encode("utf-8"))
            digest.update(json.dumps(_json_value(tensor), sort_keys=True, allow_nan=False).encode("utf-8"))
            continue
        value = tensor.detach()
        if value.is_sparse:
            value = value.to_dense()
        value = value.cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _temporary_eval_mode(module: nn.Module):
    states = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in states:
            child.training = training


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _json_value(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
