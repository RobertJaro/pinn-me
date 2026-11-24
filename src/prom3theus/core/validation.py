"""Batch tensor validity checks at a numerical pipeline boundary.

Public numerical functions validate immediately. A production forward can wrap
its already shape-checked computation in ``deferred_validation`` to collect all
device-side flags and inspect them together, without per-operation host waits.
The context stores flags only, never model states, and is reset on every exit.
"""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_PENDING = ContextVar('tensor_validation_pending', default=None)
_ENABLED = ContextVar("tensor_validation_enabled", default=True)


@contextmanager
def skip_tensor_validation():
    """Skip numerical predicates during an optimizer batch, including nested forwards."""
    token = _ENABLED.set(False)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def tensor_validation_enabled() -> bool:
    return _ENABLED.get()


def require_tensor(condition: Callable[[], torch.Tensor] | torch.Tensor, message: str, *, error=ValueError) -> None:
    """Evaluate a numerical check now and, when deferred, inspect its flag at exit.

    Pass a predicate for runtime checks so optimizer batches skip its GPU work.
    """
    if not tensor_validation_enabled():
        return
    if callable(condition):
        condition = condition()
    if condition.dtype != torch.bool or condition.numel() != 1:
        raise TypeError('Tensor validity conditions must be scalar Boolean tensors')
    pending = _PENDING.get()
    condition = condition.detach().reshape(())
    if pending is None:
        if not bool(condition):
            raise error(message)
    else:
        pending.append((condition, message, error))


@contextmanager
def deferred_validation():
    """Check accumulated flags once per device when the outer context completes.

    Numerical work may proceed before an invalid value is reported. Shape, dtype,
    indexing bounds and other checks needed to execute safely must remain eager.
    Nested contexts share their outer validation boundary.
    """
    if not tensor_validation_enabled() or _PENDING.get() is not None:
        yield
        return
    pending = []
    token = _PENDING.set(pending)
    try:
        yield
        groups = {}
        for flag, message, error in pending:
            groups.setdefault(flag.device, []).append((flag, message, error))
        for checks in groups.values():
            passed = torch.stack([flag for flag, _, _ in checks]).cpu().tolist()
            for valid, (_, message, error) in zip(passed, checks):
                if not valid:
                    raise error(message)
    finally:
        _PENDING.reset(token)
