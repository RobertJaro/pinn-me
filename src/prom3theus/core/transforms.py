"""Shared dimensionless transforms for objectives and diagnostic displays."""

import torch


def normalized_asinh(value: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """Odd asinh stretch mapping unit input to unit output.

    Callers validate positive finite scales. Scalar or per-channel scales
    broadcast to values in the same normalized intensity units. Arithmetic
    stays in the input tensor's dtype and device.
    """
    scale = torch.as_tensor(scale, dtype=value.dtype, device=value.device)
    return torch.asinh(value / scale) / torch.asinh(scale.reciprocal())
