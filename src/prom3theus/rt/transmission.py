"""Observer-first products of the layer transmission matrices."""

from typing import Optional

import torch


@torch.jit.script
def _ordered_products(layers: torch.Tensor, gradients: Optional[torch.Tensor] = None):
    """Portable compiled recurrence, also differentiable for higher derivatives."""
    count = layers.size(0)
    if gradients is None:
        state = torch.eye(4, dtype=layers.dtype, device=layers.device).expand_as(layers[0])
    else:
        state = torch.zeros_like(layers[0])
    values = [state]
    for step in range(1, count):
        if gradients is None:
            state = state @ layers[step - 1]
        else:
            i = count - step
            state = state @ layers[i].transpose(-1, -2) + gradients[i]
        values.append(state)
    result = torch.stack(values)
    return result if gradients is None else result.flip([0])


class _CudaTransmission(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layers):
        from .transmission_kernel import ordered_products

        prefixes = ordered_products(layers)
        ctx.save_for_backward(layers, prefixes)
        return prefixes

    @staticmethod
    def backward(ctx, grad_prefixes):
        from .transmission_kernel import ordered_products

        layers, prefixes = ctx.saved_tensors
        # P[i+1] = P[i] @ A[i]. Accumulate its adjoint from the bottom up.
        # Record this recurrence in autograd when a higher derivative is requested.
        scan = _ordered_products if torch.is_grad_enabled() else ordered_products
        adjoints = scan(layers, grad_prefixes)
        depth_axis = max(0, layers.ndim - 4)
        p = prefixes.movedim(0, depth_axis)
        g = adjoints.movedim(0, depth_axis)
        return (p.transpose(-1, -2) @ g).movedim(depth_axis, 0)


def transmission_above(layers):
    """Return Id, A[0], A[0] @ A[1], ... along the leading layer axis."""
    if layers.shape[0] == 0:
        raise ValueError("Transmission scan requires at least one layer")
    if layers.is_cuda and layers.numel():
        from torch.utils._triton import has_triton

        if has_triton():
            return _CudaTransmission.apply(layers)
    return _ordered_products(layers)
