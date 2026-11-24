"""One GPU kernel for an ordered transmission scan across a full ray batch."""

import math

import torch
from torch.utils._triton import has_triton_package


if has_triton_package():
    import triton
    import triton.language as tl


    @triton.jit
    def _scan(Layers, Gradients, Output, count, wavelengths,
              rows, layer_stride: tl.constexpr, grad_stride: tl.constexpr,
              out_stride: tl.constexpr, REVERSE: tl.constexpr, BLOCK: tl.constexpr):
        # Each lane owns one Stokes row at one wavelength of one ray.
        lane = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        ray = lane // (4 * wavelengths)
        wave = lane // 4 % wavelengths
        row = lane % 4
        col = tl.arange(0, 4)
        valid = lane < rows
        matrix = (ray * layer_stride[1] + wave * layer_stride[2])[:, None, None]
        matrix = (matrix + col[None, :, None] * layer_stride[3]
                  + col[None, None, :] * layer_stride[4])
        output = ((ray * out_stride[1] + wave * out_stride[2] + row * out_stride[3])[:, None]
                  + col[None, :] * out_stride[4])
        if REVERSE:
            gradient = ((ray * grad_stride[1] + wave * grad_stride[2] + row * grad_stride[3])[:, None]
                        + col[None, :] * grad_stride[4])
            state = tl.full((BLOCK, 4), 0, Layers.dtype.element_ty)
        else:
            state = (row[:, None] == col[None, :]).to(Layers.dtype.element_ty)

        # The only serial dimension is optical depth. State stays in registers.
        for step in range(count - 1):
            i = count - 1 - step if REVERSE else step
            tl.store(Output + i * out_stride[0] + output, state, valid[:, None])
            layer = tl.load(Layers + i * layer_stride[0] + matrix, valid[:, None, None], 0)
            if REVERSE:
                direct = tl.load(Gradients + i * grad_stride[0] + gradient, valid[:, None], 0)
                state = tl.sum(layer * state[:, None, :], axis=2) + direct
            else:
                state = tl.sum(state[:, :, None] * layer, axis=1)
        last = 0 if REVERSE else count - 1
        tl.store(Output + last * out_stride[0] + output, state, valid[:, None])


def ordered_products(layers, gradients=None):
    """Compute exclusive products or their reverse adjoints on the current CUDA stream."""
    count = layers.shape[0]
    waves = layers.shape[-3] if layers.ndim > 3 else 1
    rays = math.prod(layers.shape[1:-3])
    shape = (count, rays, waves, 4, 4)
    matrices = layers.reshape(shape)
    direct = matrices if gradients is None else gradients.reshape(shape)
    output = torch.empty_like(matrices)
    rows = rays * waves * 4
    block = 32
    with torch.cuda.device(layers.device):
        _scan[(triton.cdiv(rows, block),)](
            matrices, direct, output, count, waves, rows, matrices.stride(), direct.stride(),
            output.stride(), gradients is not None, block, num_warps=4,
        )
    return output.reshape(layers.shape)
