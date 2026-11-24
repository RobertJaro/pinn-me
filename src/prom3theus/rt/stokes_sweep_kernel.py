"""Fused polarized layer recurrence and its reverse adjoint on CUDA."""

import torch
from torch.utils._triton import has_triton_package


if has_triton_package():
    import triton
    import triton.language as tl


    @triton.jit
    def _load4(ptr, mask):
        return (tl.load(ptr, mask, 0), tl.load(ptr + 1, mask, 0),
                tl.load(ptr + 2, mask, 0), tl.load(ptr + 3, mask, 0))


    @triton.jit
    def _store4(ptr, vector, mask):
        for j in tl.static_range(4):
            tl.store(ptr + j, vector[j], mask)


    @triton.jit
    def _combine(a, x, b, y):
        return (a*x[0] + b*y[0], a*x[1] + b*y[1], a*x[2] + b*y[2], a*x[3] + b*y[3])


    @triton.jit
    def _dot(x, y):
        return x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3]


    @triton.jit
    def _generator(dq, du, dv, rq, ru, rv, v):
        # G(I,p) = (d.p, I d + p x rho). Transposing G reverses rho.
        i, q, u, s = v
        return (dq*q + du*u + dv*s, dq*i + u*rv - s*ru,
                du*i + s*rq - q*rv, dv*i + q*ru - u*rq)


    @triton.jit
    def _generator_gradient(left, right):
        # Partial derivatives of left^T G right with respect to d and rho.
        a, b, c, d = left
        i, q, u, v = right
        return (a*q + b*i, a*u + c*i, a*v + d*i, c*v - d*u, d*q - b*v, b*u - c*q)


    @triton.jit
    def _sweep(K, Weights, Source, Incident, Output, Upstream, DK, DW, DS,
               count, waves, lanes, BACKWARD: tl.constexpr, SAVE: tl.constexpr, BLOCK: tl.constexpr):
        lane = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        valid = lane < lanes
        base = (lane // waves) * count * waves + lane % waves
        zero = tl.full((BLOCK,), 0, K.dtype.element_ty)
        if BACKWARD:
            si, sq, su, sv = _load4(Upstream + lane*4, valid)
        else:
            si, sq, su, sv = zero, zero, zero, zero
        for step in range(count):
            state = (si, sq, su, sv)
            depth = step if BACKWARD else count - 1 - step
            layer = base + depth*waves
            k = K + layer*7
            dq = tl.load(k + 1, valid, 0)
            du = tl.load(k + 2, valid, 0)
            dv = tl.load(k + 3, valid, 0)
            rq = tl.load(k + 4, valid, 0)
            ru = tl.load(k + 5, valid, 0)
            rv = tl.load(k + 6, valid, 0)
            w = Weights + layer*5
            c0, delta0, c1, c2 = _load4(w, valid)
            c3 = tl.load(w + 4, valid, 0)
            source = _load4(Source + layer*4, valid)
            if BACKWARD:
                incoming = _load4(Incident + layer*4, valid)
            else:
                incoming = state
                if SAVE:
                    _store4(Incident + layer*4, incoming, valid)
            v = _combine(1., incoming, -1., source)
            gv = _generator(dq, du, dv, rq, ru, rv, v)
            ggv = _generator(dq, du, dv, rq, ru, rv, gv)
            h = _combine(c1, v, 1., _combine(c2, gv, c3, ggv))
            if BACKWARD:
                # Differentiate the cubic G(c1 v + c2 Gv + c3 G²v) and
                # propagate the outgoing adjoint through the transposed layer.
                gt = _generator(dq, du, dv, -rq, -ru, -rv, state)
                ggt = _generator(dq, du, dv, -rq, -ru, -rv, gt)
                grad_w = DW + layer*5
                _store4(grad_w, (_dot(state, incoming), -_dot(state, source),
                                 _dot(gt, v), _dot(gt, gv)), valid)
                tl.store(grad_w + 4, _dot(gt, ggv), valid)
                outer = _generator_gradient(state, h)
                middle = _generator_gradient(gt, _combine(c2, v, c3, gv))
                inner = _generator_gradient(ggt, v)
                grad_k = DK + layer*7
                tl.store(grad_k, zero, valid)
                for j in tl.static_range(6):
                    tl.store(grad_k + 1 + j, outer[j] + middle[j] + c3*inner[j], valid)
                ht = _combine(c1, state, 1., _combine(c2, gt, c3, ggt))
                nonidentity = _generator(dq, du, dv, -rq, -ru, -rv, ht)
                _store4(DS + layer*4, _combine(-delta0, state, -1., nonidentity), valid)
                si, sq, su, sv = _combine(c0, state, 1., nonidentity)
            else:
                nonidentity = _generator(dq, du, dv, rq, ru, rv, h)
                si, sq, su, sv = _combine(1., _combine(c0, incoming, -delta0, source), 1., nonidentity)
        if not BACKWARD:
            _store4(Output + lane*4, (si, sq, su, sv), valid)


def sweep_forward(argument, weights, source, *, save):
    count, waves = argument.shape[-3:-1]
    output = source.new_empty((*source.shape[:-3], waves, 4))
    incident = torch.empty_like(source) if save else output
    lanes = output.numel() // 4
    with torch.cuda.device(argument.device):
        _sweep[(triton.cdiv(lanes, 64),)](
            argument, weights, source, incident, output, output, output, output, output,
            count, waves, lanes, BACKWARD=False, SAVE=save, BLOCK=64)
    return output, incident


def sweep_backward(argument, weights, source, incident, gradient):
    partials = tuple(torch.empty_like(value) for value in (argument, weights, source))
    count, waves = argument.shape[-3:-1]
    lanes = gradient.numel() // 4
    with torch.cuda.device(argument.device):
        _sweep[(triton.cdiv(lanes, 64),)](
            argument, weights, source, incident, gradient, gradient.contiguous(), *partials,
            count, waves, lanes, BACKWARD=True, SAVE=False, BLOCK=64)
    return partials
