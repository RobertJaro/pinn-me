"""Bottom-to-top Stokes-vector transport using layer midpoint coefficients."""

import torch

from .polarized_exponential import exponential_coefficients, propagation_matrix


def _portable_sweep(argument, weights, source):
    # Batch the small matrix products on CPU and expose native autograd for higher derivatives.
    generator = propagation_matrix(torch.cat((torch.zeros_like(argument[..., :1]), argument[..., 1:]), -1))
    c0, delta0, c1, c2, c3 = weights.unsqueeze(-1).unbind(-2)
    squared = generator @ generator
    off = (c1.unsqueeze(-1) * generator + c2.unsqueeze(-1) * squared
           + c3.unsqueeze(-1) * (squared @ generator))
    identity = torch.eye(4, dtype=argument.dtype, device=argument.device)
    layers = c0.unsqueeze(-1) * identity + off
    emission = -(off @ source.unsqueeze(-1)).squeeze(-1) - delta0 * source
    state = torch.zeros_like(source[..., 0, :, :])
    for layer, emitted in zip(reversed(layers.unbind(-4)), reversed(emission.unbind(-3))):
        state = (layer @ state.unsqueeze(-1)).squeeze(-1) + emitted
    return state


class _CudaSweep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, argument, weights, source):
        from .stokes_sweep_kernel import sweep_forward

        result, incident = sweep_forward(argument, weights, source, save=any(ctx.needs_input_grad))
        ctx.save_for_backward(argument, weights, source, incident)
        return result

    @staticmethod
    def backward(ctx, gradient):
        argument, weights, source, incident = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Separate the three partial derivatives even when weights depend on argument.
            inputs = [value.view_as(value) for value in (argument, weights, source)]
            active = [value for value in inputs if value.requires_grad]
            partials = iter(torch.autograd.grad(_portable_sweep(*inputs), active, gradient,
                                                create_graph=True))
            return tuple(next(partials) if value.requires_grad else None for value in inputs)
        from .stokes_sweep_kernel import sweep_backward

        return sweep_backward(argument, weights, source, incident, gradient)


def polarized_sweep(argument, midpoint_source, boundary):
    """Apply exp(-K ds) to one incident Stokes vector at each layer."""
    weights = torch.stack(exponential_coefficients(argument[..., 0], argument[..., 1:4],
                                                   argument[..., 4:]), -1)
    # Evolve deviations from the incident boundary to retain exact thermal equilibrium.
    source = midpoint_source - boundary.unsqueeze(-3)
    if argument.is_cuda and argument.numel():
        from torch.utils._triton import has_triton

        if has_triton():
            argument, source = argument.contiguous(), source.contiguous()
            if not torch.is_grad_enabled():
                from .stokes_sweep_kernel import sweep_forward

                result, _ = sweep_forward(argument, weights, source, save=False)
                return boundary + result
            return boundary + _CudaSweep.apply(argument, weights, source)
    return boundary + _portable_sweep(argument, weights, source)
