import math

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dim: int = 256,
        n_layers: int = 8,
        input_scale: torch.Tensor | list[float] | None = None,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        bias: bool = True,
        initializer: str = "siren",
        c: float = 6,
        **_,
    ):
        super().__init__()
        if input_scale is None:
            self.input_scale = None
        else:
            input_scale = torch.as_tensor(input_scale, dtype=torch.float32).view(1, -1)
            if input_scale.shape[-1] != in_dim:
                raise ValueError(
                    f"input_scale must have length {in_dim}, got {input_scale.shape[-1]}"
                )
            self.register_buffer("input_scale", input_scale)
        layers = [nn.Linear(in_dim, dim, bias=bias), Sine(w0=w0_initial)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(dim, dim, bias=bias), Sine(w0=w0)])
        layers.append(nn.Linear(dim, out_dim, bias=bias))
        self.network = nn.Sequential(*layers)

        if initializer == "siren":
            for module in self.network.modules():
                if isinstance(module, nn.Linear):
                    siren_uniform_(module.weight, mode="fan_in", c=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is not None:
            x = x * self.input_scale
        return self.network(x)


def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):
    r"""Fills the input `Tensor` with values according to the method
    described in ` Implicit Neural Representations with Periodic Activation
    Functions.` - Sitzmann, Martel et al. (2020), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{\text{fan\_mode}}}
    Also known as Siren initialization.
    """
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / math.sqrt(fan)
    bound = math.sqrt(c) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
