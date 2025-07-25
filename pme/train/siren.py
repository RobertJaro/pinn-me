import numpy as np
import torch
from torch import nn
from torch.nn import Identity
from torch.nn.functional import linear

from pme.encoding import GaussianPositionalEncoding, ProgressiveFourierEncoding, PositionalEncoding


class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=1., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = in_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenModel(nn.Module):
    def __init__(self, in_dim, out_dim, dim=512, n_layers=8,
                 w0=1., w0_initial=30., encoding='default', **kwargs):
        super().__init__()

        if encoding == "positional":
            self.posenc = PositionalEncoding(num_freqs=20, d_input=in_dim)
            posenc_dim = self.posenc.d_output
        elif encoding == "gaussian":
            self.posenc = GaussianPositionalEncoding(d_input=in_dim)
            posenc_dim = self.posenc.d_output
        elif encoding == "progressive_fourier":
            self.posenc = ProgressiveFourierEncoding(d_input=in_dim)
            posenc_dim = self.posenc.d_output
        elif encoding == "identity":
            self.posenc = Identity()
            posenc_dim = in_dim
        elif encoding == "default":
            self.posenc = SirenLayer(in_dim=in_dim, out_dim=dim, w0=w0_initial, is_first=True)
            posenc_dim = dim
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        self.num_layers = n_layers
        self.dim_hidden = dim

        # initialize the input layer
        self.in_layer = SirenLayer(in_dim=posenc_dim, out_dim=dim, w0=w0)

        # initialize the hidden layers
        layers = []
        for i in range(n_layers - 1):
            layer = SirenLayer(in_dim=dim, out_dim=dim, w0=w0)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        # initialize the output layer
        self.out_layer = SirenLayer(in_dim=dim, out_dim=out_dim, w0=w0, activation=nn.Identity())

    def forward(self, x):
        x = self.posenc(x)  # apply positional encoding
        x = self.in_layer(x)
        x = self.layers(x)
        x = self.out_layer(x)
        return x

    def step(self, global_step):
        if hasattr(self.posenc, 'step'):
            self.posenc.step(global_step)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
