from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.functional import linear


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
    def __init__(self, in_dim, out_dim, dim=256, n_layers=8, w0=1., encoding_config=None, skip_layers=[]):
        super().__init__()

        encoding_config = {'type': 'default', 'w0': 30.} if encoding_config is None else encoding_config
        encoding_type = encoding_config.pop('type', 'default')

        if encoding_type == "default":
            self.posenc = SirenLayer(in_dim=in_dim, out_dim=dim, is_first=True, **encoding_config)
            posenc_dim = dim
        elif encoding_type == "multi_spectral":
            self.posenc = MultispectralEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "multi_frequency":
            self.posenc = MultiFrequencyEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        elif encoding_type == "weighted":
            enc_dim = int(encoding_config.pop('num_dims', dim))
            self.posenc = WeightedEncoding(in_dim=in_dim, dim=enc_dim, **encoding_config)
            posenc_dim = self.posenc.out_dim
        else:
            raise ValueError(f"Unknown encoding: {encoding_type}")

        self.num_layers = n_layers
        self.dim_hidden = dim
        self.skip_layers = skip_layers

        # initialize the input layer
        self.in_layer = SirenLayer(in_dim=posenc_dim, out_dim=dim, w0=w0)

        # initialize the hidden layers
        layers = []
        for i in range(n_layers - 1):
            if i in self.skip_layers:
                # this layer will receive [h, skip_ref], width increases by posenc_dim
                in_d = dim + posenc_dim
            else:
                in_d = dim
            layer = SirenLayer(in_dim=in_d, out_dim=dim, w0=w0)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # initialize the output layer
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, inp):
        inp_encoded = self.posenc(inp)  # apply positional encoding
        x = self.in_layer(inp_encoded)

        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                x = torch.cat([x, inp_encoded], dim=-1)
                x = layer(x)  # layer expects dim + ref_dim
            else:
                x = layer(x)  # standard SIREN layer

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


class WeightedEncoding(nn.Module):

    def __init__(self, in_dim, dim=512, weights=30.0):
        super().__init__()

        weights = [weights] * in_dim if not isinstance(weights, Iterable) else weights
        assert len(weights) == in_dim, 'weights length must match input dimension (in_dim)'

        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.layer = SirenLayer(in_dim=in_dim, out_dim=dim, w0=1, is_first=True)

        self.out_dim = dim

    def forward(self, x):
        # apply weights to each coordinate
        x = x * self.weights  # shape: [batch_size, in_dim]
        x = self.layer(x)
        return x

class MultispectralEncoding(nn.Module):

    def __init__(self, in_dim, num_dims=64, weights=30.0):
        super().__init__()
        num_dims = [num_dims] * in_dim if not isinstance(num_dims, Iterable) else num_dims
        weights = [weights] * in_dim if not isinstance(weights, Iterable) else weights

        assert len(num_dims) == in_dim, 'num_dims length must match input dimension (in_dim)'
        assert len(weights) == in_dim, 'weights length must match input dimension (in_dim)'

        layers = []
        for nd, w in zip(num_dims, weights):
            l = SirenLayer(in_dim=1, out_dim=nd, w0=w, is_first=True)
            layers.append(l)
        self.layers = nn.ModuleList(layers)

        self.out_dim = sum(num_dims) + in_dim

    def forward(self, x):
        encoded_coordinates = []
        for i, layer in enumerate(self.layers):
            coord = x[..., i:i + 1]
            encoded = layer(coord)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates + [x], -1)
        return encoded


class MultiFrequencyEncoding(nn.Module):

    def __init__(self, in_dim, num_dims=32, weights=None):
        super().__init__()
        weights = [1, 10, 100, 1000] if weights is None else weights

        num_dims = [num_dims] * len(weights) if not isinstance(num_dims, Iterable) else num_dims

        assert len(num_dims) == len(weights), 'num_dims length must match weights length'

        layers = []
        for nd, w in zip(num_dims, weights):
            l = SirenLayer(in_dim=in_dim, out_dim=nd, w0=w, is_first=True)
            layers.append(l)
        self.layers = nn.ModuleList(layers)

        self.out_dim = sum(num_dims)

    def forward(self, x):
        encoded_coordinates = []
        for i, layer in enumerate(self.layers):
            encoded = layer(x)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates, -1)
        return encoded