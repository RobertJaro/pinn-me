import numpy as np
import torch
from torch import nn


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class PeriodicBoundary(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, coord):
        scaled_x = coord[..., 0:1] * torch.pi  # - pi to pi
        scaled_y = coord[..., 1:2] * torch.pi  # - pi to pi
        encoded_coord = torch.cat([
            torch.sin(scaled_x), torch.cos(scaled_x),
            torch.sin(scaled_y), torch.cos(scaled_y),
            coord[..., 2:]], -1)
        return encoded_coord


class MEModel(nn.Module):

    def __init__(self, in_coords, dim, encoding='positional', num_layers=8):
        super().__init__()
        # encoding layer
        if encoding == "periodic":
            posenc = PeriodicBoundary()
            d_in = nn.Linear(in_coords + 2, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "positional":
            posenc = PositionalEncoding(num_freqs=20, d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "learned_positional":
            posenc = LearnedPositionalEncoding(num_freqs=20, d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "gaussian_positional":
            posenc = GaussianPositionalEncoding(num_freqs=20, d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "linear":
            self.d_in = nn.Linear(in_coords, dim)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # hidden layers
        lin = [nn.Linear(dim, dim) for _ in range(num_layers)]
        self.linear_layers = nn.ModuleList(lin)

        # output layer
        self.d_out = nn.Linear(dim, 9)

        # activation functions
        self.in_activation = Swish()
        self.activations = nn.ModuleList([Swish() for _ in range(num_layers)])

        # output activations
        self.softplus = nn.Softplus()
        self.register_buffer("c", torch.tensor(3e8))

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        params = self.d_out(x)
        #
        b_field = params[..., 0:1] * 1000
        theta = params[..., 1:2]# * torch.pi
        chi = params[..., 2:3]# * torch.pi
        vmac = torch.sigmoid(params[..., 3:4]) * 20e3
        damping = torch.sigmoid(params[..., 4:5]) * 1
        b0 = torch.sigmoid(params[..., 5:6])
        b1 = torch.sigmoid(params[..., 6:7])
        vdop = params[..., 7:8] * 1e4
        kl = torch.sigmoid(params[..., 8:9]) * 100
        #
        output = {
            "b_field": b_field,
            "theta": theta,
            "chi": chi,
            "vmac": vmac,
            "damping": damping,
            "b0": b0,
            "b1": b1,
            "vdop": vdop,
            "kl": kl,
        }

        return output


class ParameterModel(nn.Module):

    def __init__(self, in_coords, dim, num_layers=8, encoding='positional'):
        super().__init__()
        # encoding layer
        if encoding == "periodic":
            posenc = PeriodicBoundary()
            d_in = nn.Linear(in_coords + 2, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "positional":
            posenc = PositionalEncoding(num_freqs=20, d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "learned_positional":
            posenc = LearnedPositionalEncoding(num_freqs=20, d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "gaussian_positional":
            posenc = GaussianPositionalEncoding(num_freqs=20, d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == "linear":
            self.d_in = nn.Linear(in_coords, dim)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # hidden layers
        lin = [nn.Linear(dim, dim) for _ in range(num_layers)]
        self.linear_layers = nn.ModuleList(lin)
        # output layer
        self.d_out = nn.Linear(dim, 1)
        # activation functions
        self.in_activation = Swish()
        self.activations = nn.ModuleList([Swish() for _ in range(num_layers)])

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        return self.d_out(x)


class ParametersModel(nn.Module):

    def __init__(self, in_coords, dim, encoding='positional', num_layers=8):
        super().__init__()
        parameter_models = [ParameterModel(in_coords, dim, encoding=encoding, num_layers=num_layers) for _ in range(9)]
        self.parameter_models = nn.ModuleList(parameter_models)

    def forward(self, x):
        params = torch.cat([m(x) for m in self.parameter_models], -1)
        #
        b_field = params[..., 0:1] * 1e3
        theta = params[..., 1:2] * torch.pi
        chi = params[..., 2:3] * torch.pi
        vmac = torch.sigmoid(params[..., 3:4]) * 20e3
        damping = torch.sigmoid(params[..., 4:5]) * 1
        b0 = torch.sigmoid(params[..., 5:6])
        b1 = torch.sigmoid(params[..., 6:7])
        vdop = torch.tanh(params[..., 7:8]) * 1e4
        kl = torch.sigmoid(params[..., 8:9]) * 100
        #
        output = {
            "b_field": b_field,
            "theta": theta,
            "chi": chi,
            "vmac": vmac,
            "damping": damping,
            "b0": b0,
            "b1": b1,
            "vdop": vdop,
            "kl": kl,
        }

        return output


# revert this
class LearnedPositionalEncoding(nn.Module):

    def __init__(self, num_freqs, d_input):
        super().__init__()
        frequencies = torch.randn(num_freqs, d_input)
        self.frequencies = nn.Parameter(frequencies[None], requires_grad=True)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = x[:, None, :] * torch.pi * 2 ** self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, num_freqs, d_input):
        super().__init__()
        frequencies = torch.randn(num_freqs, d_input)
        self.frequencies = nn.Parameter(frequencies[None], requires_grad=False)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = x[:, None, :] * self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded


class PositionalEncoding(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, num_freqs: int, scale_factor: float = 2., log_space: bool = True):
        """

        Parameters
        ----------
        d_input: number of input dimensions
        num_freqs: number of frequencies used for encoding
        scale_factor: factor to adjust box size limit of 2pi (default 2; 4pi)
        log_space: use frequencies in powers of 2
        """
        super().__init__()
        self.d_input = d_input
        self.n_freqs = num_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)
            print('freq bands', freq_bands)

        self.register_buffer('freq_bands', freq_bands)
        self.scale_factor = scale_factor

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        f = self.freq_bands[None, :, None]
        enc = [x,
               (torch.sin(x[:, None, :] * f / self.scale_factor)).reshape(x.shape[0], -1),
               (torch.cos(x[:, None, :] * f / self.scale_factor)).reshape(x.shape[0], -1)
               ]
        return torch.concat(enc, dim=-1)


def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix


class NormalizationModule(nn.Module):

    def __init__(self, value_range):
        super().__init__()
        self.register_buffer("value_range", torch.tensor(value_range, dtype=torch.float32)[None, :, None, :])
        self.register_buffer("stretch", torch.tensor(np.arcsinh(1e2), dtype=torch.float32))

    def forward(self, stokes):
        stokes = stokes / self.value_range[..., 1] # normalize by max value (I = [0, 1]; QUV = [-1, 1])
        stokes = torch.asinh(stokes * 1e2) / self.stretch
        return stokes
