import torch
from torch import nn


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class MEModel(nn.Module):

    def __init__(self, in_coords, out_parameters, dim, pos_encoding=False):
        super().__init__()
        if pos_encoding:
            posenc = PositionalEncoding(8, 20)
            d_in = nn.Linear(in_coords * 40, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(in_coords, dim)
        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, out_parameters)
        self.activation = Sine()  # torch.tanh

    def forward(self, x):
        x = self.activation(self.d_in(x))
        for l in self.linear_layers:
            x = self.activation(l(x))
        b = self.d_out(x)

        

        return {'b': b}


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (batch, num_samples, in_features)
        Outputs:
            out: (batch, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2) * self.freqs.unsqueeze(dim=-1)  # (num_rays, num_samples, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # (num_rays, num_samples, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)  # (num_rays, num_samples, 2*num_freqs*in_features)
        return out
