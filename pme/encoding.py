import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


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


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, d_input, num_freqs=128, scale=64, learnable=False):
        super().__init__()
        dist = Normal(loc=0, scale=scale)
        frequencies = dist.sample([num_freqs, d_input])
        self.frequencies = nn.Parameter(2 * torch.pi * frequencies, requires_grad=learnable)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = torch.einsum('...j,ij->...ij', x, self.frequencies)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded


class ProgressiveFourierEncoding(nn.Module):
    def __init__(self, d_input, num_freqs=256, max_iter=1e4, min_freq=0, max_freq=7, transition_step=0.01):
        super().__init__()
        self.in_dim = d_input
        self.n_freqs = num_freqs
        self.max_iter = max_iter
        self.transition_step = transition_step
        frequencies = 2 ** torch.linspace(min_freq, max_freq, num_freqs, dtype=torch.float32)
        self.freq_bands = nn.Parameter(frequencies * np.pi, requires_grad=False)
        self.d_output = d_input * (num_freqs * 2 + 1)  # in_dim + n_freqs * 2
        self.global_step = nn.Parameter(torch.tensor(0, dtype=torch.int64), requires_grad=False)
        self.weights = nn.Parameter(torch.zeros(num_freqs), requires_grad=False)

    def step(self, global_step):
        t = global_step / self.max_iter
        t_l = torch.linspace(0, 1, self.n_freqs, device=self.freq_bands.device)
        weights = 1 - torch.sigmoid((t_l - t) / self.transition_step)  # smooth transition
        weights = torch.clamp(weights, min=0, max=1)
        self.weights.copy_(weights)

    def forward(self, coords):
        """
        coords: Tensor of shape [B, in_dim]
        returns: encoded features [B, in_dim * n_freqs * 2]
        """
        freqs = torch.einsum('i,...j->...ij', self.freq_bands, coords)  # [..., n_freqs, in_dim]

        sin = torch.sin(freqs) * self.weights[..., :, None]  # [..., n_freqs, in_dim]
        cos = torch.cos(freqs) * self.weights[..., :, None]  # [..., n_freqs, in_dim]
        sin = sin.view(*sin.shape[:-2], -1)  # [..., n_freqs * in_dim]
        cos = cos.view(*cos.shape[:-2], -1)  # [..., n_freqs * in_dim]
        feat = torch.cat([coords, sin, cos], dim=-1)  # [..., (2 * n_freqs + 1) * in_dim]

        return feat  # [B, out_dim]


class PositionalEncoding(nn.Module):

    def __init__(self, num_freqs, d_input, max_freq=8):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.frequencies = nn.Parameter(frequencies[None, :, None], requires_grad=False)
        self.d_output = d_input * (num_freqs * 2)

    def forward(self, x):
        encoded = x[:, None, :] * torch.pi * self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded
