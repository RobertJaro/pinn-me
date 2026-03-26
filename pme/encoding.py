from typing import Iterable

import numpy as np
import torch
import wandb
from pytorch_lightning.utilities import rank_zero_only
from torch import nn


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

    def __init__(self, d_input, num_frequencies=128, scales=64):
        super().__init__()
        num_frequencies = [num_frequencies] * d_input if not isinstance(num_frequencies, Iterable) else num_frequencies
        scales = [scales] * d_input if not isinstance(scales, Iterable) else scales

        assert len(num_frequencies) == d_input, 'num_frequencies length must match input dimension (in_dim)'
        assert len(scales) == d_input, 'scales length must match input dimension (in_dim)'

        frequencies = []
        for num_freq, scale in zip(num_frequencies, scales):
            frequency = torch.randn(num_freq, dtype=torch.float32) * scale
            frequencies.append(nn.Parameter(frequency, requires_grad=False))
        self.frequencies = nn.ParameterList(frequencies)

        self.out_dim = int(sum([n * 2 for n in num_frequencies])) + d_input

        self.num_frequencies = num_frequencies

    def forward(self, x):
        encoded_coordinates = []
        for i, frequencies in enumerate(self.frequencies):
            phases = torch.einsum('...,f->...f', x[..., i], 2 * torch.pi * frequencies)
            encoded_coordinates.append(torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1))
        encoded = torch.cat(encoded_coordinates + [x], -1)
        return encoded


class SphericalEncoding(nn.Module):

    def __init__(self):
        super().__init__()
        self.out_dim = 6  # t, r, sin_theta, cos_theta, sin_phi, cos_phi

    def forward(self, coords):
        """
        coords: Tensor of shape [B, 4] in (t, x, y, z) format
        returns: encoded features [B, 6] in (t, r, sin_theta, cos_theta, sin_phi, cos_phi) format
        """
        t, x, y, z = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-8
        sin_theta = torch.sqrt(x ** 2 + y ** 2) / r
        cos_theta = z / r
        sin_phi = y / (torch.sqrt(x ** 2 + y ** 2) + 1e-8)
        cos_phi = x / (torch.sqrt(x ** 2 + y ** 2) + 1e-8)
        spherical_coords = torch.stack([t, r, sin_theta, cos_theta, sin_phi, cos_phi], dim=-1)
        return spherical_coords


class ProgressiveFourierEncoding(nn.Module):
    def __init__(self, d_input, num_freqs=256, max_iter=1e4, min_freq=0, max_freq=7, transition_step=0.01):
        super().__init__()
        self.in_dim = d_input
        self.n_freqs = num_freqs
        self.max_iter = max_iter
        self.transition_step = transition_step
        frequencies = 2 ** torch.linspace(min_freq, max_freq, num_freqs, dtype=torch.float32)
        self.freq_bands = nn.Parameter(frequencies * np.pi, requires_grad=False)
        self.out_dim = d_input * (num_freqs * 2 + 1)  # in_dim + n_freqs * 2
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


class ProgressivePositionalEncoding(nn.Module):

    def __init__(self, in_dim, num_frequencies=64, min_frequencies=0, max_frequencies=8,
                 max_iter=1e5, transition_step=0.01, start_step=2e4):
        super().__init__()
        num_frequencies = [num_frequencies] * in_dim if not isinstance(num_frequencies, Iterable) else num_frequencies
        min_frequencies = [min_frequencies] * in_dim if not isinstance(min_frequencies, Iterable) else min_frequencies
        max_frequencies = [max_frequencies] * in_dim if not isinstance(max_frequencies, Iterable) else max_frequencies

        assert len(num_frequencies) == in_dim, 'num_frequencies length must match input dimension (in_dim)'
        assert len(min_frequencies) == in_dim, 'd_min_frequencies length must match input dimension (in_dim)'
        assert len(max_frequencies) == in_dim, 'max_frequencies length must match input dimension (in_dim)'

        frequencies = []
        for num_freq, min_freq, max_freq in zip(num_frequencies, min_frequencies, max_frequencies):
            f = 2 ** torch.linspace(min_freq, max_freq, num_freq)
            param = nn.Parameter(f, requires_grad=False)
            frequencies.append(param)
        self.frequencies = nn.ParameterList(frequencies)

        self.out_dim = int(sum([n * 2 for n in num_frequencies])) + in_dim

        self.num_frequencies = num_frequencies
        self.max_iter = max_iter
        self.start_step = start_step
        self.transition_step = transition_step
        weights = []
        for n_freq in num_frequencies:
            weights.append(nn.Parameter(torch.zeros(n_freq), requires_grad=False))
        self.weights = nn.ParameterList(weights)
        self.step(0)

    @torch.no_grad()
    @rank_zero_only
    def step(self, global_step):
        t = (global_step + self.start_step) / self.max_iter
        for i, n_freq in enumerate(self.num_frequencies):
            t_l = torch.linspace(0.0, 1.0, n_freq, device=self.weights[i].device)
            weights = 1 - torch.sigmoid((t_l - t) / self.transition_step)  # smooth transition
            self.weights[i].copy_(weights)
        # log current t value
        wandb.log({'posenc_progress': t}, commit=False)

    def forward(self, x):
        encoded_coordinates = []
        for i, frequencies in enumerate(self.frequencies):
            encoded = torch.einsum('i,...j->...ij', frequencies, x[..., i:i + 1]) * torch.pi
            encoded = encoded.reshape(x.shape[0], -1)
            sin_encoded = torch.einsum('...j,j->...j', torch.sin(encoded), self.weights[i])
            cos_encoded = torch.einsum('...j,j->...j', torch.cos(encoded), self.weights[i])
            encoded = torch.cat([sin_encoded, cos_encoded], -1)
            encoded_coordinates.append(encoded)

        encoded = torch.cat([x] + encoded_coordinates, -1)
        return encoded


class ProgressiveGaussianEncoding(nn.Module):

    def __init__(self, in_dim, num_frequencies=64, scales=2, max_iter=1e5, transition_step=0.01):
        super().__init__()
        num_frequencies = [num_frequencies] * in_dim if not isinstance(num_frequencies, Iterable) else num_frequencies
        scales = [scales] * in_dim if not isinstance(scales, Iterable) else scales

        assert len(num_frequencies) == in_dim, 'num_frequencies length must match input dimension (in_dim)'
        assert len(scales) == in_dim, 'scales length must match input dimension (in_dim)'

        frequencies = []
        for num_freq, scale in zip(num_frequencies, scales):
            f = torch.randn([num_freq]) * scale
            # sort frequencies
            idx = torch.argsort(torch.abs(f))
            f = f[idx]
            print(f)
            param = nn.Parameter(f, requires_grad=False)
            frequencies.append(param)
        self.frequencies = nn.ParameterList(frequencies)

        self.out_dim = int(sum([n * 2 for n in num_frequencies])) + in_dim

        self.num_frequencies = num_frequencies
        self.max_iter = max_iter
        self.transition_step = transition_step
        weights = []
        for n_freq in num_frequencies:
            weights.append(nn.Parameter(torch.zeros(n_freq), requires_grad=False))
        self.weights = nn.ParameterList(weights)
        self.step(0)

    @torch.no_grad()
    def step(self, global_step):
        t = global_step / self.max_iter
        for i, n_freq in enumerate(self.num_frequencies):
            t_l = torch.linspace(0.0, 1.0, n_freq, device=self.weights[i].device)
            weights = 1 - torch.sigmoid((t_l - t) / self.transition_step)  # smooth transition
            self.weights[i].copy_(weights)
        self._log_step(t)

    @rank_zero_only
    def _log_step(self, t):
        for i, frequencies in enumerate(self.frequencies):
            available_frequencies = self.weights[i] * frequencies
            print(
                f'Max available frequency for dimension {i}: {available_frequencies.min().item():.3f} -- {available_frequencies.max().item():.3f}')
        # log current t value
        wandb.log({'posenc_progress': t}, commit=False)

    def forward(self, x):
        encoded_coordinates = []
        for i, frequencies in enumerate(self.frequencies):
            encoded = torch.einsum('i,...j->...ij', torch.pi * frequencies, x[..., i:i + 1])
            encoded = encoded.reshape(*x.shape[:-1], -1)
            sin_encoded = torch.einsum('...j,j->...j', torch.sin(encoded), self.weights[i])
            cos_encoded = torch.einsum('...j,j->...j', torch.cos(encoded), self.weights[i])
            encoded = torch.cat([sin_encoded, cos_encoded], -1)
            encoded_coordinates.append(encoded)
        encoded = torch.cat([x] + encoded_coordinates, -1)  # append original coordinates
        return encoded


class BankGaussianEncoding(nn.Module):
    """
    Minimal banked Gaussian positional encoding.

    Bank spec: dict with keys
      - num_freq: int
      - dim: int          (which input dimension to apply to)
      - scale: float
      - step: int         (activate bank when global_step >= step)

    Encoding convention:
      f ~ N(0,1) * scale
      phi = pi * f * x_dim
      out: [sin(phi), cos(phi)] (per frequency)
    """

    def __init__(self, in_dim: int, banks, include_input: bool = True, seed: int | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input

        # store banks as python list (simple)
        self.banks = [
            {
                "num_freq": int(b["num_freq"]),
                "dim": int(b["dim"]),
                "scale": float(b["scale"]),
                "step": int(b["step"]),
            }
            for b in banks
        ]

        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))

        # sample frequencies and store as buffers
        freqs = []
        dims = []
        steps = []

        for b in self.banks:
            k = b["num_freq"]
            f = torch.randn(k, generator=gen) * b["scale"]
            freqs.append(f)
            dims.append(torch.full((k,), b["dim"], dtype=torch.long))
            steps.append(torch.full((k,), b["step"], dtype=torch.long))

        self.register_buffer("freqs", torch.cat(freqs, dim=0))  # [K_total]
        self.register_buffer("dims", torch.cat(dims, dim=0))  # [K_total]
        self.register_buffer("steps", torch.cat(steps, dim=0))  # [K_total]

        self.out_dim = (in_dim if include_input else 0) + 2 * self.freqs.numel()

        # current gate (0/1), updated by set_step()
        self.register_buffer("gate", torch.zeros_like(self.freqs))

    @torch.no_grad()
    def step(self, global_step: int):
        # hard activation: bank on if global_step >= step
        self.gate.copy_((global_step >= self.steps).to(self.gate.dtype))
        active_fraction = self.gate.mean().item()
        self._log_active_fraction(active_fraction)

    @rank_zero_only
    def _log_active_fraction(self, active_fraction: float):
        wandb.log({"active_frequencies_fraction": active_fraction}, commit=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        outs = [x] if self.include_input else []

        # select x per frequency: [..., K]
        x_sel = x[..., self.dims]
        phase = 2 * torch.pi * x_sel * self.freqs  # [..., K]

        g = self.gate  # [K], broadcast to leading dims
        sin = torch.sin(phase) * g
        cos = torch.cos(phase) * g

        outs.append(torch.cat([sin, cos], dim=-1))
        return torch.cat(outs, dim=-1)


class PositionalEncoding(nn.Module):

    def __init__(self, in_dim, num_frequencies=128, min_frequencies=0, max_frequencies=8):
        super().__init__()
        num_frequencies = [num_frequencies] * in_dim if not isinstance(num_frequencies, Iterable) else num_frequencies
        min_frequencies = [min_frequencies] * in_dim if not isinstance(min_frequencies, Iterable) else min_frequencies
        max_frequencies = [max_frequencies] * in_dim if not isinstance(max_frequencies, Iterable) else max_frequencies

        assert len(num_frequencies) == in_dim, 'num_frequencies length must match input dimension (in_dim)'
        assert len(min_frequencies) == in_dim, 'd_min_frequencies length must match input dimension (in_dim)'
        assert len(max_frequencies) == in_dim, 'max_frequencies length must match input dimension (in_dim)'

        frequencies = []
        for num_freq, min_freq, max_freq in zip(num_frequencies, min_frequencies, max_frequencies):
            f = 2 ** torch.linspace(min_freq, max_freq, num_freq)
            param = nn.Parameter(f, requires_grad=False)
            frequencies.append(param)
        self.frequencies = nn.ParameterList(frequencies)

        self.out_dim = int(sum([n * 2 for n in num_frequencies])) + in_dim

    def forward(self, x):
        encoded_coordinates = []
        for i, frequencies in enumerate(self.frequencies):
            encoded = torch.einsum('...,j->...j', x[..., i], 2 * torch.pi * frequencies)
            encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
            encoded_coordinates.append(encoded)

        encoded = torch.cat([x] + encoded_coordinates, -1)
        return encoded


class SpatiotemporalEncoding(nn.Module):

    def __init__(self, d_input, num_freqs=16, max_spatial_freq=8, max_temporal_freq=2):
        super().__init__()
        spatial_frequencies = 2 ** torch.linspace(0, max_spatial_freq, num_freqs)
        self.spatial_frequencies = nn.Parameter(spatial_frequencies[None, :, None], requires_grad=False)

        temporal_frequencies = 2 ** torch.linspace(0, max_temporal_freq, num_freqs)
        self.temporal_frequencies = nn.Parameter(temporal_frequencies[None, :, None], requires_grad=False)

        self.out_dim = d_input * (num_freqs * 2) + d_input

    def forward(self, x):
        t = x[..., :1]
        spatial = x[..., 1:]

        spatial_encoded = spatial[:, None, :] * torch.pi * self.spatial_frequencies
        spatial_encoded = spatial_encoded.reshape(x.shape[0], -1)
        spatial_encoded = torch.cat([torch.sin(spatial_encoded), torch.cos(spatial_encoded)], -1)

        temporal_encoded = t[:, None, :] * torch.pi * self.temporal_frequencies
        temporal_encoded = temporal_encoded.reshape(x.shape[0], -1)
        temporal_encoded = torch.cat([torch.sin(temporal_encoded), torch.cos(temporal_encoded)], -1)

        encoded = torch.cat([spatial_encoded, temporal_encoded, x], -1)
        return encoded


class ProgressiveSpatiotemporalEncoding(nn.Module):

    def __init__(self, d_input, num_freqs=128, min_freq=0, max_spatial_freq=8, max_temporal_freq=1,
                 max_iter=1e5, transition_step=0.01):
        super().__init__()
        self.num_freqs = num_freqs
        spatial_frequencies = 2 ** torch.linspace(min_freq, max_spatial_freq, num_freqs)
        self.register_buffer('spatial_frequencies', spatial_frequencies[None, :, None])

        temporal_frequencies = 2 ** torch.linspace(min_freq, max_temporal_freq, num_freqs)
        self.register_buffer('temporal_frequencies', temporal_frequencies[None, :, None])

        self.out_dim = d_input * (num_freqs * 2)

        self.max_iter = max_iter
        self.transition_step = transition_step
        self.weights = nn.Parameter(torch.zeros(num_freqs), requires_grad=False)
        self.step(0)

    @torch.no_grad()
    def step(self, global_step):
        t = global_step / self.max_iter
        t_l = torch.linspace(0, 1, self.num_freqs, device=self.spatial_frequencies.device)
        weights = 1 - torch.sigmoid((t_l - t) / self.transition_step)  # smooth transition
        self.weights.copy_(weights)
        # log current t value
        wandb.log({'posenc_progress': t}, commit=False)

    def forward(self, x):
        t = x[..., :1]
        spatial = x[..., 1:]

        weights = self.weights[None, :, None]

        spatial_encoded = spatial[:, None, :] * torch.pi * self.spatial_frequencies
        spatial_sin_encoded = torch.sin(spatial_encoded) * weights
        spatial_cos_encoded = torch.cos(spatial_encoded) * weights
        spatial_encoded = torch.cat([spatial_sin_encoded, spatial_cos_encoded], 1)
        spatial_encoded = spatial_encoded.reshape(x.shape[0], -1)

        temporal_encoded = t[:, None, :] * torch.pi * self.temporal_frequencies
        temporal_sin_encoded = torch.sin(temporal_encoded) * weights
        temporal_cos_encoded = torch.cos(temporal_encoded) * weights
        temporal_encoded = torch.cat([temporal_sin_encoded, temporal_cos_encoded], 1)
        temporal_encoded = temporal_encoded.reshape(x.shape[0], -1)

        encoded = torch.cat([spatial_encoded, temporal_encoded], -1)
        return encoded
