"""Identifiable, unbounded AIA channel-calibration nuisance model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def helmert_contrast(channel_count: int, *, dtype=torch.float64) -> torch.Tensor:
    """Return deterministic orthonormal zero-sum channel contrasts."""

    if type(channel_count) is not int or channel_count < 1:
        raise ValueError("channel_count must be a positive integer.")
    if channel_count == 1:
        return torch.empty((1, 0), dtype=dtype)
    values = torch.zeros((channel_count, channel_count - 1), dtype=dtype)
    for column in range(channel_count - 1):
        count = column + 1
        denominator = math.sqrt(count * (count + 1))
        values[:count, column] = 1.0 / denominator
        values[count, column] = -count / denominator
    return values


class AIAChannelCalibration(nn.Module):
    """Positive gains with separate common and relative log coordinates.

    ``log(g) = a 1 + Q eta`` is injective because the fixed columns of ``Q``
    span only the zero-sum channel subspace.  Neither coordinate is clipped or
    passed through a saturating activation.
    """

    def __init__(
        self,
        channels_angstrom: Sequence[int],
        *,
        enabled: bool = True,
        absolute_prior_fraction: float = 0.25,
        relative_prior_fraction: float = 0.15,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in channels_angstrom)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("AIA calibration channels must be non-empty and unique.")
        if any(value <= 0 for value in channels):
            raise ValueError("AIA channel wavelengths must be positive.")
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean.")
        for name, value in (
            ("absolute_prior_fraction", absolute_prior_fraction),
            ("relative_prior_fraction", relative_prior_fraction),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.channels_angstrom = channels
        self.enabled = enabled
        self.absolute_prior_log_sigma = math.log1p(float(absolute_prior_fraction))
        self.relative_prior_log_sigma = math.log1p(float(relative_prior_fraction))
        self.common_log_gain = nn.Parameter(
            torch.zeros((), dtype=torch.float32), requires_grad=enabled
        )
        self.relative_log_gain = nn.Parameter(
            torch.zeros(len(channels) - 1, dtype=torch.float32),
            requires_grad=enabled,
        )
        self.register_buffer(
            "contrast_matrix",
            helmert_contrast(len(channels), dtype=torch.float32),
        )

    @property
    def log_gains(self) -> torch.Tensor:
        return self.common_log_gain.expand(len(self.channels_angstrom)) + (
            self.contrast_matrix @ self.relative_log_gain
        )

    @property
    def gains(self) -> torch.Tensor:
        return torch.exp(self.log_gains)

    def forward(
        self, intensity: torch.Tensor, channel_index: torch.Tensor
    ) -> torch.Tensor:
        indices = torch.as_tensor(channel_index, device=intensity.device)
        if indices.shape != intensity.shape:
            raise ValueError("channel_index must match the intensity shape.")
        if indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("channel_index must use an integer dtype.")
        if torch.any(indices < 0) or torch.any(indices >= len(self.channels_angstrom)):
            raise IndexError("channel_index lies outside the configured channels.")
        return intensity * self.gains.to(intensity)[indices.long()]

    def prior_loss(self) -> torch.Tensor:
        common = 0.5 * (self.common_log_gain / self.absolute_prior_log_sigma).square()
        if self.relative_log_gain.numel():
            relative = (
                0.5
                * (self.relative_log_gain / self.relative_prior_log_sigma)
                .square()
                .sum()
            )
        else:
            relative = common.new_zeros(())
        return common + relative

    def metadata(self) -> dict:
        return {
            "parameterization": "log_gains = common + helmert_contrast @ relative",
            "channels_angstrom": list(self.channels_angstrom),
            "enabled": self.enabled,
            "absolute_prior_log_sigma": self.absolute_prior_log_sigma,
            "relative_prior_log_sigma": self.relative_prior_log_sigma,
            "common_log_gain": float(self.common_log_gain.detach()),
            "relative_log_gain": self.relative_log_gain.detach().tolist(),
            "gains": self.gains.detach().tolist(),
        }


__all__ = ["AIAChannelCalibration", "helmert_contrast"]
