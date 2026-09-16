"""Mean-squared channel-balanced objective for AIA count-rate images."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from prom3theus.core.transforms import normalized_asinh


class AsinhMSEImageObjective(nn.Module):
    """Channel-balanced MSE loss on asinh-transformed image intensities."""

    def __init__(
        self,
        channels_angstrom: Sequence[int],
        *,
        asinh_scales: Sequence[float],
        channel_weights: Sequence[float],
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in channels_angstrom)
        scales = tuple(float(value) for value in asinh_scales)
        weights = tuple(float(value) for value in channel_weights)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("Objective channels must be non-empty and unique.")
        if len(scales) != len(channels) or len(weights) != len(channels):
            raise ValueError("One asinh scale and weight are required per channel.")
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("Asinh scales must be finite and positive.")
        if any(not math.isfinite(value) or value < 0 for value in weights) or not any(
            value > 0 for value in weights
        ):
            raise ValueError(
                "Channel weights must be finite, non-negative, and nonzero."
            )
        self.channels_angstrom = channels
        self.register_buffer("asinh_scales", torch.tensor(scales, dtype=torch.float64))
        normalized = torch.tensor(weights, dtype=torch.float64)
        self.register_buffer("channel_weights", normalized / normalized.sum())

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        channel_index: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        require_all_channels: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if type(require_all_channels) is not bool:
            raise TypeError("require_all_channels must be boolean.")
        for name, value in (
            ("prediction", prediction),
            ("target", target),
            ("channel_index", channel_index),
        ):
            if value.shape != prediction.shape:
                raise ValueError(f"{name} must match prediction shape.")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise TypeError("Image prediction and target must be floating point.")
        if not torch.isfinite(prediction).all() or torch.any(prediction < 0):
            raise FloatingPointError("AIA prediction must be finite and non-negative.")
        if not torch.isfinite(target).all():
            raise ValueError("AIA target must be finite before applying its mask.")
        indices = channel_index.long()
        if torch.any(indices < 0) or torch.any(indices >= len(self.channels_angstrom)):
            raise IndexError("AIA channel_index lies outside configured channels.")
        valid = torch.ones_like(prediction, dtype=torch.bool)
        if valid_mask is not None:
            if (
                valid_mask.shape != prediction.shape
                or valid_mask.dtype is not torch.bool
            ):
                raise ValueError(
                    "valid_mask must be boolean and match prediction shape."
                )
            valid = valid_mask
        scales = self.asinh_scales.to(prediction)[indices]
        transformed_prediction = normalized_asinh(prediction, scales)
        transformed_target = normalized_asinh(target, scales)
        residual = transformed_prediction - transformed_target
        elementwise = residual.square()
        # Reduce every channel in two bulk scatter operations. Only the small
        # presence vector crosses to Python for error messages/metric names.
        flat_indices = indices.reshape(-1)
        counts = torch.zeros_like(
            self.channel_weights, device=prediction.device, dtype=torch.long
        ).scatter_add_(0, flat_indices, valid.reshape(-1).long())
        sums = prediction.new_zeros(len(self.channels_angstrom)).scatter_add_(
            0, flat_indices, elementwise.masked_fill(~valid, 0).reshape(-1)
        )
        from prom3theus.core.distributed import distributed_means
        means, counts = distributed_means(sums, counts)
        present = counts > 0
        presence = present.detach().cpu().tolist()
        if require_all_channels and not all(presence):
            channel = self.channels_angstrom[presence.index(False)]
            raise ValueError(f"AIA batch contains no valid {channel} Angstrom pixels.")
        if not any(presence):
            raise ValueError("AIA batch contains no valid configured-channel pixels.")
        components = {
            str(channel): component
            for channel, component, active in zip(
                self.channels_angstrom, means.unbind(), presence, strict=True
            )
            if active
        }
        weights = self.channel_weights.to(prediction) * present
        total = (weights * means).sum()
        active_weight = weights.sum()
        if not require_all_channels:
            total = torch.where(
                active_weight > 0,
                total / active_weight.clamp_min(torch.finfo(total.dtype).tiny),
                torch.zeros_like(total),
            )
        return total, components, residual

    def configuration(self) -> Mapping[str, object]:
        return {
            "type": "asinh_mse",
            "channels_angstrom": list(self.channels_angstrom),
            "asinh_scales": self.asinh_scales.tolist(),
            "channel_weights": self.channel_weights.tolist(),
        }


__all__ = ["AsinhMSEImageObjective"]
