"""Noise-standardized Stokes objectives for spectropolarimetric inversions."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch import nn


STOKES_COMPONENTS = ("I", "Q", "U", "V")


class StokesObjective(nn.Module):
    """Apply an elementwise Huber penalty to noise-standardized residuals.

    Prediction and target remain in the fixed loader-scaled atlas-continuum units.
    The fixed component sigmas describe effective measurement/model discrepancy in
    those same units; no local continuum is estimated or divided out.
    """

    def __init__(self, type="huber", *, stokes_sigmas, huber_delta=1.0):
        super().__init__()
        if type != "huber":
            raise ValueError(f'Unknown Stokes loss type {type!r}; expected "huber".')
        if not isinstance(stokes_sigmas, Mapping) or set(stokes_sigmas) != set(
            STOKES_COMPONENTS
        ):
            raise TypeError(
                f"stokes_sigmas must contain exactly {list(STOKES_COMPONENTS)}."
            )
        raw_sigmas = [stokes_sigmas[name] for name in STOKES_COMPONENTS]
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value <= 0
            for value in raw_sigmas
        ):
            raise ValueError("Stokes sigmas must be finite and strictly positive.")
        sigmas = torch.tensor(raw_sigmas, dtype=torch.float64)
        if isinstance(huber_delta, bool) or not isinstance(huber_delta, (int, float)):
            raise TypeError("huber_delta must be numeric.")
        delta = float(huber_delta)
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("huber_delta must be finite and strictly positive.")
        self.loss_type = "huber"
        self.huber_delta = delta
        self.register_buffer("stokes_sigmas", sigmas.reshape(1, 4, 1))

    def configuration(self):
        """Return the canonical artifact description of the objective."""
        return {
            "type": self.loss_type,
            "stokes_sigmas": {
                name: float(value)
                for name, value in zip(
                    STOKES_COMPONENTS, self.stokes_sigmas.flatten().tolist()
                )
            },
            "huber_delta": self.huber_delta,
        }

    @staticmethod
    def _validate_profiles(prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target Stokes profiles must have matching shapes; "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.ndim < 2 or prediction.shape[-2] != 4:
            raise ValueError(
                f"Stokes profiles must have four components on the penultimate axis; "
                f"got {tuple(prediction.shape)}."
            )

    def valid_sample_mask(self, target):
        """Return samples whose observed target can define validation metrics.

        This mask deliberately depends only on the target. A non-finite model
        prediction at an otherwise valid sample must remain visible to the loss
        reduction instead of being silently treated as missing data.
        """
        self._validate_profiles(target, target)
        return torch.isfinite(target).all(dim=(-2, -1))

    def elementwise(self, prediction, target):
        """Return an unreduced loss with the profile input shape."""
        self._validate_profiles(prediction, target)
        residual = (prediction - target) / self.stokes_sigmas.to(prediction)
        absolute = residual.abs()
        delta = self.huber_delta
        return torch.where(
            absolute <= delta,
            0.5 * residual.square(),
            delta * (absolute - 0.5 * delta),
        )

    def forward(self, prediction, target):
        return self.elementwise(prediction, target)
