"""Linear and polarization-asinh Stokes objectives."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn

from prom3theus.core.transforms import normalized_asinh

STOKES_COMPONENTS = ("I", "Q", "U", "V")


class StokesObjective(nn.Module):
    """Squared residuals, or linear I and asinh Q/U/V.

    Prediction and target remain in the fixed loader-scaled atlas-continuum units.
    For asinh_mse, Q/U/V are stretched individually before subtraction, with the
    scale in those units and division by asinh(1/scale) so unit input stays unit.
    No local continuum is estimated or divided out.
    """

    def __init__(self, type="mse", *, asinh_scale=1e-3):
        super().__init__()
        if type not in ("mse", "asinh_mse"):
            raise ValueError(f"Unknown Stokes loss type {type!r}.")
        if (
            isinstance(asinh_scale, bool)
            or not isinstance(asinh_scale, Real)
            or not math.isfinite(asinh_scale)
            or asinh_scale <= 0
        ):
            raise ValueError("Stokes asinh scale must be finite and strictly positive.")
        self.asinh_scale = float(asinh_scale)
        self.loss_type = type

    def configuration(self):
        """Return the canonical artifact description of the objective."""
        result = {"type": self.loss_type}
        if self.loss_type == "asinh_mse":
            result["asinh_scale"] = self.asinh_scale
        return result

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

    def transform(self, profiles):
        """Values compared by the loss, after synthesis and instrument integration."""
        self._validate_profiles(profiles, profiles)
        if self.loss_type == "asinh_mse":
            return torch.cat(
                (
                    profiles[..., :1, :],
                    normalized_asinh(profiles[..., 1:, :], self.asinh_scale),
                ),
                dim=-2,
            )
        return profiles

    def elementwise(self, prediction, target):
        """Return an unreduced loss with the profile input shape."""
        self._validate_profiles(prediction, target)
        residual = self.transform(prediction) - self.transform(target)
        return residual.square()

    def forward(self, prediction, target):
        return self.elementwise(prediction, target)


def weighted_stokes_loss(
    objective: StokesObjective,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    wavelength_weights: torch.Tensor,
    stokes_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the common Stokes objective over samples and active wavelengths.

    Weights are validated and applied directly without component normalization.
    Non-finite values propagate to the final objective; no device-wide scans are
    needed here. Both single-observation training and joint evaluation use this
    exact reduction, including its gradients.
    """
    if prediction.ndim < 3:
        raise ValueError("Stokes profiles must have shape [..., 4, wavelength].")
    elementwise = objective(prediction, target)
    spectral_weights = wavelength_weights.to(prediction)
    weighted = elementwise * spectral_weights
    sample_dimensions = tuple(range(prediction.ndim - 2))
    component = weighted.sum(dim=(*sample_dimensions, prediction.ndim - 1))
    sample_count = prediction.numel() // (4 * prediction.shape[-1])
    component = component / (sample_count * spectral_weights.sum())
    return component, torch.dot(component, stokes_weights.to(component))
