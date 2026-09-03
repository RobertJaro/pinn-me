"""Stokes-profile objectives used by spectropolarimetric inversions."""

from __future__ import annotations

import torch
from torch import nn


class StokesObjective(nn.Module):
    """Apply the Stokes transform followed by an elementwise squared residual.

    Prediction and target remain in the loader-scaled intensity units. The supplied
    normalization may transform individual Stokes components (for example asinh on
    Q/U/V), but this module does not estimate or divide by a local continuum.
    """

    def __init__(self, type="mse"):
        super().__init__()
        if type != "mse":
            raise ValueError(f'Unknown Stokes loss type {type!r}; expected "mse".')
        self.loss_type = "mse"

    def configuration(self):
        """Return the canonical artifact description of the objective."""
        return {"type": self.loss_type}

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

    def elementwise(self, prediction, target, normalization):
        """Return an unreduced loss with the profile input shape."""
        self._validate_profiles(prediction, target)
        prediction = normalization(prediction)
        target = normalization(target)
        return (prediction - target).square()

    def forward(self, prediction, target, normalization):
        return self.elementwise(prediction, target, normalization)
