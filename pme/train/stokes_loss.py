"""Mean-squared Stokes-profile objective."""

from __future__ import annotations

import torch
from torch import nn


class StokesLossModule(nn.Module):
    """Apply the Stokes transform followed by an elementwise squared residual.

    Optionally divide each Stokes vector by its own 90th-percentile I.
    Predictions must include the instrument response before this transform.
    Apply component scaling (linear or asinh) after continuum normalization.
    """

    def __init__(self, type='mse', continuum_normalization=False, continuum_quantile=0.9):
        super().__init__()
        if type != 'mse':
            raise ValueError(f'Unknown Stokes loss type {type!r}; expected "mse".')
        self.loss_type = 'mse'
        self.continuum_normalization = continuum_normalization
        if not 0 < continuum_quantile <= 1:
            raise ValueError('continuum_quantile must be in (0, 1].')
        self.continuum_quantile = float(continuum_quantile)

    def configuration(self):
        """Return a canonical, checkpoint-safe description of the objective."""
        config = {'type': self.loss_type}
        if self.continuum_normalization:
            config['continuum_normalization'] = True
            config['continuum_quantile'] = self.continuum_quantile
        return config

    @staticmethod
    def _validate_profiles(prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f'Prediction and target Stokes profiles must have matching shapes; '
                f'got {tuple(prediction.shape)} and {tuple(target.shape)}.'
            )
        if prediction.ndim < 2 or prediction.shape[-2] != 4:
            raise ValueError(
                f'Stokes profiles must have four components on the penultimate axis; '
                f'got {tuple(prediction.shape)}.'
            )

    def valid_sample_mask(self, target):
        """Return samples whose observed target can define validation metrics.

        This mask deliberately depends only on the target. A non-finite model
        prediction at an otherwise valid sample must remain visible to the loss
        reduction instead of being silently treated as missing data.
        """
        self._validate_profiles(target, target)
        valid = torch.isfinite(target).all(dim=(-2, -1))
        if self.continuum_normalization:
            continuum = self._continuum(target).squeeze(-1).squeeze(-1)
            valid = valid & torch.isfinite(continuum) & (continuum > 0)
        return valid

    def _continuum(self, target):
        intensity = target[..., 0:1, :]
        # torch.quantile does not support float16/bfloat16 under mixed precision.
        if intensity.dtype in (torch.float16, torch.bfloat16):
            intensity = intensity.float()
        return torch.quantile(intensity, self.continuum_quantile, dim=-1, keepdim=True)

    def normalize_profile(self, stokes):
        """Normalize all four components by this profile's own continuum proxy.

        Keep the synthetic denominator in the autograd graph. Invalid predictions
        remain non-finite rather than being masked out of the objective.
        """
        continuum = self._continuum(stokes)
        continuum = continuum.masked_fill(
            ~torch.isfinite(continuum) | (continuum <= 0), torch.nan,
        )
        return stokes / continuum

    def transform_profiles(self, prediction, target, normalization):
        """Return the common training/validation representation of both profiles."""
        self._validate_profiles(prediction, target)
        if self.continuum_normalization:
            prediction = self.normalize_profile(prediction)
            target = self.normalize_profile(target)
        return normalization(prediction), normalization(target)

    def elementwise(self, prediction, target, normalization):
        """Return an unreduced loss with the profile input shape."""
        prediction, target = self.transform_profiles(prediction, target, normalization)
        return (prediction - target).square()

    def forward(self, prediction, target, normalization):
        return self.elementwise(prediction, target, normalization)
