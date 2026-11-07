"""Linearly scaled Stokes objectives with legacy asinh artifact support."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch import nn

from prom3theus.core.transforms import normalized_asinh

STOKES_COMPONENTS = ("I", "Q", "U", "V")


@dataclass(frozen=True)
class StokesComparison:
    """Profiles in loss units and the observed continuum proxy."""

    prediction: torch.Tensor
    target: torch.Tensor
    continuum: torch.Tensor | None


class StokesObjective(nn.Module):
    """Squared residuals after optional continuum and component normalization.

    Linear stokes_scales divide I/Q/U/V independently, following SPINNME.
    They apply to both prediction and target after continuum normalization.

    Prediction and target arrive in fixed loader-scaled atlas-continuum units.
    For asinh_mse, Q/U/V are stretched individually before subtraction, with the
    scale in the comparison units and division by asinh(1/scale) so unit input
    stays unit.
    Continuum normalization divides both profiles by the observed Stokes-I
    quantile, detached from autograd, retaining sensitivity to intensity errors.
    An explicit continuum_floor preserves the shared observed-mean convention
    stored in older artifacts; new configurations leave it unset.
    """

    def __init__(
        self, type="mse", *, asinh_scale=1e-3,
        continuum_normalization=False, continuum_quantile=0.9, continuum_floor=None,
        continuum_indices=None,
        stokes_scales=None,
    ):
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
        if stokes_scales is not None and type != "mse":
            raise ValueError("Use either linear stokes_scales or asinh_mse, not both.")
        if isinstance(stokes_scales, Mapping):
            if set(stokes_scales) != set(STOKES_COMPONENTS):
                raise ValueError("stokes_scales must contain I, Q, U, V.")
            stokes_scales = [stokes_scales[name] for name in STOKES_COMPONENTS]
        scales = None if stokes_scales is None else torch.as_tensor(stokes_scales, dtype=torch.float32)
        if scales is not None:
            if scales.numel() != 4 or not torch.isfinite(scales).all() or not (scales > 0).all():
                raise ValueError("stokes_scales must contain four finite positive I/Q/U/V scales.")
            scales = scales.reshape(4, 1)
        # Derived from the archived objective configuration, not learned state.
        self.register_buffer("stokes_scales", scales, persistent=False)
        if not isinstance(continuum_normalization, bool):
            raise ValueError("continuum_normalization must be boolean")
        if (isinstance(continuum_quantile, bool)
                or not isinstance(continuum_quantile, Real)
                or not math.isfinite(continuum_quantile)
                or not 0 < continuum_quantile <= 1):
            raise ValueError("continuum_quantile must be in (0, 1]")
        self.continuum_quantile = float(continuum_quantile)
        if continuum_floor is not None and (
            isinstance(continuum_floor, bool)
            or not isinstance(continuum_floor, Real)
            or not math.isfinite(continuum_floor)
            or continuum_floor <= 0
        ):
            raise ValueError("continuum_floor must be finite and strictly positive")
        indices = torch.as_tensor([] if continuum_indices is None else continuum_indices)
        if continuum_normalization and continuum_floor is not None and (
            indices.ndim != 1
            or indices.numel() == 0
            or indices.dtype == torch.bool
            or indices.is_floating_point()
            or indices.is_complex()
            or torch.any(indices < 0)
            or indices.unique().numel() != indices.numel()
        ):
            raise ValueError(
                "continuum normalization requires distinct non-negative integer continuum_indices"
            )
        self.continuum_normalization = continuum_normalization
        self.continuum_floor = None if continuum_floor is None else float(continuum_floor)
        self.register_buffer("continuum_indices", indices.to(torch.long), persistent=False)

    def configuration(self):
        """Return the canonical artifact description of the objective."""
        result = {"type": self.loss_type}
        if self.stokes_scales is not None:
            result["stokes_scales"] = dict(zip(STOKES_COMPONENTS,
                self.stokes_scales.detach().cpu().reshape(-1).tolist()))
        if self.loss_type == "asinh_mse":
            result["asinh_scale"] = self.asinh_scale
        if self.continuum_normalization:
            result["continuum_normalization"] = True
            if self.continuum_floor is None:
                result["continuum_quantile"] = self.continuum_quantile
            else:
                result.update(continuum_floor=self.continuum_floor,
                              continuum_indices=self.continuum_indices.tolist())
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
        valid = torch.isfinite(target).all(dim=(-2, -1))
        if self.continuum_normalization and self.continuum_floor is None:
            continuum = self.profile_continuum(target)
            valid = valid & torch.isfinite(continuum) & (continuum > 0)
        return valid

    def transform(self, profiles):
        """Values compared by the loss, after synthesis and instrument integration."""
        self._validate_profiles(profiles, profiles)
        if self.stokes_scales is not None:
            return profiles / self.stokes_scales.to(profiles)
        if self.loss_type == "asinh_mse":
            return torch.cat(
                (
                    profiles[..., :1, :],
                    normalized_asinh(profiles[..., 1:, :], self.asinh_scale),
                ),
                dim=-2,
            )
        return profiles

    def profile_continuum(self, profiles):
        """Differentiable Stokes-I quantile over the detector wavelength axis."""
        intensity = profiles[..., 0, :]
        # Quantile does not support half precision. Keep production in float32.
        if intensity.dtype in (torch.float16, torch.bfloat16):
            intensity = intensity.float()
        return torch.quantile(intensity, self.continuum_quantile, dim=-1)

    @staticmethod
    def _normalize_profile(profiles, continuum):
        # Invalid observed continua must remain visible to the objective.
        continuum = continuum.masked_fill(~torch.isfinite(continuum) | (continuum <= 0), torch.nan)
        return profiles / continuum[..., None, None]

    def observed_continuum(self, target):
        """Detached observed proxy, with the old mean for legacy objectives."""
        self._validate_profiles(target, target)
        if self.continuum_floor is None:
            return self.profile_continuum(target.detach())
        if self.continuum_indices.numel() == 0:
            raise ValueError("Observed continuum requires continuum_indices.")
        return target.detach()[..., 0, :].index_select(
            -1, self.continuum_indices.to(target.device)
        ).mean(-1)

    def compare(self, prediction, target):
        """Prepare loss units once for scoring and optional fit diagnostics."""
        self._validate_profiles(prediction, target)
        continuum = None
        if self.continuum_normalization:
            continuum = self.observed_continuum(target)
            if self.continuum_floor is None:
                prediction = self._normalize_profile(prediction, continuum)
                target = self._normalize_profile(target, continuum)
            else:
                scale = continuum.clamp_min(self.continuum_floor)[..., None, None]
                prediction = prediction / scale
                target = target / scale
        return StokesComparison(self.transform(prediction), self.transform(target), continuum)

    def comparison_profiles(self, prediction, target):
        comparison = self.compare(prediction, target)
        return comparison.prediction, comparison.target

    def elementwise(self, prediction, target):
        """Return an unreduced loss with the profile input shape."""
        prediction, target = self.comparison_profiles(prediction, target)
        residual = prediction - target
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
    return reduce_stokes_loss(objective(prediction, target),
                              wavelength_weights=wavelength_weights,
                              stokes_weights=stokes_weights)


def reduce_stokes_loss(elementwise, *, wavelength_weights, stokes_weights):
    """Reduce already-computed squared residuals without repeating transforms."""
    if elementwise.ndim < 3 or elementwise.shape[-2] != 4:
        raise ValueError("Stokes profiles must have shape [..., 4, wavelength].")
    spectral_weights = wavelength_weights.to(elementwise)
    weighted = elementwise * spectral_weights
    sample_dimensions = tuple(range(elementwise.ndim - 2))
    component = weighted.sum(dim=(*sample_dimensions, elementwise.ndim - 1))
    sample_count = elementwise.numel() // (4 * elementwise.shape[-1])
    component = component / (sample_count * spectral_weights.sum())
    return component, torch.dot(component, stokes_weights.to(component))
