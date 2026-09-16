"""Strict runtime option decoding for LTE training.

This module owns value validation and normalization only.  It deliberately
does not construct trainable modules, so :mod:`prom3theus.training.joint`
can remain the lifecycle/orchestration boundary.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import torch

STOKES_COMPONENTS = ("I", "Q", "U", "V")


@dataclass(frozen=True, slots=True)
class DepthSamplingSettings:
    """Resolved coarse and adaptive depth-quadrature settings."""

    sample_count: int
    coarse_to_fine_enabled: bool
    fine_sample_count: int
    uniform_weight_floor: float
    reference_log_tau500_bounds: tuple[float, float] = (-5.0, 1.0)

    def model_configuration(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "reference_log_tau500_bounds": self.reference_log_tau500_bounds,
            "coarse_to_fine": {
                "enabled": self.coarse_to_fine_enabled,
                "fine_sample_count": self.fine_sample_count,
                "uniform_weight_floor": self.uniform_weight_floor,
            },
        }


@dataclass(frozen=True, slots=True)
class ObjectiveWeighting:
    """Validated component, wavelength, continuum, and radiance weights."""

    continuum_indices: torch.Tensor
    atlas_continuum_radiance_w_m3_sr: float
    stokes_weight_config: dict[str, float]
    stokes_weights: torch.Tensor
    wavelength_weights: torch.Tensor
    wavelength_exclude_windows_angstrom: list[list[float]]


def resolve_depth_sampling(
    config: Mapping[str, Any],
) -> DepthSamplingSettings:
    """Decode the closed depth-sampling option set."""

    if not isinstance(config, Mapping):
        raise TypeError("depth_sampling_config must be a mapping.")
    options = deepcopy(dict(config))
    bounds = tuple(
        float(v) for v in options.pop("reference_log_tau500_bounds", (-5.0, 1.0))
    )
    if (
        len(bounds) != 2
        or not bounds[0] < 0 < bounds[1]
        or not all(math.isfinite(v) for v in bounds)
    ):
        raise ValueError(
            "reference_log_tau500_bounds must be finite and straddle zero."
        )
    if set(options) != {"sample_count", "coarse_to_fine"}:
        raise TypeError(
            "depth_sampling_config must contain exactly sample_count and coarse_to_fine."
        )
    sample_value = options.pop("sample_count")
    if not isinstance(sample_value, Integral) or isinstance(sample_value, bool):
        raise TypeError("Depth sample_count must be an integer.")
    sample_count = int(sample_value)
    refinement_value = options.pop("coarse_to_fine")
    if not isinstance(refinement_value, Mapping):
        raise TypeError("coarse_to_fine must be a mapping.")
    refinement = dict(refinement_value)
    expected_refinement = {
        "enabled",
        "fine_sample_count",
        "uniform_weight_floor",
    }
    if set(refinement) != expected_refinement:
        raise TypeError(
            "coarse_to_fine must contain exactly enabled, fine_sample_count, "
            "and uniform_weight_floor."
        )
    enabled = refinement.pop("enabled")
    if type(enabled) is not bool:
        raise TypeError("coarse_to_fine.enabled must be boolean.")
    fine_sample_value = refinement.pop("fine_sample_count")
    if not isinstance(fine_sample_value, Integral) or isinstance(
        fine_sample_value, bool
    ):
        raise TypeError("fine_sample_count must be an integer.")
    fine_sample_count = int(fine_sample_value)
    floor_value = refinement.pop("uniform_weight_floor")
    if not isinstance(floor_value, Real) or isinstance(floor_value, bool):
        raise TypeError("uniform_weight_floor must be numeric.")
    uniform_weight_floor = float(floor_value)
    if sample_count < 2:
        raise ValueError("Depth sample_count must be at least two.")
    if fine_sample_count < 1:
        raise ValueError("fine_sample_count must be positive.")
    if not 0.0 <= uniform_weight_floor <= 1.0:
        raise ValueError("uniform_weight_floor must lie between zero and one.")
    return DepthSamplingSettings(
        sample_count=sample_count,
        reference_log_tau500_bounds=bounds,
        coarse_to_fine_enabled=enabled,
        fine_sample_count=fine_sample_count,
        uniform_weight_floor=uniform_weight_floor,
    )


def resolve_objective_weighting(
    wavelength_angstrom: torch.Tensor,
    *,
    weight_config: Mapping[str, Any] | None,
    wavelength_weights: Any,
    wavelength_exclude_windows_angstrom: Any,
    continuum_indices: Any,
    atlas_continuum_radiance_w_m3_sr: float,
) -> ObjectiveWeighting:
    """Validate every fixed tensor and scalar used by the Stokes objective."""

    if continuum_indices is None:
        raise TypeError("continuum_indices must be provided explicitly.")
    raw_continuum_indices = torch.as_tensor(continuum_indices)
    if (
        raw_continuum_indices.dtype is torch.bool
        or torch.is_floating_point(raw_continuum_indices)
        or torch.is_complex(raw_continuum_indices)
    ):
        raise TypeError("continuum_indices must contain integers.")
    continuum_index_tensor = raw_continuum_indices.to(dtype=torch.long)
    if continuum_index_tensor.ndim != 1 or continuum_index_tensor.numel() < 1:
        raise ValueError(
            "continuum_indices must be a non-empty one-dimensional sequence."
        )
    if torch.any(continuum_index_tensor < 0) or torch.any(
        continuum_index_tensor >= wavelength_angstrom.numel()
    ):
        raise ValueError("continuum_indices contains an out-of-range wavelength index.")
    if torch.unique(continuum_index_tensor).numel() != continuum_index_tensor.numel():
        raise ValueError("continuum_indices must not contain duplicates.")

    if not isinstance(atlas_continuum_radiance_w_m3_sr, Real) or isinstance(
        atlas_continuum_radiance_w_m3_sr, bool
    ):
        raise TypeError("atlas_continuum_radiance_w_m3_sr must be numeric.")
    radiance = float(atlas_continuum_radiance_w_m3_sr)
    if not math.isfinite(radiance) or radiance <= 0.0:
        raise ValueError(
            "atlas_continuum_radiance_w_m3_sr must be finite and positive."
        )

    if not isinstance(weight_config, Mapping):
        raise TypeError("weight_config must be a mapping.")
    component_options = deepcopy(dict(weight_config))
    if set(component_options) != set(STOKES_COMPONENTS):
        raise KeyError(
            f"Stokes weights must contain exactly {list(STOKES_COMPONENTS)}."
        )
    resolved_weights: dict[str, float] = {}
    for component in STOKES_COMPONENTS:
        value = component_options[component]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"Stokes weight {component} must be a fixed number; "
                "schedules are unsupported."
            )
        resolved_weights[component] = float(value)
    stokes_weights = torch.tensor(
        [resolved_weights[component] for component in STOKES_COMPONENTS],
        dtype=torch.float32,
    )
    if (
        stokes_weights.shape != (4,)
        or not torch.isfinite(stokes_weights).all()
        or torch.any(stokes_weights < 0)
    ):
        raise ValueError("Stokes weights must be finite and non-negative.")
    if not torch.any(stokes_weights > 0):
        raise ValueError("At least one Stokes weight must be nonzero.")
    # Component weights express relative scientific preference only. Keeping
    # their sum fixed prevents a global rescaling from silently changing the
    # balance between the data likelihood and the physical objectives.

    if wavelength_weights is None:
        spectral_weights = torch.ones_like(wavelength_angstrom)
    else:
        spectral_weights = torch.as_tensor(
            wavelength_weights,
            dtype=wavelength_angstrom.dtype,
            device=wavelength_angstrom.device,
        )
        if spectral_weights.shape != wavelength_angstrom.shape:
            raise ValueError(
                "wavelength_weights must have the same shape as wavelength_angstrom."
            )
    if not torch.isfinite(spectral_weights).all() or torch.any(spectral_weights < 0):
        raise ValueError("wavelength_weights must be finite and non-negative.")

    if wavelength_exclude_windows_angstrom is None:
        raise TypeError("wavelength_exclude_windows_angstrom must be explicit.")
    exclusion_windows: list[list[float]] = []
    for window in wavelength_exclude_windows_angstrom:
        if len(window) != 2:
            raise ValueError(
                "Each wavelength exclusion window must contain [minimum, maximum]."
            )
        minimum, maximum = map(float, window)
        if (
            not torch.isfinite(torch.tensor((minimum, maximum))).all()
            or minimum > maximum
        ):
            raise ValueError(
                "Wavelength exclusion windows must contain finite, ordered bounds."
            )
        exclusion_windows.append([minimum, maximum])
        excluded = (wavelength_angstrom >= minimum) & (wavelength_angstrom <= maximum)
        spectral_weights = spectral_weights.masked_fill(excluded, 0.0)
    if not torch.any(spectral_weights > 0):
        raise ValueError("At least one wavelength must have positive objective weight.")
    return ObjectiveWeighting(
        continuum_indices=continuum_index_tensor,
        atlas_continuum_radiance_w_m3_sr=radiance,
        stokes_weight_config=resolved_weights,
        stokes_weights=stokes_weights,
        wavelength_weights=spectral_weights,
        wavelength_exclude_windows_angstrom=exclusion_windows,
    )


__all__ = [
    "DepthSamplingSettings",
    "ObjectiveWeighting",
    "STOKES_COMPONENTS",
    "resolve_depth_sampling",
    "resolve_objective_weighting",
]
