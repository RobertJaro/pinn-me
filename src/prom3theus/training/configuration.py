"""Strict runtime option decoding for LTE training.

This module owns value validation and normalization only.  It deliberately
does not construct trainable modules, so :mod:`prom3theus.training.lightning`
can remain the lifecycle/orchestration boundary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

import torch


STOKES_COMPONENTS = ("I", "Q", "U", "V")


@dataclass(frozen=True, slots=True)
class CoordinateGrids:
    """Validated optical-depth and observed-wavelength grids."""

    log_tau500: torch.Tensor
    wavelength_angstrom: torch.Tensor


@dataclass(frozen=True, slots=True)
class VectorRegularizationSettings:
    """Resolved warm-up regularization settings."""

    enabled: bool
    magnetic_weight: float
    velocity_weight: float
    decay_steps: int

    def metadata(
        self,
        *,
        magnetic_scale_gauss: float,
        velocity_scale_m_per_s: float,
    ) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "magnetic_weight": self.magnetic_weight,
            "velocity_weight": self.velocity_weight,
            "decay_steps": self.decay_steps,
            "schedule": "linear decay to zero",
            "magnetic_scale_gauss": magnetic_scale_gauss,
            "velocity_scale_m_per_s": velocity_scale_m_per_s,
        }


@dataclass(frozen=True, slots=True)
class DepthSamplingSettings:
    """Resolved coarse and adaptive depth-quadrature settings."""

    sample_count: int
    coarse_to_fine_enabled: bool
    fine_sample_count: int
    uniform_weight_floor: float

    def model_configuration(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
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


@dataclass(frozen=True, slots=True)
class LearningRateSettings:
    """One fixed rate or a validated exponential schedule."""

    start: float
    schedule: dict[str, float | int | str] | None
    model_configuration: float | dict[str, float | int | str]


def resolve_coordinate_grids(
    log_tau500: Any,
    wavelength_angstrom: Any,
) -> CoordinateGrids:
    """Convert and validate the two one-dimensional solver grids."""

    depth = torch.as_tensor(log_tau500, dtype=torch.float32)
    wavelength = torch.as_tensor(wavelength_angstrom, dtype=torch.float32)
    if depth.ndim != 1 or depth.numel() < 2:
        raise ValueError("log_tau500 must be one-dimensional with at least two points.")
    if not torch.isfinite(depth).all() or not torch.all(depth[1:] > depth[:-1]):
        raise ValueError("log_tau500 must be finite and increase from top to bottom.")
    if wavelength.ndim != 1 or wavelength.numel() < 2:
        raise ValueError("wavelength_angstrom must be a one-dimensional observed grid.")
    if not torch.isfinite(wavelength).all() or not torch.all(
        wavelength[1:] > wavelength[:-1]
    ):
        raise ValueError("wavelength_angstrom must be finite and strictly increasing.")
    return CoordinateGrids(depth, wavelength)


def resolve_vector_regularization(
    config: Mapping[str, Any],
) -> VectorRegularizationSettings:
    """Decode the closed vector-regularization option set."""

    if not isinstance(config, Mapping):
        raise TypeError("vector_regularization_config must be a mapping.")
    options = deepcopy(dict(config))
    expected = {
        "enabled",
        "magnetic_weight",
        "velocity_weight",
        "decay_steps",
    }
    if set(options) != expected:
        raise TypeError(
            "vector_regularization_config must contain exactly enabled, "
            "magnetic_weight, velocity_weight, and decay_steps."
        )
    enabled = options["enabled"]
    if type(enabled) is not bool:
        raise TypeError("vector_regularization_config.enabled must be boolean.")
    magnetic_value = options["magnetic_weight"]
    velocity_value = options["velocity_weight"]
    for name, value in (
        ("magnetic_weight", magnetic_value),
        ("velocity_weight", velocity_value),
    ):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"vector_regularization_config.{name} must be numeric.")
    magnetic_weight = float(magnetic_value)
    velocity_weight = float(velocity_value)
    decay_value = options["decay_steps"]
    if not isinstance(decay_value, Integral) or isinstance(decay_value, bool):
        raise TypeError("vector_regularization_config.decay_steps must be an integer.")
    decay_steps = int(decay_value)
    for name, value in (
        ("magnetic_weight", magnetic_weight),
        ("velocity_weight", velocity_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"vector_regularization_config.{name} must be finite and non-negative."
            )
    if decay_steps < 0 or (enabled and decay_steps < 1):
        raise ValueError(
            "vector_regularization_config.decay_steps must be non-negative "
            "and positive when regularization is enabled."
        )
    if enabled and magnetic_weight == 0 and velocity_weight == 0:
        raise ValueError(
            "Enabled vector regularization requires a positive magnetic or "
            "velocity weight."
        )
    if not enabled and any(
        value != 0 for value in (magnetic_weight, velocity_weight, decay_steps)
    ):
        raise ValueError(
            "Disabled vector regularization requires zero weights and decay_steps."
        )
    return VectorRegularizationSettings(
        enabled=enabled,
        magnetic_weight=magnetic_weight,
        velocity_weight=velocity_weight,
        decay_steps=decay_steps,
    )


def resolve_depth_sampling(
    config: Mapping[str, Any],
) -> DepthSamplingSettings:
    """Decode the closed depth-sampling option set."""

    if not isinstance(config, Mapping):
        raise TypeError("depth_sampling_config must be a mapping.")
    options = deepcopy(dict(config))
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
    stokes_weights = stokes_weights / stokes_weights.sum()

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


def resolve_learning_rate(
    learning_rate: Mapping[str, Any],
) -> LearningRateSettings:
    """Resolve the exact exponential learning-rate schedule mapping."""

    if not isinstance(learning_rate, Mapping):
        raise TypeError("learning_rate must be a schedule mapping.")
    options = dict(learning_rate)
    if set(options) != {"start", "end", "iterations"}:
        raise TypeError(
            "learning_rate must contain exactly start, end, and iterations."
        )
    start_value = options["start"]
    end_value = options["end"]
    iterations = options["iterations"]
    if any(
        not isinstance(value, Real) or isinstance(value, bool)
        for value in (start_value, end_value)
    ):
        raise TypeError("learning_rate.start and end must be numeric.")
    start = float(start_value)
    end = float(end_value)
    if iterations == "auto":
        pass
    elif type(iterations) is not int or iterations <= 0:
        raise ValueError("learning_rate.iterations must be positive or 'auto'.")
    schedule: dict[str, float | int | str] = {
        "start": start,
        "end": end,
        "iterations": iterations,
    }
    configuration: dict[str, float | int | str] = deepcopy(schedule)
    if any(not math.isfinite(value) or value <= 0 for value in (start, end)):
        raise ValueError("Learning rates must be finite and positive.")
    return LearningRateSettings(
        start=start,
        schedule=schedule,
        model_configuration=configuration,
    )


__all__ = [
    "CoordinateGrids",
    "DepthSamplingSettings",
    "LearningRateSettings",
    "ObjectiveWeighting",
    "STOKES_COMPONENTS",
    "VectorRegularizationSettings",
    "resolve_coordinate_grids",
    "resolve_depth_sampling",
    "resolve_learning_rate",
    "resolve_objective_weighting",
    "resolve_vector_regularization",
]
