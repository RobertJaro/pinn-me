"""Depth-quadrature settings for development-only numerical reference solvers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

@dataclass(frozen=True, slots=True)
class DepthSamplingSettings:
    """Resolved coarse and adaptive depth-quadrature settings."""

    sample_count: int
    coarse_to_fine_enabled: bool
    fine_sample_count: int
    uniform_weight_floor: float
    reference_log_tau500_bounds: tuple[float, float] = (-5.0, 1.0)
    randomize_training: bool = True
    height_sampling: str = "reference"

    def model_configuration(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "reference_log_tau500_bounds": self.reference_log_tau500_bounds,
            "randomize_training": self.randomize_training,
            "height_sampling": self.height_sampling,
            "coarse_to_fine": {
                "enabled": self.coarse_to_fine_enabled,
                "fine_sample_count": self.fine_sample_count,
                "uniform_weight_floor": self.uniform_weight_floor,
            },
        }


def resolve_depth_sampling(
    config: Mapping[str, Any],
) -> DepthSamplingSettings:
    """Decode the closed depth-sampling option set."""

    if not isinstance(config, Mapping):
        raise TypeError("depth_sampling_config must be a mapping.")
    options = deepcopy(dict(config))
    randomize_training = options.pop("randomize_training", True)
    if type(randomize_training) is not bool:
        raise TypeError("randomize_training must be boolean.")
    height_sampling = options.pop("height_sampling", "reference")
    if height_sampling not in ("reference", "uniform"):
        raise ValueError("height_sampling must be 'reference' or 'uniform'.")
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
        randomize_training=randomize_training,
        height_sampling=height_sampling,
    )

