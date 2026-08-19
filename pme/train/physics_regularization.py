"""Composition of continuous physics constraints with the Stokes objective.

This module owns the small amount of orchestration between an explicit global
collocation domain, the collocation sampler, constraint residuals, and their
weight schedules.  It intentionally knows nothing about the Stokes forward
model or data-loader batches.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence
import warnings

import torch

from pme.train.collocation import (
    LongitudeInterval,
    SmoothSphericalCollocationSampler,
    SphericalCollocationConfig,
    SphericalCollocationDomain,
)
from pme.train.physics import (
    CONSTRAINT_NAMES,
    PhysicsConstraintModule,
    PhysicsWeightSchedule,
)


STOKES_COMPONENTS = ("I", "Q", "U", "V")
PHYSICS_CONSTRAINTS = frozenset(CONSTRAINT_NAMES)


@dataclass(frozen=True)
class PhysicsRegularizationResult:
    """The rank-local DDP-scaled physics objective and logging metrics."""

    total: torch.Tensor
    metrics: dict[str, torch.Tensor]


def split_stokes_and_physics_config(
    lambda_config: Mapping[str, Any] | None,
    physics_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Separate legacy mixed lambdas into Stokes and physics configuration.

    New configurations put constraint schedules under ``physics.constraints``.
    Physics keys in the historical top-level ``lambda`` map remain supported so
    existing experiments fail neither mysteriously nor halfway through parsing.
    """
    lambda_config = copy.deepcopy(dict(lambda_config or {}))
    physics_config = (
        None if physics_config is None else copy.deepcopy(dict(physics_config))
    )

    unknown = set(lambda_config).difference((*STOKES_COMPONENTS, *PHYSICS_CONSTRAINTS))
    if unknown:
        raise ValueError(
            f"Unknown lambda keys {sorted(unknown)}; expected Stokes components "
            f"{STOKES_COMPONENTS} or physics constraints {sorted(PHYSICS_CONSTRAINTS)}."
        )

    legacy_constraints = {
        name: lambda_config.pop(name)
        for name in tuple(lambda_config)
        if name in PHYSICS_CONSTRAINTS
    }
    if not legacy_constraints:
        return lambda_config, physics_config

    warnings.warn(
        "Physics weights in the top-level lambda mapping are deprecated; move "
        "them to physics.constraints.",
        DeprecationWarning,
        stacklevel=2,
    )
    physics_config = {} if physics_config is None else physics_config
    configured_constraints = physics_config.get("constraints")
    if configured_constraints:
        overlap = set(configured_constraints).intersection(legacy_constraints)
        if overlap:
            raise ValueError(
                f"Physics constraints {sorted(overlap)} are configured in both lambda "
                "and physics.constraints."
            )
        configured_constraints = {**legacy_constraints, **configured_constraints}
    else:
        configured_constraints = legacy_constraints
    physics_config["constraints"] = configured_constraints
    return lambda_config, physics_config


def _minimum_covering_longitude(
    bounds: Sequence[Mapping[str, Any]],
) -> LongitudeInterval:
    """Return the smallest circular arc containing every dataset interval."""
    two_pi = 2 * math.pi
    segments = []
    for item in bounds:
        offset_min, offset_max = map(float, item["longitude_offset_rad"])
        width = offset_max - offset_min
        if width < 0 or width > two_pi + 1e-12:
            raise ValueError(f"Invalid dataset longitude width: {width} rad.")
        if width >= two_pi - 1e-12:
            return LongitudeInterval(center=0.0, width=two_pi)

        start = (float(item["longitude_center_rad"]) + offset_min) % two_pi
        stop = start + width
        if stop <= two_pi:
            segments.append((start, stop))
        else:
            segments.extend(((start, two_pi), (0.0, stop - two_pi)))

    segments.sort()
    merged = []
    for start, stop in segments:
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    if not merged:
        raise ValueError("The training datasets contain no longitude coverage.")

    gaps = [
        (merged[index][1], merged[index + 1][0]) for index in range(len(merged) - 1)
    ]
    gaps.append((merged[-1][1], merged[0][0] + two_pi))
    gap_start, gap_stop = max(gaps, key=lambda gap: gap[1] - gap[0])
    uncovered_width = gap_stop - gap_start
    if uncovered_width <= 1e-12:
        return LongitudeInterval(center=0.0, width=two_pi)

    covering_width = two_pi - uncovered_width
    if covering_width <= 1e-12:
        raise ValueError(
            "The combined training longitude range must have nonzero width."
        )
    covering_start = gap_stop % two_pi
    return LongitudeInterval(
        center=covering_start + covering_width / 2,
        width=covering_width,
    )


def build_spherical_collocation_domain(
    train_datasets: Sequence,
    Rs_per_ds: float,
    radius_range_Rs: Sequence[float] | None = None,
) -> SphericalCollocationDomain:
    """Build one seam-safe global domain from all TRAIN acquisitions."""
    if not train_datasets:
        raise ValueError(
            "Physics regularization requires at least one training dataset."
        )
    Rs_per_ds = float(Rs_per_ds)
    if not math.isfinite(Rs_per_ds) or Rs_per_ds <= 0:
        raise ValueError("Rs_per_ds must be finite and positive.")

    missing_bounds = [
        getattr(dataset, "ds_id", str(index))
        for index, dataset in enumerate(train_datasets)
        if not getattr(dataset, "physics_bounds", None)
    ]
    if missing_bounds:
        raise RuntimeError(
            "The cached training data predates global physics-domain metadata for "
            f"datasets {missing_bounds[:5]}. Rebuild it with --reload or use a fresh "
            "work_directory before enabling physics regularization."
        )
    bounds = [copy.deepcopy(dataset.physics_bounds) for dataset in train_datasets]
    longitude = _minimum_covering_longitude(bounds)

    latitude_min = min(item["latitude_rad"][0] for item in bounds)
    latitude_max = max(item["latitude_rad"][1] for item in bounds)
    if latitude_max <= latitude_min:
        raise ValueError(
            "The combined training latitude range must have nonzero width."
        )
    time_min = min(item["normalized_time"] for item in bounds)
    time_max = max(item["normalized_time"] for item in bounds)

    if radius_range_Rs is None:
        radius_min_Rs = min(item["radius_Rs"][0] for item in bounds)
        radius_max_Rs = max(item["radius_Rs"][1] for item in bounds)
    else:
        if len(radius_range_Rs) != 2:
            raise ValueError("physics.radius_range_Rs must contain exactly two values.")
        radius_min_Rs, radius_max_Rs = map(float, radius_range_Rs)
    if not (math.isfinite(radius_min_Rs) and math.isfinite(radius_max_Rs)):
        raise ValueError("physics.radius_range_Rs values must be finite.")
    if radius_min_Rs <= 0 or radius_max_Rs < radius_min_Rs:
        raise ValueError("physics.radius_range_Rs must be positive and ordered.")

    return SphericalCollocationDomain(
        normalized_time_range=(time_min, time_max),
        latitude_range=(latitude_min, latitude_max),
        longitude=longitude,
        normalized_radius_range=(radius_min_Rs / Rs_per_ds, radius_max_Rs / Rs_per_ds),
        surface_radius=1.0 / Rs_per_ds,
    )


class PhysicsRegularization:
    """Evaluate scheduled constraints at independent continuous coordinates."""

    _CONFIG_KEYS = {
        "constraints",
        "num_points",
        "seed",
        "radial_measure",
        "normalization",
        "epsilon",
        "domain",
        "radius_range_Rs",
    }

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        domain: SphericalCollocationDomain | None,
        *,
        vector_potential: bool,
    ):
        self._enabled = config is not None
        if not self._enabled:
            self.config = None
            self.domain = None
            self.sampler = None
            self.constraint_module = None
            self.schedules = {}
            return

        config = copy.deepcopy(dict(config))
        unknown = set(config).difference(self._CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Unknown physics configuration keys: {sorted(unknown)}.")
        if config.get("domain", "full_train") != "full_train":
            raise ValueError("physics.domain currently supports only 'full_train'.")
        if domain is None:
            raise ValueError(
                "Physics regularization requires an explicit global training domain."
            )

        constraints = config.get("constraints")
        if not isinstance(constraints, Mapping) or not constraints:
            raise ValueError(
                "physics.constraints must contain at least one weighted constraint."
            )
        unknown_constraints = set(constraints).difference(PHYSICS_CONSTRAINTS)
        if unknown_constraints:
            raise ValueError(
                f"Unknown physics constraints: {sorted(unknown_constraints)}."
            )
        if "gauge" in constraints and not vector_potential:
            raise ValueError(
                "The gauge constraint requires model.vector_potential: true."
            )
        if vector_potential and "divergence" in constraints:
            warnings.warn(
                "divergence is analytically redundant when B=curl(A); the sampled "
                "penalty only measures automatic-differentiation roundoff.",
                UserWarning,
                stacklevel=2,
            )
        if domain.normalized_time_range[0] == domain.normalized_time_range[1] and {
            "induction",
            "dB_dt",
        }.intersection(constraints):
            raise ValueError(
                "Temporal physics constraints require more than one training time."
            )

        self.domain = domain
        self.schedules = {
            name: PhysicsWeightSchedule.from_config(weight)
            for name, weight in constraints.items()
        }
        sampler_config = SphericalCollocationConfig(
            global_size=config.get("num_points", 4096),
            seed=config.get("seed", 0),
            radial_measure=config.get("radial_measure", "volume"),
        )
        self.sampler = SmoothSphericalCollocationSampler(domain, sampler_config)
        self.constraint_module = PhysicsConstraintModule(
            normalization=config.get("normalization", "raw"),
            epsilon=float(config.get("epsilon", 1e-6)),
        )
        radius_range_Rs = config.get("radius_range_Rs")
        self.config = {
            "domain": "full_train",
            "num_points": sampler_config.global_size,
            "seed": sampler_config.seed,
            "radial_measure": sampler_config.radial_measure,
            "normalization": self.constraint_module.normalization,
            "epsilon": self.constraint_module.epsilon,
            "radius_range_Rs": None
            if radius_range_Rs is None
            else list(radius_range_Rs),
            "constraints": {
                name: schedule.configuration()
                for name, schedule in self.schedules.items()
            },
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configuration(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        metadata = {
            **copy.deepcopy(self.config),
            "collocation_domain": self.domain.canonical_metadata(),
        }
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        metadata["fingerprint"] = hashlib.sha256(payload).hexdigest()
        return metadata

    def evaluate(
        self,
        parameter_model,
        *,
        global_step: int,
        reference: torch.Tensor,
    ) -> PhysicsRegularizationResult:
        zero = reference.new_zeros(())
        if not self.enabled:
            return PhysicsRegularizationResult(zero, {})

        weights = {
            name: schedule.value_at(global_step)
            for name, schedule in self.schedules.items()
        }
        active = tuple(name for name, weight in weights.items() if weight > 0)
        if not active:
            return PhysicsRegularizationResult(zero, {"physics.loss": zero})

        collocation = self.sampler.sample(
            int(global_step),
            device=reference.device,
            dtype=reference.dtype,
        )
        output = parameter_model(collocation.coords)
        b = torch.cat([output["b_x"], output["b_y"], output["b_z"]], dim=-1)
        v = torch.cat([output["v_x"], output["v_y"], output["v_z"]], dim=-1)
        result = self.constraint_module(
            b=b,
            v=v,
            coords=collocation.coords,
            selected=active,
            a_jac_matrix=output.get("a_jac_matrix"),
        )

        total = zero
        metrics = {}
        for name in active:
            loss = collocation.ddp_scaled_mean(result.losses[name])
            weighted = loss * weights[name]
            total = total + weighted
            metrics[f"physics.{name}"] = loss
        if not torch.isfinite(total):
            raise RuntimeError("Encountered a non-finite physics regularization loss.")
        metrics["physics.loss"] = total
        return PhysicsRegularizationResult(total, metrics)
