"""Plain multimodal forward model shared by setup checks and joint fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .data_terms import (
    DataTermBatchResult,
    ObservationDataTerm,
    SharedObjectiveTerm,
    SharedTermResult,
)


@dataclass(frozen=True, slots=True)
class JointEvaluation:
    """Fully assembled objective without any training-framework lifecycle."""

    loss: torch.Tensor
    streams: Mapping[str, DataTermBatchResult]
    weighted_likelihoods: Mapping[str, torch.Tensor]
    nuisance_losses: Mapping[str, torch.Tensor]
    shared_terms: Mapping[str, SharedTermResult]
    stream_weights: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def training_metrics(self) -> dict[str, torch.Tensor]:
        """Actual weighted contributions, with no disabled or duplicate losses."""
        values = {"loss": self.loss}
        values.update(
            {
                f"streams.{name}.data_loss": value
                for name, value in self.weighted_likelihoods.items()
            }
        )
        values.update(
            {
                f"streams.{name}.weight": value
                for name, value in self.stream_weights.items()
            }
        )
        for name, result in self.streams.items():
            for metric in (
                "qu_weight_factor",
                "line_of_sight_velocity_correction_m_per_s",
                "phase_rotation_active",
                "phase_binary_weight",
                "phase_binary_mean_sin2",
                "phase_binary_loss",
            ):
                if metric in result.metrics:
                    values[f"streams.{name}.{metric}"] = result.metrics[metric]
        for name, result in self.shared_terms.items():
            if not result.active:
                continue
            components = result.component_losses or {"loss": result.loss}
            values.update(
                {
                    f"shared.{name}.{component}": value
                    for component, value in components.items()
                }
            )
        values.update(
            {f"nuisance.{name}": value for name, value in self.nuisance_losses.items()}
        )
        return values

    def validation_metrics(self) -> dict[str, torch.Tensor]:
        """One data loss and one fit-error measure per observation stream."""
        values = {}
        for name, result in self.streams.items():
            values[f"streams.{name}.data_loss"] = result.likelihood_loss
            for metric in (
                "residual_rms",
                "asinh_rmse",
                "line_of_sight_velocity_correction_m_per_s",
                "phase_rotation_active",
                "phase_binary_mean_sin2",
            ):
                if metric in result.metrics:
                    values[f"streams.{name}.{metric}"] = result.metrics[metric]
        return values

    def scalar_metrics(self) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {"loss": self.loss}
        for stream_id, result in self.streams.items():
            values[f"streams.{stream_id}.likelihood"] = result.likelihood_loss
            if stream_id in self.weighted_likelihoods:
                values[f"streams.{stream_id}.weighted_likelihood"] = (
                    self.weighted_likelihoods[stream_id]
                )
            for name, value in result.component_losses.items():
                values[f"streams.{stream_id}.components.{name}"] = value
            for name, value in result.metrics.items():
                values[f"streams.{stream_id}.metrics.{name}"] = value
        for name, value in self.nuisance_losses.items():
            values[f"nuisance.{name}"] = value
        values.update(
            {
                f"streams.{name}.weight": value
                for name, value in self.stream_weights.items()
            }
        )
        for term_id, result in self.shared_terms.items():
            values[f"shared.{term_id}.loss"] = result.loss
            for name, value in result.component_losses.items():
                values[f"shared.{term_id}.components.{name}"] = value
            for name, value in result.metrics.items():
                values[f"shared.{term_id}.metrics.{name}"] = value
        return values


class JointForwardModel(nn.Module):
    """Register one atmosphere and an open set of observation data terms."""

    def __init__(
        self,
        atmosphere_model: nn.Module,
        terms: Mapping[str, ObservationDataTerm],
        weights: Mapping[str, float],
        shared_terms: Mapping[str, SharedObjectiveTerm] | None = None,
        *,
        weight_schedules: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(atmosphere_model, nn.Module):
            raise TypeError("atmosphere_model must be a torch module.")
        if not terms or set(terms) != set(weights):
            raise ValueError("terms and weights must have the same non-empty keys.")
        invalid_ids = [
            name
            for name in terms
            if not isinstance(name, str)
            or not name
            or not name.replace("_", "a").isalnum()
            or not name[0].isalpha()
            or not name.islower()
        ]
        if invalid_ids:
            raise ValueError(f"Invalid joint stream identifiers: {invalid_ids}.")
        if any(not isinstance(term, ObservationDataTerm) for term in terms.values()):
            raise TypeError("Every joint term must be an ObservationDataTerm.")
        shared_terms = {} if shared_terms is None else dict(shared_terms)
        if set(terms).intersection(shared_terms):
            raise ValueError(
                "Observation and shared-term identifiers must be disjoint."
            )
        if any(
            not isinstance(name, str)
            or not name
            or not name.replace("_", "a").isalnum()
            or not name[0].isalpha()
            or not name.islower()
            for name in shared_terms
        ):
            raise ValueError("Shared-term identifiers must be safe lowercase IDs.")
        if any(
            not isinstance(term, SharedObjectiveTerm) for term in shared_terms.values()
        ):
            raise TypeError("Every shared term must be a SharedObjectiveTerm.")
        normalized_weights = {name: float(value) for name, value in weights.items()}
        if any(
            not torch.isfinite(torch.tensor(value)) or value < 0.0
            for value in normalized_weights.values()
        ):
            raise ValueError("Joint stream weights must be finite and non-negative.")
        self.atmosphere_model = atmosphere_model
        self.terms = nn.ModuleDict(dict(terms))
        self.shared_objectives = nn.ModuleDict(shared_terms)
        self.weights = normalized_weights
        schedules = {} if weight_schedules is None else dict(weight_schedules)
        unknown_schedules = set(schedules) - set(terms)
        if unknown_schedules:
            raise ValueError(
                "Weight schedules must identify configured streams; unknown: "
                f"{sorted(unknown_schedules)}"
            )
        self.weight_schedules = {}
        for name, schedule in schedules.items():
            if not isinstance(schedule, Mapping):
                raise TypeError(f"Weight schedule for {name!r} must be a mapping.")
            required = {"initial_weight", "initial_steps", "ramp_steps"}
            if set(schedule) != required:
                raise ValueError(
                    f"Weight schedule for {name!r} must contain exactly "
                    f"{sorted(required)}."
                )
            initial_weight = float(schedule["initial_weight"])
            initial_steps = schedule["initial_steps"]
            ramp_steps = schedule["ramp_steps"]
            if not torch.isfinite(torch.tensor(initial_weight)) or initial_weight < 0.0:
                raise ValueError(
                    f"Initial weight schedule value for {name!r} must be finite "
                    "and non-negative."
                )
            if type(initial_steps) is not int or initial_steps < 0:
                raise ValueError(
                    f"Initial weight schedule steps for {name!r} must be a "
                    "non-negative integer."
                )
            if type(ramp_steps) is not int or ramp_steps < 0:
                raise ValueError(
                    f"Weight schedule ramp steps for {name!r} must be a "
                    "non-negative integer."
                )
            if initial_steps == 0 and ramp_steps == 0:
                raise ValueError(
                    f"Weight schedule for {name!r} requires initial_steps or "
                    "ramp_steps."
                )
            self.weight_schedules[name] = {
                "initial_weight": initial_weight,
                "initial_steps": initial_steps,
                "ramp_steps": ramp_steps,
            }
        self.current_step = 0
        # Kept outside the module state because these are optimizer policy
        # variables, not scientific model parameters. They are checkpointed
        # explicitly by the Lightning wrapper.
        self._adaptive_multipliers: dict[str, float] = {}

    def adaptive_multiplier(self, key: str) -> float:
        """Return the current policy multiplier for one objective component."""

        return float(self._adaptive_multipliers.get(key, 1.0))

    def set_adaptive_multiplier(self, key: str, value: float) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("Adaptive multiplier keys must be non-empty strings.")
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise ValueError("Adaptive multipliers must be finite and positive.")
        self._adaptive_multipliers[key] = value

    def adaptive_multiplier_state(self) -> dict[str, float]:
        """Return a detached serializable snapshot of balancing policy state."""

        return dict(self._adaptive_multipliers)

    def load_adaptive_multiplier_state(self, state: Mapping[str, float] | None) -> None:
        if state is None:
            return
        if not isinstance(state, Mapping):
            raise TypeError("Adaptive multiplier state must be a mapping.")
        self._adaptive_multipliers = {}
        for key, value in state.items():
            self.set_adaptive_multiplier(str(key), value)

    def set_step(self, step: int) -> None:
        """Set the optimizer-policy step used by stream weight schedules."""

        if type(step) is not int or step < 0:
            raise ValueError("Joint objective step must be a non-negative integer.")
        self.current_step = step
        atmosphere_set_step = getattr(self.atmosphere_model, "set_step", None)
        if atmosphere_set_step is not None:
            atmosphere_set_step(step)

    def effective_stream_weight(self, stream_id: str) -> float:
        """Return a stream's scheduled weight at the current fitting step."""

        base = self.weights[stream_id]
        schedule = self.weight_schedules.get(stream_id)
        if schedule is None:
            return base
        initial = schedule["initial_weight"]
        hold = schedule["initial_steps"]
        ramp = schedule["ramp_steps"]
        if self.current_step < hold:
            return initial
        if ramp <= 0:
            return base
        fraction = min(1.0, max(0.0, (self.current_step - hold) / ramp))
        return initial + (base - initial) * fraction

    def evaluate_batches(
        self,
        batches: Mapping[str, Mapping[str, Any]],
    ) -> JointEvaluation:
        """Evaluate each supplied stream and assemble its ready-to-enable loss."""

        if set(batches) != set(self.terms):
            raise KeyError(
                "Joint batches must match configured streams exactly; "
                f"got {sorted(batches)}, expected {sorted(self.terms)}."
            )
        results: dict[str, DataTermBatchResult] = {}
        weighted: dict[str, torch.Tensor] = {}
        nuisance: dict[str, torch.Tensor] = {}
        shared: dict[str, SharedTermResult] = {}
        stream_weights: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for stream_id, term in self.terms.items():
            stream_weight = self.effective_stream_weight(stream_id)
            active = stream_weight > 0.0
            # A zero-weight stream remains available for forward diagnostics,
            # but neither its likelihood nor its calibration priors may fit.
            with torch.set_grad_enabled(torch.is_grad_enabled() and active):
                result = term.evaluate_batch(batches[stream_id])
            likelihood = result.likelihood_loss
            if not getattr(term, "distributed_channel_reduction", False):
                from prom3theus.core.distributed import distributed_mean
                likelihood = distributed_mean(likelihood, result.sample_count)
            if not active:
                likelihood = likelihood.detach()
            contribution = (
                likelihood
                * stream_weight
                * self.adaptive_multiplier(f"streams.{stream_id}")
            )
            if stream_id in self.weight_schedules:
                stream_weights[stream_id] = likelihood.new_tensor(stream_weight)
            results[stream_id] = result
            if active:
                weighted[stream_id] = contribution
            total = contribution if total is None else total + contribution
            if not active:
                continue
            for name, value in term.nuisance_losses().items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Nuisance-loss names must be non-empty strings.")
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise TypeError("Nuisance losses must be scalar tensors.")
                qualified = f"{stream_id}.{name}"
                contribution = value * self.adaptive_multiplier(f"streams.{stream_id}")
                nuisance[qualified] = contribution
                total = contribution if total is None else total + contribution
        # Refresh shared reference data before any objective consumes it.
        for term in self.shared_objectives.values():
            prepare = getattr(term, "prepare", None)
            if prepare is not None:
                prepare(self.atmosphere_model)
        for term_id, term in self.shared_objectives.items():
            result = term.evaluate(self.atmosphere_model)
            if result.component_losses:
                components = {
                    name: value
                    * self.adaptive_multiplier(f"shared.{term_id}.{name}")
                    for name, value in result.component_losses.items()
                }
                shared_result = SharedTermResult(
                    loss=sum(components.values(), start=result.loss.new_zeros(())),
                    component_losses=components,
                    metrics=result.metrics,
                    active=result.active,
                )
            else:
                loss = result.loss * self.adaptive_multiplier(f"shared.{term_id}.loss")
                shared_result = SharedTermResult(
                    loss=loss,
                    component_losses={},
                    metrics=result.metrics,
                    active=result.active,
                )
            shared[term_id] = shared_result
            total = shared_result.loss if total is None else total + shared_result.loss
        if total is None:  # pragma: no cover - constructor rejects empty terms
            raise RuntimeError("Joint evaluation produced no objective.")
        if not torch.isfinite(total):
            raise FloatingPointError("The assembled joint objective is non-finite.")
        return JointEvaluation(
            loss=total,
            streams=results,
            weighted_likelihoods=weighted,
            nuisance_losses=nuisance,
            shared_terms=shared,
            stream_weights=stream_weights,
        )


__all__ = ["JointEvaluation", "JointForwardModel"]
