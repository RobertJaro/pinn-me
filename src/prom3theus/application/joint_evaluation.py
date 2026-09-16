"""Evaluate joint objectives, gradient probes, and quadrature convergence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from prom3theus.config.joint_schema import AIAOpticallyThinDataTermConfig
from prom3theus.inversion.data_terms import (
    AtmosphereRegularizationTerm,
    PhysicsConstraintTerm,
    SharedObjectiveTerm,
)

from prom3theus.inversion.data_terms.potential_boundary import ProgressivePotentialBoundary

from .batches import _batch_length, _indexed_batch, _slice_batch
from .joint_contracts import JointRuntime, JointRuntimeEvaluation
from .runtime import _preserve_rng, _temporary_eval_mode, state_dict_sha256


def _gradient_batch(batch: Mapping[str, Any], count: int) -> dict[str, Any]:
    length = _batch_length(batch)
    count = min(int(count), length)
    channel = batch.get("channel_index")
    if not isinstance(channel, torch.Tensor) or channel.ndim != 1:
        return _slice_batch(batch, count)
    groups = torch.unique(channel, sorted=True)
    if count < groups.numel():
        return _slice_batch(batch, count)
    base, extra = divmod(count, groups.numel())
    selected = []
    for index, group in enumerate(groups):
        group_indices = torch.nonzero(channel == group, as_tuple=False).flatten()
        selected.append(group_indices[: base + (index < extra)])
    return _indexed_batch(batch, torch.cat(selected))


@contextmanager
def _temporary_physics_derivative_graph(module: nn.Module, *, enabled: bool):
    """Temporarily control higher-order PDE graphs without changing readiness."""

    if type(enabled) is not bool:
        raise TypeError("enabled must be boolean.")
    terms = [
        child for child in module.modules() if isinstance(child, PhysicsConstraintTerm)
    ]
    states = [term.create_graph for term in terms]
    for term in terms:
        term.create_graph = enabled
    try:
        yield
    finally:
        for term, state in zip(terms, states, strict=True):
            term.create_graph = state


def _scalar(value: torch.Tensor, name: str) -> float:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise TypeError(f"Reported metric {name!r} must be a scalar tensor.")
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError(f"Reported metric {name!r} is non-finite.")
    return result


def _capped_physics_counts(
    point_count: int,
    height_count: int,
    maximum_count: int,
) -> tuple[int, int]:
    """Keep a bounded, valid grouped sample shape for a physics probe."""

    if point_count == 0:
        return 0, 0
    target = min(point_count, maximum_count)
    capped_heights = min(height_count, target)
    original_points_per_height = point_count // height_count
    points_per_height = min(
        original_points_per_height,
        max(1, target // capped_heights),
    )
    return capped_heights * points_per_height, capped_heights


@contextmanager
def _bounded_shared_gradient_samples(
    term: SharedObjectiveTerm,
    *,
    maximum_count: int,
):
    """Bound shared-term probe memory while retaining its fitting semantics."""

    if isinstance(term, ProgressivePotentialBoundary):
        # Exercise the future fitting path during warmup without changing the
        # live targets or schedule. Swap buffers so graph references stay valid.
        buffers = term._buffers.copy()
        state, options, training = term.get_extra_state(), term.options, term.training
        term._buffers = {
            name: value if name == "potential_operator" else value.clone()
            for name, value in buffers.items()
        }
        term.options = {
            **options, "batch_size": min(options["batch_size"], maximum_count),
            "photosphere": {**options["photosphere"], "batch_size": min(options["photosphere"]["batch_size"], maximum_count)},
        }
        term.training = True
        term.set_step(max(term.step, options["start_step"] + options["ramp_steps"]))
        try:
            yield {"mode": "potential_after_warmup", "sample_count_per_face": term.options["batch_size"]}
        finally:
            operator = term.potential_operator
            term._buffers = buffers
            term.set_extra_state(state)
            term.potential_operator = operator
            term.options, term.training = options, training
        return

    if isinstance(term, PhysicsConstraintTerm):
        # Higher-order PDE parameter gradients are materially more expensive
        # than observation or value-prior gradients.  Eight samples per active
        # family are enough to prove the graph while keeping this setup check
        # safe even when gradient_sample_count is configured very large.
        physics_maximum_count = min(maximum_count, 8)
        names = (
            "volume_points_per_step",
            "height_layers_per_step",
            "upper_volume_points_per_step",
            "upper_height_layers_per_step",
            "upper_boundary_points_per_step",
            "side_boundary_points_per_step",
        )
        values = {name: getattr(term, name) for name in names}
        training = term.training
        term.volume_points_per_step, term.height_layers_per_step = (
            _capped_physics_counts(
                values["volume_points_per_step"],
                values["height_layers_per_step"],
                physics_maximum_count,
            )
        )
        term.upper_volume_points_per_step, term.upper_height_layers_per_step = (
            _capped_physics_counts(
                values["upper_volume_points_per_step"],
                values["upper_height_layers_per_step"],
                physics_maximum_count,
            )
        )
        term.upper_boundary_points_per_step = min(
            values["upper_boundary_points_per_step"], physics_maximum_count
        )
        side_count = values["side_boundary_points_per_step"]
        if side_count:
            side_count = min(side_count, max(4, physics_maximum_count))
            side_count -= side_count % 4
        term.side_boundary_points_per_step = side_count
        # Exercise the actual future-fitting path (random domain samples and
        # higher-order parameter graphs).  _gradient_smoke isolates and fixes
        # the RNG, so these small samples are reproducible.
        term.training = True
        metadata = {
            "mode": "training_sampler",
            "volume": term.volume_points_per_step,
            "upper_volume": term.upper_volume_points_per_step,
            "upper_boundary": term.upper_boundary_points_per_step,
            "side_boundary": term.side_boundary_points_per_step,
        }
        try:
            yield metadata
        finally:
            term.training = training
            for name, value in values.items():
                setattr(term, name, value)
        return

    if isinstance(term, AtmosphereRegularizationTerm):
        position = term.position_m
        time = term.time_hours
        flat_position = position.reshape(-1, 3)
        flat_time = time.reshape(-1, 1)
        count = min(maximum_count, flat_position.shape[0])
        indices = (
            torch.linspace(
                0,
                flat_position.shape[0] - 1,
                count,
                device=flat_position.device,
            )
            .round()
            .to(torch.long)
        )
        term.position_m = flat_position.index_select(0, indices)
        term.time_hours = flat_time.index_select(0, indices.to(flat_time.device))
        try:
            yield {"mode": "fixed_evenly_spaced", "sample_count": count}
        finally:
            term.position_m = position
            term.time_hours = time
        return

    yield {"mode": "term_owned"}


def _probe_scalar_losses(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[tuple[str, nn.Parameter]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Probe every scalar independently without populating ``.grad`` fields."""

    outcomes: dict[str, dict[str, Any]] = {}
    eligible = [
        name
        for name, loss in losses.items()
        if isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and loss.requires_grad
        and bool(torch.isfinite(loss.detach()))
    ]
    remaining = len(eligible)
    used_union: set[str] = set()
    parameter_values = tuple(parameter for _, parameter in parameters)
    for name, loss in losses.items():
        outcome: dict[str, Any] = {"success": False}
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            outcome["error"] = "loss is not a scalar tensor"
            outcomes[name] = outcome
            continue
        outcome["value"] = float(loss.detach().cpu())
        outcome["requires_grad"] = bool(loss.requires_grad)
        outcome["finite_loss"] = bool(torch.isfinite(loss.detach()))
        if not outcome["finite_loss"]:
            outcome["error"] = "loss is non-finite"
            outcomes[name] = outcome
            continue
        if not loss.requires_grad:
            outcome["error"] = "loss is detached from autograd"
            outcomes[name] = outcome
            continue
        remaining -= 1
        try:
            gradients = torch.autograd.grad(
                loss,
                parameter_values,
                allow_unused=True,
                retain_graph=remaining > 0,
            )
        except RuntimeError as error:
            outcome["error"] = str(error)
            outcomes[name] = outcome
            continue
        used = [
            parameter_name
            for (parameter_name, _), gradient in zip(parameters, gradients, strict=True)
            if gradient is not None
        ]
        finite = [
            gradient is None or bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ]
        norms = [
            float(gradient.detach().norm().cpu())
            for gradient in gradients
            if gradient is not None
        ]
        outcome.update(
            success=bool(used) and all(finite),
            used_parameter_count=len(used),
            unused_parameter_count=len(parameters) - len(used),
            all_finite=all(finite),
            global_l2_norm=math.sqrt(sum(value * value for value in norms)),
        )
        if not used:
            outcome["error"] = "loss is not connected to any model parameter"
        elif not all(finite):
            outcome["error"] = "loss produced non-finite parameter gradients"
        used_union.update(used)
        outcomes[name] = outcome
    return outcomes, used_union


def _gradient_smoke(
    runtime: JointRuntime,
) -> dict[str, Any]:
    parameters = [
        (name, parameter)
        for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad
    ]
    grad_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in parameters
    }
    if not parameters:
        raise RuntimeError(
            "Gradient smoke failed: the configured objective has no trainable "
            "parameters."
        )
    batches = {
        name: _gradient_batch(batch, runtime.config.dry_run.gradient_sample_count)
        for name, batch in runtime.validation_batches.items()
    }
    probes: dict[str, dict[str, Any]] = {}
    used_parameters: set[str] = set()
    shared_samples: dict[str, Any] = {}
    failure: Exception | None = None
    try:
        with _preserve_rng(), torch.enable_grad():
            for stream_id, term in runtime.model.terms.items():
                result = term.evaluate_batch(batches[stream_id])
                losses = {
                    f"streams.{stream_id}.likelihood": result.likelihood_loss,
                    **{
                        f"nuisance.{stream_id}.{name}": value
                        for name, value in term.nuisance_losses().items()
                    },
                }
                stream_probes, used = _probe_scalar_losses(losses, parameters)
                probes.update(stream_probes)
                used_parameters.update(used)
            for term_id, term in runtime.model.shared_objectives.items():
                with _bounded_shared_gradient_samples(
                    term,
                    maximum_count=runtime.config.dry_run.gradient_sample_count,
                ) as sample_metadata:
                    result = term.evaluate(runtime.model.atmosphere_model)
                shared_samples[term_id] = sample_metadata
                losses = {
                    f"shared.{term_id}.loss": result.loss,
                    **{
                        f"shared.{term_id}.components.{name}": value
                        for name, value in result.component_losses.items()
                    },
                }
                shared_probes, used = _probe_scalar_losses(losses, parameters)
                probes.update(shared_probes)
                used_parameters.update(used)
    except Exception as error:  # preserve .grad invariants before making it fatal
        failure = error
    grad_unchanged = all(
        (
            parameter.grad is None
            if grad_before[name] is None
            else parameter.grad is not None
            and torch.equal(grad_before[name], parameter.grad)
        )
        for name, parameter in parameters
    )
    if not grad_unchanged:
        raise RuntimeError("Gradient smoke evaluation mutated parameter .grad fields.")
    if failure is not None:
        raise RuntimeError(f"Gradient smoke evaluation failed: {failure}") from failure
    failed = [name for name, result in probes.items() if not result["success"]]
    if failed:
        details = "; ".join(
            f"{name}: {probes[name].get('error', 'unknown failure')}" for name in failed
        )
        raise RuntimeError(f"Gradient smoke failed for configured losses: {details}")
    outcome: dict[str, Any] = {
        "enabled": True,
        "success": True,
        "scope": (
            "every stream likelihood, nuisance prior, shared objective, and "
            "shared loss component"
        ),
        "parameter_count": len(parameters),
        "used_parameter_count": len(used_parameters),
        "unused_parameter_count": len(parameters) - len(used_parameters),
        "all_finite": True,
        "loss_count": len(probes),
        "losses": probes,
        "sample_count_by_stream": {
            name: _batch_length(batch) for name, batch in batches.items()
        },
        "shared_samples": shared_samples,
        "parameter_grad_fields_unchanged": True,
    }
    return outcome


def _quadrature_report(runtime: JointRuntime) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    stream_configs = {stream.id: stream for stream in runtime.config.streams}
    batch_size = runtime.config.diagnostics.visualization.ray_sampling.batch_size
    for stream_id, term in runtime.model.terms.items():
        stream = stream_configs[stream_id]
        if not isinstance(stream.data_term, AIAOpticallyThinDataTermConfig):
            continue
        predictions: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for sample_count in runtime.config.dry_run.quadrature_samples:
                batch = runtime.validation_batches[stream_id]
                parts = []
                for start in range(0, _batch_length(batch), batch_size):
                    indices = torch.arange(
                        start,
                        min(start + batch_size, _batch_length(batch)),
                        device=runtime.device,
                    )
                    result = term.synthesize(
                        _indexed_batch(batch, indices),
                        ray_samples=sample_count,
                        refine=False,
                    )
                    parts.append(result.prediction.detach())
                prediction = torch.cat(parts, dim=0)
                if prediction.shape[0] != _batch_length(batch):
                    raise RuntimeError(
                        f"AIA {stream_id!r} quadrature batching lost rays."
                    )
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(
                        f"AIA {stream_id!r} quadrature prediction is non-finite."
                    )
                predictions[sample_count] = prediction
        finest_count = runtime.config.dry_run.quadrature_samples[-1]
        finest = predictions[finest_count]
        denominator = float(finest.norm().cpu())
        denominator = max(denominator, torch.finfo(finest.dtype).tiny)
        reports[stream_id] = {
            "prediction": "raw_precalibration",
            "ray_batch_size": batch_size,
            "reference_ray_samples": finest_count,
            "samples": {
                str(sample_count): {
                    "relative_l2_to_finest": float((prediction - finest).norm().cpu())
                    / denominator,
                }
                for sample_count, prediction in predictions.items()
            },
        }
    return reports


def _runtime_metadata(runtime: JointRuntime) -> dict[str, Any]:
    stream_metadata = {}
    for stream_id, loaded in runtime.streams.items():
        term = runtime.model.terms[stream_id]
        term_metadata = getattr(term, "metadata", None)
        composition = getattr(term, "_composition", None)
        stream_metadata[stream_id] = {
            "prepared": loaded.prepared.metadata(),
            "loader": loaded.data_module.run_metadata(),
            "setup": dict(loaded.setup_metadata),
            "term": term_metadata() if callable(term_metadata) else {},
            "validation_sample_count": _batch_length(
                runtime.validation_batches[stream_id]
            ),
            "uses_shared_atmosphere": (
                getattr(composition, "atmosphere_model", runtime.model.atmosphere_model)
                is runtime.model.atmosphere_model
            ),
        }
    coordinate = runtime.scene.atmosphere_coordinate_metadata
    return {
        "resource_sets": list(runtime.resources),
        "resources": runtime.resources,
        "scene": coordinate,
        "streams": stream_metadata,
        "shared_objectives": list(runtime.model.shared_objectives),
        "shared_objective_contracts": {
            term_id: {
                "type": type(term).__name__,
                **(
                    {
                        "parameter_gradient_create_graph": bool(term.create_graph),
                        "dry_run_override": False,
                    }
                    if isinstance(term, PhysicsConstraintTerm)
                    else {}
                ),
            }
            for term_id, term in runtime.model.shared_objectives.items()
        },
        "shared_sampling": runtime.shared_metadata,
        "stream_weights": runtime.model.weights,
        "stream_weight_schedules": runtime.model.weight_schedules,
    }


def evaluate_joint_runtime(
    runtime: JointRuntime,
    *,
    evaluate_gradients: bool | None = None,
    evaluate_quadrature: bool = True,
) -> JointRuntimeEvaluation:
    """Forward-evaluate the configured objective and prove state immutability."""

    if not isinstance(runtime, JointRuntime):
        raise TypeError("evaluate_joint_runtime requires JointRuntime.")
    if evaluate_gradients is None:
        evaluate_gradients = runtime.config.dry_run.evaluate_gradients
    if type(evaluate_gradients) is not bool or type(evaluate_quadrature) is not bool:
        raise TypeError("Evaluation flags must be boolean.")
    before = state_dict_sha256(runtime.model)
    with _temporary_eval_mode(runtime.model):
        # Numerical setup checks need first spatial derivatives but no graph of
        # those derivatives.  The term reverts to its training-safe setting on
        # leaving this context.
        with _temporary_physics_derivative_graph(runtime.model, enabled=False):
            with torch.no_grad():
                evaluation = runtime.model.evaluate_batches(runtime.validation_batches)
        quadrature = _quadrature_report(runtime) if evaluate_quadrature else {}
        gradients = (
            _gradient_smoke(runtime)
            if evaluate_gradients
            else {"enabled": False, "success": None}
        )
    after = state_dict_sha256(runtime.model)
    report = {
        "format": "prom3theus.joint_dry_run",
        "version": 1,
        "schema_version": runtime.config.schema_version,
        "mode": "setup_and_forward_only",
        "optimization_started": False,
        "checkpoint_written": False,
        "device": str(runtime.device),
        "state": {
            "sha256_at_build": runtime.state_sha256_at_build,
            "sha256_before_evaluation": before,
            "sha256_after_evaluation": after,
            "unchanged": runtime.state_sha256_at_build == before == after,
        },
        "objective": {
            "loss": _scalar(evaluation.loss, "loss"),
            "metrics": {
                name: _scalar(value, name)
                for name, value in evaluation.scalar_metrics().items()
            },
            "sample_count_by_stream": {
                name: result.sample_count for name, result in evaluation.streams.items()
            },
        },
        "gradient_smoke": gradients,
        "quadrature_convergence": quadrature,
        "runtime": _runtime_metadata(runtime),
    }
    if runtime.state_sha256_at_build != before:
        raise RuntimeError(
            "Joint model state changed after setup and before evaluation."
        )
    if before != after:
        raise RuntimeError("Setup-only joint evaluation mutated model state.")
    return JointRuntimeEvaluation(evaluation=evaluation, report=report)
