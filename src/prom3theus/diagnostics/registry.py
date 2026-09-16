"""Lifecycle-independent routing for observation-stream diagnostics.

Renderers receive only the diagnostic payload attached to a
:class:`~prom3theus.inversion.data_terms.base.DataTermBatchResult`.  This keeps
plotting independent of Lightning, the optimizer, and observation-specific
forward-model implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from prom3theus.inversion.data_terms.base import DataTermBatchResult


@runtime_checkable
class DiagnosticRenderer(Protocol):
    """Structural interface implemented by one observation-kind renderer."""

    observation_kind: str

    def render(
        self,
        payloads: Sequence[Mapping[str, Any]],
        *,
        output_directory: Path,
        label: str,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Render collected payloads and return a JSON-serializable report."""


def _finite_scalar(value: torch.Tensor, *, name: str) -> float | None:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError(f"{name} must be a scalar tensor.")
    result = float(value.detach().cpu())
    return result if math.isfinite(result) else None


def _weighted_scalars(
    results: Sequence[DataTermBatchResult],
    accessor,
    *,
    name: str,
) -> float | None:
    numerator = 0.0
    denominator = 0
    for result in results:
        value = accessor(result)
        if value is None:
            continue
        scalar = _finite_scalar(value, name=name)
        if scalar is None:
            return None
        numerator += scalar * result.sample_count
        denominator += result.sample_count
    return numerator / denominator if denominator else None


def _configured_component_likelihood(
    components: Mapping[str, float | None],
    context: Mapping[str, Any] | None,
    *,
    stream_id: str,
) -> tuple[float | None, dict[str, Any] | None]:
    """Reduce per-component means with explicitly configured scientific weights.

    Native-grid image diagnostics are commonly streamed one channel at a time.
    Sample-weighting those already reduced, single-channel likelihoods would
    implicitly weight channels by their valid-pixel counts.  An observation
    renderer may instead provide ``likelihood_component_weights`` in its
    context so the diagnostic report reproduces the configured likelihood.
    """

    if not isinstance(context, Mapping):
        return None, None
    configured = context.get("likelihood_component_weights")
    if configured is None:
        return None, None
    if not isinstance(configured, Mapping) or not configured:
        raise TypeError(
            f"{stream_id}.likelihood_component_weights must be a non-empty mapping."
        )
    weights: dict[str, float] = {}
    for name, raw_weight in configured.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{stream_id}.likelihood_component_weights keys must be "
                "non-empty strings."
            )
        if isinstance(raw_weight, bool):
            raise TypeError(
                f"{stream_id}.likelihood_component_weights.{name} must be numeric."
            )
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{stream_id}.likelihood_component_weights.{name} must be numeric."
            ) from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{stream_id}.likelihood_component_weights.{name} must be "
                "finite and non-negative."
            )
        weights[name] = weight
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError(
            f"{stream_id}.likelihood_component_weights must contain a positive weight."
        )
    missing = sorted(
        name
        for name, weight in weights.items()
        if weight > 0 and components.get(name) is None
    )
    if missing:
        raise KeyError(
            f"{stream_id} has no aggregate component losses for configured "
            f"likelihood components {missing}."
        )
    normalized = {
        name: weight / total_weight for name, weight in sorted(weights.items())
    }
    likelihood = sum(
        normalized[name] * float(components[name])
        for name in normalized
        if components.get(name) is not None
    )
    return likelihood, {
        "type": "configured_component_balanced",
        "normalized_component_weights": normalized,
    }


def _normalize_results(
    value: DataTermBatchResult | Sequence[DataTermBatchResult],
    *,
    stream_id: str,
) -> tuple[DataTermBatchResult, ...]:
    if isinstance(value, DataTermBatchResult):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"Results for stream {stream_id!r} must be a DataTermBatchResult "
            "or a sequence of them."
        )
    normalized = tuple(value)
    if not normalized or any(
        not isinstance(result, DataTermBatchResult) for result in normalized
    ):
        raise TypeError(
            f"Results for stream {stream_id!r} must contain "
            "DataTermBatchResult instances."
        )
    return normalized


class DiagnosticRegistry:
    """Map stable observation kinds to independent diagnostic renderers."""

    def __init__(self, renderers: Sequence[DiagnosticRenderer] = ()) -> None:
        self._renderers: dict[str, DiagnosticRenderer] = {}
        for renderer in renderers:
            self.register(renderer)

    def register(
        self,
        renderer: DiagnosticRenderer,
        *,
        replace: bool = False,
    ) -> None:
        kind = getattr(renderer, "observation_kind", None)
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(
                "A diagnostic renderer needs a non-empty observation_kind."
            )
        if not callable(getattr(renderer, "render", None)):
            raise TypeError("A diagnostic renderer must define render().")
        kind = kind.strip()
        if kind in self._renderers and not replace:
            raise ValueError(f"A renderer is already registered for {kind!r}.")
        self._renderers[kind] = renderer

    def renderer_for(self, observation_kind: str) -> DiagnosticRenderer:
        try:
            return self._renderers[observation_kind]
        except KeyError as error:
            raise KeyError(
                f"No diagnostic renderer is registered for {observation_kind!r}."
            ) from error

    @property
    def observation_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._renderers))

    def render_results(
        self,
        results: Mapping[str, DataTermBatchResult | Sequence[DataTermBatchResult]],
        *,
        stream_kinds: Mapping[str, str],
        output_directory: str | Path,
        label: str,
        contexts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Aggregate scalar outputs and route diagnostic payloads per stream.

        Component and metric reductions are sample-count weighted.  Likelihood
        uses the same reduction unless a stream context explicitly supplies
        ``likelihood_component_weights``; this lets channel-split image
        diagnostics reproduce a configured channel-balanced objective.  Missing
        component or metric keys are permitted across individual batches.
        """

        if not isinstance(label, str) or not label.strip():
            raise ValueError("Diagnostic label must be a non-empty string.")
        unknown_streams = set(results) - set(stream_kinds)
        if unknown_streams:
            raise KeyError(
                "No observation kind was supplied for streams "
                f"{sorted(unknown_streams)}."
            )
        directory = Path(output_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        context_by_stream = {} if contexts is None else dict(contexts)

        streams: dict[str, Any] = {}
        for stream_id in sorted(results):
            batches = _normalize_results(results[stream_id], stream_id=stream_id)
            payloads = tuple(
                result.diagnostics
                for result in batches
                if result.diagnostics is not None
            )
            if not payloads:
                raise ValueError(
                    f"Stream {stream_id!r} produced no diagnostic payloads."
                )
            component_names = sorted(
                {name for result in batches for name in result.component_losses}
            )
            metric_names = sorted(
                {name for result in batches for name in result.metrics}
            )
            pixel_weighted_likelihood = _weighted_scalars(
                batches,
                lambda result: result.likelihood_loss,
                name=f"{stream_id}.likelihood_loss",
            )
            components = {
                name: _weighted_scalars(
                    batches,
                    lambda result, key=name: result.component_losses.get(key),
                    name=f"{stream_id}.component_losses.{name}",
                )
                for name in component_names
            }
            metrics = {
                name: _weighted_scalars(
                    batches,
                    lambda result, key=name: result.metrics.get(key),
                    name=f"{stream_id}.metrics.{name}",
                )
                for name in metric_names
            }
            stream_context = context_by_stream.get(stream_id)
            (
                configured_likelihood,
                configured_reduction,
            ) = _configured_component_likelihood(
                components,
                stream_context,
                stream_id=stream_id,
            )
            if configured_reduction is None:
                likelihood = pixel_weighted_likelihood
                likelihood_reduction = {"type": "sample_count_weighted_batches"}
            else:
                likelihood = configured_likelihood
                likelihood_reduction = configured_reduction
            kind = stream_kinds[stream_id]
            renderer = self.renderer_for(kind)
            rendered = dict(
                renderer.render(
                    payloads,
                    output_directory=directory / stream_id,
                    label=label,
                    context=stream_context,
                )
            )
            stream_report = {
                "observation_kind": kind,
                "sample_count": sum(result.sample_count for result in batches),
                "losses": {
                    "likelihood": likelihood,
                    "likelihood_reduction": likelihood_reduction,
                    "components": components,
                },
                "metrics": metrics,
                "diagnostics": rendered,
            }
            streams[stream_id] = stream_report

        report = {"label": label, "streams": streams}
        try:
            json.dumps(report, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Diagnostic renderers must return strictly JSON-serializable values."
            ) from error
        return report


def default_diagnostic_registry(
    *,
    aia_max_pixels_per_image: int = 65_536,
    aia_max_contribution_rays: int = 16,
    dpi: int = 180,
) -> DiagnosticRegistry:
    """Build the standard registry without importing plotting at module load."""

    from .aia_euv import AIAEUVDiagnosticRenderer

    return DiagnosticRegistry(
        (
            AIAEUVDiagnosticRenderer(
                max_pixels_per_image=aia_max_pixels_per_image,
                max_contribution_rays=aia_max_contribution_rays,
                dpi=dpi,
            ),
        )
    )


__all__ = [
    "DiagnosticRegistry",
    "DiagnosticRenderer",
    "default_diagnostic_registry",
]
