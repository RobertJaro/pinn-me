"""Render native-grid joint diagnostics through the diagnostic registry."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from prom3theus.observations.loading import buffered_dataloader
from prom3theus.observations.bulk import grid_indices
from torch.utils.data import Subset

from prom3theus.config.joint_schema import AIAOpticallyThinDataTermConfig
from prom3theus.diagnostics.registry import default_diagnostic_registry
from prom3theus.diagnostics.sampling import subsample_grid
from prom3theus.inversion.data_terms import DataTermBatchResult

from .joint_contracts import JointRuntime
from prom3theus.inversion.joint import JointEvaluation
from .runtime import _move_value, _temporary_eval_mode, state_dict_sha256


def _render_stokes_diagnostics(runtime, trainer, output_directory, *, stream_ids=None):
    """Reuse the original LTE renderers with the joint run's shared atmosphere."""
    from prom3theus.diagnostics.evaluation import (
        AtmosphereEvaluator,
        AtmosphereSampling,
    )
    from prom3theus.diagnostics.providers import DiagnosticOutput
    from prom3theus.diagnostics.rendering import AtmosphereRenderer
    from prom3theus.diagnostics.sampling import ValidationSampleCollector, display_grid
    from prom3theus.inversion.data_terms.stokes import StokesObservationTerm

    # The common lifecycle uploads returned artifacts exactly once.
    trainer = SimpleNamespace(global_step=trainer.global_step, logger=None)
    options = runtime.config.diagnostics.visualization
    if not options.enabled:
        return DiagnosticOutput([])
    sampling = AtmosphereSampling.from_options(
        ray_sampling=options.ray_sampling.to_dict(),
        slice_sampling=options.slice_sampling.to_dict(),
        meridional_slice=options.meridional_slice.to_dict(),
    )
    renderer = AtmosphereRenderer(
        output_directory / "diagnostics",
        dpi=options.dpi,
        evaluator=AtmosphereEvaluator(sampling),
    )
    stream_configs = {stream.id: stream for stream in runtime.config.streams}
    paths = []
    for stream_id, term in runtime.model.terms.items():
        if stream_ids is not None and stream_id not in stream_ids:
            continue
        if not isinstance(term, StokesObservationTerm):
            continue
        data = runtime.streams[stream_id].data_module
        raster = data.raster
        rows, columns = display_grid(
            SimpleNamespace(datamodule=data), raster, sampling.max_ray_pixels
        )
        label = stream_id
        disambiguation_options = getattr(term, "disambiguation_options", None)
        phase_for_diagnostics = getattr(term, "phase_for_diagnostics", None)
        phase_enabled = (
            isinstance(disambiguation_options, Mapping)
            and bool(disambiguation_options.get("enabled", False))
            and callable(phase_for_diagnostics)
        )
        with _temporary_eval_mode(term), torch.no_grad():
            if options.render_streams or phase_enabled:
                dataset = data.evaluation_dataset()
                indices = grid_indices(dataset, rows, columns)
                loader = buffered_dataloader(
                    Subset(dataset, indices.tolist()),
                    batch_size=sampling.ray_evaluation_batch_size,
                    num_workers=getattr(data, "num_workers", 2),
                    collate_fn=data.evaluation_collate_fn(),
                )
                collector = (
                    ValidationSampleCollector(sampling.max_profile_samples)
                    if options.render_streams
                    else None
                )
                phase_payloads = []
                for batch in loader:
                    device_batch = _move_value(batch, runtime.device)
                    if options.render_streams:
                        result = term.evaluate_batch(device_batch)
                        payload = result.diagnostics
                        collector.add(
                            term.objective.transform(payload["prediction"]),
                            term.objective.transform(payload["target"]),
                            payload["pixel_index"],
                            term.wavelength_angstrom,
                            term.wavelength_weights,
                        )
                    if phase_enabled:
                        pixel_index = device_batch.get("pixel_index")
                        if pixel_index is None:
                            raise ValueError(
                                "Disambiguation phase validation requires pixel_index values."
                            )
                        phase = phase_for_diagnostics(device_batch["coordinates"])
                        if phase is None:
                            raise RuntimeError(
                                "Enabled disambiguation did not return a diagnostic phase."
                            )
                        phase_payloads.append(
                            {
                                "phase_rad": phase.detach().float().cpu().reshape(-1),
                                "pixel_index": pixel_index.detach().cpu(),
                            }
                        )
                if options.render_streams:
                    paths.append(
                        renderer.render_stokes_validation(
                            trainer,
                            collector.local_payload(),
                            raster,
                            term.wavelength_angstrom,
                            label,
                            rows=rows,
                            columns=columns,
                            objective_config=term.objective.configuration(),
                            exclusion_windows_angstrom=stream_configs[
                                stream_id
                            ].data_term.synthesis.excluded_wavelength_windows_angstrom,
                            line_centers_angstrom=[
                                line.wavelength_air_angstrom
                                for line in term.synthesizer.lines
                            ],
                        )
                    )
                if phase_enabled and phase_payloads:
                    phase_payload = {
                        key: torch.cat([payload[key] for payload in phase_payloads], dim=0)
                        for key in ("phase_rad", "pixel_index")
                    }
                    paths.append(
                        renderer.render_disambiguation_phase(
                            trainer,
                            phase_payload,
                            raster,
                            label,
                            rows=rows,
                            columns=columns,
                            step=int(getattr(term, "step", trainer.global_step)),
                            cold_steps=int(disambiguation_options["cold_steps"]),
                            handoff_step=int(disambiguation_options["handoff_step"]),
                        )
                    )
            if options.render_atmosphere:
                view = SimpleNamespace(
                    atmosphere_model=runtime.model.atmosphere_model,
                    forward_composition=term._composition,
                    synthesizer=term.synthesizer,
                    sample_depth_grid=lambda randomize=False: (
                        term._composition.sample_depth_grid(
                            term.coarse_depth_grid, randomize=randomize
                        )
                    ),
                )
                paths.extend(
                    renderer.render_atmosphere(
                        trainer,
                        view,
                        raster,
                        label=label,
                        rows=rows,
                        columns=columns,
                    )
                )
    paths = tuple(str(path) for path in paths)
    return DiagnosticOutput(
        list(paths), paths,
        media_groups={key: tuple(images) for key, images in renderer.media_groups.items()},
    )


_AIA_RENDER_FIELDS = (
    "prediction_raw",
    "prediction_calibrated",
    "target",
    "channel_angstrom",
    "channel_index",
    "pixel_index",
    "image_index",
    "asinh_residual",
    "fractional_residual",
)


_AIA_CONTRIBUTION_FIELDS = (
    "contribution",
    "height_m",
    "channel_angstrom",
    "channel_index",
    "pixel_index",
    "image_index",
)


def _detached_aia_render_result(
    result: DataTermBatchResult,
) -> DataTermBatchResult:
    """Keep only CPU tensors required to reconstruct an AIA image."""

    payload = result.diagnostics
    if not isinstance(payload, Mapping):
        raise ValueError("AIA diagnostic evaluation produced no payload.")
    missing = sorted(set(_AIA_RENDER_FIELDS) - set(payload))
    if missing:
        raise KeyError(f"AIA diagnostic payload is missing fields: {missing}.")
    if payload["pixel_index"] is None:
        raise ValueError("AIA native-grid rendering requires pixel_index values.")
    return DataTermBatchResult(
        likelihood_loss=result.likelihood_loss.detach().cpu(),
        component_losses={
            name: value.detach().cpu()
            for name, value in result.component_losses.items()
        },
        metrics={name: value.detach().cpu() for name, value in result.metrics.items()},
        sample_count=result.sample_count,
        diagnostics={name: payload[name].detach().cpu() for name in _AIA_RENDER_FIELDS},
    )


def _detached_aia_contribution_payload(
    result: DataTermBatchResult,
) -> dict[str, torch.Tensor]:
    """Detach only the bounded ray profiles selected for formation diagnostics."""

    payload = result.diagnostics
    if not isinstance(payload, Mapping):
        raise ValueError("AIA contribution evaluation produced no payload.")
    missing = sorted(set(_AIA_CONTRIBUTION_FIELDS) - set(payload))
    if missing:
        raise KeyError(f"AIA contribution payload is missing fields: {missing}.")
    if payload["pixel_index"] is None:
        raise ValueError("AIA contribution diagnostics require pixel_index values.")
    return {name: payload[name].detach().cpu() for name in _AIA_CONTRIBUTION_FIELDS}


def _balanced_contribution_counts(
    capacities: Mapping[int, int],
    maximum_count: int,
) -> dict[int, int]:
    """Distribute one bounded ray budget deterministically across channels."""

    counts = {channel: 0 for channel in sorted(capacities)}
    remaining = min(maximum_count, sum(capacities.values()))
    while remaining:
        progressed = False
        for channel in counts:
            if counts[channel] >= capacities[channel]:
                continue
            counts[channel] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            break
    return counts


def _aia_render_results(
    runtime: JointRuntime,
    evaluation: JointEvaluation,
    stream_id: str,
) -> tuple[tuple[DataTermBatchResult, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Forward-render bounded native-grid AIA pixels in memory-safe chunks."""

    loaded = runtime.streams[stream_id]
    data_module = loaded.data_module
    term = runtime.model.terms[stream_id]
    datasets = getattr(data_module, "validation_datasets", None)
    evaluator = getattr(term, "evaluate_diagnostic_batch", None)
    collator = getattr(data_module, "collator", None)
    if not isinstance(datasets, Mapping) or not datasets or not callable(evaluator):
        # Replaceable test/factory terms need not implement the concrete native
        # image path.  Their already evaluated payload remains renderable.
        return (evaluation.streams[stream_id],), ()
    if not callable(collator):
        raise TypeError("AIA image data modules must expose a callable collator.")

    visualization = runtime.config.diagnostics.visualization
    maximum = visualization.ray_sampling.max_pixels
    batch_size = visualization.ray_sampling.batch_size
    results: list[DataTermBatchResult] = []
    for channel in sorted(datasets):
        dataset = datasets[channel]
        rows, columns = subsample_grid(*dataset.raster.spatial_shape, maximum)
        indices = grid_indices(dataset, rows, columns)
        loader = buffered_dataloader(
            Subset(dataset, indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(data_module, "num_workers", 2),
            pin_memory=False,
            collate_fn=collator,
        )
        for batch in loader:
            device_batch = _move_value(batch, runtime.device)
            with torch.no_grad():
                result = evaluator(device_batch)
            results.append(_detached_aia_render_result(result))
    if not results:
        raise ValueError(f"AIA stream {stream_id!r} produced no render batches.")

    capacities = {channel: len(datasets[channel]) for channel in sorted(datasets)}
    contribution_counts = _balanced_contribution_counts(
        capacities,
        visualization.contribution_ray_count,
    )
    contribution_payloads: list[dict[str, torch.Tensor]] = []
    for channel, count in contribution_counts.items():
        if not count:
            continue
        dataset = datasets[channel]
        indices = dataset.diagnostic_dataset_indices(count)
        loader = buffered_dataloader(
            Subset(dataset, indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(data_module, "num_workers", 2),
            pin_memory=False,
            collate_fn=collator,
        )
        for batch in loader:
            device_batch = _move_value(batch, runtime.device)
            with torch.no_grad():
                result = evaluator(device_batch)
            contribution_payloads.append(_detached_aia_contribution_payload(result))
    return tuple(results), tuple(contribution_payloads)


def _render_aia_diagnostics(
    runtime: JointRuntime,
    evaluation: JointEvaluation,
    output_directory: Path,
    *,
    stream_ids=None,
) -> dict[str, Any]:
    visualization = runtime.config.diagnostics.visualization
    if not visualization.enabled or not visualization.render_streams:
        return {
            "enabled": False,
            "reason": "stream rendering disabled by configuration",
        }
    stream_configs = {stream.id: stream for stream in runtime.config.streams}
    aia_stream_ids = tuple(
        stream_id
        for stream_id in evaluation.streams
        if (stream_ids is None or stream_id in stream_ids)
        and isinstance(
            stream_configs[stream_id].data_term,
            AIAOpticallyThinDataTermConfig,
        )
    )
    if not aia_stream_ids:
        return {"enabled": False, "reason": "no AIA streams configured"}
    state_before = state_dict_sha256(runtime.model)
    side_paths = []
    with _temporary_eval_mode(runtime.model):
        render_outputs = {
            stream_id: _aia_render_results(runtime, evaluation, stream_id)
            for stream_id in aia_stream_ids
        }
        from prom3theus.diagnostics.aia_side_view import (
            render_side_view,
            synthesize_side_view,
        )
        from prom3theus.inversion.data_terms.aia_euv import AIAEUVObservationTerm
        from prom3theus.inversion.sampling import SphericalShellDomain

        for stream_id in aia_stream_ids:
            term = runtime.model.terms[stream_id]
            if not isinstance(term, AIAEUVObservationTerm):
                continue  # Replaceable setup/test terms have no volumetric renderer.
            domain = SphericalShellDomain.from_observation_bounds(
                runtime.shared_metadata["observation_bounds"],
                (
                    runtime.config.atmosphere.geometry.outer_height_megameter,
                    runtime.config.atmosphere.geometry.inner_height_megameter,
                ),
            )
            absolute_time = runtime.validation_batches[stream_id][
                "absolute_tai_seconds"
            ]
            time_hours = (
                float(absolute_time.reshape(-1)[0])
                - runtime.scene.reference_time_tai_seconds
            ) / 3600
            image, extent = synthesize_side_view(
                runtime.model.atmosphere_model,
                term,
                domain,
                time_hours,
                max_pixels=visualization.ray_sampling.max_pixels,
                batch_size=visualization.ray_sampling.batch_size,
            )
            path = output_directory / "diagnostics" / f"{stream_id}_aia_side_view.png"
            upper = runtime.config.atmosphere.upper_atmosphere
            markers = [
                runtime.config.atmosphere.geometry.line_formation_outer_height_megameter
            ]
            if upper is not None:
                markers.append(upper.transition_region_top_megameter)
            render_side_view(
                image,
                extent,
                term.channels_angstrom,
                path,
                time_hours=time_hours,
                height_markers=markers,
                dpi=visualization.dpi,
            )
            side_paths.append(str(path))
    state_after = state_dict_sha256(runtime.model)
    if state_before != state_after:
        raise RuntimeError("AIA diagnostic rendering mutated joint model state.")
    registry = default_diagnostic_registry(
        aia_max_pixels_per_image=visualization.ray_sampling.max_pixels,
        aia_max_contribution_rays=visualization.contribution_ray_count,
        dpi=visualization.dpi,
    )
    results = {stream_id: output[0] for stream_id, output in render_outputs.items()}
    rendered = registry.render_results(
        results,
        stream_kinds={stream_id: "aia_euv" for stream_id in results},
        output_directory=output_directory / "diagnostics",
        label="validation",
        contexts={
            stream_id: {
                "rasters": runtime.streams[stream_id].rasters,
                "data_module": runtime.streams[stream_id].data_module,
                "asinh_scales": dict(
                    zip(
                        runtime.model.terms[stream_id].channels_angstrom,
                        runtime.model.terms[stream_id]
                        .objective.asinh_scales.detach()
                        .cpu()
                        .tolist(),
                        strict=True,
                    )
                ),
                "intensity_scales": dict(
                    zip(
                        runtime.model.terms[stream_id].channels_angstrom,
                        runtime.model.terms[stream_id]
                        .intensity_scales.detach()
                        .cpu()
                        .tolist(),
                        strict=True,
                    )
                ),
                "contribution_payloads": render_outputs[stream_id][1],
                "likelihood_component_weights": {
                    f"channel_{item.channel_angstrom}": item.weight
                    for item in stream_configs[
                        stream_id
                    ].data_term.objective.channel_weights
                },
            }
            for stream_id in results
        },
    )
    return {
        "enabled": True,
        "registry": "default_diagnostic_registry",
        "side_views": side_paths,
        "output_directory": str((output_directory / "diagnostics").resolve()),
        "forward_render": {
            "selection": (
                "regular native-pixel grid using the shared LTE ray_sampling.max_pixels bound"
            ),
            "max_pixels_per_channel": visualization.ray_sampling.max_pixels,
            "batch_size": visualization.ray_sampling.batch_size,
            "contribution_rays": {
                "requested": visualization.contribution_ray_count,
                "rendered_by_stream": {
                    stream_id: sum(
                        int(payload["contribution"].shape[0])
                        for payload in render_outputs[stream_id][1]
                    )
                    for stream_id in render_outputs
                },
                "selection": (
                    "deterministic evenly distributed native-grid rays, "
                    "balanced across channels"
                ),
            },
            "state_unchanged": True,
        },
        "report": rendered,
    }
