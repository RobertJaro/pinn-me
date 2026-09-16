"""Explicit diagnostic providers return artifacts to the common lifecycle."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiagnosticOutput:
    """A provider report and the exact image artifacts it created."""

    report: Any
    paths: tuple[str | Path, ...] = ()
    media_groups: dict[str, tuple[str | Path, ...]] | None = None


_PROVIDERS = {}


def register_diagnostic_provider(term_type, renderer):
    if term_type in _PROVIDERS:
        raise ValueError(f"Diagnostic provider already registered: {term_type}")
    _PROVIDERS[term_type] = renderer


def _stokes(runtime, trainer, evaluation, output, stream_id):
    from prom3theus.application.joint_rendering import _render_stokes_diagnostics

    return _render_stokes_diagnostics(
        runtime, trainer, output, stream_ids=(stream_id,)
    )


def _aia(runtime, trainer, evaluation, output, stream_id):
    from prom3theus.application.joint_rendering import _render_aia_diagnostics

    report = _render_aia_diagnostics(
        runtime, evaluation, output, stream_ids=(stream_id,)
    )
    if not report["enabled"]:
        return DiagnosticOutput(report)
    # An enabled renderer must declare its comparison images. Do not silently
    # drop uploads when a report contract changes.
    diagnostics = report["report"]["streams"][stream_id]["diagnostics"]
    comparison_paths = diagnostics["paths"]
    if not comparison_paths:
        raise RuntimeError(
            f"AIA validation rendered no comparison images for {stream_id}"
        )
    paths = [
        *report["side_views"],
        *comparison_paths,
        *diagnostics["contribution_paths"],
    ]
    return DiagnosticOutput(
        report,
        tuple(paths),
        media_groups={
            "AIA observation comparison": tuple(comparison_paths),
            "AIA side view": tuple(report["side_views"]),
        },
    )


_PROVIDERS.update(lte_stokes=_stokes, aia_optically_thin=_aia)


def render_diagnostics(runtime, trainer, evaluation, output):
    reports = {}
    artifacts = []

    for stream in runtime.config.streams:
        renderer = _PROVIDERS.get(stream.data_term.type)
        if renderer is None:
            reports[stream.id] = {
                "enabled": False,
                "reason": "no diagnostic provider registered",
            }
            continue
        result = renderer(runtime, trainer, evaluation, output, stream.id)
        if not isinstance(result, DiagnosticOutput):
            raise TypeError("Diagnostic providers must return DiagnosticOutput")
        reports[stream.id] = result.report
        media_keys = {
            str(path): key
            for key, paths in (result.media_groups or {}).items()
            for path in paths
        }
        artifacts.extend(
            {
                "stream_id": stream.id,
                "path": path,
                **(
                    {"media_key": media_keys.get(path)}
                    if result.media_groups is not None
                    else {}
                ),
            }
            for path in dict.fromkeys(str(path) for path in result.paths)
        )
    # Shared physics is rendered once, independently of observation streams.
    if hasattr(runtime, "model") and hasattr(runtime.config, "diagnostics"):
        from .physics_equations import render_physics_equations
        result = render_physics_equations(runtime, trainer, output)
        reports["physics_equations"] = result.report
        for key, paths in (result.media_groups or {}).items():
            artifacts.extend({"stream_id": "physics", "path": str(path), "media_key": key}
                             for path in paths)
    return {"streams": reports, "artifacts": artifacts}
