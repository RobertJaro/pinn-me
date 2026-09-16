"""Orchestrate joint setup, evaluation, diagnostics, and report persistence."""

from __future__ import annotations

import json
import os
from types import MappingProxyType, SimpleNamespace
from typing import Any

from prom3theus.config.joint_schema import JointInversionConfig

from .joint_assembly import build_joint_runtime
from .joint_contracts import (
    JointDryRun,
    JointRunnerFactories,
    JointRuntimeEvaluation,
)
from .joint_evaluation import evaluate_joint_runtime
from prom3theus.diagnostics.providers import render_diagnostics
from .runtime import _json_value


def run_joint_dry_run(
    config: JointInversionConfig,
    *,
    rebuild_observations: bool = False,
    factories: JointRunnerFactories | None = None,
) -> JointDryRun:
    """Assemble, render, smoke-test gradients, and atomically write JSON."""

    runtime = build_joint_runtime(
        config,
        rebuild_observations=rebuild_observations,
        factories=factories,
    )
    evaluation = evaluate_joint_runtime(runtime)
    output_directory = config.solver.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = (output_directory / config.dry_run.report_filename).resolve()
    diagnostics = render_diagnostics(
        runtime,
        SimpleNamespace(global_step=0, logger=None),
        evaluation.evaluation,
        config.solver.work_directory / "validation" / "setup",
    )
    report = dict(evaluation.report)
    report["diagnostics"] = diagnostics
    report["report_path"] = str(report_path)
    evaluation = JointRuntimeEvaluation(
        evaluation=evaluation.evaluation,
        report=MappingProxyType(report),
    )
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                _json_value(evaluation.report),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return JointDryRun(
        runtime=runtime,
        evaluation=evaluation,
        report_path=report_path,
    )


def dry_run_joint_inversion(*args, **kwargs) -> dict[str, Any]:
    """Return the JSON-ready report expected by command-line callers."""

    result = run_joint_dry_run(*args, **kwargs)
    report = _json_value(result.evaluation.report)
    if not isinstance(report, dict):  # pragma: no cover - report invariant
        raise TypeError("Joint dry-run report must be a JSON object.")
    return report


__all__ = ["dry_run_joint_inversion", "run_joint_dry_run"]
