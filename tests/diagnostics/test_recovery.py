"""Smoke coverage for the differentiable LTE recovery diagnostic."""

from __future__ import annotations

import math

from prom3theus.diagnostics.recovery import run_recovery


def test_recovery_remains_finite_across_optimizer_steps():
    result = run_recovery(steps=5)

    assert math.isfinite(result["initial_loss"])
    assert math.isfinite(result["final_loss"])
    assert math.isfinite(result["max_scaled_error"])
    assert result["final_loss"] < result["initial_loss"]
    assert result["loss_reduction"] > 1.0
