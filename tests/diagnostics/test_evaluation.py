"""Strict option parsing for atmosphere diagnostics."""

from __future__ import annotations

import pytest

from prom3theus.diagnostics.evaluation import (
    AtmosphereEvaluator,
    AtmosphereSampling,
)


def test_sampling_counts_are_not_silently_truncated():
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        AtmosphereSampling.from_options(ray_sampling={"batch_size": 1.5})


def test_meridional_enabled_requires_an_actual_boolean():
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        AtmosphereSampling.from_options(
            meridional_slice={"enabled": "false", "longitude_deg": 0.0}
        )


def test_ray_grid_indices_must_be_supplied_as_a_pair():
    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options())

    with pytest.raises(ValueError, match="supplied together"):
        evaluator.evaluate_ray_optical_depth(object(), object(), rows=None, columns=[0])


def test_shell_height_levels_must_be_finite():
    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options())

    with pytest.raises(ValueError, match="finite vector"):
        evaluator.evaluate_shell_layers(object(), object(), [0.0, float("nan")])
