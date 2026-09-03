"""Tests for bounded diagnostic sampling independent of Lightning hooks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prom3theus.diagnostics.sampling import (
    ValidationSampleCollector,
    integrated_stokes,
    map_coordinates,
    subsample_grid,
)


def test_rectangular_subsample_never_exceeds_limit():
    rows, columns = subsample_grid(101, 79, maximum=127)

    assert rows.dtype == np.int64
    assert columns.dtype == np.int64
    assert rows.size * columns.size <= 127
    assert rows[0] == columns[0] == 0


@pytest.mark.parametrize(
    ("wavelength", "weights", "message"),
    [
        (torch.tensor([2.0, 1.0]), torch.ones(2), "strictly increasing"),
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, -1.0]), "non-negative"),
        (torch.tensor([1.0, float("nan")]), torch.ones(2), "finite"),
    ],
)
def test_integrated_stokes_rejects_invalid_spectral_quadrature(
    wavelength, weights, message
):
    with pytest.raises(ValueError, match=message):
        integrated_stokes(torch.ones(1, 4, 2), wavelength, weights)


def test_integrated_stokes_rejects_nonfinite_and_integer_stokes():
    wavelength = torch.tensor([1.0, 2.0])
    weights = torch.ones(2)
    with pytest.raises(ValueError, match="only finite"):
        integrated_stokes(torch.full((1, 4, 2), float("nan")), wavelength, weights)
    with pytest.raises(TypeError, match="floating-point"):
        integrated_stokes(torch.ones(1, 4, 2, dtype=torch.int64), wavelength, weights)


def test_validation_collector_keeps_all_integrals_and_bounded_profiles():
    collector = ValidationSampleCollector(max_profile_samples=3)
    wavelength = torch.tensor((6300.0, 6301.0, 6302.0, 6303.0))
    weights = torch.tensor((1.0, 1.0, 0.0, 1.0))
    prediction = torch.ones(5, 4, 4)
    reference = 2.0 * prediction
    pixel_index = torch.tensor(((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)))

    collector.add(prediction, reference, pixel_index, wavelength, weights)
    payload = collector.local_payload()

    assert payload is not None
    assert payload["integrated_prediction"].shape == (5, 4)
    assert payload["stokes_pred"].shape == (3, 4, 4)
    torch.testing.assert_close(
        payload["integrated_prediction"],
        integrated_stokes(prediction, wavelength, weights),
    )


def test_map_coordinates_uses_canonical_observation_field():
    coordinates = torch.tensor(
        (
            ((1.0, 10.0, 0.0), (2.0, 20.0, 0.0)),
            ((3.0, 30.0, 0.0), (4.0, 40.0, 0.0)),
        )
    )

    x_mm, y_mm = map_coordinates(
        SimpleNamespace(coordinates=coordinates),
        np.array((0, 1)),
        np.array((0, 1)),
    )

    np.testing.assert_array_equal(x_mm, ((1.0, 2.0), (3.0, 4.0)))
    np.testing.assert_array_equal(y_mm, ((10.0, 20.0), (30.0, 40.0)))
