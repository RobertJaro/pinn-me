"""Tests for the explicit LTE optimizer-batch tensor contract."""

from __future__ import annotations

import pytest
import torch

from prom3theus.training.batch import validate_batch_tensors


def test_nested_batch_tensors_match_the_model_dtype_and_are_finite():
    reference = torch.ones((), dtype=torch.float32)
    validate_batch_tensors(
        {
            "coordinates": torch.zeros(2, 3),
            "instrument_response": {"weights": torch.ones(2, 4)},
            "pixel_index": torch.zeros(2, 2, dtype=torch.long),
            "observation_id": ["observation", "observation"],
        },
        reference=reference,
    )

    with pytest.raises(TypeError, match="batch.coordinates must use torch.float32"):
        validate_batch_tensors(
            {"coordinates": torch.zeros(2, 3, dtype=torch.float64)},
            reference=reference,
        )
    with pytest.raises(FloatingPointError, match="batch.stokes"):
        validate_batch_tensors(
            {"stokes": torch.tensor([float("nan")])},
            reference=reference,
        )
