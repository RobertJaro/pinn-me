"""Rendering evaluates shared LTE modules without changing their training modes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from prom3theus.artifacts.stokes import evaluate_stokes


@pytest.mark.parametrize("fail_prediction", [False, True])
@pytest.mark.parametrize("register_shared", [False, True])
def test_stokes_export_evaluates_shared_modules(fail_prediction, register_shared):
    class ObservationTerm(nn.Module):
        def __init__(self):
            super().__init__()
            assign = setattr if register_shared else object.__setattr__
            assign(self, "atmosphere_model", nn.Sequential(nn.Linear(3, 1)))
            assign(self, "synthesis_window", nn.Sequential(nn.Identity()))
            self.synthesis_window[0].eval()

        def predict(self, batch):
            assert not self.training
            assert not self.atmosphere_model.training
            assert not self.synthesis_window.training
            if fail_prediction:
                raise RuntimeError("prediction failed")
            return torch.zeros(len(batch["coordinates"]), 4, 3)

    module = ObservationTerm()
    raster = SimpleNamespace(
        spatial_shape=(1, 2),
        coordinates=torch.zeros(1, 2, 3),
        ray_direction=torch.tensor([0., 0., -1.]).expand(1, 2, 3),
        stokes_basis=torch.eye(3).expand(1, 2, 3, 3),
        wavelength_angstrom=torch.tensor([6301.884179, 6301.905728, 6301.927277], dtype=torch.float64),
        stokes=torch.zeros(1, 2, 4, 3),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        auxiliary={},
    )
    spec = SimpleNamespace(
        velocity_synthesis_mode=SimpleNamespace(value="carrington_observer_relative"),
        instrument_type="hinode_sp",
    )
    rng_before = torch.random.get_rng_state()
    if fail_prediction:
        with pytest.raises(RuntimeError, match="prediction failed"):
            evaluate_stokes(module, raster, spec, batch_size=1, storage_dtype="float32")
    else:
        result = evaluate_stokes(module, raster, spec, batch_size=1, storage_dtype="float32")
        np.testing.assert_array_equal(result["predicted_stokes"], raster.stokes.numpy())
        assert result["wavelength_angstrom"].dtype == np.float64
        np.testing.assert_array_equal(result["wavelength_angstrom"], raster.wavelength_angstrom.numpy())
    assert module.training
    assert module.atmosphere_model.training
    assert module.synthesis_window.training
    assert not module.synthesis_window[0].training
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)


def test_export_input_reads_are_bounded_by_batch_size(monkeypatch):
    import prom3theus.artifacts.stokes as module

    raster = SimpleNamespace(
        spatial_shape=(1, 5), coordinates=torch.arange(15).reshape(1, 5, 3).float(),
        ray_direction=torch.tensor([0., 0., -1.]).expand(1, 5, 3),
        stokes_basis=torch.eye(3).expand(1, 5, 3, 3),
        auxiliary={
            "instrument_response:spectral_weights": torch.ones(6, 3).expand(1, 5, 6, 3),
            "instrument_response:continuum_weights": torch.zeros(6).expand(1, 5, 6),
        },
    )
    read = module.read_flat_samples
    reads = []

    def bounded(value, indices):
        assert len(indices) <= 2
        reads.append(indices.tolist())
        return read(value, indices)

    monkeypatch.setattr(module, "read_flat_samples", bounded)
    selected = torch.tensor([0, 2, 4])
    batches = list(module.stokes_batches(raster, selected, 2,
                                        "carrington_observer_relative", "hmi_filter_profiles"))
    assert torch.equal(torch.cat([indices for _, indices in batches]), selected)
    assert reads and all(len(indices) <= 2 for indices in reads)
    torch.testing.assert_close(torch.cat([batch["coordinates"] for batch, _ in batches]),
                               raster.coordinates[0, selected])
