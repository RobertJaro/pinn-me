"""Small deterministic tests for optional training diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prom3theus.diagnostics.evaluation import AtmosphereEvaluator
from prom3theus.diagnostics.rendering import AtmosphereRenderer
from prom3theus.diagnostics.sampling import ValidationSampleCollector
from prom3theus.training.callbacks import AtmosphereVisualizationCallback


def test_visualization_configuration_is_strict(tmp_path):
    callback = AtmosphereVisualizationCallback(
        tmp_path,
        every_n_validations=2,
        ray_sampling={"batch_size": 8, "max_pixels": 16, "max_profile_samples": 4},
        slice_sampling={
            "batch_size": 8,
            "layer_count": 3,
            "longitude_points": 2,
            "latitude_points": 2,
            "radial_points": 2,
        },
        dpi=70,
        include_initial=False,
    )
    assert callback.output_directory == tmp_path.resolve()
    assert callback.every_n_validations == 2
    assert isinstance(callback.evaluator, AtmosphereEvaluator)
    assert isinstance(callback.renderer, AtmosphereRenderer)
    assert isinstance(callback.collector, ValidationSampleCollector)
    with pytest.raises(TypeError, match="Unknown ray-sampling options"):
        AtmosphereVisualizationCallback(tmp_path, ray_sampling={"mystery": 1})
    with pytest.raises(ValueError, match="requires longitude_deg"):
        AtmosphereVisualizationCallback(
            tmp_path,
            meridional_slice={"enabled": True, "longitude_deg": None},
        )


def test_validation_collection_is_bounded_and_uses_the_loss_mask(tmp_path):
    callback = AtmosphereVisualizationCallback(
        tmp_path,
        every_n_validations=1,
        ray_sampling={"max_profile_samples": 3},
    )
    module = SimpleNamespace(
        wavelength_angstrom=torch.tensor((6300.0, 6301.0, 6302.0, 6303.0)),
        wavelength_weights=torch.tensor((1.0, 1.0, 0.0, 1.0)),
    )
    callback.on_validation_epoch_start(SimpleNamespace(sanity_checking=False), module)
    for column in range(2):
        profiles = torch.ones(2, 4, 4)
        callback.on_validation_batch_end(
            None,
            module,
            {
                "stokes_pred": profiles,
                "stokes_reference": profiles,
                "pixel_index": torch.tensor(((0, column), (1, column))),
            },
            None,
            column,
        )

    payload = callback._local_validation_payload()
    assert payload is not None
    assert payload["integrated_prediction"].shape == (4, 4)
    torch.testing.assert_close(payload["integrated_prediction"], torch.ones(4, 4))
    assert payload["stokes_pred"].shape == (3, 4, 4)
    assert payload["stokes_reference"].shape == (3, 4, 4)


def test_validation_cadence_is_local_to_the_current_fit(tmp_path):
    trainer = SimpleNamespace(sanity_checking=False)
    callback = AtmosphereVisualizationCallback(
        tmp_path, every_n_validations=3, include_initial=False
    )
    callback.on_validation_epoch_start(trainer, None)
    callback.on_validation_epoch_start(trainer, None)
    callback.on_validation_epoch_start(trainer, None)

    assert callback._validation_event_count == 3
    assert callback._active_validation_event == 3
    assert callback._collect_validation_event is True
    assert "state_dict" not in AtmosphereVisualizationCallback.__dict__
    assert "load_state_dict" not in AtmosphereVisualizationCallback.__dict__
