"""Loading-time reduction preserves native artifacts and physical geometry."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from prom3theus.muram.binning import bin_image_raster, bin_spatial, bin_stokes_raster
from prom3theus.muram.geometry import build_disk_center_geometry, SOLAR_RADIUS_M
from prom3theus.muram.synthesize import SyntheticHMIResult
from prom3theus.observations import ObservationStore
from prom3theus.observations.image_contracts import ImageObservationRaster


def synthetic_result():
    geometry = build_disk_center_geometry(8, 4, 2e5, 3e5, reference_time_hours=2)
    stokes = torch.arange(4 * 8 * 4 * 3, dtype=torch.float64).reshape(4, 8, 4, 3)
    return SyntheticHMIResult(
        stokes=stokes, raw_stokes=stokes,
        observed_wavelength_angstrom=torch.tensor([6301., 6302., 6303.]),
        raw_wavelength_angstrom=torch.tensor([6301., 6302., 6303.]),
        geometry=geometry,
        dx_m=2e5, dy_m=3e5, instrument_config={"type": "hinode_sp"},
        response={"weights": torch.tensor([0.2, 0.8])},
    )


def test_store_loader_bins_without_changing_native_source(tmp_path):
    from prom3theus.muram.observation import _build_data

    result = synthetic_result()
    raster = result.to_observation_raster()
    path = tmp_path / "native"
    ObservationStore.save(path, raster, source_signature="a" * 64)
    data = _build_data({"directory": path, "loader": {}, "validation_raster_index": 0, "binning": 2})
    actual = data.raster
    expected_geometry = build_disk_center_geometry(8, 4, 2e5, 3e5, binning=2, reference_time_hours=2)
    for y in range(2):
        for x in range(4):
            torch.testing.assert_close(actual.stokes[y, x], raster.stokes[2*y:2*y+2, 2*x:2*x+2].mean((0, 1)))
    torch.testing.assert_close(actual.coordinates, expected_geometry.coordinates)
    torch.testing.assert_close(actual.surface_position_m, expected_geometry.surface_position_m)
    torch.testing.assert_close(actual.stokes_basis, expected_geometry.stokes_basis)
    torch.testing.assert_close(actual.auxiliary["instrument_response:weights"], torch.tensor([0.2, 0.8]).expand(2, 4, 2))
    assert actual.metadata["spatial_binning"]["dx_m"] == 4e5
    assert actual.metadata["spatial_binning"]["dy_m"] == 6e5
    assert len(actual.metadata["observer_velocity_correction"]["removed_solar_los_velocity_m_per_s"]) == 4
    assert actual.metadata["muram_synthesis"]["spatial_binning"] == 1
    native = ObservationStore.load(path)
    assert native.spatial_shape == (4, 8)
    torch.testing.assert_close(native.stokes[:], raster.stokes)
    assert "spatial_binning" not in native.metadata
    data.setup()  # No double application on repeated setup.
    assert data.raster.spatial_shape == (2, 4)


def test_invalid_pixels_invalidate_whole_blocks():
    raster = synthetic_result().to_observation_raster()
    mask = raster.valid_mask.clone()
    mask[0, 0] = False
    values = raster.stokes.clone()
    values[0, 0] = float("nan")
    actual = bin_stokes_raster(replace(raster, stokes=values, valid_mask=mask), 2)
    assert not actual.valid_mask[0, 0]
    assert actual.valid_mask.sum() == 7


def test_image_mean_uncertainty_and_geometry():
    result = synthetic_result()
    raster = ImageObservationRaster(
        intensity=result.stokes[..., 0, 0],
        uncertainty=torch.full((4, 8), 2., dtype=torch.float64),
        ray_direction=result.geometry.ray_direction,
        surface_position_m=result.geometry.surface_position_m,
        valid_mask=torch.ones((4, 8), dtype=torch.bool),
        absolute_tai_seconds=123., channel_angstrom=171, exposure_group="muram_000000",
        metadata={"ray_geometry": {"solar_radius_m": SOLAR_RADIUS_M},
                  "provenance": {"synthetic": True, "dx_m": 2e5, "dy_m": 3e5}},
    )
    actual = bin_image_raster(raster, 2)
    assert actual.spatial_shape == (2, 4)
    torch.testing.assert_close(actual.intensity, result.stokes[..., 0, 0].reshape(2, 2, 4, 2).mean((1, 3)))
    torch.testing.assert_close(actual.uncertainty, torch.ones((2, 4), dtype=torch.float64))
    expected = build_disk_center_geometry(8, 4, 2e5, 3e5, binning=2)
    torch.testing.assert_close(actual.surface_position_m, expected.surface_position_m)
    assert actual.metadata["provenance"]["spatial_binning"]["dx_m"] == 4e5


@pytest.mark.parametrize("factor", [0, -1, True, 1.5, 3, None])
def test_invalid_loading_factors(factor):
    with pytest.raises(ValueError, match="binning"):
        bin_stokes_raster(synthetic_result().to_observation_raster(), factor)


def test_prediction_binning_retains_gradients_and_default_is_identity():
    raster = synthetic_result().to_observation_raster()
    assert bin_stokes_raster(raster) is raster
    prediction = raster.stokes.clone().requires_grad_()
    bin_spatial(prediction, 2).sum().backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 0.25))


def test_prepared_cache_keeps_resolutions_separate(tmp_path):
    from prom3theus.muram.provenance import stokes_forward_provenance
    from prom3theus.observations.registry import load_or_prepare_observation

    result = replace(synthetic_result(), forward_model_provenance=stokes_forward_provenance("hinode_sp"))
    path = tmp_path / "native"
    ObservationStore.save(path, result.to_observation_raster(), source_signature="a" * 64)
    config = {
        "type": "muram_stored", "directory": path, "validation_raster_index": 0,
        "loader": {"batch_size": 2, "validation_batch_size": 2,
                   "validation_stride": 1, "workers": 0, "pin_memory": False},
    }
    native, _, _ = load_or_prepare_observation({**config, "binning": 1}, {}, tmp_path / "cache")
    binned, _, _ = load_or_prepare_observation({**config, "binning": 2}, {}, tmp_path / "cache")
    cached, _, _ = load_or_prepare_observation({**config, "binning": 2}, {}, tmp_path / "cache")
    assert native.observation_store_path != binned.observation_store_path
    assert binned.observation_store_path == cached.observation_store_path
    assert native.raster.spatial_shape == (4, 8)
    assert cached.raster.spatial_shape == (2, 4)
    torch.testing.assert_close(cached.raster.stokes[:], bin_spatial(native.raster.stokes, 2))


@pytest.mark.parametrize("factor", [0, -1, True, 1.5])
def test_schema_rejects_invalid_binning(factor):
    from prom3theus.muram import config as _registration  # noqa: F401
    from prom3theus.config import load_config

    for name in ("configs/muram/hinode/mhs.yaml", "configs/muram/sdo/joint.yaml"):
        for stream in load_config(name).streams:
            with pytest.raises((TypeError, ValueError), match="binning"):
                replace(stream.observation, binning=factor)


def test_native_scripts_and_loading_presets():
    from pathlib import Path
    from prom3theus.muram import config as _registration  # noqa: F401
    from prom3theus.config import load_config

    root = Path(__file__).resolve().parents[2]
    for name in ("scripts/muram/synthesize_snapshots.sh", "scripts/muram_hinode/synthesize.sh"):
        assert "--binning" not in (root / name).read_text()
        assert "--stride" not in (root / name).read_text()
    for path in (root / "configs/muram").glob("*/*.yaml"):
        if path.name == "synthesis.yaml":
            continue
        for stream in load_config(path).streams:
            assert stream.observation.binning == 4
