"""Stokes height selection preserves native values and geometric spacing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prom3theus.muram.atmosphere_builder import build_native_atmosphere, stokes_depth_count
from prom3theus.muram.synthesize import synthesize_hmi, synthesize_aia, synthesize_sdo
from tests.muram.test_muram_synthesis import snapshot


def test_native_layers_preserve_values_order_and_spacing():
    q = snapshot().synthesis_ready_quantities()
    q.pop("tau")  # Optical-depth labels are not needed to build the atmosphere.
    atmosphere = build_native_atmosphere(q)
    nz = q["temperature"].shape[-1]
    for field, original in (("temperature", "temperature"), ("gas_pressure", "pressure")):
        np.testing.assert_array_equal(getattr(atmosphere, field), q[original][..., ::-1].reshape(-1, nz))
    for field, names in (("magnetic_field", ("bx", "by", "bz")),
                         ("velocity_field", ("vx", "vy", "vz"))):
        expected = np.stack([q[name][..., ::-1].reshape(-1, nz) for name in names], axis=-1)
        np.testing.assert_array_equal(getattr(atmosphere, field), expected)
    np.testing.assert_array_equal(atmosphere.depth_coordinate, np.arange(nz))
    np.testing.assert_array_equal(atmosphere.geometric_height_m,
                                 np.broadcast_to(np.arange(nz)[::-1] * q["ds_m"][2], (4, nz)))


def test_synthesis_sends_every_native_layer_to_transfer(monkeypatch):
    from prom3theus.validation.reference_synthesis import LTESynthesizer

    snap = snapshot()
    q = snap.synthesis_ready_quantities()
    expected = q["temperature"][..., ::-1].reshape(4, -1).astype(np.float32)
    original_forward = LTESynthesizer.forward
    seen = []

    def checked_forward(self, atmosphere, *args, **kwargs):
        seen.extend(atmosphere.temperature.cpu().numpy())
        assert atmosphere.temperature.shape[-1] == expected.shape[-1]
        return original_forward(self, atmosphere, *args, **kwargs)

    monkeypatch.setattr(LTESynthesizer, "forward", checked_forward)
    first = synthesize_hmi(snap, batch_size=1, progress=False, keep_raw_stokes=True)
    np.testing.assert_array_equal(seen, expected)
    seen.clear()
    other = synthesize_hmi(snap, batch_size=3, progress=False, keep_raw_stokes=True)
    np.testing.assert_array_equal(seen, expected)
    torch.testing.assert_close(first.raw_stokes, other.raw_stokes)
    assert torch.isfinite(first.stokes).all()
    assert first.native_depth_samples == expected.shape[-1]
    assert first.native_vertical_spacing_m == q["ds_m"][2]
    raster = first.to_observation_raster()
    assert "ground_truth_bz_gauss" not in raster.auxiliary
    contract = raster.metadata["muram_synthesis"]
    assert "bottom_boundary" not in contract
    assert "native_log_tau_grid" not in contract
    assert contract["depth_sampling"] == "native cells within Stokes height range; no vertical resampling"
    assert contract["stokes_depth_samples"] == expected.shape[-1]


@pytest.mark.parametrize("height,count", [(3.0, 31), (3.05, 31), (20., 100), (None, 100)])
def test_stokes_height_keeps_native_centers(height, count):
    assert stokes_depth_count(100, 1e5, height) == count


@pytest.mark.parametrize("height", [0., -1., float("nan"), float("inf"), True, 0.01])
def test_invalid_stokes_height(height):
    with pytest.raises(ValueError, match="stokes_height_megameters"):
        stokes_depth_count(100, 1e5, height)


def test_uniform_shell_nodes_use_registered_native_heights_and_ray_distances(monkeypatch):
    from pathlib import Path
    import yaml
    from prom3theus.muram.geometry import SOLAR_RADIUS_M
    from prom3theus.muram.synthesize import synthesize_stokes
    from prom3theus.validation.reference_synthesis import LTESynthesizer
    from prom3theus.validation.reference_transfer import RayDistancePath

    snap = snapshot()
    q = snap.synthesis_ready_quantities()
    # Wide pixels exercise the longer off-center characteristic.
    q["ds_m"] = (1.9e8, 1.9e8, 64000.)
    native_z = np.arange(15) * q["ds_m"][2]
    q["temperature"][:] = 5000 + 1e-3 * native_z
    config = Path(__file__).resolve().parents[2] / "configs/muram/hinode/synthesis.yaml"
    shell = dict(depth_count=32, shell_height_bounds_megameter=[-.2, .2],
                 photosphere_height_megameters=.5)
    captured = []
    original = LTESynthesizer.forward

    def capture(self, atmosphere, *args, path, **kwargs):
        assert isinstance(path, RayDistancePath)
        captured.append((atmosphere.temperature.clone(), path.distance_m.clone()))
        return original(self, atmosphere, *args, path=path, **kwargs)

    monkeypatch.setattr(LTESynthesizer, "forward", capture)
    result = synthesize_stokes(snap, **yaml.safe_load(config.read_text()), uniform_shell_grid=shell,
                               dtype=torch.float64, device="cpu", progress=False)
    temperature, distance = captured[0]
    surface = result.geometry.surface_position_m[0, 0]
    impact_sq = surface[1:].square().sum()
    top = ((SOLAR_RADIUS_M + 2e5)**2 - impact_sq).sqrt()
    bottom = ((SOLAR_RADIUS_M - 2e5)**2 - impact_sq).sqrt()
    expected_distance = torch.linspace(0., 1., 32, dtype=distance.dtype) * (top - bottom)
    native_height = (impact_sq + (top - expected_distance).square()).sqrt() - SOLAR_RADIUS_M + 5e5
    torch.testing.assert_close(distance, expected_distance.expand(4, -1))
    torch.testing.assert_close(temperature, (5000 + 1e-3 * native_height).expand(4, -1))
    metadata = result.observation_metadata()["muram_synthesis"]
    assert metadata["stokes_depth_samples"] == 32
    np.testing.assert_allclose(metadata["stokes_height_range_m"], [3e5, 7e5])
    with pytest.raises(ValueError, match="outside the native cell centers"):
        synthesize_stokes(snap, **yaml.safe_load(config.read_text()),
                          uniform_shell_grid={**shell, "photosphere_height_megameters": 7.5})


def test_stokes_crops_only_transfer_and_preserves_aia_and_diagnostics():
    snap = snapshot()
    q = snap.synthesis_ready_quantities()
    q["ds_m"] = (1e5, 1e5, 4e5)  # Full box: 5.6 megameters; retained centers: 0..2.8 megameters.
    originals = {key: value.copy() for key, value in q.items() if key != "ds_m"}
    # Locate tau=1 above the Stokes cutoff to exercise full-column diagnostics.
    q["tau"][:] = np.geomspace(1e4, 0.1, 15)
    originals["tau"] = q["tau"].copy()
    lower = {key: value[..., :8].copy() if key != "ds_m" else value for key, value in q.items()}
    cropped = SimpleNamespace(time_s=snap.time_s, synthesis_ready_quantities=lambda: lower)
    expected = synthesize_hmi(cropped, stokes_height_megameters=None, diagnostics=False, progress=False)
    hmi, aia = synthesize_sdo(snap, progress=False)
    torch.testing.assert_close(hmi.stokes, expected.stokes)
    torch.testing.assert_close(aia.intensity, synthesize_aia(snap, progress=False).intensity)
    assert hmi.native_depth_samples == 15
    assert hmi.stokes_depth_samples == 8
    assert hmi.observation_metadata()["muram_synthesis"]["stokes_height_range_m"] == [0., 2.8e6]
    assert np.all(hmi.diagnostics.maps["height_m"] > 3e6)
    full = synthesize_hmi(snap, stokes_height_megameters=None, progress=False)
    assert full.stokes_depth_samples == 15
    for name, value in originals.items():
        np.testing.assert_array_equal(q[name], value)
