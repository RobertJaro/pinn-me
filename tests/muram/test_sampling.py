"""Synthesis preserves full native columns and their geometric spacing."""

import numpy as np
import torch

from prom3theus.muram.atmosphere_builder import build_native_atmosphere
from prom3theus.muram.synthesize import synthesize_hmi
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
    first = synthesize_hmi(snap, batch_size=1, progress=False)
    np.testing.assert_array_equal(seen, expected)
    seen.clear()
    other = synthesize_hmi(snap, batch_size=3, progress=False)
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
    assert contract["depth_sampling"] == "all native cells; no vertical resampling"
