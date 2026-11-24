"""Surface diagnostics share source batches and can be repaired independently."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import prom3theus.muram.diagnostics as diagnostics
from prom3theus.muram.columns import ArrayColumns
from prom3theus.muram.diagnostics import DiagnosticAccumulator, ensure_diagnostics, load_diagnostic_maps
from prom3theus.muram.synthesize import synthesize_hmi, synthesize_sdo
from tests.muram.test_sdo_fits import snapshot


def accumulator(shape):
    return DiagnosticAccumulator(shape, (10., 20., 100.), native_shape=shape, stride=1,
                                 synthesis_binning=1, native_reference={}, time_s=3600.)


@pytest.mark.parametrize("tau,status,height", [
    ([10., .1, .01], 0, 50.), ([.1, 10., 100.], 0, 50.),
    ([10., 1., .1], 0, 100.), ([1., .1, .01], 0, 0.), ([10., 2., 1.], 0, 200.),
    ([10., 1., 10.], 0, 100.),  # One exact sample, without a strict crossing.
    ([2., 0., 0.], 0, 50.),  # Zero endpoint uses explicit linear-tau fallback.
    ([10., 2., 1.1], 1, None), ([.1, .01, .001], 1, None),
    ([10., .1, 10.], 2, None), ([1., 1., .1], 2, None),
    ([1., .1, 1.], 2, None), ([1., np.nan, .1], 3, None),
    ([10., -1., .1], 3, None), ([np.inf, 1., .1], 3, None),
])
def test_unique_surface_interpolation_and_status(tau, status, height):
    current = accumulator((1, 1, 3))
    fields = {name: np.array([[[100., 200., 300.]]]) for name in current.fields}
    fields["tau"] = np.array(tau)[None, None, :]
    before = {name: values.copy() for name, values in fields.items()}
    current(np.array([0]), np.array([0]), fields)
    result = current.result()
    assert result.status.item() == status
    if height is None:
        assert all(np.isnan(values).all() for values in result.maps.values())
    else:
        assert result.maps["height_m"].item() == pytest.approx(height)
        for name in current.fields[:-1]:
            assert result.maps[name].item() == pytest.approx(height + 100.)
    for name in fields:
        np.testing.assert_array_equal(fields[name], before[name])


def test_surface_values_are_validated_without_invalidating_other_columns():
    current = accumulator((2, 1, 3))
    fields = {name: np.full((2, 1, 3), 100.) for name in current.fields}
    fields["tau"][:] = [10., 1., .1]
    fields["temperature"][0, 0, 1] = -1
    current(np.arange(2), np.zeros(2, dtype=int), fields)
    np.testing.assert_array_equal(current.status, [[4, 0]])
    assert np.isnan(current.maps["height_m"][0, 0])
    assert current.maps["height_m"][0, 1] == 100.


def test_stokes_transfer_without_diagnostics_does_not_require_tau():
    snap = snapshot()
    snap.synthesis_ready_quantities().pop("tau")
    result = synthesize_hmi(snap, diagnostics=False, progress=False)
    assert result.diagnostics is None
    assert torch.isfinite(result.stokes).all()


def test_maps_keep_xy_order_native_frame_and_prebinning_resolution(monkeypatch):
    snap = snapshot((4, 4, 30))
    q = snap.synthesis_ready_quantities()
    q["vx"][:] = 3000
    for x in range(4):
        for y in range(4):
            q["bx"][x, y] = 100 * x + y
    options = dict(stride=2, binning=2, batch_size=3, progress=False, device="cpu")
    plain = synthesize_hmi(snap, diagnostics=False, **options)
    passes = []
    original = ArrayColumns.batches

    def counted(self, *args, **kwargs):
        passes.append(1)
        yield from original(self, *args, **kwargs)

    monkeypatch.setattr(ArrayColumns, "batches", counted)
    hmi, aia = synthesize_sdo(snap, **options)
    assert len(passes) == 1
    assert hmi.diagnostics is aia.diagnostics
    np.testing.assert_array_equal(hmi.diagnostics.maps["bx"], [[0., 200.], [2., 202.]])
    np.testing.assert_allclose(hmi.diagnostics.maps["vx"], 3000.)
    assert hmi.stokes.shape[:2] == (1, 1)
    assert hmi.diagnostics.status.shape == (2, 2)
    assert hmi.diagnostics.metadata["sampling"]["map_binning"] == 1
    assert hmi.diagnostics.metadata["sampling"]["synthesis_binning"] == 2
    torch.testing.assert_close(hmi.stokes, plain.stokes, rtol=0, atol=0)
    assert plain.diagnostics is None


def test_missing_overview_is_repaired_from_maps_after_plot_failure(tmp_path, monkeypatch):
    import prom3theus.muram.plotting as plotting
    from prom3theus.muram.cli import main

    result = synthesize_hmi(snapshot(), progress=False).diagnostics
    original_plot = plotting.write_parameter_overview

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted image rendering")

    monkeypatch.setattr(plotting, "write_parameter_overview", fail)
    with pytest.raises(RuntimeError, match="image rendering"):
        ensure_diagnostics(tmp_path, 10, {}, result=result)
    directory = tmp_path / "diagnostics"
    sidecar = directory / "muram_000010.npz"
    before = sidecar.read_bytes()
    restored = load_diagnostic_maps(sidecar)
    assert restored.metadata["units"]["bz"] == "G"
    assert restored.metadata["surface"]["value"] == 1.
    for name in result.maps:
        np.testing.assert_array_equal(restored.maps[name], result.maps[name])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["snapshots"]["10"]["state"] == "maps_saved"
    assert "RuntimeError" in manifest["snapshots"]["10"]["error"]

    def forbidden(*args, **kwargs):
        pytest.fail("Repairing an overview must neither open cubes nor synthesize spectra")

    monkeypatch.setattr(diagnostics, "collect_diagnostic_maps", forbidden)
    monkeypatch.setattr(diagnostics, "open_native_reference", forbidden)
    monkeypatch.setattr(plotting, "write_parameter_overview", original_plot)
    main(["diagnostics", "--out", str(tmp_path), "--iterations", "10", "--no-progress"])
    assert sidecar.read_bytes() == before
    assert (directory / "muram_000010.jpg").is_file()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["snapshots"]["10"]["state"] == "complete"
    assert "error" not in manifest["snapshots"]["10"]
    # An existing valid overview is also left byte-for-byte untouched.
    monkeypatch.setattr(plotting, "write_parameter_overview", forbidden)
    ensure_diagnostics(tmp_path, 10, {})


def test_missing_maps_backfill_only_one_source_pass(tmp_path, monkeypatch):
    from prom3theus.validation.reference_synthesis import LTESynthesizer
    from prom3theus.muram.cli import main

    snap = snapshot((4, 4, 30))
    reference = {"iteration": 10, "pixel_mapping": {"stride": 2, "binning": 2}}
    seen = []
    closed = []
    source = ArrayColumns(snap.synthesis_ready_quantities())
    proxy = SimpleNamespace(synthesis_columns=lambda: source, time_s=snap.time_s,
                            close=lambda: closed.append(True))
    monkeypatch.setattr(diagnostics, "open_native_reference", lambda ref: proxy)
    monkeypatch.setattr(diagnostics, "verify_native_reference", lambda ref: None)
    original = ArrayColumns.batches

    def once(self, *args, **kwargs):
        assert not seen
        for ix, iy, batch in original(self, *args, **kwargs):
            seen.extend(zip(ix, iy))
            yield ix, iy, batch

    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostic-only backfill must not run LTE")

    monkeypatch.setattr(ArrayColumns, "batches", once)
    monkeypatch.setattr(LTESynthesizer, "forward", forbidden)
    (tmp_path / "synthesis_status.json").write_text(json.dumps({"native_references": {"10": reference}}))
    status_before = (tmp_path / "synthesis_status.json").read_bytes()
    main(["diagnostics", "--out", str(tmp_path), "--batch-size", "3", "--no-progress"])
    assert seen == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert closed == [True]
    assert (tmp_path / "synthesis_status.json").read_bytes() == status_before
    restored = load_diagnostic_maps(tmp_path / "diagnostics/muram_000010.npz")
    assert restored.metadata["native_reference"] == reference
    assert restored.status.shape == (2, 2)


def test_sidecar_reference_mismatch_is_not_silently_reused(tmp_path):
    result = synthesize_hmi(snapshot(), progress=False).diagnostics
    ensure_diagnostics(tmp_path, 10, {}, result=result)
    with pytest.raises(ValueError, match="reference mismatch"):
        ensure_diagnostics(tmp_path, 10, {"different": True})
