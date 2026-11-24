"""Selected renderers share one input pass and execute in a defined order."""

import torch

import prom3theus.muram.synthesize as synthesis
from prom3theus.muram.columns import ArrayColumns
from tests.muram.test_sdo_fits import snapshot


def test_sdo_runs_sequential_renderers_once_per_loaded_batch(monkeypatch):
    snap = snapshot((4, 4, 30))
    options = dict(stride=2, binning=2, batch_size=3, progress=False, device="cpu")
    expected_hmi = synthesis.synthesize_hmi(snap, **options)
    expected_aia = synthesis.synthesize_aia(snap, **options)
    reads, events, descriptor_calls = [], [], []
    descriptor = {"test_native_snapshot": 10}

    def native_reference(**kwargs):
        descriptor_calls.append(kwargs)
        return descriptor

    snap.native_reference = native_reference
    original_batches = ArrayColumns.batches
    stokes = synthesis._StokesRenderer.__call__
    aia = synthesis.AIAAccumulator.__call__
    diagnostic = synthesis.DiagnosticAccumulator.__call__

    def once(self, batch_size, fields=None):
        assert not reads, "The snapshot must have only one source pass"
        assert set(fields) == {"temperature", "pressure", "tau", "bx", "by", "bz", "vx", "vy", "vz"}
        for ix, iy, columns in original_batches(self, batch_size, fields):
            reads.extend(zip(ix.tolist(), iy.tolist()))
            yield ix, iy, columns

    def render_stokes(self, *args):
        events.append("stokes")
        return stokes(self, *args)

    def render_aia(self, *args):
        events.append("aia")
        return aia(self, *args)

    def accumulate(self, *args):
        events.append("diagnostics")
        return diagnostic(self, *args)

    monkeypatch.setattr(ArrayColumns, "batches", once)
    monkeypatch.setattr(synthesis._StokesRenderer, "__call__", render_stokes)
    monkeypatch.setattr(synthesis.AIAAccumulator, "__call__", render_aia)
    monkeypatch.setattr(synthesis.DiagnosticAccumulator, "__call__", accumulate)
    actual_hmi, actual_aia = synthesis.synthesize_sdo(snap, **options)
    assert reads == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert events == ["diagnostics", "stokes", "aia", "diagnostics", "stokes", "aia"]
    assert descriptor_calls == [{"stride": 2, "binning": 2}]
    assert actual_hmi.native_reference == actual_aia.native_reference == descriptor
    assert actual_hmi.observation_metadata()["muram_synthesis"]["native_reference"] == descriptor
    assert actual_hmi.raw_stokes is None
    assert actual_aia.response_resource == synthesis.AIA_RESPONSE_RESOURCE
    torch.testing.assert_close(actual_hmi.stokes, expected_hmi.stokes)
    torch.testing.assert_close(actual_aia.intensity, expected_aia.intensity)
