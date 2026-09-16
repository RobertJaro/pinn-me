"""Calibration cache tests using small real Astropy tables and no network."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import QTable
from astropy.time import Time

from prom3theus.instruments.aia_euv import calibration_download as module


class Backend:
    def __init__(self, *, version=10, intervals=None):
        self.calls = []
        self.version = version
        self.intervals = intervals or [
            ("2011-02-14T00:00:00", "2011-02-16T00:00:00"),
        ]

    def fetch_correction_table(self):
        self.calls.append("correction")
        return QTable({"VER_NUM": [self.version]})

    def fetch_pointing_table(self, **kwargs):
        self.calls.append(("pointing", kwargs))
        return QTable({
            "T_START": Time([a for a, _ in self.intervals]),
            "T_STOP": Time([b for _, b in self.intervals]),
        })


def download(tmp_path, backend, **kwargs):
    options = dict(
        output_directory=tmp_path / "calibration",
        start_utc="2011-02-14T00:00:00", end_utc="2011-02-15T06:00:00",
        backend=backend,
    )
    options.update(kwargs)
    return module.download_aia_preprocessing_calibration(**options)


def test_fetches_two_tables_and_reuses_without_network(tmp_path, monkeypatch):
    checksum = module._sha256_file

    def check_final_path(path):
        assert path.parent == tmp_path / "calibration"
        return checksum(path)

    monkeypatch.setattr(module, "_sha256_file", check_final_path)
    backend = Backend()
    path = download(tmp_path, backend)
    assert len(backend.calls) == 2
    assert path.is_file()
    assert set(tmp_path.iterdir()) == {tmp_path / "calibration"}
    correction = QTable.read(path.parent / module.AIA_CORRECTION_ECSV, format="ascii.ecsv")
    pointing = QTable.read(path.parent / module.AIA_POINTING_ECSV, format="ascii.ecsv")
    np.testing.assert_array_equal(correction["VER_NUM"], [10])
    assert isinstance(pointing["T_START"], Time)
    # Requests longer than 24 hours are supported.
    assert download(tmp_path, backend) == path
    assert len(backend.calls) == 2
    assert module.load_aia_calibration_manifest(path.parent)["tables"]["pointing"]["sha256"]


def test_existing_pointing_interval_must_cover_request(tmp_path):
    backend = Backend()
    download(tmp_path, backend)
    with pytest.raises(ValueError, match="does not cover"):
        download(tmp_path, backend, end_utc="2011-02-16T00:00:00")
    assert len(backend.calls) == 2
    download(tmp_path, backend, end_utc="2011-02-16T00:00:00", overwrite=True)
    assert len(backend.calls) == 4


@pytest.mark.parametrize("intervals", [
    [("2011-02-14T01:00:00", "2011-02-16T00:00:00")],
    [("2011-02-14T00:00:00", "2011-02-15T00:00:00")],
    [("2011-02-14T00:00:00", "2011-02-14T03:00:00"),
     ("2011-02-14T06:00:00", "2011-02-16T00:00:00")],
])
def test_rejects_incomplete_pointing_coverage_before_publication(tmp_path, intervals):
    with pytest.raises(ValueError, match="gap or does not cover"):
        download(tmp_path, Backend(intervals=intervals))
    assert not (tmp_path / "calibration").exists()


def test_rejects_missing_v10_without_output(tmp_path):
    with pytest.raises(ValueError, match="V10"):
        download(tmp_path, Backend(version=9))
    assert not (tmp_path / "calibration").exists()


def test_changed_cache_is_not_silently_reused(tmp_path):
    backend = Backend()
    path = download(tmp_path, backend)
    (path.parent / module.AIA_CORRECTION_ECSV).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        download(tmp_path, backend)
    assert len(backend.calls) == 2


def test_invalid_interval_does_not_fetch(tmp_path):
    backend = Backend()
    with pytest.raises(ValueError, match="later"):
        download(tmp_path, backend, end_utc="2011-02-14T00:00:00")
    assert not backend.calls


def test_fetch_failure_preserves_existing_cache(tmp_path):
    backend = Backend()
    path = download(tmp_path, backend)
    before = path.read_bytes()

    def fail(**kwargs):
        raise RuntimeError("network unavailable")

    backend.fetch_pointing_table = fail
    with pytest.raises(RuntimeError, match="network unavailable"):
        download(tmp_path, backend, overwrite=True)
    assert path.read_bytes() == before
    module.load_aia_calibration_manifest(path.parent)


def test_legacy_cache_manifest_can_be_reused(tmp_path):
    backend = Backend()
    path = download(tmp_path, backend)
    manifest = json.loads(path.read_text())
    manifest.update(format="legacy", version=1, implementation={"aiapy": "0.12.1"})
    path.write_text(json.dumps(manifest))
    download(tmp_path, backend)
    assert len(backend.calls) == 2


def test_real_backend_delegates_to_standard_aiapy_getters(monkeypatch):
    import sys
    calls = []
    utils = SimpleNamespace(
        get_correction_table=lambda **kw: calls.append(("correction", kw)) or "correction",
        get_pointing_table=lambda **kw: calls.append(("pointing", kw)) or "pointing",
    )
    monkeypatch.setitem(sys.modules, "aiapy.calibrate.utils", utils)
    backend = module.AiapyCalibrationAcquisitionBackend()
    assert backend.fetch_correction_table() == "correction"
    assert backend.fetch_pointing_table(
        start_utc="2011-02-14T00:00:00+00:00", end_utc="2011-02-15T00:00:00+00:00",
    ) == "pointing"
    assert calls[0] == ("correction", {"source": "SSW"})
    assert calls[1][1]["source"] == "jsoc"
    assert all(isinstance(t, Time) for t in calls[1][1]["time_range"])
