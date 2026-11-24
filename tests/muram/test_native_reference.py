"""Native references retain scientific registration without copying cube data."""

import builtins
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from prom3theus.muram.reader import MURaMSnapshot
from prom3theus.muram.reference import open_native_reference, verify_native_reference
from tests.muram.test_column_loading import _write_snapshot


@pytest.mark.parametrize("boundary", ["equal", "nan", "inf", "negative"])
def test_ambiguous_tau_orientation_is_rejected(tmp_path, boundary):
    source = tmp_path / "source"
    shape = (4, 5, 6)
    _write_snapshot(source, shape, 1)
    tau = np.fromfile(source / "tau.000010", dtype=np.float32).reshape(shape, order="F")
    if boundary == "equal":
        tau[:, -1, :] = tau[:, 0, :]
    else:
        tau[:, 0, :] = {"nan": np.nan, "inf": np.inf, "negative": -1}[boundary]
    tau.ravel(order="F").tofile(source / "tau.000010")
    snapshot = MURaMSnapshot(str(source), 10)
    with pytest.raises(ValueError, match="Cannot determine vertical orientation.*iteration 10"):
        snapshot.synthesis_columns()
    snapshot.close()


def test_reference_records_resolved_frame_and_native_pixel_mapping(tmp_path):
    source = tmp_path / "source"
    _write_snapshot(source, (8, 5, 12), 1)
    snapshot = MURaMSnapshot(str(source), 10)
    reference = json.loads(json.dumps(snapshot.native_reference(stride=2, binning=2)))
    geometry = reference["native_geometry"]
    assert reference["iteration"] == 10
    assert reference["time_s"] == 3600
    assert geometry["header_shape"] == [8, 5, 12]
    assert geometry["oriented_shape_xyz"] == [8, 12, 5]
    assert geometry["header_axes_for_local_xyz"] == [0, 2, 1]
    assert geometry["vector_components_for_local_xyz"] == [0, 2, 1]
    assert geometry["index_direction_xyz"] == [1, -1, 1]
    assert geometry["vector_sign_xyz"] == [1, -1, 1]
    assert geometry["oriented_spacing_m"] == [1e5, 1e5, 1e4]
    assert geometry["height_range_m"] == [0, 4e4]
    assert geometry["photosphere_height_m"] is None
    assert "not the photosphere" in geometry["height_origin"]
    assert reference["pixel_mapping"]["output_shape_yx"] == [3, 2]
    assert reference["pixel_mapping"]["stride"] == 2
    assert reference["pixel_mapping"]["binning"] == 2
    assert set(reference["files"]) == {"header", "temperature", "pressure", "tau", "bx", "by", "bz", "vx", "vy", "vz"}
    assert len(json.dumps(reference)) < 10000
    assert not snapshot._arrays
    assert set(snapshot._mappings) == {"tau"}
    snapshot.close()


def test_reference_verification_only_stats_files_and_detects_changes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 5, 6), 1)
    snapshot = MURaMSnapshot(str(source), 10)
    reference = snapshot.native_reference()
    snapshot.close()

    def no_read(*args, **kwargs):
        pytest.fail("Reference verification must not read file contents")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", no_read)
        patch.setattr(Path, "open", no_read)
        assert verify_native_reference(reference) == source.resolve()
    path = source / "eosT.000010"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="Native source identity changed"):
        verify_native_reference(reference)


def test_reopen_is_lazy_and_supports_explicit_relocation(tmp_path):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 5, 6), 1, increasing=True)
    original = MURaMSnapshot(str(source), 10, axis_order=(1, 0, 2), component_order=(2, 0, 1))
    reference = original.native_reference()
    expected = original.synthesis_columns().dataset[0]
    relocated = tmp_path / "copy"
    shutil.copytree(source, relocated, copy_function=shutil.copy2)
    reopened = open_native_reference(reference, source_path=relocated)
    try:
        assert not reopened._arrays and not reopened._mappings and not reopened._synthesis_cache
        assert reopened.axis_order == original.axis_order
        assert reopened.component_order == original.component_order
        actual = reopened.synthesis_columns().dataset[0]
        for name, values in expected["fields"].items():
            np.testing.assert_array_equal(actual["fields"][name], values)
    finally:
        reopened.close()
        original.close()


def test_reference_rejects_a_new_preferred_filename(tmp_path):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 5, 6), 1)
    snapshot = MURaMSnapshot(str(source), 10)
    reference = snapshot.native_reference()
    snapshot.close()
    shutil.copy2(source / "eosT.000010", source / "eosT.10")
    with pytest.raises(ValueError, match="file selection changed"):
        open_native_reference(reference)
