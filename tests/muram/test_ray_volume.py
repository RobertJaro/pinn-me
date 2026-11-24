"""Eager volume sampling preserves source values, coordinates, and boundaries."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from prom3theus.muram.atmosphere_builder import NATIVE_ATMOSPHERE_FIELDS
from prom3theus.muram.columns import NativeColumnDataset
from prom3theus.muram.ray_volume import EagerVolume
from prom3theus.muram.reader import MURaMSnapshot
from tests.muram.test_column_loading import _legacy_quantities, _write_snapshot


def analytic_snapshot(shape=(6, 5, 7)):
    spacing = (30., 40., 10.)
    x = (np.arange(shape[0]) - (shape[0] - 1) / 2) * spacing[0]
    y = (np.arange(shape[1]) - (shape[1] - 1) / 2) * spacing[1]
    z = np.arange(shape[2]) * spacing[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # Trilinear polynomial is represented exactly even on a nonuniform grid.
    values = 1000. + 2 * xx + 3 * yy + 4 * zz + .001 * xx * yy * zz
    quantities = {name: values.copy() for name in NATIVE_ATMOSPHERE_FIELDS}
    quantities["ds_m"] = spacing
    return SimpleNamespace(synthesis_ready_quantities=lambda: quantities), quantities


def test_stride_retains_original_coordinate_endpoints_and_interpolates():
    snapshot, original = analytic_snapshot()
    volume = EagerVolume.from_snapshot(snapshot, stride=3, vertical_stride=4, progress=False)
    assert volume.shape == (3, 3, 3)
    assert volume.metadata["retained_native_indices"] == [[0, 3, 5], [0, 3, 4], [0, 4, 6]]
    np.testing.assert_array_equal(volume.bounds_m, [[-75., -80., 0.], [75., 80., 60.]])
    points = np.array([[-75, -80, 0], [75, 80, 60], [5., 10., 50.], [0., 0., 20.]])
    sampled, valid = volume.sample(points.reshape(2, 2, 3))
    assert valid.shape == (2, 2)
    assert valid.all()
    x, y, z = points.T
    expected = (1000 + 2*x + 3*y + 4*z + .001*x*y*z).reshape(2, 2)
    for name in NATIVE_ATMOSPHERE_FIELDS:
        np.testing.assert_allclose(sampled[name], expected, rtol=1e-6)
        assert sampled[name].dtype == np.float32
        assert not volume.fields[name].flags.writeable
        assert not np.shares_memory(volume.fields[name], original[name])
    assert volume.nbytes == 8 * 3**3 * 4 + 3 * 3 * 8


def test_outside_queries_are_flagged_without_extrapolation():
    snapshot, _ = analytic_snapshot()
    volume = EagerVolume.from_snapshot(snapshot, progress=False)
    points = np.array([[0., 0., -1.], [76., 0., 50.], [0., 0., 61.], [np.nan, 0., 0.], [0., 0., 20.]])
    sampled, valid = volume.sample(points)
    np.testing.assert_array_equal(valid, [False, False, False, False, True])
    for field in sampled.values():
        assert np.isnan(field[:4]).all()
        assert np.isfinite(field[-1])


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_invalid_stride(stride):
    snapshot, _ = analytic_snapshot()
    with pytest.raises(ValueError, match="strides"):
        EagerVolume.from_snapshot(snapshot, stride=stride, progress=False)


@pytest.mark.parametrize("vertical", [0, 1, 2])
@pytest.mark.parametrize("increasing", [False, True])
def test_native_eager_load_is_one_field_read_and_preserves_sources(tmp_path, monkeypatch, vertical, increasing):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 5, 6), vertical, increasing)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    snapshot = MURaMSnapshot(str(source), 10)
    expected = _legacy_quantities(snapshot)
    reads = []
    mapped = NativeColumnDataset._mapped

    def count(self, name):
        reads.append(name)
        return mapped(self, name)

    monkeypatch.setattr(NativeColumnDataset, "_mapped", count)
    volume = EagerVolume.from_snapshot(snapshot, progress=False)
    assert reads == list(NATIVE_ATMOSPHERE_FIELDS)
    for name in NATIVE_ATMOSPHERE_FIELDS:
        np.testing.assert_array_equal(volume.fields[name], expected[name])
    assert not snapshot._arrays  # No duplicate eager atmosphere in the native reader.
    assert volume.native_reference["iteration"] == 10
    snapshot.close()
    for path in source.iterdir():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
    points = np.stack(np.meshgrid(*volume.axes_m, indexing="ij"), axis=-1)
    sampled, valid = volume.sample(points)
    assert valid.all()
    np.testing.assert_array_equal(sampled["bx"], expected["bx"])



@pytest.mark.parametrize("vertical_stride", [1, 4])
@pytest.mark.parametrize("increasing", [False, True])
def test_lower_volume_is_cropped_before_copying_native_fields(tmp_path, monkeypatch, vertical_stride, increasing):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 5, 9), vertical=2, increasing=increasing)
    header = source / "Header.000010"
    values = np.loadtxt(header)
    values[3:6] = [1e8, 1e8, 5e7]  # 0.5 megameter depth spacing; full height 4.
    np.savetxt(header, values)
    snapshot = MURaMSnapshot(str(source), 10)
    expected = _legacy_quantities(snapshot)
    copied_shapes = []
    copyto = np.copyto

    def record_copy(destination, source, **kwargs):
        copied_shapes.append(source.shape)
        assert source.shape == (4, 5, 7)
        return copyto(destination, source, **kwargs)

    monkeypatch.setattr(np, "copyto", record_copy)
    volume = EagerVolume.from_snapshot(snapshot, vertical_stride=vertical_stride, progress=False)
    indices = [0, 4, 6] if vertical_stride == 4 else list(range(7))
    assert volume.metadata["retained_native_indices"][2] == indices
    assert volume.metadata["native_shape"] == [4, 5, 9]
    assert volume.metadata["volume_height_megameters"] == 3.
    assert volume.metadata["loaded_height_range_m"] == [0., 3e6]
    for name in NATIVE_ATMOSPHERE_FIELDS:
        np.testing.assert_array_equal(volume.fields[name], expected[name][..., indices])
    assert copied_shapes == ([(4, 5, 7)] * 8 if vertical_stride == 1 else [])
    assert not snapshot.synthesis_columns()._owner.fields  # Never materialize full original cubes.
    assert not volume.sample([[0., 0., 3.1e6]])[1].any()
    snapshot.close()


@pytest.mark.parametrize("height", [0., -1., float("nan"), float("inf"), True, 1e-6])
def test_invalid_volume_height(height):
    snapshot, _ = analytic_snapshot()
    with pytest.raises(ValueError, match="volume_height_megameters"):
        EagerVolume.from_snapshot(snapshot, volume_height_megameters=height, progress=False)
