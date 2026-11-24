"""Read-only eager cubes, sequential column batches, and profile parity."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prom3theus.muram.reader import MURaMSnapshot, _FIELD_IDS, _UNIT_SCALE
from prom3theus.muram.diagnostics import collect_diagnostic_maps
from prom3theus.muram.synthesize import synthesize_aia, synthesize_hmi, synthesize_stokes


def _write_snapshot(path, shape, vertical, increasing=False, physical=False):
    path.mkdir()
    spacing = [1e7, 1e7, 1e7]
    spacing[vertical] = 1e6
    np.savetxt(path / "Header.000010", [*shape, *spacing, 3600.])
    index = np.indices(shape)
    level = index[vertical] / (shape[vertical] - 1)
    if increasing:
        level = 1 - level
    for number, name in enumerate(("bx", "by", "bz", "vx", "vy", "vz", "pressure", "temperature", "tau")):
        if name == "tau":
            values = 10. ** (2 - 8 * level)
        elif physical and name == "temperature":
            values = 6500 - 2000 * level + 10 * index[(vertical + 1) % 3]
        elif physical and name == "pressure":
            values = 2e6 * (10 / 2e6) ** level
        else:
            values = 100 * (number + 1) + 3 * index[0] + 5 * index[1] + 7 * index[2]
        (values / _UNIT_SCALE[name]).astype("float32").ravel(order="F").tofile(
            path / f"{_FIELD_IDS[name]}.000010"
        )


def _legacy_quantities(snapshot):
    """Independent eager pre-change reader, including conversion before reflection."""
    fields = {}
    for name in ("bx", "by", "bz", "vx", "vy", "vz", "pressure", "temperature", "tau"):
        raw = np.fromfile(snapshot._field_path(_FIELD_IDS[name]), dtype=np.float32)
        native = raw.reshape(snapshot._native_shape[::-1]).transpose(2, 1, 0)
        fields[name] = native.transpose(snapshot.axis_order).astype(np.float32) * np.float32(_UNIT_SCALE[name])
    vertical = snapshot.detect_vertical_axis()
    horizontal = tuple(axis for axis in range(3) if axis != vertical)
    profile = fields["tau"].mean(axis=horizontal)
    z_sign = 1 if profile[0] > profile[-1] else -1
    order = horizontal + (vertical,)
    components = tuple(snapshot.component_order[snapshot.axis_order[axis]] for axis in order)
    y_sign = round(np.linalg.det(np.eye(3)[list(components)])) * z_sign
    result = {}
    for name in fields:
        sign, source = 1, name
        if name[0] in ("b", "v") and name[1:] in "xyz":
            axis = "xyz".index(name[1])
            source = name[0] + "xyz"[components[axis]]
            sign = (1, y_sign, z_sign)[axis]
        values = fields[source].transpose(order)
        if z_sign < 0:
            values = np.flip(values, -1)
        if y_sign < 0:
            values = np.flip(values, 1)
        result[name] = values if sign == 1 else -values
    result["ds_m"] = tuple(snapshot.ds_m[axis] for axis in order)
    return result


@pytest.mark.parametrize("vertical", [0, 1, 2])
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("axis_order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("component_order", [(0, 1, 2), (1, 0, 2)])
def test_flattened_dataset_preserves_every_native_column(tmp_path, vertical, increasing, axis_order, component_order):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 5, 6), vertical, increasing)
    snapshot = MURaMSnapshot(str(source), 10, axis_order=axis_order, component_order=component_order)
    expected = _legacy_quantities(snapshot)
    columns = snapshot.synthesis_columns()
    assert snapshot.synthesis_columns() is columns
    dataset = columns.dataset
    assert not dataset._mappings
    assert set(snapshot._mappings) == {"tau"}
    nx, ny, nz = dataset.shape
    assert len(dataset) == nx * ny
    for index in range(len(dataset)):
        sample = dataset[index]
        ix, iy = divmod(index, ny)
        assert sample["index"] == index
        for name, value in sample["fields"].items():
            assert value.shape == (nz,)
            assert value.dtype == np.float32
            np.testing.assert_array_equal(value, expected[name][ix, iy])
    assert not snapshot._arrays
    assert all(not value.flags.writeable for value in dataset._mappings.values())
    first, second = snapshot.synthesis_ready_quantities(), snapshot.synthesis_ready_quantities()
    for name in dataset.paths:
        np.testing.assert_array_equal(first[name], expected[name])
        assert first[name] is second[name]


def test_orientation_reads_only_boundaries_and_is_cached(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 5, 6), 1)
    snapshot = MURaMSnapshot(str(source), 10)
    mean = np.memmap.mean
    shapes = []

    def counted_mean(array, *args, **kwargs):
        shapes.append(array.shape)
        assert array.ndim == 2
        return mean(array, *args, **kwargs)

    monkeypatch.setattr(np.memmap, "mean", counted_mean)
    first = snapshot.synthesis_columns()
    second = snapshot.synthesis_columns()
    snapshot.synthesis_ready_quantities()
    assert shapes == [(4, 6), (4, 6)]
    assert first is second
    assert not first._owner.fields


def test_dataset_pickles_descriptors_without_cube_contents(tmp_path):
    import pickle

    source = tmp_path / "source"
    _write_snapshot(source, (8, 9, 10), 1)
    snapshot = MURaMSnapshot(str(source), 10)
    dataset = snapshot.synthesis_columns().dataset
    sample = dataset[0]
    payload = pickle.dumps(dataset)
    assert len(payload) < 10000
    restored = pickle.loads(payload)
    assert restored._mappings == {}
    assert restored._pid is None
    for name, value in sample["fields"].items():
        np.testing.assert_array_equal(restored[0]["fields"][name], value)


def test_eager_cubes_load_once_before_sequential_batches(tmp_path, monkeypatch):

    source = tmp_path / "source"
    _write_snapshot(source, (5, 6, 7), 1)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    snapshot = MURaMSnapshot(str(source), 10)
    expected = _legacy_quantities(snapshot)
    columns = snapshot.synthesis_columns()
    reads = []
    mapped = columns.dataset._mapped

    def counted_map(name):
        reads.append(name)
        return mapped(name)

    def forbidden_loader(*args, **kwargs):
        pytest.fail("Original synthesis must not construct a DataLoader")

    monkeypatch.setattr(columns.dataset, "_mapped", counted_map)
    monkeypatch.setattr(torch.utils.data, "DataLoader", forbidden_loader)
    try:
        for step, batch_size, fields in [(1, 4, tuple(columns.dataset.paths)),
                                         (2, 3, ("tau",)),
                                         (1, 7, ("temperature", "pressure", "tau"))]:
            selected = columns.strided(step)
            flattened = {name: np.empty((selected.shape[0] * selected.shape[1], selected.shape[2]),
                                        dtype=np.float32) for name in fields}
            indices = []
            for ix, iy, batch in selected.batches(batch_size, fields):
                assert len(ix) <= batch_size
                assert set(columns._owner.fields) == set(columns.dataset.paths)
                flat_index = ix * selected.shape[1] + iy
                indices.extend(flat_index.tolist())
                for name in fields:
                    flattened[name][flat_index] = batch[name][:, 0, :]
                assert batch["ds_m"] == expected["ds_m"]
            assert indices == list(range(len(indices)))
            for name in fields:
                np.testing.assert_array_equal(flattened[name].reshape(selected.shape),
                                              expected[name][::step, ::step, :])
            assert sorted(reads) == sorted(columns.dataset.paths)
            assert all(not array.flags.writeable for array in columns._owner.fields.values())
        assert not snapshot._arrays
    finally:
        columns.close()
    assert not columns._owner.fields
    assert not columns.dataset._mappings
    assert original == {path.name: path.read_bytes() for path in source.iterdir()}


@pytest.mark.parametrize("instrument", ["hmi", "hinode", "aia"])
@pytest.mark.parametrize("stride,binning", [(None, 2), (2, None)])
def test_dataset_synthesis_matches_eager_profiles(tmp_path, monkeypatch, instrument, stride, binning):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 4, 15), 2, physical=True)
    original_files = {path.name: path.read_bytes() for path in source.iterdir()}
    snapshot = MURaMSnapshot(str(source), 10)
    expected = _legacy_quantities(snapshot)
    eager = SimpleNamespace(time_s=snapshot.time_s, iteration=10,
                            component_order=snapshot.component_order,
                            synthesis_ready_quantities=lambda: expected)
    kwargs = dict(stride=stride, binning=binning, batch_size=3, device="cpu", progress=False)
    if instrument == "aia":
        synthesize = synthesize_aia
    else:
        kwargs["keep_raw_stokes"] = True
        synthesize = synthesize_hmi
        if instrument == "hinode":
            synthesize = synthesize_stokes
            kwargs.update(
                instrument={"type": "hinode_sp", "spectral_psf": {
                    "type": "gaussian", "fwhm_angstrom": .025, "oversample": 2, "truncate_sigma": 4.,
                }},
                observed_wavelength_angstrom=np.linspace(6301., 6302.9, 21),
                line_ids=["FeI_6301.5008", "FeI_6302.4932"],
            )
    baseline = synthesize(eager, **kwargs)

    def no_eager_load(*args, **kwargs):
        pytest.fail("Native synthesis must use the shared volume cache, not a duplicate eager reader cache")

    monkeypatch.setattr(MURaMSnapshot, "_load", no_eager_load)
    from prom3theus.muram.columns import NativeColumns
    original_batches = NativeColumns.batches
    reads = []

    def counted_batches(self, batch_size, fields=None):
        for ix, iy, columns in original_batches(self, batch_size, fields):
            assert "tau" in columns  # Shared diagnostics use native tau for every instrument.
            reads.extend(zip(ix.tolist(), iy.tolist()))
            yield ix, iy, columns

    monkeypatch.setattr(NativeColumns, "batches", counted_batches)
    try:
        loaded = synthesize(snapshot, **kwargs)
    finally:
        snapshot.synthesis_columns().close()
    selected_shape = snapshot.synthesis_columns().strided(stride or 1).shape
    assert reads == list(itertools.product(range(selected_shape[0]), range(selected_shape[1])))
    if instrument == "aia":
        torch.testing.assert_close(loaded.intensity, baseline.intensity, rtol=2e-6, atol=1e-8)
    else:
        torch.testing.assert_close(loaded.stokes, baseline.stokes, rtol=2e-6, atol=1e-8)
        torch.testing.assert_close(loaded.raw_stokes, baseline.raw_stokes, rtol=2e-6, atol=1e-8)
        torch.testing.assert_close(loaded.geometry.coordinates, baseline.geometry.coordinates)
        assert loaded.native_depth_samples == baseline.native_depth_samples == 15
        assert loaded.native_vertical_spacing_m == baseline.native_vertical_spacing_m
    assert not snapshot._arrays
    assert original_files == {path.name: path.read_bytes() for path in source.iterdir()}


def test_failed_synthesis_releases_cached_cubes(tmp_path):
    source = tmp_path / "source"
    _write_snapshot(source, (4, 4, 15), 2, physical=True)
    path = source / "eosT.000010"
    temperature = np.fromfile(path, dtype=np.float32).reshape((4, 4, 15), order="F")
    temperature[0, 0, 5] = np.nan
    temperature.ravel(order="F").tofile(path)
    snapshot = MURaMSnapshot(str(source), 10)
    columns = snapshot.synthesis_columns()
    try:
        with pytest.raises(ValueError, match="finite"):
            synthesize_hmi(snapshot, batch_size=2, progress=False, device="cpu")
        assert columns._owner.fields
    finally:
        snapshot.close()
    assert not columns._owner.fields
    assert not columns.dataset._mappings


def test_hmi_and_aia_share_one_column_pass(monkeypatch):
    from tests.muram.test_sdo_fits import snapshot
    from prom3theus.muram.columns import ArrayColumns, snapshot_columns, strided_columns
    from prom3theus.muram.synthesize import synthesize_sdo

    snap = snapshot((4, 4, 30))
    options = dict(stride=2, binning=2, batch_size=3, progress=False, device="cpu")
    expected_hmi = synthesize_hmi(snap, **options)
    expected_aia = synthesize_aia(snap, **options)
    source = snapshot_columns(snap)
    batches = ArrayColumns.batches
    reads = []

    def once(self, batch_size, fields=None):
        assert not reads, "Synthesis must not start a second source pass"
        for ix, iy, columns in batches(self, batch_size, fields):
            reads.extend(zip(ix.tolist(), iy.tolist()))
            yield ix, iy, columns

    monkeypatch.setattr(ArrayColumns, "batches", once)
    actual_hmi, actual_aia = synthesize_sdo(snap, **options)
    assert reads == [(0, 0), (0, 1), (1, 0), (1, 1)]
    torch.testing.assert_close(actual_hmi.stokes, expected_hmi.stokes)
    torch.testing.assert_close(actual_aia.intensity, expected_aia.intensity)
    assert actual_aia.dx_m == expected_aia.dx_m
    assert actual_aia.dy_m == expected_aia.dy_m


@pytest.mark.parametrize("increasing", [False, True])
def test_stokes_crops_lazy_views_before_copying_and_repairs_same_diagnostics(tmp_path, monkeypatch, increasing):
    source = tmp_path / "source"
    _write_snapshot(source, (5, 6, 9), vertical=2, increasing=increasing, physical=True)
    header = source / "Header.000010"
    values = np.loadtxt(header)
    values[3:6] = [1e8, 1e8, 5e7]  # 4 megameters high; 7 centers lie at/below 3.
    np.savetxt(header, values)
    snapshot = MURaMSnapshot(str(source), 10)
    expected = _legacy_quantities(snapshot)
    lower = {name: value[..., :7] if isinstance(value, np.ndarray) else value for name, value in expected.items()}
    eager = SimpleNamespace(time_s=snapshot.time_s, synthesis_ready_quantities=lambda: lower)
    baseline = synthesize_hmi(eager, stride=2, progress=False, device="cpu")
    copied = []
    array = np.array

    def selected_array(value, *args, **kwargs):
        if isinstance(value, np.memmap) and value.ndim == 3:
            copied.append(value.shape)
            assert value.shape == (3, 3, 7), "Slice horizontal stride and height before allocating RAM"
        return array(value, *args, **kwargs)

    monkeypatch.setattr(np, "array", selected_array)
    result = synthesize_hmi(snapshot, stride=2, batch_size=3, progress=False, device="cpu")
    assert copied == [(3, 3, 7)] * 9
    assert not snapshot.synthesis_columns()._owner.fields  # No uncropped eager cache.
    assert result.native_depth_samples == 9
    assert result.stokes_depth_samples == 7
    assert result.native_reference["column_selection"]["height_range_m"] == [0., 3e6]
    torch.testing.assert_close(result.stokes, baseline.stokes)
    repaired = collect_diagnostic_maps(result.native_reference, batch_size=2, progress=False)
    assert copied == [(3, 3, 7)] * 18
    assert repaired.metadata["sampling"]["sampled_height_range_m"] == [0., 3e6]
    np.testing.assert_array_equal(repaired.status, result.diagnostics.status)
    for name in repaired.maps:
        np.testing.assert_array_equal(repaired.maps[name], result.diagnostics.maps[name])
    snapshot.close()
