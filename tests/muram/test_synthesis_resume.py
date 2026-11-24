"""Resume and replacement must preserve completed products and native inputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prom3theus.muram.checkpoint import prepare_synthesis_output, save_completed_raster


def test_source_identity_reads_metadata_only_and_tracks_changes(tmp_path, monkeypatch):
    import builtins
    import os
    from prom3theus.muram.cli import _source_file_metadata

    path = tmp_path / "result_prim_1.1000000"
    path.write_bytes(b"native data")
    stat = path.stat()

    def unexpected_read(*args, **kwargs):
        pytest.fail("Source identity must not read cube contents")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", unexpected_read)
        patch.setattr(Path, "open", unexpected_read)
        original = _source_file_metadata([path])
    assert original == [{
        "path": str(path.resolve()), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
    }]
    assert _source_file_metadata([path]) == original
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert _source_file_metadata([path]) != original
    with pytest.raises(FileNotFoundError):
        _source_file_metadata([tmp_path / "missing"])


def _args(source, output, **overrides):
    return SimpleNamespace(
        **(dict(source_path=source, out=output, overwrite=False,
                output_format="store", stride=None, binning=None, dtype="float32",
                vertical_axis=None, component_order=[0, 1, 2]) | overrides)
    )


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("ancestor", [False, True])
def test_output_cannot_replace_native_source(tmp_path, overwrite, ancestor):
    source = tmp_path / "source"
    source.mkdir()
    native = source / "Header.000010"
    native.write_bytes(b"native snapshot")
    with pytest.raises(ValueError, match="source directory"):
        prepare_synthesis_output(_args(source, tmp_path if ancestor else source,
                                      overwrite=overwrite), {})
    assert native.read_bytes() == b"native snapshot"


def test_empty_output_can_resume_and_changed_contract_cannot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    args = _args(source, output)
    prepare_synthesis_output(args, {"line_ids": ["first"]})
    recorded = (output / "synthesis_run.json").read_bytes()
    with pytest.raises(ValueError, match="settings changed"):
        prepare_synthesis_output(args, {"line_ids": ["second"]})
    assert (output / "synthesis_run.json").read_bytes() == recorded


def test_resume_rejects_changed_stokes_height(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"
    prepare_synthesis_output(_args(source, output), {})
    with pytest.raises(ValueError, match="settings changed.*stokes_height_megameters"):
        prepare_synthesis_output(_args(source, output, stokes_height_megameters=4.), {})


def test_retry_replaces_only_unpublished_raster_arrays(tmp_path, monkeypatch):
    import prom3theus.muram.checkpoint as checkpoint
    from prom3theus.muram.synthesize import synthesize_hmi
    from prom3theus.observations.store import ObservationStore
    from tests.muram.test_muram_synthesis import snapshot

    raster = synthesize_hmi(snapshot(), progress=False).to_observation_raster()
    out = tmp_path / "out"
    out.mkdir()
    replace = checkpoint.os.replace
    count = 0

    def interrupted_replace(source, target):
        nonlocal count
        if Path(target).parent == out:
            count += 1
            if count == 2:
                raise OSError("interrupted array publication")
        return replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_completed_raster(out, raster, name="muram_000010",
                              source_signature="0" * 64, metadata={})
    assert list(out.glob("muram_000010*"))
    assert not (out / "manifest.json").exists()
    monkeypatch.setattr(checkpoint.os, "replace", replace)
    save_completed_raster(out, raster, name="muram_000010",
                          source_signature="0" * 64, metadata={})
    ObservationStore.load_sequence(out, verify=True)
    with pytest.raises(FileExistsError, match="already published"):
        save_completed_raster(out, raster, name="muram_000010",
                              source_signature="0" * 64, metadata={})


def test_sdo_resumes_partial_snapshot_and_preparation_then_overwrites(tmp_path, monkeypatch):
    import prom3theus.muram.sdo as sdo
    from prom3theus.muram.cli import main
    from tests.muram.test_sdo_fits import snapshot

    source = tmp_path / "source"
    source.mkdir()
    import numpy as np
    from dataclasses import replace
    from prom3theus.muram.reader import MURaMSnapshot, _FIELD_IDS, _UNIT_SCALE
    native = snapshot().synthesis_ready_quantities()
    for iteration in (10, 20):
        np.savetxt(source / f"Header.{iteration:06d}", [3, 4, 30, 1e7, 1e7, 1e6, iteration * 360.])
        for name, value in native.items():
            if name != "ds_m":
                (value / _UNIT_SCALE[name]).astype("float32").ravel(order="F").tofile(
                    source / f"{_FIELD_IDS[name]}.{iteration:06d}")
    marker = source / "native"
    marker.write_bytes(b"unchanged")
    root = tmp_path / "out"
    snap = snapshot()
    hmi, aia = sdo.synthesize_sdo(snap, progress=False)
    calls = []
    fail_aia = True

    def load_snapshot(path, iteration, **kwargs):
        return SimpleNamespace(iteration=iteration, time_s=iteration * 360.0,
                               synthesis_ready_quantities=snap.synthesis_ready_quantities)

    def render_sdo(proxy, **kwargs):
        calls.append(proxy.iteration)
        native_snapshot = MURaMSnapshot(str(source), proxy.iteration)
        reference = native_snapshot.native_reference()
        native_snapshot.close()
        diagnostics = replace(hmi.diagnostics, metadata={**hmi.diagnostics.metadata,
                              "native_reference": reference, "time_s": proxy.time_s})
        return (replace(hmi, native_reference=reference, diagnostics=diagnostics),
                replace(aia, native_reference=reference, diagnostics=diagnostics))

    original_write_aia = sdo.write_aia_fits

    def write_aia(*args, **kwargs):
        if fail_aia and kwargs["iteration"] == 20:
            raise RuntimeError("interrupted after HMI")
        return original_write_aia(*args, **kwargs)

    monkeypatch.setattr(sdo, "MURaMSnapshot", load_snapshot)
    monkeypatch.setattr(sdo, "synthesize_sdo", render_sdo)
    monkeypatch.setattr(sdo, "write_aia_fits", write_aia)
    args = ["synthesize", "--source-path", str(source), "--out", str(root),
            "--iterations", "10", "20", "--device", "cpu", "--no-progress"]
    with pytest.raises(RuntimeError, match="after HMI"):
        main(args)
    status = json.loads((root / "synthesis_status.json").read_text())
    assert status["completed_iterations"] == [10]
    legacy_status = {key: value for key, value in status.items() if key != "snapshot_products"}
    recovered = sdo._completed_products(root, legacy_status)
    assert {key: set(paths) for key, paths in recovered.items()} == {
        key: set(paths) for key, paths in status["snapshot_products"].items()
    }
    preserved = {path: (root / path).read_bytes()
                 for path in status["snapshot_products"]["10"]}
    fail_aia = False
    original_prepare = sdo.prepare_aia_fits

    def fail_preparation(paths, output, **kwargs):
        output.mkdir(parents=True)
        (output / "partial").write_text("interrupted")
        raise RuntimeError("interrupted preparation")

    monkeypatch.setattr(sdo, "prepare_aia_fits", fail_preparation)
    with pytest.raises(RuntimeError, match="preparation"):
        main([*args, "--batch-size", "32"])
    assert calls == [10, 20, 20]
    assert all((root / path).read_bytes() == content for path, content in preserved.items())
    monkeypatch.setattr(sdo, "prepare_aia_fits", original_prepare)
    main(args)
    assert calls == [10, 20, 20]
    maps = root / "diagnostics" / "muram_000010.npz"
    saved_maps = maps.read_bytes()
    overview = maps.with_suffix(".jpg")
    overview.unlink()
    main(args)
    assert calls == [10, 20, 20]
    assert overview.is_file()
    assert maps.read_bytes() == saved_maps
    assert (root / "aia" / "prepared" / "manifest.json").is_file()
    assert not (root / "aia" / "prepared" / "partial").exists()
    main(args)
    assert calls == [10, 20, 20]
    # Source descriptors also live in the FITS contract: losing the status copy
    # must not require resynthesis or discard evaluation provenance.
    status_path = root / "synthesis_status.json"
    status = json.loads(status_path.read_text())
    status.pop("native_references")
    status_path.write_text(json.dumps(status))
    main(args)
    assert calls == [10, 20, 20]
    main([*args, "--overwrite"])
    assert calls == [10, 20, 20, 10, 20]
    assert marker.read_bytes() == b"unchanged"

    # An external writer changing a native input during synthesis must prevent
    # publishing the newly rendered products with a stale source descriptor.
    import os

    def render_changed_source(proxy, **kwargs):
        result = render_sdo(proxy, **kwargs)
        path = source / f"eosT.{proxy.iteration:06d}"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(sdo, "synthesize_sdo", render_changed_source)
    with pytest.raises(ValueError, match="Native source identity changed"):
        main([*args, "--overwrite"])
    assert not list(root.rglob("*.fits"))


def test_native_column_run_rejects_legacy_depth_sampling(tmp_path):
    args = _args(tmp_path / "source", tmp_path / "out")
    prepare_synthesis_output(args, {})
    contract = args.out / "synthesis_run.json"
    old = json.loads(contract.read_text())
    old.pop("depth_sampling")
    contract.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="settings changed.*depth_sampling"):
        prepare_synthesis_output(args, {})
