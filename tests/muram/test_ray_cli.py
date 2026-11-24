"""Ray CLI publishes sequential snapshots, resumes safely, and preserves sources."""

import hashlib
import json
import weakref

import numpy as np
import pytest

from prom3theus.muram import synthesize_rays as command
from prom3theus.muram.ray_volume import EagerVolume
from tests.muram.test_column_loading import _write_snapshot


def source_snapshots(path):
    _write_snapshot(path, (2, 2, 5), vertical=2, increasing=True, physical=True)
    for original in list(path.iterdir()):
        (path / original.name.replace("000010", "000020")).write_bytes(original.read_bytes())


def arguments(source, output, workers=0):
    return ["--source-path", str(source), "--out", str(output), "--iterations", "10", "20",
            "--bottom-height-megameters", "-0.02", "--depth-samples", "8", "--batch-size", "3",
            "--volume-height-megameters", "0.02",
            "--device", "cpu", "--workers", str(workers), "--no-progress"]


def file_hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("workers", [0, 2])
def test_ray_cli_sequential_resume_repair_and_source_integrity(tmp_path, monkeypatch, workers):
    source, output = tmp_path / "source", tmp_path / "output"
    source_snapshots(source)
    original_hashes = file_hashes(source)
    calls, previous = [], []
    original = EagerVolume.from_snapshot

    def load(cls, snapshot, **kwargs):
        assert all(item() is None for item in previous), "Previous snapshot cube must be released"
        volume = original(snapshot, **kwargs)
        calls.append(snapshot.iteration)
        previous[:] = [weakref.ref(volume)]
        return volume

    monkeypatch.setattr(EagerVolume, "from_snapshot", classmethod(load))
    argv = arguments(source, output, workers)
    command.main(argv)
    assert calls == [10, 20]
    assert all(item() is None for item in previous)
    for iteration in (10, 20):
        arrays, metadata = command.load_ray_product(output / f"muram_{iteration:06d}")
        assert metadata["native_reference"]["iteration"] == iteration
        assert metadata["native_reference"]["source_path"] == str(source.resolve())
        assert metadata["depth_samples"] == 8
        assert metadata["volume"]["volume_height_megameters"] == 0.02
        assert metadata["volume"]["selected_shape"] == [2, 2, 3]
        assert metadata["volume"]["loaded_height_range_m"] == [0., 2e4]
        assert arrays["stokes"].shape[:3] == (2, 2, 4)
        assert arrays["instrument_stokes"].shape == (2, 2, 4, 6)
        assert arrays["valid_mask"].all()
        assert np.isfinite(arrays["stokes"]).all()
        assert not arrays["stokes"].flags.writeable
    del arrays
    before = file_hashes(output / "muram_000010")
    command.main(argv)
    assert calls == [10, 20], "Resume must not materialize source fields"
    assert file_hashes(output / "muram_000010") == before
    # Missing product arrays regenerate only the affected snapshot.
    (output / "muram_000020" / "instrument_stokes.npy").unlink()
    command.main(argv)
    assert calls == [10, 20, 20]
    assert file_hashes(output / "muram_000010") == before
    assert json.loads((output / "ray_synthesis_status.json").read_text())["snapshots"] == {"10": "complete", "20": "complete"}
    assert file_hashes(source) == original_hashes
    changed = argv.copy()
    changed[changed.index("--depth-samples") + 1] = "16"
    with pytest.raises(ValueError, match="settings changed"):
        command.main(changed)
    assert calls == [10, 20, 20]
    changed = argv.copy()
    changed[changed.index("--volume-height-megameters") + 1] = "0.03"
    with pytest.raises(ValueError, match="settings changed"):
        command.main(changed)
    # Same-sized native edits cannot be silently combined with old spectra.
    native = source / "eosT.000010"
    value = bytearray(native.read_bytes())
    value[0] ^= 1
    native.write_bytes(value)
    with pytest.raises(ValueError, match="Native source changed"):
        command.main(argv)
    assert calls == [10, 20, 20]


def test_ray_cli_preflight_does_not_destroy_output(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source_snapshots(source)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve this output")
    argv = arguments(source, output)
    argv[argv.index("20")] = "30"  # The second requested snapshot is missing.
    with pytest.raises(FileNotFoundError):
        command.main([*argv, "--overwrite"])
    assert sentinel.read_text() == "preserve this output"


def test_ray_overwrite_rejects_truncated_field_before_deleting_output(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source_snapshots(source)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve this output")
    temperature = source / "eosT.000010"
    temperature.write_bytes(temperature.read_bytes()[:-4])
    with pytest.raises(ValueError, match="invalid file size"):
        command.main([*arguments(source, output), "--overwrite"])
    assert sentinel.read_text() == "preserve this output"


def test_loader_rejects_empty_array_manifest(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"schema": command.SCHEMA, "arrays": {}}))
    with pytest.raises(ValueError, match="missing required arrays"):
        command.load_ray_product(tmp_path)
