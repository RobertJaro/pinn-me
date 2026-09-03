"""HMI response preparation uses an explicit phase-map identity only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prom3theus.core import sha256_file
from prom3theus.instruments.hmi import preparation


class _Row(dict):
    @property
    def index(self):
        return tuple(self)


class _Indexer:
    def __init__(self, frame, *, positional: bool):
        self.frame = frame
        self.positional = positional

    def __getitem__(self, value):
        if self.positional:
            return _Row(self.frame.rows[value])
        return _Frame(
            [
                row
                for row, selected in zip(self.frame.rows, value, strict=True)
                if selected
            ]
        )


class _Series(list):
    def astype(self, dtype):
        return _Series(dtype(value) for value in self)

    def __eq__(self, other):
        return [value == other for value in self]


class _Frame:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return _Series(row[key] for row in self.rows)

    @property
    def loc(self):
        return _Indexer(self, positional=False)

    @property
    def iloc(self):
        return _Indexer(self, positional=True)


class _Client:
    def __init__(self):
        self.queries = []

    def query(self, record, *, key, seg):
        self.queries.append((record, key, seg))
        keys = _Frame(
            [
                {
                    "HCAMID": 2,
                    "NX": 4096,
                    "T_START": "2023.10.10_19:25:31_TAI",
                    "T_REC": "2023.10.10_19:26:26_TAI",
                    "T_STOP": "2023.10.10_19:27:22_TAI",
                },
                {
                    "HCAMID": 3,
                    "NX": 4096,
                    "T_START": "2023.10.10_19:25:31_TAI",
                    "T_REC": "2023.10.10_19:26:26_TAI",
                    "T_STOP": "2023.10.10_19:27:22_TAI",
                },
            ]
        )
        segments = _Frame(
            [
                {"phases": "/phase-camera-2.fits"},
                {"phases": "/phase-camera-3.fits"},
            ]
        )
        return keys, segments


def test_transmission_metadata_serialization_is_exact_and_finite(tmp_path):
    output = tmp_path / "profile.npz"
    preparation.write_transmission_file(
        output,
        {"offsets": np.array([0.0], dtype=np.float32)},
        {"array": np.array([1, 2]), "path": tmp_path, "scalar": np.float32(1.5)},
    )
    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata == {
        "array": [1, 2],
        "path": str(tmp_path),
        "scalar": 1.5,
    }

    with pytest.raises(ValueError, match="must be finite"):
        preparation.write_transmission_file(
            tmp_path / "nonfinite.npz",
            {"offsets": np.array([0.0], dtype=np.float32)},
            {"value": float("nan")},
        )
    with pytest.raises(TypeError, match="require string keys"):
        preparation.write_transmission_file(
            tmp_path / "integer-key.npz",
            {"offsets": np.array([0.0], dtype=np.float32)},
            {1: "coerced"},
        )
    with pytest.raises(TypeError, match="Unsupported HMI metadata"):
        preparation.write_transmission_file(
            tmp_path / "unsupported.npz",
            {"offsets": np.array([0.0], dtype=np.float32)},
            {"value": object()},
        )


def test_preparation_uses_explicit_phase_map_for_each_acquisition_camera(
    tmp_path, monkeypatch
):
    acquisitions = [
        {
            "acquisition_key": "2024.03.24_00:00:00_TAI|HCAMID=2",
            "record_time": "2024.03.24_00:00:00_TAI",
            "observation_time": "2024.03.24_00:00:00_TAI",
            "hcamid": 2,
        },
        {
            "acquisition_key": "2024.03.24_00:12:00_TAI|HCAMID=3",
            "record_time": "2024.03.24_00:12:00_TAI",
            "observation_time": "2024.03.24_00:12:00_TAI",
            "hcamid": 3,
        },
    ]
    monkeypatch.setattr(
        preparation, "discover_hmi_acquisitions", lambda _: acquisitions
    )

    written = []

    def write_profile(**options):
        output = options["output"]
        metadata = options["archive_metadata"]
        offsets = np.zeros((6, 2), dtype=np.float32)
        weights = np.full((2, 2, 6, 2), 0.5, dtype=np.float32)
        continuum = np.zeros((2, 2, 6), dtype=np.float32)
        np.savez(
            output,
            offsets=offsets,
            weights=weights,
            continuum_weights=continuum,
            metadata=np.array(
                json.dumps(
                    {
                        **metadata,
                        "format": "prom3theus.hmi_response.v1",
                        "half_width": 0.65,
                        "wavelength_unit": "Angstrom",
                        "ccd_size": 4096,
                        "phase_map_shape": [2, 2],
                        "samples": 2,
                    }
                )
            ),
        )
        written.append(metadata)

    monkeypatch.setattr(preparation, "_write_hmi_transmission_profile", write_profile)
    client = _Client()
    manifest_path = preparation.prepare_hmi_response_directory(
        [tmp_path / "unused-input"],
        tmp_path / "responses",
        "scientist@example.test",
        phase_map_fsn=4242,
        client=client,
    )

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["profiles"]) == {
        "INVPHMAP=4242|HCAMID=2",
        "INVPHMAP=4242|HCAMID=3",
    }
    assert {entry["profile"] for entry in manifest["acquisitions"].values()} == set(
        manifest["profiles"]
    )
    assert {entry["phase_map_fsn"] for entry in written} == {4242}
    assert all(
        entry["sha256"] == sha256_file(manifest_path.parent / entry["file"])
        for entry in manifest["profiles"].values()
    )
    assert {entry["HCAMID"] for entry in written} == {2, 3}
    assert [record for record, _, _ in client.queries] == [
        "hmi.phasemaps_extended[4242]",
        "hmi.phasemaps_extended[4242]",
    ]


def test_detune_sequence_times_do_not_limit_phase_map_observation_time():
    metadata = preparation.resolve_hmi_phase_map(
        _Client(),
        observation_time="2024.03.24_00:59:58_TAI",
        hcamid=2,
        phase_map_fsn=4242,
    )

    assert metadata["T_START"] == "2023.10.10_19:25:31_TAI"
    assert metadata["T_REC"] == "2023.10.10_19:26:26_TAI"
    assert metadata["T_STOP"] == "2023.10.10_19:27:22_TAI"
    assert metadata["observation_time"] == "2024.03.24_00:59:58_TAI"


def test_response_directory_requires_explicit_atomic_replacement(tmp_path, monkeypatch):
    output = tmp_path / "responses"
    output.mkdir()
    marker = output / "old"
    marker.write_text("preserve")
    monkeypatch.setattr(
        preparation,
        "discover_hmi_acquisitions",
        lambda _: pytest.fail("existing output must be rejected before input access"),
    )

    with pytest.raises(FileExistsError, match="complete prepared directory"):
        preparation.prepare_hmi_response_directory(
            [tmp_path / "unused-input"],
            output,
            "scientist@example.test",
            phase_map_fsn=4242,
            client=_Client(),
        )
    assert marker.read_text() == "preserve"


def test_response_output_must_not_contain_source_fits(tmp_path, monkeypatch):
    output = tmp_path / "stokes"
    output.mkdir()
    source = output / "hmi.S_720s.example.3.I0.fits"
    source.write_bytes(b"source")
    acquisition = {
        "acquisition_key": "2024.03.24_01:00:00_TAI|HCAMID=3",
        "record_time": "2024.03.24_01:00:00_TAI",
        "observation_time": "2024.03.24_00:59:58_TAI",
        "hcamid": 3,
        "path": str(source),
    }
    monkeypatch.setattr(
        preparation,
        "discover_hmi_acquisitions",
        lambda _: [acquisition],
    )

    with pytest.raises(ValueError, match="must not equal or contain source FITS"):
        preparation.prepare_hmi_response_directory(
            [source],
            output,
            "scientist@example.test",
            phase_map_fsn=4242,
            overwrite=True,
            client=_Client(),
        )

    assert source.read_bytes() == b"source"


def test_unrestorable_response_backup_is_retained(tmp_path, monkeypatch):
    output = tmp_path / "responses"
    output.mkdir()
    marker = output / "old"
    marker.write_text("preserve")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new").write_text("replacement")
    real_replace = preparation.os.replace

    def fail_publication_and_restoration(source, destination):
        source = Path(source)
        if source == output:
            return real_replace(source, destination)
        if source == staged or source.name.startswith(
            f".{output.name}.response-backup-"
        ):
            raise OSError("simulated rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        preparation.os,
        "replace",
        fail_publication_and_restoration,
    )

    with pytest.raises(RuntimeError, match="previous directory was retained"):
        preparation._publish_response_directory(
            staged,
            output,
            overwrite=True,
        )

    backups = list(tmp_path.glob(f".{output.name}.response-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / marker.name).read_text() == "preserve"
