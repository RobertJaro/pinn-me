import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from prom3theus.artifacts.model import (
    ARTIFACT_SCHEMA_VERSION,
    OBSERVATION_DIRECTORY_NAME,
    ArtifactManifest,
    UnsupportedArtifactVersion,
    load_manifest,
    load_state_dict,
    save_artifact,
)
from prom3theus.core import sha256_file
from prom3theus.observations import (
    OBSERVATION_STORE_FORMAT,
    OBSERVATION_STORE_VERSION,
    ObservationRaster,
    ObservationStore,
)


_SIGNATURE = "a" * 64
_SPEC = {"observation_id": "test", "observation_type": "hinode_sp"}


def _stokes_basis():
    return torch.tensor(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))).reshape(
        1, 1, 3, 3
    )


def _manifest():
    return ArtifactManifest.create(
        package_version="test",
        resolved_config={
            "schema_version": 2,
            "solver": {"kind": "lte"},
            "resources": {"bundle": "packaged"},
        },
        resources={"manifest_sha256": "abc"},
        observation={
            "store": {
                "path": OBSERVATION_DIRECTORY_NAME,
                "format": OBSERVATION_STORE_FORMAT,
                "version": OBSERVATION_STORE_VERSION,
                "source_signature": _SIGNATURE,
            },
            "spec": _SPEC,
        },
        model={"hidden_features": 8},
    )


def _observation_store(tmp_path):
    raster = ObservationRaster(
        stokes=torch.zeros(1, 1, 4, 2),
        wavelength_angstrom=torch.tensor([6301.0, 6302.0]),
        coordinates=torch.zeros(1, 1, 3),
        ray_direction=torch.tensor([[[-1.0, 0.0, 0.0]]]),
        surface_position_m=torch.tensor([[[6.96e8, 0.0, 0.0]]]),
        stokes_basis=_stokes_basis(),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        metadata={},
    )
    return ObservationStore.save(
        tmp_path / "prepared.observation",
        raster,
        source_signature=_SIGNATURE,
        metadata={"observation": _SPEC, "validation_raster_index": 0},
    )


def test_artifact_round_trip_is_manifest_plus_tensor_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    artifact = tmp_path / "artifact"
    source_store = _observation_store(tmp_path)
    save_artifact(
        artifact,
        model=model,
        manifest=_manifest(),
        observation_store=source_store,
    )

    assert sorted(path.name for path in artifact.iterdir()) == [
        "manifest.json",
        OBSERVATION_DIRECTORY_NAME,
        "weights.pt",
    ]
    stored = ObservationStore.load(artifact / OBSERVATION_DIRECTORY_NAME)
    assert stored.spatial_shape == (1, 1)
    manifest = load_manifest(artifact)
    assert manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert manifest.weights_sha256 == sha256_file(artifact / "weights.pt")
    state = load_state_dict(artifact, manifest)
    assert set(state) == set(model.state_dict())
    assert all(
        torch.equal(state[name], value) for name, value in model.state_dict().items()
    )


def test_artifact_schema_version_must_match_current_contract(tmp_path):
    payload = _manifest().to_dict()
    payload["schema_version"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedArtifactVersion, match="expected exactly"):
        load_manifest(tmp_path)


def test_unknown_artifact_fields_are_rejected(tmp_path):
    payload = _manifest().to_dict()
    payload["unexpected_field"] = "not-supported"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown artifact manifest fields"):
        load_manifest(tmp_path)


def test_missing_or_noncanonical_weights_digest_is_rejected(tmp_path):
    payload = _manifest().to_dict()
    payload.pop("weights_sha256")
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_manifest(tmp_path)

    payload["weights_sha256"] = "A" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_manifest(tmp_path)


def test_weights_tampering_is_rejected_before_tensor_load(tmp_path):
    model = torch.nn.Linear(3, 2)
    artifact = tmp_path / "artifact"
    save_artifact(
        artifact,
        model=model,
        manifest=_manifest(),
        observation_store=_observation_store(tmp_path),
    )
    manifest = load_manifest(artifact)
    weights_path = artifact / "weights.pt"
    payload = bytearray(weights_path.read_bytes())
    payload[-1] ^= 1
    weights_path.write_bytes(payload)

    with pytest.raises(RuntimeError, match="integrity verification"):
        load_state_dict(artifact, manifest)


def test_nonfinite_model_weights_are_not_persisted(tmp_path):
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")

    with pytest.raises(ValueError, match="contains non-finite"):
        save_artifact(
            tmp_path / "artifact",
            model=model,
            manifest=_manifest(),
            observation_store=_observation_store(tmp_path),
        )


def test_weights_symlink_must_not_escape_artifact_directory(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    external = tmp_path / "external.pt"
    torch.save(torch.nn.Linear(1, 1).state_dict(), external)
    (artifact / "weights.pt").symlink_to(external)
    manifest = replace(_manifest(), weights_sha256=sha256_file(external))

    with pytest.raises(ValueError, match="stay inside"):
        load_state_dict(artifact, manifest)


def test_boolean_artifact_schema_version_is_rejected(tmp_path):
    payload = _manifest().to_dict()
    payload["schema_version"] = True
    payload["weights_sha256"] = "a" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedArtifactVersion, match="expected exactly"):
        load_manifest(tmp_path)


def test_artifacts_package_import_does_not_load_export_or_training():
    project_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    code = """
import sys
import prom3theus.artifacts

forbidden = {
    'prom3theus.artifacts.export',
    'prom3theus.artifacts.validation',
    'prom3theus.training.lightning',
    'lightning',
    'pytorch_lightning',
}
loaded = sorted(forbidden.intersection(sys.modules))
if loaded:
    raise SystemExit(f'unexpected eager imports: {loaded}')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        check=True,
    )
