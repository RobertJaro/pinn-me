import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import prom3theus.artifacts.export as exporter
import prom3theus.artifacts.evaluation as artifact_evaluation
import prom3theus.artifacts.stokes as stokes_evaluation
import prom3theus.artifacts.validation as artifact_validation
from prom3theus.artifacts import ArtifactExportError as PackageArtifactExportError
from prom3theus.artifacts.model import (
    OBSERVATION_DIRECTORY_NAME,
    ArtifactManifest,
    save_artifact,
)
from prom3theus.core import sha256_file
from prom3theus.observations import (
    CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    OBSERVATION_STORE_FORMAT,
    OBSERVATION_STORE_VERSION,
    ObservationRaster,
    ObservationSpec,
    ObservationStore,
    observation_store_signature,
)


def _raster(*, auxiliary=None, mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY):
    metadata = {
        "ray_geometry": {"scene_basis_rows": np.eye(3).tolist()},
        "stokes_order": ["I", "Q", "U", "V"],
        "normalization": {
            "indices": [0, 1],
            "radiometric_calibration": {
                "atlas_disk_center_continuum_radiance_w_m3_sr": 1.0
            },
        },
    }
    if mode == CARRINGTON_REGISTERED_RELATIVE_VELOCITY:
        metadata["observer_velocity_correction"] = {
            "removed_solar_los_velocity_m_per_s": [10.0, 20.0]
        }
    return ObservationRaster(
        stokes=torch.zeros(1, 2, 4, 2),
        wavelength_angstrom=torch.tensor([6301.0, 6302.0]),
        coordinates=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        ray_direction=torch.tensor([[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
        surface_position_m=torch.tensor([[[6.96e8, 0.0, 0.0], [6.96e8, 0.0, 0.0]]]),
        stokes_basis=torch.tensor(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
        .expand(1, 2, 3, 3)
        .clone(),
        valid_mask=torch.tensor([[True, True]]),
        metadata=metadata,
        auxiliary={} if auxiliary is None else auxiliary,
    )


def _spec():
    return ObservationSpec(
        observation_id="hinode-test",
        observation_type="hinode_sp",
        instrument_type="hinode_sp",
        wavelength_angstrom=torch.tensor([6301.0, 6302.0]),
        continuum_indices=(0, 1),
        radiance_scale_w_m3_sr=1.0,
        velocity_synthesis_mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
        required_line_ids=("Fe_I_6301",),
        line_support_angstrom=(6300.0, 6303.0),
    )


def _model_config():
    return {
        "log_tau500": [-5.0, 1.0],
        "wavelength_angstrom": [6301.0, 6302.0],
        "atmosphere_config": {},
        "synthesizer_config": {"line_ids": ["Fe_I_6301"]},
        "instrument_config": {"type": "hinode_sp"},
        "normalization_config": {},
        "stokes_loss_config": {},
        "weight_config": {},
        "wavelength_weights": None,
        "wavelength_exclude_windows_angstrom": [],
        "continuum_indices": [0, 1],
        "atlas_continuum_radiance_w_m3_sr": 1.0,
        "depth_sampling_config": {},
        "physics_config": {},
        "learning_rate": 1.0e-5,
        "run_metadata": {},
        "observation_id": "hinode-test",
        "velocity_synthesis_mode": CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
        "instrument_radial_velocity_correction_m_per_s": 0.0,
        "vector_regularization_config": None,
    }


class _FakeAtmosphere(torch.nn.Module):
    def __init__(self, log_tau500):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("log_tau500", torch.as_tensor(log_tau500))


class _FakeLTE(torch.nn.Module):
    last_kwargs = None
    last_instance = None

    def __init__(self, **kwargs):
        super().__init__()
        type(self).last_kwargs = kwargs
        type(self).last_instance = self
        self.atmosphere_model = _FakeAtmosphere(kwargs["log_tau500"])
        self.synthesizer = SimpleNamespace(continuum_opacity=object())
        self.velocity_synthesis_mode = kwargs["velocity_synthesis_mode"]
        self.instrument_radial_velocity_correction_m_per_s = torch.tensor(0.0)
        self.loaded_strict = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_strict = strict
        return super().load_state_dict(state_dict, strict=strict)


def _write_artifact(tmp_path, *, observation=None, signature=None):
    artifact = tmp_path / "artifact"
    resources = {
        "instrument": "Hinode/SOT-SP",
        "resource_sha256": {"lines.json": "abc"},
    }
    observation_config = {"type": "hinode_sp", "directory": "/source"}
    resolved_config = {
        "schema_version": 1,
        "observation": observation_config,
        "resources": {"bundle": "packaged"},
    }
    signature = signature or observation_store_signature(
        observation_config,
        resources,
        source_files_sha256="a" * 64,
    )
    spec = _spec()
    canonical_observation = {
        "store": {
            "path": OBSERVATION_DIRECTORY_NAME,
            "format": OBSERVATION_STORE_FORMAT,
            "version": OBSERVATION_STORE_VERSION,
            "source_signature": signature,
        },
        "spec": spec.metadata(),
    }
    requested_observation = observation or canonical_observation
    model_config = _model_config()
    manifest = ArtifactManifest.create(
        package_version="test",
        resolved_config=resolved_config,
        resources=resources,
        observation=canonical_observation,
        model=model_config,
    )
    model = _FakeLTE(**model_config)
    prepared_store = ObservationStore.save(
        tmp_path / "prepared.observation",
        _raster(),
        source_signature=signature,
        metadata={
            "adapter": "hinode_sp",
            "observation": spec.metadata(),
            "validation_raster_index": 0,
        },
    )
    save_artifact(
        artifact,
        model=model,
        manifest=manifest,
        observation_store=prepared_store,
    )
    if requested_observation != canonical_observation:
        payload = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        payload["observation"] = requested_observation
        (artifact / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    config = SimpleNamespace(
        resources=SimpleNamespace(bundle="packaged"),
        observation=SimpleNamespace(type="hinode_sp"),
        instrument=SimpleNamespace(type="hinode_sp"),
        synthesis=SimpleNamespace(line_ids=("Fe_I_6301",)),
    )
    return artifact, resources, config, model_config


def _patch_external_contracts(monkeypatch, resources, config):
    monkeypatch.setattr(artifact_validation, "LTEInversionModule", _FakeLTE)
    monkeypatch.setattr(
        artifact_validation, "parse_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        artifact_validation, "validate_resource_bundle", lambda: resources
    )
    monkeypatch.setattr(
        artifact_validation,
        "get_observation_adapter",
        lambda name: SimpleNamespace(name=name),
    )


def test_export_reconstructs_exact_manifest_model_and_stored_observation(
    tmp_path, monkeypatch
):
    artifact, resources, config, model_config = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    monkeypatch.setattr(
        exporter,
        "evaluate_atmosphere",
        lambda module, raster, **kwargs: {
            "temperature_k": np.zeros((1, 2, 2), dtype=np.float32),
            "valid_mask": raster.valid_mask.numpy(),
        },
    )

    output = exporter.export_artifact(
        artifact, tmp_path / "result.npz", depth_samples=2, device="cpu"
    )

    assert output == (tmp_path / "result.npz").resolve()
    assert _FakeLTE.last_kwargs == model_config
    assert _FakeLTE.last_instance.loaded_strict is True
    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["schema_version"] == 1
        assert metadata["artifact"]["weights_sha256"] == sha256_file(
            artifact / "weights.pt"
        )
        assert metadata["observation"]["store"]["path"] == "observations"
        assert metadata["evaluation"]["depth_samples"] == 2
        assert metadata["evaluation"]["raster_role"] == "validation"
        assert metadata["evaluation"]["raster_name"] == "raster_0000"


def test_export_rejects_nonfinite_model_values_at_valid_pixels(tmp_path, monkeypatch):
    artifact, resources, config, _ = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    monkeypatch.setattr(
        exporter,
        "evaluate_atmosphere",
        lambda module, raster, **kwargs: {
            "temperature_k": np.full((1, 2, 2), np.nan, dtype=np.float32),
            "valid_mask": raster.valid_mask.numpy(),
        },
    )

    with pytest.raises(exporter.ArtifactExportError, match="non-finite"):
        exporter.export_artifact(
            artifact, tmp_path / "result.npz", depth_samples=2, device="cpu"
        )


def test_export_rejects_object_arrays_that_require_pickle(tmp_path, monkeypatch):
    artifact, resources, config, _ = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    monkeypatch.setattr(
        exporter,
        "evaluate_atmosphere",
        lambda module, raster, **kwargs: {
            "temperature_k": np.asarray([[["not-numeric"]]], dtype=object),
            "valid_mask": raster.valid_mask.numpy(),
        },
    )

    with pytest.raises(exporter.ArtifactExportError, match="numeric or boolean"):
        exporter.export_artifact(
            artifact, tmp_path / "result.npz", depth_samples=2, device="cpu"
        )


def test_observation_manifest_requires_store_contract(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = ArtifactManifest.create(
        package_version="test",
        resolved_config={},
        resources={},
        observation=_spec().metadata(),
        model={},
    )
    payload = manifest.to_dict()
    payload["weights_sha256"] = "a" * 64
    (artifact / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(exporter.ArtifactExportError, match="store is required"):
        exporter.export_artifact(artifact, tmp_path / "result.npz")


def test_observation_store_path_must_stay_inside_artifact(tmp_path):
    observation = {
        "store": {
            "path": "../observation",
            "format": OBSERVATION_STORE_FORMAT,
            "version": OBSERVATION_STORE_VERSION,
            "source_signature": "a" * 64,
        },
        "spec": _spec().metadata(),
    }
    artifact, _, _, _ = _write_artifact(tmp_path, observation=observation)

    with pytest.raises(exporter.ArtifactExportError, match="must be exactly"):
        exporter.export_artifact(artifact, tmp_path / "result.npz")


def test_manifest_signature_must_match_embedded_store(tmp_path, monkeypatch):
    artifact, resources, config, _ = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    manifest_path = artifact / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["observation"]["store"]["source_signature"] = "a" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(exporter.ArtifactExportError, match="source_signature"):
        exporter.export_artifact(artifact, tmp_path / "result.npz")


def test_unknown_model_contract_field_is_rejected(tmp_path, monkeypatch):
    artifact, resources, config, _ = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    manifest_path = artifact / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model"]["unexpected_field"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(exporter.ArtifactExportError, match="unknown"):
        exporter.export_artifact(artifact, tmp_path / "result.npz")


def test_model_continuum_indices_must_be_actual_integers(tmp_path, monkeypatch):
    artifact, resources, config, _ = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    manifest_path = artifact / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model"]["continuum_indices"] = [False, True]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(exporter.ArtifactExportError, match="sequence of integers"):
        exporter.export_artifact(artifact, tmp_path / "result.npz")


def test_raster_normalization_indices_must_be_actual_integers():
    raster = _raster()
    raster.metadata["normalization"]["indices"] = [False, True]

    with pytest.raises(exporter.ArtifactExportError, match="indices must be integers"):
        artifact_validation.validate_raster_scientific_contract(raster, _spec())


def test_hmi_stokes_requires_stored_per_pixel_response():
    raster = _raster(
        auxiliary={"observer_los_velocity_m_per_s": torch.zeros(1, 2)},
        mode=CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
    )
    flat_indices = torch.tensor([0, 1])

    with pytest.raises(exporter.ArtifactExportError, match="HMI Stokes export"):
        next(
            stokes_evaluation.stokes_batches(
                raster,
                flat_indices,
                2,
                CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
                "hmi_filter_profiles",
            )
        )


def test_registered_los_export_uses_the_forward_solver_sign():
    base_los = np.asarray([[[-100.0, -101.0]]])
    removed_from_toward = np.asarray([[30.0]])

    restored = artifact_evaluation._apply_positive_redshift_los_offset(
        base_los, removed_from_toward
    )

    np.testing.assert_array_equal(restored, [[[-70.0, -71.0]]])


def test_export_facade_exposes_the_current_api():
    assert exporter.__all__ == ["ArtifactExportError", "export_artifact"]


def test_package_and_export_module_share_one_export_error_type():
    assert PackageArtifactExportError is exporter.ArtifactExportError
