from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import prom3theus.artifacts.export as exporter
import prom3theus.artifacts.evaluation as artifact_evaluation
import prom3theus.artifacts.checkpoint as checkpoint_artifact
import prom3theus.artifacts.loader as p3s_loader
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
    VelocitySynthesisMode,
    observation_store_signature,
)


def _raster(
    *, auxiliary=None, mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY, height=1
):
    metadata = {
        "ray_geometry": {
            "scene_basis_rows": np.eye(3).tolist(),
            "solar_radius_m": 6.96e8,
        },
        "times": ["2024-03-24T01:00:00.000"],
        "coordinates": {"time_scale": "TAI"},
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
        stokes=torch.zeros(height, 2, 4, 2),
        wavelength_angstrom=torch.tensor([6301.0, 6302.0]),
        coordinates=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        .expand(height, -1, -1)
        .clone(),
        ray_direction=torch.tensor([[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
        .expand(height, -1, -1)
        .clone(),
        surface_position_m=torch.tensor(
            [[[6.96e8, 0.0, 0.0], [6.96e8, 0.0, 0.0]]]
        )
        .expand(height, -1, -1)
        .clone(),
        stokes_basis=torch.tensor(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
        .expand(height, 2, 3, 3)
        .clone(),
        valid_mask=torch.ones(height, 2, dtype=torch.bool),
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
        "instrument_line_of_sight_velocity_correction_m_per_s": 0.0,
        "optimize_instrument_line_of_sight_velocity_correction": True,
        "vector_regularization_config": None,
    }


class _FakeAtmosphere(torch.nn.Module):
    def __init__(self, log_tau500):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("log_tau500", torch.as_tensor(log_tau500))

    def evaluate_at_height(self, coordinates, heights):
        scalar = self.scale * torch.ones_like(heights)
        vector = torch.zeros(
            coordinates.shape[0], 3, dtype=coordinates.dtype, device=coordinates.device
        )
        vector[:, 0] = self.scale
        return {
            "temperature": 5_000.0 * scalar,
            "microturbulence": 1_000.0 * scalar,
            "gas_pressure": 100.0 * scalar,
            "magnetic_field": vector,
            "velocity_field": 2.0 * vector,
        }

    def evaluate_chart_height_points(self, coordinates, heights):
        return self.evaluate_at_height(coordinates, heights)


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
        self.instrument_line_of_sight_velocity_correction_m_per_s = torch.tensor(0.0)
        self.loaded_strict = None
        self.save_state_context = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_strict = strict
        return super().load_state_dict(state_dict, strict=strict)

    def set_save_state_context(self, context):
        self.save_state_context = dict(context)

    def synthesize(self, coordinates, **options):
        assert options["ray_direction"].shape == coordinates.shape
        assert options["stokes_basis"].shape == (*coordinates.shape[:-1], 3, 3)
        return {
            "stokes": self.atmosphere_model.scale
            * torch.ones(
                coordinates.shape[0],
                4,
                2,
                dtype=coordinates.dtype,
                device=coordinates.device,
            )
        }


class _EvaluationAtmosphere(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.thermodynamic_eos = SimpleNamespace(
            mass_density=lambda temperature, gas_pressure: torch.ones_like(temperature)
        )

    def trace_rays(self, coordinates, ray_direction, depth_grid):
        del ray_direction
        batch_size = coordinates.shape[0]
        depth = depth_grid.numel()
        shape = (batch_size, depth)
        scalar = torch.ones(shape, dtype=self.scale.dtype, device=self.scale.device)
        zeros = torch.zeros(
            (*shape, 3), dtype=self.scale.dtype, device=self.scale.device
        )
        velocity = zeros.clone()
        velocity[..., 0] = 100.0
        position = zeros.clone()
        position[..., 0] = 6.96e8
        atmosphere = SimpleNamespace(
            temperature=5_000.0 * scalar,
            microturbulence=1_000.0 * scalar,
            gas_pressure=1_000.0 * scalar,
            magnetic_field=zeros,
            velocity_field=velocity,
            geometric_height_m=torch.linspace(
                1_000.0,
                0.0,
                depth,
                dtype=self.scale.dtype,
                device=self.scale.device,
            ).expand(shape),
        )
        return atmosphere, SimpleNamespace(position_m=position)


class _EvaluationContinuumOpacity:
    @staticmethod
    def volume_extinction_at_5000(temperature, gas_pressure):
        del gas_pressure
        return torch.full_like(temperature, 1.0e-5)


def _evaluation_module(mode):
    return SimpleNamespace(
        atmosphere_model=_EvaluationAtmosphere(),
        synthesizer=SimpleNamespace(continuum_opacity=_EvaluationContinuumOpacity()),
        velocity_synthesis_mode=mode,
        instrument_line_of_sight_velocity_correction_m_per_s=torch.tensor(5.0),
    )


def _write_artifact(tmp_path, *, observation=None, signature=None):
    artifact = tmp_path / "artifact"
    resources = {
        "instrument": "Hinode/SOT-SP",
        "resource_sha256": {"lines.json": "abc"},
    }
    observation_config = {"type": "hinode_sp", "directory": "/source"}
    resolved_config = {
        "schema_version": 2,
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
        observation=SimpleNamespace(
            type="hinode_sp",
            to_dict=lambda: {
                **observation_config,
                "loader": {
                    "batch_size": 2,
                    "validation_batch_size": 2,
                    "validation_stride": 1,
                },
            },
        ),
        instrument=SimpleNamespace(type="hinode_sp"),
        synthesis=SimpleNamespace(line_ids=("Fe_I_6301",)),
        solver=SimpleNamespace(work_directory=tmp_path / "work"),
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


def test_p3s_round_trip_reconstructs_model_and_evaluation_raster(
    tmp_path, monkeypatch
):
    artifact, resources, config, model_config = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    monkeypatch.setattr(checkpoint_artifact, "LTEInversionModule", _FakeLTE)
    stored_rasters, stored_names, stored_metadata = ObservationStore.load_sequence(
        artifact / "observations"
    )
    bounds = {
        "surface_longitude_center_rad": 0.0,
        "surface_longitude_offset_rad": [0.0, 0.0],
        "surface_latitude_rad": [0.0, 0.0],
        "time_hours": [0.0, 0.0],
        "solar_radius_m": 6.96e8,
    }
    context = checkpoint_artifact.p3s_context(
        package_version="test",
        resolved_config={
            "schema_version": 2,
            "observation": {"type": "hinode_sp", "directory": "/source"},
            "resources": {"bundle": "packaged"},
        },
        resources=resources,
        observation_spec=_spec().metadata(),
        source_signature=ObservationStore.manifest(artifact / "observations")[
            "source_signature"
        ],
        raster_names=stored_names,
        validation_raster_index=stored_metadata["validation_raster_index"],
        times=[{"values": ["2024-03-24T01:00:00.000"], "scale": "tai"}],
        bounds=bounds,
        model=model_config,
    )
    original = _FakeLTE(**model_config)
    save_state_path = tmp_path / "state.p3s"
    torch.save(
        {
            "epoch": 7,
            "global_step": 12_345,
            "parameters": dict(original.named_parameters()),
            "prom3theus_save_state": context,
        },
        save_state_path,
    )

    loaded = checkpoint_artifact.load_validated_save_state(save_state_path)

    assert loaded.epoch == 7
    assert loaded.global_step == 12_345
    assert loaded.observation.raster_names == ("raster_0000",)
    assert loaded.observation.bounds == bounds
    assert loaded.module.save_state_context == context
    torch.testing.assert_close(
        loaded.module.atmosphere_model.scale,
        original.atmosphere_model.scale,
    )
    raster = stored_rasters[0]
    rendered = loaded.module.synthesize(
        raster.coordinates.reshape(-1, 3),
        ray_direction=raster.ray_direction.reshape(-1, 3),
        stokes_basis=raster.stokes_basis.reshape(-1, 3, 3),
    )["stokes"]
    assert rendered.shape == (2, 4, 2)
    assert torch.isfinite(rendered).all()

    cache_path = (
        tmp_path
        / "work"
        / "observation-cache"
        / f"hinode_sp-{loaded.observation.source_signature}.observation"
    )
    ObservationStore.save_sequence(
        cache_path,
        stored_rasters,
        raster_names=stored_names,
        source_signature=loaded.observation.source_signature,
        metadata=stored_metadata,
    )
    loader = p3s_loader.P3SLoader(save_state_path, device="cpu")
    assert loader._data_module is None
    assert loader.raster_names == ("raster_0000",)
    assert loader.raster_count == 1
    assert loader._data_module is None
    assert loader.select_raster().name == "raster_0000"
    assert loader._data_module.observation_store_path == cache_path
    assert loader.select_raster(index=0).index == 0
    assert loader.select_raster(name="raster_0000").index == 0
    assert loader._data_module.observation_store_path == cache_path
    height_fields = loader.raster_fields_at_height(0.0, batch_size=1)
    assert height_fields["magnetic_field_gauss"].shape == (1, 2, 3)
    assert np.isfinite(height_fields["temperature_k"]).all()

    monkeypatch.setattr(p3s_loader, "depth_grid", lambda module, count: count)
    monkeypatch.setattr(
        p3s_loader,
        "evaluate_atmosphere",
        lambda module, raster, **kwargs: {"kind": "cube", **kwargs},
    )
    monkeypatch.setattr(
        p3s_loader, "full_shell_height_grid", lambda module, count: count
    )
    monkeypatch.setattr(
        p3s_loader,
        "evaluate_full_shell_atmosphere",
        lambda module, raster, **kwargs: {"kind": "full_cube", **kwargs},
    )
    monkeypatch.setattr(
        p3s_loader,
        "evaluate_stokes",
        lambda module, raster, spec, **kwargs: {"kind": "stokes", **kwargs},
    )
    assert loader.cube(depth_samples=17)["kind"] == "cube"
    assert loader.full_cube(height_samples=23)["kind"] == "full_cube"
    assert loader.stokes()["kind"] == "stokes"

    evaluator = SimpleNamespace(
        evaluate_shell_layers=lambda module, raster, heights: {
            "kind": "shell_slices",
            "heights": heights,
        },
        evaluate_meridional_slice=lambda module, raster: {
            "kind": "meridional_slice"
        },
    )
    monkeypatch.setattr(loader, "_slice_evaluator", lambda **kwargs: evaluator)
    assert loader.shell_slices([0.0, 1.0])["kind"] == "shell_slices"
    assert loader.meridional_slice(215.0)["kind"] == "meridional_slice"


def test_p3s_rejects_nonprimitive_metadata(tmp_path, monkeypatch):
    artifact, resources, config, model_config = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    context = checkpoint_artifact.p3s_context(
        package_version="test",
        resolved_config={
            "schema_version": 2,
            "resources": {"bundle": "packaged"},
        },
        resources=resources,
        observation_spec=_spec().metadata(),
        source_signature=ObservationStore.manifest(artifact / "observations")[
            "source_signature"
        ],
        raster_names=("raster_0000",),
        validation_raster_index=0,
        times=[{"values": ["2024-03-24T01:00:00.000"], "scale": "tai"}],
        bounds={
            "surface_longitude_center_rad": 0.0,
            "surface_longitude_offset_rad": [0.0, 0.0],
            "surface_latitude_rad": [0.0, 0.0],
            "time_hours": [0.0, 0.0],
            "solar_radius_m": 6.96e8,
        },
        model=model_config,
    )
    incompatible = deepcopy(context)
    incompatible["model"]["velocity_synthesis_mode"] = (
        VelocitySynthesisMode.CARRINGTON_REGISTERED_RELATIVE
    )
    path = tmp_path / "nonprimitive.p3s"
    torch.save(
        {
            "epoch": 0,
            "global_step": 0,
            "parameters": {},
            "prom3theus_save_state": incompatible,
        },
        path,
    )

    with pytest.raises(PackageArtifactExportError, match="Could not safely load"):
        checkpoint_artifact.load_validated_save_state(path)


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


def _evaluate_velocity_export(raster, mode):
    return artifact_evaluation.evaluate_atmosphere(
        _evaluation_module(mode),
        raster,
        depth_grid=torch.tensor([-5.0, 1.0]),
        batch_size=2,
        storage_dtype="float32",
    )


def _assert_velocity_export(result, expected_observation_los):
    np.testing.assert_allclose(result["v_los_solar_inertial_m_per_s"], -100.0)
    np.testing.assert_allclose(
        result["v_los_instrument_corrected_m_per_s"], -95.0
    )
    np.testing.assert_allclose(result["v_los_m_per_s"], expected_observation_los)
    np.testing.assert_allclose(
        result["v_los_m_per_s"],
        -result["velocity_field_observer_m_per_s"][..., 2],
    )
    np.testing.assert_allclose(
        result["v_los_instrument_corrected_m_per_s"],
        -result["velocity_field_synthesis_observer_m_per_s"][..., 2],
    )


def test_registered_atmosphere_export_broadcasts_metadata_offset_once():
    raster = _raster(height=2)

    result = _evaluate_velocity_export(
        raster, CARRINGTON_REGISTERED_RELATIVE_VELOCITY
    )

    expected = np.asarray(
        [
            [[-85.0, -85.0], [-75.0, -75.0]],
            [[-85.0, -85.0], [-75.0, -75.0]],
        ]
    )
    _assert_velocity_export(result, expected)
    np.testing.assert_array_equal(
        result["removed_solar_los_velocity_m_per_s"],
        [[10.0, 20.0], [10.0, 20.0]],
    )


def test_registered_atmosphere_export_accepts_matching_duplicate_auxiliary():
    duplicate = torch.tensor([[10.0, 20.0], [10.0, 20.0]])
    raster = _raster(
        auxiliary={"removed_solar_los_velocity_m_per_s": duplicate},
        height=2,
    )

    result = _evaluate_velocity_export(
        raster, CARRINGTON_REGISTERED_RELATIVE_VELOCITY
    )

    np.testing.assert_allclose(
        result["v_los_m_per_s"],
        [[[-85.0, -85.0], [-75.0, -75.0]]] * 2,
    )


def test_registered_atmosphere_export_rejects_mismatched_duplicate_auxiliary():
    duplicate = torch.tensor([[10.0, 20.0], [10.0, 21.0]])
    raster = _raster(
        auxiliary={"removed_solar_los_velocity_m_per_s": duplicate},
        height=2,
    )

    with pytest.raises(
        exporter.ArtifactExportError, match="does not match the authoritative"
    ):
        _evaluate_velocity_export(raster, CARRINGTON_REGISTERED_RELATIVE_VELOCITY)


def test_observer_atmosphere_export_accepts_trailing_singleton_los_offset():
    observer_los = torch.tensor(
        [[[10.0], [20.0]], [[30.0], [40.0]]]
    )
    raster = _raster(
        auxiliary={"observer_los_velocity_m_per_s": observer_los},
        mode=CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
        height=2,
    )

    result = _evaluate_velocity_export(raster, CARRINGTON_OBSERVER_RELATIVE_VELOCITY)

    expected = np.asarray(
        [
            [[-85.0, -85.0], [-75.0, -75.0]],
            [[-65.0, -65.0], [-55.0, -55.0]],
        ]
    )
    _assert_velocity_export(result, expected)
    np.testing.assert_array_equal(
        result["observer_los_velocity_m_per_s"],
        [[10.0, 20.0], [30.0, 40.0]],
    )


def test_full_shell_export_is_opt_in_and_self_described(tmp_path, monkeypatch):
    artifact, resources, config, _ = _write_artifact(tmp_path)
    _patch_external_contracts(monkeypatch, resources, config)
    monkeypatch.setattr(
        exporter,
        "evaluate_atmosphere",
        lambda module, raster, **kwargs: {
            "temperature_k": np.zeros((1, 2, 2), dtype=np.float32),
            "valid_mask": raster.valid_mask.numpy(),
        },
    )
    requested_samples = []

    def shell_grid(module, samples):
        requested_samples.append(samples)
        return torch.tensor([20.0e6, -0.1e6])

    monkeypatch.setattr(exporter, "full_shell_height_grid", shell_grid)
    monkeypatch.setattr(
        exporter,
        "evaluate_full_shell_atmosphere",
        lambda module, raster, **kwargs: {
            "full_shell_geometric_height_m": np.asarray(
                [20.0e6, -0.1e6], dtype=np.float32
            ),
            "full_shell_temperature_k": np.zeros((1, 2, 2), dtype=np.float32),
        },
    )

    output = exporter.export_artifact(
        artifact,
        tmp_path / "full-shell.npz",
        depth_samples=2,
        include_full_shell=True,
        full_shell_samples=2,
        device="cpu",
    )

    assert requested_samples == [2]
    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert "full_shell_temperature_k" in archive
        assert metadata["evaluation"]["full_shell"]["coordinate"] == (
            "geometric_height_m"
        )
        assert metadata["evaluation"]["full_shell"]["height_bounds_m"] == [
            20.0e6,
            -0.1e6,
        ]
        assert metadata["evaluation"]["full_shell"]["spatial_sampling"] == (
            "radial_carrington_columns"
        )


def test_export_facade_exposes_the_current_api():
    assert exporter.__all__ == ["ArtifactExportError", "export_artifact"]


def test_package_and_export_module_share_one_export_error_type():
    assert PackageArtifactExportError is exporter.ArtifactExportError
