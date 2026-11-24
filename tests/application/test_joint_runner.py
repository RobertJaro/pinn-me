"""Joint training and validation tests with deliberately cheap forward fakes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from prom3theus.application import joint_rendering, joint_streams
from prom3theus.application.joint_contracts import (
    JointRunnerFactories,
    LoadedJointStream,
    SharedTermAssembly,
)
from prom3theus.application.joint_assembly import build_joint_runtime
from prom3theus.application.batches import deterministic_validation_batch
from prom3theus.config import JointInversionConfig, parse_config
from prom3theus.inversion.data_terms import (
    DataTermBatchResult,
    ObservationDataTerm,
)
from prom3theus.observations import (
    ImageObservationRaster,
    ObservationDescriptor,
    ObservationKind,
    PreparedObservationStream,
    SceneContract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolate_runner_protocol_from_lte_physics(monkeypatch):
    class ProtocolWindow(torch.nn.Module):
        def __init__(self, atmosphere, *, configuration):
            super().__init__()
            self.construction = configuration
    monkeypatch.setattr("prom3theus.inversion.lte.LTEWindow", ProtocolWindow)


class _DiagnosticPixelDataset(torch.utils.data.Dataset):
    pixel_indices = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"pixel_index": self.pixel_indices[index], "stokes": torch.full((4, 6), .2)}


@pytest.mark.parametrize("normalize", [False, True])
def test_joint_stokes_diagnostics_render_shared_radiation_and_atmosphere(tmp_path, monkeypatch, normalize):
    from unittest.mock import Mock
    from prom3theus.config import load_config
    from prom3theus.diagnostics.rendering import AtmosphereRenderer
    from prom3theus.inversion.data_terms.stokes import StokesObservationTerm

    term = StokesObservationTerm.__new__(StokesObservationTerm)
    nn.Module.__init__(term)
    term.wavelength_angstrom = torch.arange(6, dtype=torch.float32) + 6173
    term.wavelength_weights = torch.ones(6)
    from prom3theus.inversion.objective import StokesObjective
    term.objective = StokesObjective(type="asinh_mse", asinh_scale=1e-3,
                                    continuum_normalization=normalize, continuum_indices=[0, 5])
    window = nn.Sequential(nn.Identity())
    window[0].eval()
    window.coefficients = SimpleNamespace(lines=[])
    object.__setattr__(term, "synthesis_window", window)

    def predict(batch):
        assert not term.training
        assert not window.training
        assert not atmosphere.training
        return batch["stokes"] * 0.9

    term.predict = predict
    raster = SimpleNamespace(spatial_shape=(2, 2))
    dataset = _DiagnosticPixelDataset()
    data = SimpleNamespace(
        raster=raster,
        validation_dataset=torch.utils.data.Subset(dataset, [0]),
        evaluation_dataset=lambda: dataset,
        evaluation_collate_fn=lambda: None,
        num_workers=0,
    )
    atmosphere = nn.Linear(1, 1)
    runtime = SimpleNamespace(
        config=load_config(PROJECT_ROOT / "configs/sdo/hmi_aia_dynamic.yaml"),
        device=torch.device("cpu"),
        model=SimpleNamespace(
            terms={"photospheric_stokes": term}, atmosphere_model=atmosphere
        ),
        streams={"photospheric_stokes": SimpleNamespace(
            data_module=data, specification=SimpleNamespace(radiance_scale_w_m3_sr=1.e13)
        )},
    )
    stokes = Mock(return_value=tmp_path / "stokes.png")
    profiles = Mock(return_value=tmp_path / "profiles.png")
    fields = Mock(return_value=[tmp_path / "fields.png"])
    monkeypatch.setattr(AtmosphereRenderer, "render_stokes_validation", stokes)
    monkeypatch.setattr(AtmosphereRenderer, "render_stokes_profiles", profiles)
    monkeypatch.setattr(AtmosphereRenderer, "render_atmosphere", fields)
    paths = joint_rendering._render_stokes_diagnostics(
        runtime, SimpleNamespace(global_step=12), tmp_path
    )
    assert len(paths.paths) == 3
    stokes.assert_called_once()
    profiles.assert_called_once()
    assert profiles.call_args.kwargs["radiance_scale_w_m3_sr"] == 1.e13
    fields.assert_called_once()
    assert stokes.call_args.args[0].logger is None
    assert stokes.call_args.args[4] == "photospheric_stokes"
    assert fields.call_args.kwargs["label"] == "photospheric_stokes"
    assert stokes.call_args.kwargs["rows"].tolist() == [0, 1]
    assert stokes.call_args.kwargs["columns"].tolist() == [0, 1]
    assert fields.call_args.kwargs["rows"].tolist() == [0]
    assert stokes.call_args.args[1]["stokes_pred"].shape == (4, 4, 6)
    from prom3theus.diagnostics.sampling import integrated_stokes
    payload = stokes.call_args.args[1]
    torch.testing.assert_close(payload["raw_stokes_pred"], torch.full((4, 4, 6), .18))
    torch.testing.assert_close(payload["raw_stokes_reference"], torch.full((4, 4, 6), .2))
    compared = term.objective.transform(torch.full((4, 4, 6), .9 if normalize else .18))
    torch.testing.assert_close(payload["stokes_pred"], compared)
    torch.testing.assert_close(payload["integrated_prediction"], integrated_stokes(
        compared, term.wavelength_angstrom, term.wavelength_weights))
    assert stokes.call_args.kwargs["objective_config"] == term.objective.configuration()
    view = fields.call_args.args[1]
    assert view.atmosphere_model is atmosphere
    assert view.synthesis_window is term.synthesis_window
    assert not hasattr(view, "forward_composition")
    assert term.training
    assert window.training
    assert not window[0].training
    assert atmosphere.training


def test_joint_online_wandb_logger_and_validation_uploads(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from prom3theus.application.joint_training import _online_logger, _log_validation
    from prom3theus.config import load_config
    import pytorch_lightning.loggers

    config = load_config(PROJECT_ROOT / "configs/sdo/hmi_aia_dynamic.yaml")
    logger = Mock()
    constructor = Mock(return_value=logger)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(pytorch_lightning.loggers, "WandbLogger", constructor)
    assert _online_logger(config) is logger
    options = constructor.call_args.kwargs
    assert options["mode"] == "online"
    assert options["offline"] is False
    assert options["project"] == "PROM3THEUS"
    logger.log_hyperparams.assert_called_once()

    image = tmp_path / "diagnostics" / "initial_aia_group_comparison.png"
    image.parent.mkdir(parents=True)
    image.touch()
    _log_validation(
        logger,
        {
            "metrics": {"loss": 1.5},
            "diagnostics": {"internal.raw_loss": 99.0},
            "rendering": {"artifacts": [{"stream_id": "aia", "path": str(image)}]},
        },
        12,
    )
    logger.log_metrics.assert_called_once_with({"valid.loss": 1.5}, step=12)
    logger.log_image.assert_called_once_with(
        key="Observation comparison", images=[str(image)], caption=[image.stem], step=12
    )
    _log_validation(None, {"metrics": {}}, 12)


def test_default_stokes_builder_converts_yaml_component_keys(monkeypatch):
    from prom3theus.application import joint_assembly
    from prom3theus.components import forward as forward_components
    from prom3theus.config import load_config
    from prom3theus.inversion.configuration import resolve_objective_weighting
    from prom3theus.inversion.objective import StokesObjective

    config = load_config(PROJECT_ROOT / "configs/sdo/hmi_aia_dynamic.yaml")
    stream = config.streams[0]
    original = stream.data_term.objective.to_dict()
    specification = SimpleNamespace(
        observation_id="hmi",
        wavelength_angstrom=torch.arange(6, dtype=torch.float32),
        continuum_indices=[0],
        radiance_scale_w_m3_sr=1.0,
        velocity_synthesis_mode=SimpleNamespace(value="local"),
    )
    monkeypatch.setattr(
        forward_components, "_stokes_instrument_options", lambda *args: ({}, 0.0, False, 10000.0)
    )

    def build_term(**options):
        weighting = resolve_objective_weighting(
            options["wavelength_angstrom"],
            weight_config=options["weight_config"],
            wavelength_weights=None,
            wavelength_exclude_windows_angstrom=[],
            continuum_indices=options["continuum_indices"],
            atlas_continuum_radiance_w_m3_sr=1.0,
        )
        objective_options = dict(options["objective_config"])
        objective_options.pop("qu_warmup_steps", None)  # Owned by StokesObservationTerm.
        objective = StokesObjective(**objective_options, continuum_indices=options["continuum_indices"])
        expected = torch.tensor(list(original["stokes_weights"].values()))
        torch.testing.assert_close(weighting.stokes_weights, expected)
        assert objective.loss_type == original["type"]
        assert objective.asinh_scale == original["asinh_scale"]
        return objective

    monkeypatch.setattr(forward_components, "StokesObservationTerm", build_term)
    joint_assembly._default_term_builder(
        stream, SimpleNamespace(specification=specification), nn.Identity(), None,
        synthesis_windows={stream.data_term.synthesis_window: nn.Identity()},
    )
    assert stream.data_term.objective.to_dict() == original


def _config(
    tmp_path: Path, *, render_streams: bool = False
) -> JointInversionConfig:
    legacy = yaml.safe_load(
        (PROJECT_ROOT / "configs/sdo/hmi_aia_dynamic.yaml").read_text(encoding="utf-8")
    )
    document = {
        "schema_version": 5,
        "solver": {
            "kind": "joint",
            "output_directory": "output",
            "work_directory": "work",
        },
        "scene": {"reference_stream": "photospheric_stokes"},
        "atmosphere": deepcopy(legacy["atmosphere"]),
        "physics": deepcopy(legacy["physics"]),
        "synthesis": deepcopy(legacy["synthesis"]),
        "atmosphere_regularization": [],
        "streams": [
            {
                "id": "photospheric_stokes",
                "observation": deepcopy(legacy["streams"][0]["observation"]),
                "data_term": {
                    "type": "lte_stokes",
                    "weight": 1.0,
                    "synthesis_window": legacy["streams"][0]["data_term"]["synthesis_window"],
                    "instrument": deepcopy(
                        legacy["streams"][0]["data_term"]["instrument"]
                    ),
                    "objective": deepcopy(
                        legacy["streams"][0]["data_term"]["objective"]
                    ),
                },
            },
            {
                "id": "coronal_euv",
                "observation": {
                    "type": "aia_euv",
                    "directory": "prepared-aia",
                    "channels_angstrom": [171, 193, 211],
                    "selection": {"validation_exposure_group": "group-0"},
                    "loader": {
                        "batch_size": 4,
                        "validation_batch_size": 4,
                        "workers": 0,
                        "pin_memory": False,
                    },
                },
                "data_term": {
                    "type": "aia_optically_thin",
                    "weight": 0.25,
                    "synthesis": {
                        "response_resource": "aia_euv_v1:aia_temperature_response",
                        "ray_end": "atmosphere_outer_shell",
                        "ray_samples": 4,
                        "height_sampling_power": 2.0,
                    },
                    "objective": {
                        "type": "asinh_mse",
                        "channel_weights": [
                            {"channel_angstrom": 171, "weight": 1.0},
                            {"channel_angstrom": 193, "weight": 1.0},
                            {"channel_angstrom": 211, "weight": 1.0},
                        ],
                        "calibration": {
                            "enabled": True,
                            "absolute_prior_fraction": 0.25,
                            "relative_prior_fraction": 0.15,
                        },
                    },
                },
            },
        ],
        "diagnostics": {
            "visualization": {
                "enabled": render_streams,
                "render_streams": render_streams,

            }
        },
        "training": {
            "device": "cpu",
            "max_steps": 3,
            "learning_rate": 0.01,
            "final_learning_rate": 0.01,
            "validation_samples_per_stream": 9,
        },
    }
    config = parse_config(document, base_directory=tmp_path, environ={})
    assert isinstance(config, JointInversionConfig)
    return config


class _StokesData:
    def __init__(self) -> None:
        self.setup_calls = []
        self.batch = {
            "x": torch.arange(1.0, 13.0),
            "coordinates": torch.zeros(12, 3),
            "ray_direction": torch.tensor([[0.0, 0.0, -1.0]]).expand(12, 3),
            "target": torch.zeros(12),
            "instrument_response": {"profile": torch.arange(24.0).reshape(12, 2)},
        }

    def setup(self, stage=None) -> None:
        self.setup_calls.append(stage)

    def val_dataloader(self):
        return [self.batch]

    def run_metadata(self):
        return {"kind": "stokes", "setup_calls": list(self.setup_calls)}


class _ImageData:
    def __init__(self) -> None:
        self.setup_calls = []
        self.loaders = {
            channel: [
                {
                    "x": torch.arange(1.0, 6.0) + index,
                    "target": torch.full((5,), float(index)),
                }
            ]
            for index, channel in enumerate((171, 193, 211))
        }

    def setup(self, stage=None) -> None:
        self.setup_calls.append(stage)

    def validation_dataloaders(self):
        return self.loaders

    def val_dataloader(self):
        return list(self.loaders.values())

    def run_metadata(self):
        return {"kind": "image", "setup_calls": list(self.setup_calls)}


class _Atmosphere(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.field = nn.Parameter(torch.tensor(0.75))


class _Term(ObservationDataTerm):
    observation_kind = "fake"

    def __init__(self, atmosphere: _Atmosphere, stream_id: str) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))
        self.stream_id = stream_id
        if stream_id == "photospheric_stokes":
            self.synthesis_window_id = "fe_i_6173"
        self._composition = SimpleNamespace(atmosphere_model=atmosphere)

    def _prediction(self, batch, ray_samples=None, refine=True):
        factor = 1.0 if ray_samples is None else 1.0 + 1.0 / ray_samples
        return (
            self._composition.atmosphere_model.field * batch["x"] * factor + self.bias
        )

    def predict(self, batch):
        return self._prediction(batch)[:, None, None].expand(-1, 4, 2)

    def evaluate_batch(self, batch):
        prediction = self._prediction(batch)
        residual = prediction - batch["target"]
        return DataTermBatchResult(
            likelihood_loss=residual.square().mean(),
            component_losses={"data": residual.abs().mean()},
            metrics={"prediction_mean": prediction.mean(),
                     "stokes_outgoing_admissibility_max": prediction.new_zeros(())},
            sample_count=int(prediction.numel()),
            diagnostics={"prediction": prediction},
        )

    def synthesize(self, batch, *, ray_samples=None, refine=True):
        return SimpleNamespace(prediction=self._prediction(batch, ray_samples))

    def metadata(self):
        return {"type": "fake", "stream_id": self.stream_id}


def _factories(tmp_path: Path):
    calls = {"resources": None, "rebuild": []}

    def resources(set_ids):
        calls["resources"] = tuple(set_ids)
        return {
            "legacy_lte_v2": {"fake": True},
            "aia_euv_v1": {
                "scientific_contract": {
                    "calibration_convention_id": "sha256:" + "a" * 64
                }
            },
        }

    def stream_loader(
        stream, resource_metadata, work_directory, *, rebuild_observations
    ):
        del resource_metadata, work_directory
        calls["rebuild"].append((stream.id, rebuild_observations))
        kind = "image" if stream.id == "coronal_euv" else "stokes"
        data = _ImageData() if kind == "image" else _StokesData()
        store = tmp_path / f"{stream.id}.store"
        store.mkdir(exist_ok=True)
        descriptor = ObservationDescriptor(
            observation_id=stream.id,
            observation_type=stream.observation.type,
            instrument_type="fake",
            observation_kind=kind,
        )
        prepared = PreparedObservationStream(
            name=stream.id,
            descriptor=descriptor,
            data_module=data,
            store_path=store,
            source_signature=("a" if kind == "stokes" else "b") * 64,
        )
        return LoadedJointStream(
            prepared=prepared,
            specification=SimpleNamespace(observation_id=stream.id),
            rasters=(SimpleNamespace(),),
            setup_metadata={"asinh_scales": (1.0, 2.0, 3.0)} if kind == "image" else {},
        )

    def scene_builder(config, streams):
        assert set(streams) == {config.scene.reference_stream}
        geometry = config.atmosphere.geometry
        return SceneContract(
            scene_basis=torch.eye(3),
            solar_radius_m=6.957e8,
            reference_time_tai_seconds=1.7e9,
            spatial_coordinate_center_megameter=(0.0, 0.0),
            spatial_coordinate_scale_megameter=(10.0, 10.0),
            time_coordinate_center_hours=0.0,
            time_coordinate_scale_hours=1.0,
            height_bounds_m=(
                geometry.inner_height_megameter * 1.0e6,
                geometry.outer_height_megameter * 1.0e6,
            ),
        )

    def atmosphere_builder(config, scene, resources):
        del config, scene, resources
        return _Atmosphere()

    def term_builder(stream, loaded, atmosphere, scene):
        del loaded, scene
        return _Term(atmosphere, stream.id)

    def shared_builder(config, streams, scene, resources):
        del config, streams, scene, resources
        return SharedTermAssembly(
            {},
            {"sampling": "fake deterministic"},
        )

    return (
        JointRunnerFactories(
            resource_validator=resources,
            stream_loader=stream_loader,
            scene_builder=scene_builder,
            atmosphere_builder=atmosphere_builder,
            term_builder=term_builder,
            shared_terms_builder=shared_builder,
        ),
        calls,
    )


@pytest.mark.parametrize("automatic_resume", [False, True])
@pytest.mark.parametrize("aia_weight", [0.0, 0.1])
def test_joint_inversion_optimizes_both_streams_and_resumes(
    tmp_path, monkeypatch, aia_weight, automatic_resume
):
    import pytorch_lightning
    from dataclasses import replace
    from torch.utils.data import DataLoader
    from prom3theus.config.joint_schema import JointTrainingConfig
    from prom3theus.application.joint_training import run_joint_inversion
    from prom3theus.application.joint_training import JointSaveStateCallback

    saved_steps = []
    original_save = JointSaveStateCallback._save

    def record_save(callback, trainer, module):
        original_save(callback, trainer, module)
        saved_steps.append(torch.load(callback.path, weights_only=True)["global_step"])

    monkeypatch.setattr(JointSaveStateCallback, "_save", record_save)

    def reject_seed(*args, **kwargs):
        pytest.fail("Joint runs must not explicitly seed random generators.")

    monkeypatch.setattr(torch, "manual_seed", reject_seed)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", reject_seed)
    monkeypatch.setattr(pytorch_lightning, "seed_everything", reject_seed)

    monkeypatch.setattr(
        _StokesData,
        "train_dataloader",
        lambda self: DataLoader([self.batch], batch_size=None),
        raising=False,
    )
    monkeypatch.setattr(
        _ImageData,
        "train_dataloader",
        lambda self: DataLoader(
            [batch for batches in self.loaders.values() for batch in batches],
            batch_size=None,
        ),
        raising=False,
    )
    config = _config(tmp_path)
    config = replace(
        config,
        streams=tuple(
            replace(stream, data_term=replace(stream.data_term, weight=aia_weight))
            if stream.id == "coronal_euv"
            else stream
            for stream in config.streams
        ),
    )
    config = replace(
        config,
        training=JointTrainingConfig(
            max_steps=3,
            learning_rate=0.01,
            final_learning_rate=0.01,
            device="cpu",
            validation_samples_per_stream=9,
            validation_every_n_steps=2,
            checkpoint_every_n_steps=2,
            log_every_n_steps=1,
        ),
    )
    factories, _ = _factories(tmp_path)
    monkeypatch.setattr(
        "prom3theus.application.joint_training.snapshot_context",
        lambda runtime: {"test_runtime_max_steps": runtime.config.training.max_steps},
    )
    run = run_joint_inversion(config, factories=factories)
    assert saved_steps == [2, 3]
    assert run.trainer.global_step == 3
    assert run.module.model.atmosphere_model.field.item() < 0.75
    assert run.module.model.terms["photospheric_stokes"].bias.item() < 0.1
    if aia_weight > 0:
        assert run.module.model.terms["coronal_euv"].bias.item() < 0.1
    else:
        torch.testing.assert_close(
            run.module.model.terms["coronal_euv"].bias,
            torch.tensor(0.1),
            rtol=0,
            atol=0,
        )
        assert run.module.model.terms["coronal_euv"].bias.grad is None
    checkpoint = torch.load(run.checkpoint_path, weights_only=False)
    assert checkpoint["global_step"] == 3
    assert checkpoint["optimizer_states"]
    assert checkpoint["lr_schedulers"]
    assert "prom3theus_joint" in checkpoint
    assert sorted(
        path.name for path in config.solver.output_directory.rglob("*.ckpt")
    ) == ["last.ckpt"]
    assert not (config.solver.output_directory / "validation").exists()
    assert not (config.solver.work_directory / "validation/step_00000000").exists()
    saved = torch.load(run.save_state_path, weights_only=True)
    assert saved["format"] == "prom3theus.stream_state"
    assert saved["global_step"] == 3
    assert "optimizer_states" not in saved
    assert "lr_schedulers" not in saved
    assert (
        config.solver.work_directory / "validation/step_00000002/metrics.json"
    ).is_file()
    assert (
        config.solver.work_directory / "validation/step_00000003/metrics.json"
    ).is_file()
    trained_field = run.module.model.atmosphere_model.field.item()
    # Enable AIA by YAML-equivalent weight change while restoring the same checkpoint.
    config = replace(
        config,
        streams=tuple(
            replace(stream, data_term=replace(stream.data_term, weight=0.1))
            if stream.id == "coronal_euv"
            else stream
            for stream in config.streams
        ),
    )
    config = replace(
        config,
        training=replace(
            config.training,
            max_steps=5,
            resume_from_checkpoint=None if automatic_resume else run.checkpoint_path,
        ),
    )
    resumed = run_joint_inversion(config, factories=factories)
    assert saved_steps == [2, 3, 4, 5]
    assert resumed.trainer.global_step == 5
    assert resumed.module.model.atmosphere_model.field.item() < trained_field
    assert resumed.module.model.terms["coronal_euv"].bias.item() < 0.1
    assert torch.load(resumed.save_state_path, weights_only=True)["global_step"] == 5
    assert sorted(
        path.name for path in config.solver.output_directory.rglob("*.ckpt")
    ) == ["last.ckpt"]
    assert sorted(
        path.name for path in config.solver.output_directory.glob("*.p3s")
    ) == ["state.p3s"]

    completed = torch.load(resumed.save_state_path, weights_only=True)
    for snapshot_condition in ("missing", "stale"):
        if snapshot_condition == "missing":
            resumed.save_state_path.unlink()
        else:
            # The public snapshot can lag behind the resumable checkpoint.
            torch.save(saved, resumed.save_state_path)
        no_steps = run_joint_inversion(config, factories=factories)
        regenerated = torch.load(no_steps.save_state_path, weights_only=True)
        assert no_steps.trainer.global_step == regenerated["global_step"] == 5
        assert regenerated["context"]["test_runtime_max_steps"] == 5
        assert set(regenerated["state_dict"]) == set(completed["state_dict"])
        for name, expected in completed["state_dict"].items():
            torch.testing.assert_close(regenerated["state_dict"][name], expected, rtol=0, atol=0)
            torch.testing.assert_close(no_steps.module.model.state_dict()[name], expected, rtol=0, atol=0)
    assert saved_steps == [2, 3, 4, 5, 5, 5]


def test_build_assembles_one_atmosphere_and_balanced_deterministic_batches(tmp_path):
    config = _config(tmp_path)
    factories, calls = _factories(tmp_path)

    runtime = build_joint_runtime(
        config, rebuild_observations=True, factories=factories
    )

    assert calls["resources"] == ("legacy_lte_v2", "aia_euv_v1")
    assert calls["rebuild"] == [
        ("photospheric_stokes", True),
        ("coronal_euv", True),
    ]
    assert (
        set(runtime.streams)
        == set(runtime.model.terms)
        == {
            "photospheric_stokes",
            "coronal_euv",
        }
    )
    assert all(
        term._composition.atmosphere_model is runtime.model.atmosphere_model
        for term in runtime.model.terms.values()
    )
    assert {
        name: batch["x"].numel() for name, batch in runtime.validation_batches.items()
    } == {"photospheric_stokes": 9, "coronal_euv": 9}
    assert runtime.validation_batches["photospheric_stokes"]["instrument_response"][
        "profile"
    ].shape == (9, 2)
    # Three image loaders each contribute their first three samples.
    torch.testing.assert_close(
        runtime.validation_batches["coronal_euv"]["x"],
        torch.tensor([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]),
    )

    repeated = deterministic_validation_batch(
        runtime.streams["coronal_euv"].data_module, max_samples=9
    )
    torch.testing.assert_close(
        repeated["x"], runtime.validation_batches["coronal_euv"]["x"].cpu()
    )


def test_validation_routes_aia_payload_and_native_rasters_to_default_registry(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    config = _config(tmp_path, render_streams=True)
    factories, _ = _factories(tmp_path)
    original_builder = factories.term_builder

    def term_builder(*args):
        term = original_builder(*args)
        term.channels_angstrom = (211, 171, 193)
        term.objective = SimpleNamespace(asinh_scales=torch.tensor([30.0, 10.0, 20.0]))
        term.intensity_scales = torch.tensor([3000.0, 1000.0, 2000.0])
        return term

    factories = replace(factories, term_builder=term_builder)
    captured = {}

    class _Registry:
        def render_results(
            self,
            results,
            *,
            stream_kinds,
            output_directory,
            label,
            contexts,
        ):
            captured.update(
                results=results,
                stream_kinds=stream_kinds,
                output_directory=output_directory,
                label=label,
                contexts=contexts,
            )
            output_directory.mkdir(parents=True)
            (output_directory / "initial_aia.png").write_bytes(b"fake image")
            return {
                "label": label,
                "streams": {
                    "coronal_euv": {
                        "diagnostics": {
                            "paths": [str(output_directory / "initial_aia.png")],
                            "contribution_paths": [],
                        }
                    }
                },
            }

    def registry_factory(**options):
        captured["options"] = options
        return _Registry()

    from prom3theus.application import joint_rendering

    monkeypatch.setattr(
        joint_rendering, "default_diagnostic_registry", registry_factory
    )

    runtime = build_joint_runtime(config, factories=factories)
    evaluation = runtime.model.evaluate_batches(runtime.validation_batches)
    result = joint_rendering._render_aia_diagnostics(runtime, evaluation, tmp_path)

    assert set(captured["results"]) == {"coronal_euv"}
    assert captured["stream_kinds"] == {"coronal_euv": "aia_euv"}
    assert captured["label"] == "validation"
    assert captured["contexts"]["coronal_euv"]["rasters"] == (
        runtime.streams["coronal_euv"].rasters
    )
    assert captured["options"] == {
        "aia_max_pixels_per_image": (
            config.diagnostics.visualization.ray_sampling.max_pixels
        ),
        "aia_max_contribution_rays": (
            config.diagnostics.visualization.contribution_ray_count
        ),
        "dpi": config.diagnostics.visualization.dpi,
    }
    assert captured["contexts"]["coronal_euv"]["contribution_payloads"] == ()
    assert captured["contexts"]["coronal_euv"]["asinh_scales"] == {
        171: 10.0,
        193: 20.0,
        211: 30.0,
    }
    assert captured["contexts"]["coronal_euv"]["intensity_scales"] == {
        171: 1000.0,
        193: 2000.0,
        211: 3000.0,
    }
    assert captured["contexts"]["coronal_euv"]["likelihood_component_weights"] == {
        "channel_171": 1.0,
        "channel_193": 1.0,
        "channel_211": 1.0,
    }
    assert result["registry"] == ("default_diagnostic_registry")
    assert (
        tmp_path / "diagnostics/initial_aia.png"
    ).is_file()


def test_validation_cap_must_include_each_native_image_loader():
    with pytest.raises(ValueError, match="at least one sample"):
        deterministic_validation_batch(_ImageData(), max_samples=2)


def test_contribution_ray_budget_is_global_balanced_and_capacity_aware():
    counts = joint_rendering._balanced_contribution_counts(
        {171: 100, 193: 2, 211: 100},
        maximum_count=16,
    )

    assert counts == {171: 7, 193: 2, 211: 7}
    assert sum(counts.values()) == 16


def test_physics_bounds_conservatively_cover_complete_aia_ray_support():
    radius = 6.957e8
    latitude = torch.deg2rad(torch.tensor(-10.0, dtype=torch.float64))
    surface_vector = (
        torch.tensor(
            [torch.cos(latitude), 0.0, torch.sin(latitude)], dtype=torch.float64
        )
        * radius
    )
    raster = ImageObservationRaster(
        intensity=torch.ones(1, 1),
        uncertainty=torch.ones(1, 1),
        ray_direction=torch.tensor([[[-1.0, 0.0, 0.0]]]),
        surface_position_m=surface_vector.reshape(1, 1, 3).to(torch.float32),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        absolute_tai_seconds=1_700_000_000.0,
        channel_angstrom=171,
        exposure_group="target-0000",
        metadata={
            "intensity_unit": "DN s-1 pixel-1",
            "ray_geometry": {"solar_radius_m": radius},
        },
    )
    scene = SceneContract(
        scene_basis=torch.eye(3, dtype=torch.float64),
        solar_radius_m=radius,
        reference_time_tai_seconds=1_700_000_000.0,
        spatial_coordinate_center_megameter=(0.0, 0.0),
        spatial_coordinate_scale_megameter=(10.0, 10.0),
        time_coordinate_center_hours=0.0,
        time_coordinate_scale_hours=1.0,
        height_bounds_m=(-1.0e5, 50.0e6),
    )
    loaded = SimpleNamespace(
        prepared=SimpleNamespace(
            descriptor=SimpleNamespace(observation_kind=ObservationKind.IMAGE)
        ),
        rasters=(raster,),
    )

    bounds = joint_streams._joint_observation_bounds({"aia": loaded}, scene)
    endpoint = scene.image_ray_outer_endpoints(raster)[0, 0]
    fractions = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    support = surface_vector + fractions[:, None] * (endpoint - surface_vector)
    longitude = torch.atan2(support[:, 1], support[:, 0])
    latitude = torch.atan2(
        support[:, 2], torch.linalg.vector_norm(support[:, :2], dim=-1)
    )
    center = bounds["surface_longitude_center_rad"]
    offset = torch.atan2(torch.sin(longitude - center), torch.cos(longitude - center))

    assert float(offset.amin()) >= bounds["surface_longitude_offset_rad"][0]
    assert float(offset.amax()) <= bounds["surface_longitude_offset_rad"][1]
    assert float(latitude.amin()) >= bounds["surface_latitude_rad"][0]
    assert float(latitude.amax()) <= bounds["surface_latitude_rad"][1]
    assert bounds["maximum_image_ray_angular_span_rad"] > 0.0
    assert "complete AIA" in bounds["angular_support"]
