"""Public snapshots reconstruct evaluation without training or observation I/O."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from prom3theus.config import load_config
from prom3theus.application.joint_assembly import _default_atmosphere_builder
from prom3theus.components.forward import _default_term_builder
from prom3theus.artifacts.checkpoint import (
    EvaluationModel,
    save_state,
    snapshot_context,
    load_validated_save_state,
)
from prom3theus.artifacts.loader import P3SLoader
from prom3theus.artifacts.errors import ArtifactExportError
from prom3theus.observations import SceneContract
from prom3theus.resources import validate_resource_sets, LEGACY_LTE_RESOURCE_SET_ID

ROOT = Path(__file__).resolve().parents[2]


def _runtime(tmp_path):
    config = load_config(ROOT / "configs/hmi_aia_dynamic.yaml")
    stream = config.streams[1]
    sampling = replace(
        stream.data_term.synthesis,
        ray_samples=6,
        coarse_to_fine=replace(
            stream.data_term.synthesis.coarse_to_fine, fine_sample_count=4
        ),
    )
    stream = replace(stream, data_term=replace(stream.data_term, synthesis=sampling))
    config = replace(
        config,
        scene=replace(config.scene, reference_stream=stream.id),
        streams=(stream,),
        physics=replace(
            config.physics,
            equations=replace(
                config.physics.equations,
                induction=replace(
                    config.physics.equations.induction, enabled=False, weight=0.0
                ),
            ),
        ),
        atmosphere=replace(
            config.atmosphere,
            network=replace(
                config.atmosphere.network, hidden_dimension=8, hidden_layers=1
            ),
        ),
        training=replace(config.training, reference_stream=stream.id),
        dry_run=replace(config.dry_run, quadrature_samples=(6, 12, 24)),
    )
    radius = 6.957e8
    scene = SceneContract(
        torch.eye(3, dtype=torch.float64),
        radius,
        1_700_000_000.0,
        (0.0, 0.0),
        (100.0, 100.0),
        0.0,
        1.0,
        (
            config.atmosphere.geometry.inner_height_megameter * 1e6,
            config.atmosphere.geometry.outer_height_megameter * 1e6,
        ),
    )
    resources = validate_resource_sets((LEGACY_LTE_RESOURCE_SET_ID, "aia_euv_v1"))
    atmosphere = _default_atmosphere_builder(config, scene, resources)
    spec = SimpleNamespace(
        observation_id="aia", metadata=lambda: {"observation_id": "aia"}
    )
    loaded = SimpleNamespace(
        specification=spec,
        setup_metadata={
            "asinh_scales": [1.0, 1.0, 1.0],
            "intensity_scales": [1.0, 1.0, 1.0],
        },
        store_metadata={},
        rasters=(),
        data_module=SimpleNamespace(),
        prepared=SimpleNamespace(
            store_path=tmp_path / "missing-observations",
            source_signature="a" * 64,
            descriptor=SimpleNamespace(metadata=lambda: {}),
        ),
    )
    term = _default_term_builder(stream, loaded, atmosphere, scene)
    model = EvaluationModel(atmosphere, {stream.id: term}).eval()
    runtime = SimpleNamespace(
        config=config,
        scene=scene,
        resources=resources,
        model=model,
        streams={stream.id: loaded},
    )
    batch = {
        "surface_position_m": torch.tensor(
            [[0.0, 0.0, radius]] * 3, dtype=torch.float64
        ),
        "ray_direction": torch.tensor([[0.0, 0.0, -1.0]] * 3, dtype=torch.float64),
        "absolute_tai_seconds": torch.full(
            (3,), scene.reference_time_tai_seconds, dtype=torch.float64
        ),
        "channel_index": torch.arange(3),
    }
    return runtime, batch


def test_fixed_magnetic_reference_survives_p3s_reconstruction(tmp_path):
    runtime, _ = _runtime(tmp_path)
    atmosphere = runtime.config.atmosphere
    runtime.config = replace(runtime.config, atmosphere=replace(atmosphere,
        parameters=replace(atmosphere.parameters, magnetic_field=replace(
            atmosphere.parameters.magnetic_field, reference_height_megameter=.15))))
    fixed = _default_atmosphere_builder(runtime.config, runtime.scene, runtime.resources)
    fixed.load_state_dict(runtime.model.atmosphere_model.state_dict(), strict=True)
    runtime.model.atmosphere_model = fixed
    path = tmp_path / "fixed-reference.p3s"
    save_state(path, runtime.model, snapshot_context(runtime), epoch=0, global_step=12)
    restored = load_validated_save_state(path)
    assert restored.module.atmosphere_model.magnetic_reference_height_megameter == .15
    assert restored.context["atmosphere"]["magnetic_reference_height_megameter"] == .15
    coords = torch.tensor([[.3, -.2, .1], [-.6, .4, .7]])
    heights = torch.tensor([[0., .1e6, .3e6]]).expand(2, -1)
    with torch.no_grad():
        expected = fixed.evaluate_at_height(coords, heights)
        actual = restored.module.atmosphere_model.evaluate_at_height(coords, heights)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    torch.testing.assert_close(actual["magnetic_field"][:, 0], actual["magnetic_field"][:, 2],
                               atol=0, rtol=0)


@pytest.mark.parametrize("legacy_objective", [False, True])
def test_public_p3s_roundtrip_without_observations_or_trainer(tmp_path, legacy_objective):
    runtime, batch = _runtime(tmp_path)
    path = tmp_path / "state.p3s"
    context = snapshot_context(runtime)
    if legacy_objective:
        objective = context["configuration"]["streams"][0]["data_term"]["objective"]
        objective.update(type="asinh_huber", huber_delta=1.0)
        context["terms"]["coronal_euv"]["options"]["huber_delta"] = 1.0
    with torch.no_grad():
        expected = runtime.model.terms["coronal_euv"].predict(batch)
    save_state(path, runtime.model, context, epoch=2, global_step=17)
    loader = P3SLoader(path, device="cpu")
    assert loader.config.streams[0].data_term.objective.type == "asinh_mse"
    if legacy_objective:
        assert context["configuration"]["streams"][0]["data_term"]["objective"]["type"] == "asinh_huber"
    assert loader.stream_ids == ("coronal_euv",)
    assert loader.global_step == 17
    torch.testing.assert_close(
        loader.predict("coronal_euv", batch), expected, rtol=0, atol=0
    )
    assert not (tmp_path / "missing-observations").exists()
    assert set(torch.load(path, weights_only=True)) == {
        "format",
        "version",
        "context",
        "state_dict",
        "epoch",
        "global_step",
    }
    save_state(path, runtime.model, context, epoch=3, global_step=18)
    assert P3SLoader(path, device="cpu").global_step == 18
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.p3s"]


def test_snapshot_rejects_old_format(tmp_path):
    path = tmp_path / "old.p3s"
    torch.save({"format": "prom3theus.joint_save_state", "version": 1}, path)
    with pytest.raises(ArtifactExportError, match="Unsupported"):
        load_validated_save_state(path)


def test_nonfinite_save_keeps_previous_snapshot(tmp_path):
    runtime, _ = _runtime(tmp_path)
    path = tmp_path / "state.p3s"
    context = snapshot_context(runtime)
    save_state(path, runtime.model, context, epoch=0, global_step=1)
    before = path.read_bytes()
    with torch.no_grad():
        next(runtime.model.parameters()).fill_(float("nan"))
    with pytest.raises(ArtifactExportError, match="non-finite"):
        save_state(path, runtime.model, context, epoch=0, global_step=2)
    assert path.read_bytes() == before


def test_third_observation_type_registers_without_runner_changes(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from typing import Literal
    from prom3theus.config.schema import ConfigNode
    from prom3theus.config.registry import (
        register_stream_schema,
        OBSERVATION_SCHEMAS,
        TERM_SCHEMAS,
        COMPATIBLE_PAIRS,
    )
    from prom3theus.config import parse_config
    from prom3theus.components import forward
    from prom3theus.inversion.data_terms.base import (
        ObservationDataTerm,
        DataTermBatchResult,
    )
    from prom3theus.inversion.joint import JointForwardModel
    from prom3theus.training.joint import JointInversionModule
    from prom3theus.config.joint_schema import JointTrainingConfig

    @dataclass(frozen=True)
    class TemperatureObservation(ConfigNode):
        type: Literal["temperature_probe"]

    @dataclass(frozen=True)
    class TemperatureTermConfig(ConfigNode):
        type: Literal["temperature_sample"]
        weight: float

    class TemperatureTerm(ObservationDataTerm):
        observation_kind = "point"

        def __init__(self, atmosphere, scale):
            super().__init__()
            object.__setattr__(self, "atmosphere", atmosphere)
            self.scale = torch.nn.Parameter(torch.tensor(scale))

        def predict(self, batch):
            return (
                self.atmosphere.evaluate_chart_height_points(
                    batch["coordinates"], batch["height"]
                )["temperature"]
                * self.scale
            )

        def evaluate_batch(self, batch):
            loss = (self.predict(batch) - batch["target"]).square().mean()
            return DataTermBatchResult(loss, {}, {}, len(batch["target"]))

    def build(stream, loaded, atmosphere, scene):
        term = TemperatureTerm(atmosphere, 1.0)
        term.construction = {"type": "temperature_sample", "options": {"scale": 1.0}}
        return term

    monkeypatch.setitem(
        forward._PROVIDERS,
        "temperature_sample",
        forward.ForwardProvider(
            build,
            lambda options, atmosphere, scene: TemperatureTerm(atmosphere, **options),
        ),
    )
    monkeypatch.setitem(
        OBSERVATION_SCHEMAS, "temperature_probe", TemperatureObservation
    )
    monkeypatch.setitem(TERM_SCHEMAS, "temperature_sample", TemperatureTermConfig)
    monkeypatch.setattr(
        "prom3theus.config.registry.COMPATIBLE_PAIRS",
        COMPATIBLE_PAIRS | {("temperature_probe", "temperature_sample")},
    )
    runtime, _ = _runtime(tmp_path)
    document = runtime.config.to_dict()
    document["scene"]["reference_stream"] = "probe"
    document["training"]["reference_stream"] = "probe"
    document["streams"] = [
        {
            "id": "probe",
            "observation": {"type": "temperature_probe"},
            "data_term": {"type": "temperature_sample", "weight": 1.0},
        }
    ]
    config = parse_config(document, base_directory=tmp_path)
    term = forward._default_term_builder(
        config.streams[0], None, runtime.model.atmosphere_model, runtime.scene
    )
    model = JointForwardModel(
        runtime.model.atmosphere_model, {"probe": term}, {"probe": 1.0}
    )
    batch = {
        "coordinates": torch.zeros(2, 3),
        "height": torch.full((2,), 1e7),
        "target": torch.zeros(2),
    }
    loss = model.evaluate_batches({"probe": batch}).loss
    loss.backward()
    assert term.scale.grad is not None
    assert any(p.grad is not None for p in model.atmosphere_model.parameters())
    context = snapshot_context(runtime)
    context["configuration"] = config.to_dict()
    context["terms"] = {"probe": term.construction}
    context["streams"] = {"probe": context["streams"]["coronal_euv"]}
    path = tmp_path / "probe.p3s"
    save_state(path, model, context, epoch=0, global_step=1)
    with torch.no_grad():
        expected = term.predict(batch)
    loader = P3SLoader(path, device="cpu")
    torch.testing.assert_close(loader.predict("probe", batch), expected)

    # The same provider also participates in the actual runner, including
    # a non-raster observation source, validation, and both durable outputs.
    from prom3theus.components import observations as observation_components
    from prom3theus.observations import ObservationDescriptor, PreparedObservationStream
    from prom3theus.application.joint_contracts import (
        LoadedJointStream,
        JointRunnerFactories,
        SharedTermAssembly,
    )
    from prom3theus.application.joint_training import run_joint_inversion
    from prom3theus.config.schema import LoggingConfig

    class PointData:
        def setup(self, stage=None):
            pass

        def run_metadata(self):
            return {"sample_count": 2}

        def train_dataloader(self):
            return [batch]

        def val_dataloader(self):
            return [batch]

    store = tmp_path / "point-store"
    store.mkdir()
    prepared = PreparedObservationStream(
        "probe",
        ObservationDescriptor(
            "probe", "temperature_probe", "temperature_sensor", "point"
        ),
        PointData(),
        store,
        "b" * 64,
    )

    def load_point(stream, resources, work_directory, **kwargs):
        return LoadedJointStream(
            prepared,
            SimpleNamespace(metadata=lambda: {"type": "temperature_probe"}),
            (),
            scene_contract=runtime.scene,
            sampling_support=lambda scene, chunk_size: iter(
                [(torch.tensor([[0.0, 0.0, scene.solar_radius_m]]), torch.zeros(1))]
            ),
        )

    monkeypatch.setitem(
        observation_components._PROVIDERS,
        "temperature_probe",
        observation_components.ObservationProvider(load_point, lambda stream: ()),
    )
    run_config = replace(
        config,
        solver=replace(
            config.solver,
            output_directory=tmp_path / "output",
            work_directory=tmp_path / "work",
        ),
        logging=LoggingConfig(type="disabled"),
        training=replace(
            config.training,
            max_steps=2,
            max_epochs=-1,
            device="cpu",
            checkpoint_every_n_steps=1,
            validation_every_n_steps=1,
        ),
    )
    run = run_joint_inversion(
        run_config,
        factories=JointRunnerFactories(
            atmosphere_builder=lambda *args: runtime.model.atmosphere_model,
            shared_terms_builder=lambda *args: SharedTermAssembly({}),
        ),
    )
    assert run.trainer.global_step == 2
    assert sorted(p.name for p in (tmp_path / "output").iterdir()) == [
        "last.ckpt",
        "state.p3s",
    ]
    restored = P3SLoader(run.save_state_path, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(
            restored.predict("probe", batch),
            run.module.model.terms["probe"].predict(batch),
        )
    assert (tmp_path / "work/validation/step_00000002/metrics.json").is_file()


def test_stokes_snapshot_restores_prediction_and_export_facade(tmp_path):
    runtime, _ = _runtime(tmp_path)
    config = load_config(ROOT / "configs/hinode_lte_mhs.yaml")
    config = replace(
        config,
        atmosphere=replace(
            config.atmosphere,
            network=replace(
                config.atmosphere.network, hidden_dimension=8, hidden_layers=1
            ),
        ),
    )
    stream = config.streams[0]
    config = replace(
        config,
        streams=(
            replace(
                stream,
                data_term=replace(
                    stream.data_term,
                    depth_sampling=replace(
                        stream.data_term.depth_sampling,
                        sample_count=8,
                        coarse_to_fine=replace(
                            stream.data_term.depth_sampling.coarse_to_fine,
                            fine_sample_count=4,
                        ),
                    ),
                ),
            ),
        ),
    )
    stream = config.streams[0]
    scene = replace(
        runtime.scene,
        height_bounds_m=(
            config.atmosphere.geometry.inner_height_megameter * 1e6,
            config.atmosphere.geometry.outer_height_megameter * 1e6,
        ),
    )
    atmosphere = _default_atmosphere_builder(config, scene, runtime.resources)
    loaded = next(iter(runtime.streams.values()))
    loaded.specification = SimpleNamespace(
        observation_id="hinode",
        instrument_options={},
        wavelength_angstrom=torch.linspace(6301.0, 6303.0, 25),
        continuum_indices=(0, 24),
        radiance_scale_w_m3_sr=1e13,
        velocity_synthesis_mode=SimpleNamespace(value="carrington_observer_relative"),
        metadata=lambda: {"observation_id": "hinode"},
    )
    term = _default_term_builder(stream, loaded, atmosphere, scene)
    runtime = SimpleNamespace(
        config=config,
        scene=scene,
        resources=runtime.resources,
        model=EvaluationModel(atmosphere, {stream.id: term}).eval(),
        streams={stream.id: loaded},
    )
    batch = {
        "coordinates": torch.zeros(2, 3),
        "ray_direction": torch.tensor([[0.0, 0.0, -1.0]] * 2),
        "stokes_basis": torch.eye(3).expand(2, 3, 3),
        "observer_los_velocity_m_per_s": torch.zeros(2),
    }
    with torch.no_grad():
        expected = term.predict(batch)
    assert torch.isfinite(expected).all()
    path = tmp_path / "state.p3s"
    save_state(path, runtime.model, snapshot_context(runtime), epoch=0, global_step=1)
    loader = P3SLoader(path, device="cpu")
    torch.testing.assert_close(
        loader.predict(stream.id, batch), expected, rtol=0, atol=0
    )
    with torch.no_grad():
        exported = loader.module.synthesize(
            batch["coordinates"],
            ray_direction=batch["ray_direction"],
            stokes_basis=batch["stokes_basis"],
            observer_los_velocity_m_per_s=batch["observer_los_velocity_m_per_s"],
        )["stokes"]
    torch.testing.assert_close(exported, expected, rtol=0, atol=0)
    assert loader.module.sample_depth_grid().numel() == 8


def test_old_potential_sampling_does_not_prevent_p3s_queries(monkeypatch, tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace

    from prom3theus.artifacts import loader as implementation
    from prom3theus.config import load_config

    configuration = load_config('configs/hmi_aia_dynamic.yaml').to_dict()
    potential = configuration['physics']['potential_boundary']
    for key in ('photosphere', 'top_grid_size', 'side_horizontal_points', 'side_height_points', 'seed'):
        potential.pop(key)
    potential.update(interior={'enabled': True, 'points': 1024}, top_points=256,
                     side_points=512, normalization_height_count=32, normalization_points_per_height=64)
    original = deepcopy(configuration)
    state = SimpleNamespace(context={'configuration': configuration}, path=tmp_path / 'state.p3s',
                            module=SimpleNamespace(terms={'photospheric_stokes': None}))
    monkeypatch.setattr(implementation, 'load_validated_save_state', lambda path: state)
    monkeypatch.setattr(implementation, 'StreamEvaluationView', lambda *args: SimpleNamespace(eval=lambda: None))
    loaded = implementation.P3SLoader(state.path)
    assert loaded.config.atmosphere.to_dict() == configuration['atmosphere']
    assert loaded.state.context['configuration'] == original
