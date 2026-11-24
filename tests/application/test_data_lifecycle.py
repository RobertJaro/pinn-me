"""One source stream must be finished before the next source is loaded."""

from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

from prom3theus.application.data import load_joint_data
from prom3theus.application.joint_contracts import LoadedJointStream
from prom3theus.application.joint_streams import _default_scene_builder
from prom3theus.observations import (
    ImageObservationRaster,
    ImageObservationSpec,
    StoredImageDataModule,
    ObservationDescriptor,
    PreparedObservationStream,
    SceneContract,
)
from prom3theus.observations.arrays import ArrayRef
from prom3theus.observations.tensor_dataset import TensorDiskDataset


@pytest.mark.parametrize("for_training", [True, False])
def test_finish_each_stream_before_loading_next_and_release_raw_payload(
    tmp_path, monkeypatch, for_training
):
    events, raw_payloads = [], []
    scene = SceneContract(
        scene_basis=torch.eye(3),
        solar_radius_m=6.957e8,
        reference_time_tai_seconds=1.7e9,
        spatial_coordinate_center_megameter=(0.0, 0.0),
        spatial_coordinate_scale_megameter=(10.0, 10.0),
        time_coordinate_center_hours=0.0,
        time_coordinate_scale_hours=1.0,
        height_bounds_m=(0.0, 1e7),
    )

    def option(name):
        return SimpleNamespace(
            id=name,
            observation=SimpleNamespace(
                to_dict=lambda: {"type": "fixture", "id": name}
            ),
        )

    config = SimpleNamespace(
        streams=(option("second"), option("reference")),
        scene=SimpleNamespace(
            reference_stream="reference",
            to_dict=lambda: {"reference_stream": "reference"},
        ),
        atmosphere=SimpleNamespace(geometry=SimpleNamespace(to_dict=lambda: {})),
        solver=SimpleNamespace(work_directory=tmp_path),
        training=SimpleNamespace(
            device="cpu",
            reader_workers=2,
            loader_block_bytes=4096,
            loader_cache_bytes=8192,
            loader_prefetch_batches=2,
            loader_max_batch_bytes=8192,
            loader_seed=3,
        ),
    )

    def load(stream, *args, **kwargs):
        if stream.id == "second":
            assert events[-1] == ("persisted", "reference")
            gc.collect()
            assert all(ref() is None for ref in raw_payloads)
        events.append(("load", stream.id))
        intensity = torch.arange(64).float().reshape(2, 32)
        raw_payloads.append(weakref.ref(intensity))
        ray = torch.zeros(2, 32, 3)
        ray[..., 2] = -1
        raster = ImageObservationRaster(
            intensity=intensity,
            uncertainty=None,
            ray_direction=ray,
            surface_position_m=-ray * scene.solar_radius_m,
            valid_mask=torch.ones(2, 32, dtype=torch.bool),
            absolute_tai_seconds=1.7e9,
            channel_angstrom=171,
            exposure_group="one",
            metadata={
                "intensity_unit": "DN s-1 pixel-1",
                "ray_geometry": {"solar_radius_m": scene.solar_radius_m},
            },
        )
        spec = ImageObservationSpec(
            observation_id=stream.id,
            observation_type="aia_euv",
            instrument_type="aia_temperature_response",
            intensity_unit="DN s-1 pixel-1",
            channels_angstrom=(171,),
            exposure_groups=("one",),
            calibration_convention={"degradation_correction": "applied_once"},
            geometry_convention={"ray_direction": "observer_to_sun"},
            required_resource_sets=("aia_euv_v1",),
        )
        data = StoredImageDataModule(
            [raster], ["one"], spec, validation_exposure_group="one", batch_size=8
        )
        (tmp_path / stream.id).mkdir()
        prepared = PreparedObservationStream(
            name=stream.id,
            descriptor=ObservationDescriptor.from_spec(spec, observation_kind="image"),
            data_module=data,
            store_path=tmp_path / stream.id,
            source_signature="a" * 64,
        )
        return LoadedJointStream(prepared, spec, (raster,), scene_contract=scene)

    original_build = TensorDiskDataset.from_batches.__func__

    def packed(cls, *args, **kwargs):
        events.append(("shuffle", events[-1][1]))
        return original_build(cls, *args, **kwargs)

    monkeypatch.setattr(TensorDiskDataset, "from_batches", classmethod(packed))
    from prom3theus.observations import persistence

    original_persist = persistence.persist_prepared_stream

    def persist(stream, *args, **kwargs):
        result = original_persist(stream, *args, **kwargs)
        assert isinstance(result[0].rasters[0].intensity, ArrayRef)
        events.append(("persisted", stream.prepared.name))
        return result

    monkeypatch.setattr(persistence, "persist_prepared_stream", persist)
    module = load_joint_data(
        config,
        {},
        load,
        _default_scene_builder,
        for_training=for_training,
        compute_summaries=False,
    )
    expected = [
        ("load", "reference"),
        ("shuffle", "reference"),
        ("persisted", "reference"),
        ("load", "second"),
        ("shuffle", "second"),
        ("persisted", "second"),
    ]
    assert events == [
        event for event in expected if for_training or event[0] != "shuffle"
    ]
    if not for_training:
        module = load_joint_data(
            config,
            {},
            lambda *a, **k: pytest.fail("reloaded source for training upgrade"),
            _default_scene_builder,
            for_training=True,
            compute_summaries=False,
        )
    gc.collect()
    assert all(ref() is None for ref in raw_payloads)
    assert list(module.streams) == ["second", "reference"]
    for loader in module.train_dataloader().values():
        try:
            values = torch.cat([batch["intensity"] for batch in loader])
            torch.testing.assert_close(values.sort().values, torch.arange(64).float())
            assert not torch.equal(values, torch.arange(64).float())
        finally:
            loader.close()
    restored = load_joint_data(
        config,
        {},
        lambda *a, **k: pytest.fail("reloaded source"),
        _default_scene_builder,
        for_training=True,
        compute_summaries=False,
    )
    assert restored.generation == module.generation


def test_training_starts_logger_before_preparation_and_launch(tmp_path, monkeypatch):
    from prom3theus.application import joint_training
    from prom3theus.core import distributed
    from unittest.mock import Mock

    events = []
    logger = Mock()
    config = SimpleNamespace(
        training=SimpleNamespace(device="cpu", devices=1, resume_from_checkpoint=None),
        solver=SimpleNamespace(output_directory=tmp_path),
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("prom3theus.observations.persistence.global_rank", lambda: 0)

    def start_logger(config):
        events.append("logger")
        return logger

    def launch(*args, **kwargs):
        events.append("launch")
        return 0, 1

    class StopAfterAssembly(Exception):
        pass

    def assemble(config, *, before_data_handoff, **kwargs):
        events.append("data")
        before_data_handoff()
        events.append("model")
        raise StopAfterAssembly

    monkeypatch.setattr(joint_training, "_online_logger", start_logger)
    monkeypatch.setattr(distributed, "initialize_distributed", launch)
    monkeypatch.setattr(joint_training, "build_joint_runtime", assemble)
    with pytest.raises(StopAfterAssembly):
        joint_training.run_joint_inversion(config)
    assert events == ["logger", "data", "launch", "model"]
    logger.finalize.assert_called_once_with("failed")
    logger.log_metrics.assert_not_called()


def test_failed_preparation_does_not_launch_workers(tmp_path):
    from prom3theus.observations.persistence import restore_or_create_data_module
    from unittest.mock import Mock

    launch = Mock()

    def build():
        raise ValueError("invalid input")

    with pytest.raises(ValueError, match="invalid input"):
        restore_or_create_data_module(tmp_path / "data_module.pt", build,
                                      before_handoff=launch)
    launch.assert_not_called()


def test_enabling_potential_backfills_only_missing_summaries(tmp_path, monkeypatch):
    from dataclasses import replace
    from prom3theus.application import data
    from prom3theus.config import load_config
    from prom3theus.observations.persistence import JointDataModule

    config = load_config("configs/hinode/mhs_extrapolation.yaml")
    config = replace(config, solver=replace(config.solver, work_directory=tmp_path))
    stream = config.streams[0]
    stored = SimpleNamespace(
        prepared=SimpleNamespace(descriptor=SimpleNamespace(observation_kind="stokes")),
        data_module=SimpleNamespace(),
    )
    module = JointDataModule({stream.id: stored}, object(), summaries={},
                             contract=data.data_contract(config, {}))
    scans, changes = [], []

    def restore(path, build, *, finalize, **kwargs):
        changes.append(finalize(module))
        return module

    monkeypatch.setattr(data, "restore_or_create_data_module", restore)
    monkeypatch.setattr(data, "_joint_observation_bounds", lambda *args: scans.append("bounds") or {"x": 1})
    monkeypatch.setattr(data, "_observation_times_hours", lambda *args: scans.append("times") or [0.])
    load_joint_data(config, {}, None, None)
    assert scans == ["bounds"]
    potential = replace(config.physics.potential_boundary, enabled=True)
    # Isolate the preparation path from the preset's incompatible outer equations.
    physics = replace(config.physics, potential_boundary=potential, equations=type(config.physics.equations)())
    config = replace(config, physics=physics)
    load_joint_data(config, {}, None, None)
    assert scans == ["bounds"] and module.summaries["stokes_bounds"] == {"x": 1}
    config = replace(config, atmosphere=replace(config.atmosphere, geometry=replace(
        config.atmosphere.geometry, time_dependent=True)))
    module.contract = data.data_contract(config, {})
    load_joint_data(config, {}, None, None)
    load_joint_data(config, {}, None, None)
    assert scans == ["bounds", "times"]
    assert changes == [True, True, True, False]
