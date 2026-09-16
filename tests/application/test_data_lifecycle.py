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
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(10.0, 10.0),
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
