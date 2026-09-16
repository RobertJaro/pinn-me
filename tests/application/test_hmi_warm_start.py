"""Explicit HMI network transfers must not restore an old physical model."""

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prom3theus.application import hmi_warm_start as implementation
from prom3theus.config import load_config


class Atmosphere(torch.nn.Module):
    def __init__(self, marker):
        super().__init__()
        self.network = torch.nn.Linear(2, 1)
        self.register_buffer("geometry_marker", torch.tensor(marker))


@pytest.fixture
def warm_start(tmp_path, monkeypatch):
    source_path = tmp_path / "source" / "state.p3s"
    source_path.parent.mkdir()
    source_path.write_bytes(b"fixed source snapshot")
    source_config = load_config(Path(__file__).resolve().parents[2] / "configs/hmi_local_fine.yaml")
    target = replace(source_config,
        solver=replace(source_config.solver, output_directory=tmp_path / "new_run"),
        atmosphere=replace(source_config.atmosphere, parameters=replace(
            source_config.atmosphere.parameters, magnetic_field=replace(
                source_config.atmosphere.parameters.magnetic_field, reference_height_megameter=0.15))),
    )
    scene = SimpleNamespace(atmosphere_coordinate_metadata={"basis": [[1.0, 0.0]], "height_scale": 3e6})
    source_model = Atmosphere(17)
    with torch.no_grad():
        source_model.network.weight.fill_(2.5)
        source_model.network.bias.fill_(-0.4)
    source = SimpleNamespace(config=source_config, state=SimpleNamespace(scene=scene),
        module=SimpleNamespace(atmosphere_model=source_model), global_step=4000, epoch=1,
        stream_id=source_config.scene.reference_stream, observation=SimpleNamespace(source_signature="signature"))

    def load(path, *, device):
        assert path == source_path and device == "cpu"
        return source

    def build(config, scene, resources):
        model = Atmosphere(99)
        model.reference_height = config.atmosphere.parameters.magnetic_field.reference_height_megameter
        return model

    monkeypatch.setattr(implementation, "P3SLoader", load)
    monkeypatch.setattr(implementation, "_default_atmosphere_builder", build)
    return target, source_path, scene, source


def test_transfers_only_network_weights_and_records_fresh_run_provenance(warm_start):
    target, path, scene, source = warm_start
    factories, provenance = implementation.build_hmi_warm_start_factories(target, path)
    model = factories.atmosphere_builder(target, scene, {})
    for name, expected in source.module.atmosphere_model.network.state_dict().items():
        torch.testing.assert_close(model.network.state_dict()[name], expected, rtol=0, atol=0)
    assert model.reference_height == 0.15
    assert model.geometry_marker.item() == 99
    assert provenance["fresh_optimizer"] is True
    assert provenance["source_global_step"] == 4000
    assert provenance["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["target_reference_height_megameter"] == 0.15
    assert not torch.optim.Adam(model.parameters()).state


@pytest.mark.parametrize("change", ["network", "geometry", "observation"])
def test_rejects_incompatible_scientific_configuration(warm_start, change):
    target, path, _, _ = warm_start
    if change == "network":
        target = replace(target, atmosphere=replace(target.atmosphere,
            network=replace(target.atmosphere.network, hidden_dimension=129)))
    elif change == "geometry":
        target = replace(target, atmosphere=replace(target.atmosphere,
            geometry=replace(target.atmosphere.geometry, height_input_scale_m=1e6)))
    else:
        stream = target.streams[0]
        target = replace(target, streams=(replace(stream,
            observation=replace(stream.observation, directory=path.parent / "different_crop")),))
    with pytest.raises(ValueError, match="must match"):
        implementation.build_hmi_warm_start_factories(target, path)


@pytest.mark.parametrize("condition", ["checkpoint", "explicit_resume", "same_directory"])
def test_rejects_resume_or_source_output_reuse(warm_start, condition):
    target, path, _, _ = warm_start
    if condition == "checkpoint":
        target.solver.output_directory.mkdir()
        (target.solver.output_directory / "last.ckpt").write_bytes(b"existing checkpoint")
    elif condition == "explicit_resume":
        target = replace(target, training=replace(target.training, resume_from_checkpoint=path))
    else:
        target = replace(target, solver=replace(target.solver, output_directory=path.parent))
    with pytest.raises(ValueError, match="checkpoint|separate"):
        implementation.build_hmi_warm_start_factories(target, path)


def test_rejects_changed_runtime_coordinate_geometry(warm_start):
    target, path, _, _ = warm_start
    factories, _ = implementation.build_hmi_warm_start_factories(target, path)
    scene = SimpleNamespace(atmosphere_coordinate_metadata={"basis": [[0.0, 1.0]], "height_scale": 3e6})
    with pytest.raises(ValueError, match="coordinate geometry"):
        factories.atmosphere_builder(target, scene, {})
