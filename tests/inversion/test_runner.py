"""Application-runner integration at the typed observation/artifact boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

import prom3theus.inversion.runner as runner
from prom3theus.artifacts.model import load_manifest
from prom3theus.config import load_config
from prom3theus.observations import (
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    ObservationRaster,
    ObservationSpec,
    ObservationStore,
    observation_store_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _raster() -> ObservationRaster:
    return ObservationRaster(
        stokes=torch.zeros(1, 2, 4, 2),
        wavelength_angstrom=torch.tensor([6301.0, 6302.0]),
        coordinates=torch.zeros(1, 2, 3),
        ray_direction=torch.tensor([[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
        surface_position_m=torch.tensor([[[6.96e8, 0.0, 0.0], [6.96e8, 0.0, 0.0]]]),
        stokes_basis=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        .expand(1, 2, 3, 3)
        .clone(),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        metadata={
            "coordinates": {
                "network_affine": {
                    "center_mm": [0.0, 0.0],
                    "scale_mm": [1.0, 1.0],
                },
                "time_affine": {"center_hours": 0.0, "scale_hours": 1.0},
            },
            "ray_geometry": {
                "solar_radius_m": 6.96e8,
                "scene_basis_rows": torch.eye(3).tolist(),
            },
        },
    )


class _DataModule:
    def __init__(self, raster: ObservationRaster, store: Path):
        self.raster = raster
        self.observation_store_path = store

    @staticmethod
    def run_metadata() -> dict:
        return {"prepared_by": "runner-test"}


class _Module(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.synthesizer = SimpleNamespace(lines=())
        values = deepcopy(kwargs)
        values["wavelength_angstrom"] = torch.as_tensor(
            values["wavelength_angstrom"]
        ).tolist()
        self.hparams = values


class _Trainer:
    latest = None

    def __init__(self, **kwargs):
        self.options = kwargs
        self.fit_call = None
        type(self).latest = self

    def fit(self, module, *, datamodule):
        self.fit_call = (module, datamodule)


def test_runner_embeds_the_exact_observation_store(tmp_path, monkeypatch):
    loaded = load_config(PROJECT_ROOT / "configs" / "hinode_lte_mhs.yaml")
    config = replace(
        loaded,
        solver=replace(
            loaded.solver,
            output_directory=tmp_path / "runs",
            work_directory=tmp_path / "work",
        ),
        runtime=replace(
            loaded.runtime,
            validation_check_interval_steps=None,
        ),
    )
    resources = {
        "bundle_schema_version": 2,
        "stic_lookup_bounds": {
            "temperature_log10_k": [3.4, 4.0],
            "gas_pressure_log10_pa": [-1.5, 6.0],
        },
        "falc_top_boundary": {"gravity_cm_s2": 27_542.287},
    }
    spec = ObservationSpec(
        observation_id="runner-test",
        observation_type="hinode_sp",
        instrument_type="hinode_sp",
        wavelength_angstrom=torch.tensor([6301.0, 6302.0]),
        continuum_indices=(0, 1),
        radiance_scale_w_m3_sr=1.0,
        velocity_synthesis_mode=CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    )
    signature = observation_store_signature(
        config.observation.to_dict(),
        resources,
        source_files_sha256="a" * 64,
    )
    source_store = ObservationStore.save(
        tmp_path / "source.observation",
        _raster(),
        source_signature=signature,
        metadata={
            "adapter": "hinode_sp",
            "observation": spec.metadata(),
            "validation_raster_index": 0,
        },
    )
    data = _DataModule(_raster(), source_store)
    call = {}

    def load_observation(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return data, spec, SimpleNamespace(name="hinode_sp")

    monkeypatch.setattr(runner, "validate_resource_bundle", lambda: resources)
    monkeypatch.setattr(runner, "load_or_prepare_observation", load_observation)
    monkeypatch.setattr(
        runner,
        "resolve_instrument_config",
        lambda value, **_: dict(value),
    )
    monkeypatch.setattr(runner, "_physics_sampling", lambda *args: None)
    monkeypatch.setattr(runner, "_logger", lambda config: False)
    monkeypatch.setattr(runner, "_visualization_callback", lambda config: None)
    monkeypatch.setattr(runner, "LTEInversionModule", _Module)
    monkeypatch.setattr(runner, "Trainer", _Trainer)

    result = runner.run_inversion(config, rebuild_observations=True)

    assert call["kwargs"]["rebuild"] is True
    assert _Trainer.latest.fit_call == (result.module, data)
    assert _Trainer.latest.options["gradient_clip_val"] == 0.1
    assert "val_check_interval" not in _Trainer.latest.options
    manifest = load_manifest(result.artifact_directory)
    assert manifest.observation["store"]["path"] == "observations"
    assert manifest.model["run_metadata"]["prepared_by"] == "runner-test"
    assert manifest.model["run_metadata"]["adapter"] == "hinode_sp"
    embedded, names, metadata = ObservationStore.load_sequence(
        result.artifact_directory / "observations"
    )
    assert len(embedded) == 1
    assert names == ["raster_0000"]
    assert metadata["observation"] == spec.metadata()


def test_extrapolation_builds_distinct_full_and_upper_physics_domains():
    config = load_config(PROJECT_ROOT / "configs" / "hinode_lte_mhs_extrapolation.yaml")
    atmosphere = runner._atmosphere_config(config)
    physics = runner._physics_config(config)
    data = SimpleNamespace(
        observation_sampling_bounds={
            "surface_longitude_center_rad": 0.2,
            "surface_longitude_offset_rad": [-0.1, 0.1],
            "surface_latitude_rad": [-0.05, 0.05],
            "time_hours": [0.0, 0.0],
            "solar_radius_m": 695_700_000.0,
        }
    )

    metadata = runner._physics_sampling(physics, data, atmosphere)

    assert physics["sampling_domain"]["height_Mm"] == [-0.1, 20.0]
    assert physics["upper_sampling_domain"]["height_Mm"] == [1.5, 20.0]
    assert metadata["line_formation_height_bounds_Mm"] == [1.5, -0.1]


def test_visualization_outputs_are_kept_with_the_durable_run(tmp_path):
    loaded = load_config(PROJECT_ROOT / "configs" / "hmi_lte_dynamic.yaml")
    config = replace(
        loaded,
        solver=replace(
            loaded.solver,
            output_directory=tmp_path / "output",
            work_directory=tmp_path / "scratch",
        ),
    )

    callback = runner._visualization_callback(config)

    assert callback.output_directory == (tmp_path / "output" / "diagnostics").resolve()
