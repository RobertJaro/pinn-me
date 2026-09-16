"""Strict, additive configuration checks for setup-only joint runs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from prom3theus.config import (
    AIAObservationConfig,
    AIAOpticallyThinDataTermConfig,
    ConfigError,
    JointInversionConfig,
    LTEStokesDataTermConfig,
    load_config,
    parse_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"




def test_coronal_energy_requires_coronal_domain_and_dynamic_model():
    document = yaml.safe_load((CONFIG_DIRECTORY / "hmi_aia_dynamic.yaml").read_text())
    energy = document["physics"]["equations"]["coronal_energy"]
    energy.update(enabled=True, weight=0.1, cooling_table="cooling.json")
    config = parse_config(document, base_directory=CONFIG_DIRECTORY)
    assert config.physics.equations.coronal_energy.enabled
    energy["minimum_height_megameter"] = 2.0
    with pytest.raises(ConfigError, match="above the transition region"):
        parse_config(document, base_directory=CONFIG_DIRECTORY)


@pytest.fixture()
def joint_document() -> dict:
    legacy = yaml.safe_load(
        (CONFIG_DIRECTORY / "hmi_lte_dynamic.yaml").read_text(encoding="utf-8")
    )
    return {
        "schema_version": 3,
        "solver": {
            "kind": "joint",
            "output_directory": "../runs/hmi-aia-dry-run",
            "work_directory": "../runs/hmi-aia-dry-run-work",
        },
        "scene": {"reference_stream": "photospheric_stokes"},
        "atmosphere": deepcopy(legacy["atmosphere"]),
        "physics": deepcopy(legacy["physics"]),
        "atmosphere_regularization": [
            {
                "id": "lte_support",
                "type": "stic_table_support",
                "temperature_weight": 0.01,
                "gas_pressure_weight": 0.01,
                "sample_count": 64,
                "height_layers": 8,
            }
        ],
        "streams": [
            {
                "id": "photospheric_stokes",
                "observation": deepcopy(legacy["streams"][0]["observation"]),
                "data_term": {
                    "type": "lte_stokes",
                    "weight": 1.0,
                    "synthesis": deepcopy(legacy["streams"][0]["data_term"]["synthesis"]),
                    "instrument": deepcopy(legacy["streams"][0]["data_term"]["instrument"]),
                    "objective": deepcopy(legacy["streams"][0]["data_term"]["objective"]),
                    "depth_sampling": deepcopy(legacy["streams"][0]["data_term"]["depth_sampling"]),
                },
            },
            {
                "id": "coronal_euv",
                "observation": {
                    "type": "aia_euv",
                    "directory": "../data/sdo/sample/prepared/aia",
                    "channels_angstrom": [171, 193, 211],
                    "selection": {
                        "validation_exposure_group": "20240323T221158388-hmi"
                    },
                    "loader": {
                        "batch_size": 256,
                        "validation_batch_size": 512,
                        "workers": 0,
                        "pin_memory": False,
                    },
                },
                "data_term": {
                    "type": "aia_optically_thin",
                    "weight": 0.1,
                    "synthesis": {
                        "response_resource": ("aia_euv_v1:aia_temperature_response"),
                        "ray_end": "atmosphere_outer_shell",
                        "ray_samples": 192,
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
                "enabled": True,
                "render_atmosphere": True,
                "render_streams": True,
                "ray_sampling": {
                    "batch_size": 512,
                    "max_pixels": 4096,
                    "max_profile_samples": 2048,
                },
                "slice_sampling": {
                    "batch_size": 512,
                    "layer_count": 6,
                    "longitude_points": 64,
                    "latitude_points": 64,
                    "radial_points": 96,
                },
                "meridional_slice": {
                    "enabled": True,
                    "longitude_deg": 215.0,
                },
                "contribution_ray_count": 8,
                "dpi": 120,
            }
        },
        "dry_run": {
            "device": "cpu",
            "evaluate_gradients": True,
            "gradient_sample_count": 16,
            "max_samples_per_stream": 1024,
            "quadrature_samples": [96, 192, 384],
            "report_filename": "dry_run.json",
        },
    }


def test_complete_joint_contract_is_typed_and_json_serializable(joint_document):
    config = parse_config(
        joint_document,
        base_directory=CONFIG_DIRECTORY,
        environ={},
    )

    assert isinstance(config, JointInversionConfig)
    assert config.schema_version == 3
    assert config.solver.kind == "joint"
    assert config.solver.output_directory == PROJECT_ROOT / "runs/hmi-aia-dry-run"
    assert config.scene.reference_stream == "photospheric_stokes"
    assert isinstance(config.streams[0].data_term, LTEStokesDataTermConfig)
    assert isinstance(config.streams[1].observation, AIAObservationConfig)
    assert isinstance(config.streams[1].data_term, AIAOpticallyThinDataTermConfig)
    assert config.streams[1].observation.channels_angstrom == (171, 193, 211)
    assert config.dry_run.quadrature_samples == (96, 192, 384)
    assert config.atmosphere_regularization[0].id == "lte_support"
    json.dumps(config.to_dict())


def test_load_config_dispatches_schema_three(tmp_path, joint_document):
    path = tmp_path / "joint.yaml"
    path.write_text(yaml.safe_dump(joint_document), encoding="utf-8")

    config = load_config(path, environ={})

    assert isinstance(config, JointInversionConfig)
    assert (
        config.solver.output_directory
        == (tmp_path / "../runs/hmi-aia-dry-run").resolve()
    )




@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["streams"][1].update({"id": "Coronal-EUV"}),
            "stream id",
        ),
        (
            lambda value: value["streams"][1].update({"id": "photospheric_stokes"}),
            "stream ids must be unique",
        ),
        (
            lambda value: value["scene"].update({"reference_stream": "missing"}),
            "reference_stream",
        ),
        (
            lambda value: value["streams"][1]["data_term"].update(
                {"type": "unsupported"}
            ),
            "data_term.type",
        ),
    ],
)
def test_stream_identity_and_discriminators_are_strict(
    joint_document,
    mutate,
    message,
):
    mutate(joint_document)

    with pytest.raises(ConfigError, match=message):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


@pytest.mark.parametrize(
    "channels",
    ([171, 211], [171, 193, 193], [171, 193, 335]),
)
def test_aia_objective_must_cover_the_exact_supported_channel_set(
    joint_document,
    channels,
):
    joint_document["streams"][1]["observation"]["channels_angstrom"] = channels

    with pytest.raises(ConfigError, match="channels|channel_weights"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_aia_channel_weights_cannot_hide_native_pixel_count(joint_document):
    weights = joint_document["streams"][1]["data_term"]["objective"]["channel_weights"]
    weights.pop()

    with pytest.raises(ConfigError, match="exactly every configured"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_joint_schema_rejects_mismatched_stokes_instrument(joint_document):
    joint_document["streams"][0]["data_term"]["instrument"] = {
        "type": "hinode_sp",
        "spectral_psf": {
            "type": "gaussian",
            "fwhm_angstrom": 0.0243,
            "oversample": 4,
            "truncate_sigma": 4.0,
        },
    }

    with pytest.raises(ConfigError, match="incompatible observation/instrument"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_dry_run_quadrature_includes_the_operational_ray_grid(joint_document):
    joint_document["dry_run"]["quadrature_samples"] = [96, 384]

    with pytest.raises(ConfigError, match="include every configured AIA ray_samples"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


@pytest.mark.parametrize(
    "filename", ["reports/dry_run.json", "dry_run.txt", "../x.json"]
)
def test_dry_run_report_is_a_local_json_basename(joint_document, filename):
    joint_document["dry_run"]["report_filename"] = filename

    with pytest.raises(ConfigError, match="JSON basename"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_unknown_schema_versions_are_rejected_before_dataclass_decoding(joint_document):
    joint_document["schema_version"] = 4

    with pytest.raises(ConfigError, match=r"schema_version.*3"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_schema_version_must_be_present_and_an_integer(joint_document):
    joint_document.pop("schema_version")
    with pytest.raises(ConfigError, match="schema_version is required"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})

    joint_document["schema_version"] = True
    with pytest.raises(ConfigError, match="must be an integer"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_single_stream_uses_the_joint_contract():
    config = load_config(CONFIG_DIRECTORY / "hinode_lte_mhs.yaml", environ={})

    assert type(config).__name__ == "JointInversionConfig"
    assert config.schema_version == 3
    assert len(config.streams) == 1


def test_atmosphere_rejects_removed_optical_depth_grid(joint_document):
    joint_document['atmosphere']['depth_grid'] = {
        'coordinate': 'log_tau500', 'minimum': -5.0, 'maximum': 1.0, 'count': 51,
    }
    with pytest.raises(ConfigError, match='depth_grid'):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY)
