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
from prom3theus.muram import config as _register_muram_schema


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"


@pytest.mark.parametrize("boundary", ["upper", "side"])
def test_current_free_boundaries_allow_static_atmospheres(boundary):
    document = yaml.safe_load(
        (CONFIG_DIRECTORY / "muram/hinode/mhs.yaml").read_text()
    )
    document["atmosphere"]["geometry"]["time_dependent"] = False
    name = f"{boundary}_boundary_current_free"
    document["physics"]["equations"] = {
        name: {"enabled": True, "weight": 0.0001},
    }

    config = parse_config(document, base_directory=CONFIG_DIRECTORY)

    assert not config.atmosphere.geometry.time_dependent
    assert getattr(config.physics.equations, name).enabled


@pytest.mark.parametrize("devices", [0, -1, True, 1.5, "all"])
def test_training_devices_requires_positive_integer(devices):
    from prom3theus.config.joint_schema import JointTrainingConfig

    with pytest.raises((ValueError, TypeError), match="devices"):
        JointTrainingConfig(max_steps=2, learning_rate=1e-4,
                            final_learning_rate=1e-5, devices=devices)


def test_training_uses_all_visible_devices_by_default():
    from prom3theus.config.joint_schema import JointTrainingConfig

    assert JointTrainingConfig(max_steps=2, learning_rate=1e-4,
                               final_learning_rate=1e-5).devices == "auto"


@pytest.mark.parametrize("max_steps,max_epochs", [(-1, -1), (100, -1), (-1, 3)])
def test_training_accepts_disabled_stopping_limits(max_steps, max_epochs):
    from prom3theus.config.joint_schema import JointTrainingConfig

    settings = JointTrainingConfig(
        max_steps=max_steps, max_epochs=max_epochs,
        learning_rate=1e-4, final_learning_rate=1e-5,
    )
    assert settings.max_steps == max_steps
    assert settings.max_epochs == max_epochs


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_learning_rate_decay_requires_positive_integer_steps(steps):
    from prom3theus.config.joint_schema import JointTrainingConfig

    with pytest.raises((ValueError, TypeError), match="learning_rate_decay_steps"):
        JointTrainingConfig(max_steps=-1, learning_rate=1e-4,
                            final_learning_rate=1e-5,
                            learning_rate_decay_steps=steps)


def test_induction_accepts_known_fixed_velocity_calibration(joint_document):
    joint_document["streams"][0]["data_term"]["instrument"]["optimize_line_of_sight_velocity_correction"] = False
    config = parse_config(joint_document, base_directory=CONFIG_DIRECTORY)
    assert config.physics.equations.induction.enabled
    instrument = config.streams[0].data_term.instrument
    assert not instrument.optimize_line_of_sight_velocity_correction
    assert instrument.line_of_sight_velocity_correction_m_per_s == 0.0


@pytest.mark.parametrize("names", [[""], ["shared.physics.momentum"] * 2])
def test_loss_balance_rejects_invalid_inclusions(joint_document, names):
    joint_document["training"] = {
        "max_steps": 1, "learning_rate": 0.01, "final_learning_rate": 0.001,
        "loss_balance": {"included_objectives": names},
    }
    with pytest.raises(ConfigError, match="included_objectives"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY)




@pytest.mark.parametrize("path", sorted(path for path in CONFIG_DIRECTORY.rglob("*.yaml")
                                      if path.name not in {"synthesis.yaml", "surrogate.yaml"}),
                         ids=lambda p: p.relative_to(CONFIG_DIRECTORY).as_posix())
def test_presets_explicitly_include_only_nonboundary_objectives(path):
    config = load_config(path, environ={})
    if config.training is None or not config.training.loss_balance.enabled:
        return
    included = config.training.loss_balance.included_objectives
    assert included
    assert all("boundary" not in key for key in included)




def test_coronal_energy_requires_coronal_domain_and_dynamic_model():
    document = load_config(CONFIG_DIRECTORY / "sdo/hmi_aia_energy.yaml").to_dict()
    energy = document["physics"]["equations"]["coronal_energy"]
    energy.update(enabled=True, weight=0.1, minimum_height_megameter=4.0)
    config = parse_config(document, base_directory=CONFIG_DIRECTORY)
    assert config.physics.equations.coronal_energy.enabled
    energy["minimum_height_megameter"] = 2.0
    with pytest.raises(ConfigError, match="above the transition region"):
        parse_config(document, base_directory=CONFIG_DIRECTORY)


@pytest.fixture()
def joint_document() -> dict:
    legacy = load_config(CONFIG_DIRECTORY / "sdo/hmi_aia_dynamic.yaml").to_dict()
    return {
        "schema_version": 5,
        "solver": {
            "kind": "joint",
            "output_directory": "../runs/hmi-aia",
            "work_directory": "../runs/hmi-aia-work",
        },
        "scene": {"reference_stream": "photospheric_stokes"},
        "atmosphere": deepcopy(legacy["atmosphere"]),
        "physics": deepcopy(legacy["physics"]),
        "synthesis": deepcopy(legacy["synthesis"]),
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
                    "synthesis_window": legacy["streams"][0]["data_term"]["synthesis_window"],
                    "instrument": deepcopy(legacy["streams"][0]["data_term"]["instrument"]),
                    "objective": deepcopy(legacy["streams"][0]["data_term"]["objective"]),

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
    }


def test_complete_joint_contract_is_typed_and_json_serializable(joint_document):
    config = parse_config(
        joint_document,
        base_directory=CONFIG_DIRECTORY,
        environ={},
    )

    assert isinstance(config, JointInversionConfig)
    assert config.schema_version == 5
    assert config.solver.kind == "joint"
    assert config.solver.output_directory == PROJECT_ROOT / "runs/hmi-aia"
    assert config.scene.reference_stream == "photospheric_stokes"
    assert isinstance(config.streams[0].data_term, LTEStokesDataTermConfig)
    assert isinstance(config.streams[1].observation, AIAObservationConfig)
    assert isinstance(config.streams[1].data_term, AIAOpticallyThinDataTermConfig)
    assert config.streams[1].observation.channels_angstrom == (171, 193, 211)
    assert config.atmosphere_regularization[0].id == "lte_support"
    json.dumps(config.to_dict())


def test_load_config_reads_joint_yaml(tmp_path, joint_document):
    path = tmp_path / "joint.yaml"
    path.write_text(yaml.safe_dump(joint_document), encoding="utf-8")

    config = load_config(path, environ={})

    assert isinstance(config, JointInversionConfig)
    assert (
        config.solver.output_directory
        == (tmp_path / "../runs/hmi-aia").resolve()
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


def test_removed_dry_run_options_are_rejected(joint_document):
    joint_document["dry_run"] = {}
    with pytest.raises(ConfigError, match="dry_run"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_unknown_schema_versions_are_rejected_before_dataclass_decoding(joint_document):
    joint_document["schema_version"] = 3

    with pytest.raises(ConfigError, match=r"schema_version.*5"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_schema_version_must_be_present_and_an_integer(joint_document):
    joint_document.pop("schema_version")
    with pytest.raises(ConfigError, match="schema_version is required"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})

    joint_document["schema_version"] = True
    with pytest.raises(ConfigError, match="must be an integer"):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY, environ={})


def test_single_stream_uses_the_joint_contract():
    config = load_config(CONFIG_DIRECTORY / "hinode/mhs.yaml", environ={})

    assert type(config).__name__ == "JointInversionConfig"
    assert config.schema_version == 5
    assert len(config.streams) == 1


def test_atmosphere_rejects_removed_optical_depth_grid(joint_document):
    joint_document['atmosphere']['depth_grid'] = {
        'coordinate': 'log_tau500', 'minimum': -5.0, 'maximum': 1.0, 'count': 51,
    }
    with pytest.raises(ConfigError, match='depth_grid'):
        parse_config(joint_document, base_directory=CONFIG_DIRECTORY)
