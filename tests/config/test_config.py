from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from prom3theus.config import (
    ConfigError,
    HMIObservationConfig,
    HinodeObservationConfig,
    EnvironmentResolutionError,
    expand_environment,
    load_config,
    parse_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"
CONFIG_PATHS = sorted(CONFIG_DIRECTORY.glob("*.yaml"))


@pytest.fixture(scope="module")
def raw_hinode() -> dict:
    return yaml.safe_load((CONFIG_DIRECTORY / "hinode_lte_mhs.yaml").read_text())


def test_exactly_two_public_configs_load_with_explicit_discriminators():
    assert [path.name for path in CONFIG_PATHS] == [
        "hinode_lte_mhs.yaml",
        "hmi_lte_subframe.yaml",
    ]
    environ = {
        "PROM3THEUS_RUNS_DIR": "/must/not/redirect/maintained/configs",
        "PROM3THEUS_WORK_DIR": "/must/not/redirect/maintained/configs",
        "PROM3THEUS_DATA_DIR": "/must/not/redirect/maintained/configs",
        "PROM3THEUS_CALIBRATION_DIR": "/must/not/redirect/maintained/configs",
    }

    hinode = load_config(CONFIG_DIRECTORY / "hinode_lte_mhs.yaml", environ=environ)
    hmi = load_config(CONFIG_DIRECTORY / "hmi_lte_subframe.yaml", environ=environ)

    assert hinode.schema_version == hmi.schema_version == 1
    assert hinode.solver.kind == hmi.solver.kind == "lte"
    assert isinstance(hinode.observation, HinodeObservationConfig)
    assert isinstance(hmi.observation, HMIObservationConfig)
    assert hinode.observation.type == hinode.instrument.type == "hinode_sp"
    assert hmi.observation.type == "hmi_stokes"
    assert hmi.instrument.type == "hmi_filter_profiles"
    assert hinode.solver.output_directory == Path(
        "/glade/work/rjarolim/lte/hinode_mhs_20110214_000004_v03"
    )
    assert hinode.solver.work_directory == Path(
        "/glade/derecho/scratch/rjarolim/lte/hinode_mhs_20110214_000004_v03"
    )
    assert hinode.observation.directory == Path(
        "/glade/work/rjarolim/data/inversion/hinode_2011_02/20110214_000004"
    )
    assert hmi.solver.output_directory == Path(
        "/glade/work/rjarolim/spinn_me/hmi/subframe_20240323_physics_mlp_v09"
    )
    assert hmi.solver.work_directory == Path(
        "/glade/derecho/scratch/rjarolim/spinn_me/subframe_20240323_mlp_v08"
    )
    assert hmi.observation.directory == Path(
        "/glade/work/rjarolim/data/hmi_stokes/20240323_720s_subframe"
    )
    assert hmi.observation.calibration.transmission_profile_directory == Path(
        "/glade/work/rjarolim/data/hmi_calibration/20240323"
    )


def test_hmi_server_paths_are_literal_and_environment_independent():
    config = load_config(CONFIG_DIRECTORY / "hmi_lte_subframe.yaml", environ={})

    assert config.solver.output_directory == Path(
        "/glade/work/rjarolim/spinn_me/hmi/subframe_20240323_physics_mlp_v09"
    )
    assert config.resources.bundle == "packaged"
    assert config.observation.directory == Path(
        "/glade/work/rjarolim/data/hmi_stokes/20240323_720s_subframe"
    )


def test_to_dict_returns_a_plain_runtime_mapping():
    config = load_config(CONFIG_DIRECTORY / "hinode_lte_mhs.yaml", environ={})
    runtime = config.to_dict()

    assert runtime["schema_version"] == 1
    assert runtime["solver"]["kind"] == "lte"
    assert runtime["observation"]["type"] == "hinode_sp"
    assert runtime["instrument"]["type"] == "hinode_sp"
    assert runtime["atmosphere"]["network"]["activation"] == "silu"
    assert runtime["observation"]["selection"] == {
        "slit_slice": {"start": 0, "stop": 512}
    }
    assert runtime["atmosphere"]["geometry"]["time_dependent"] is False
    assert runtime["training"]["physics"]["equations"][
        "magnetohydrostatic_equilibrium"
    ] == {"enabled": True, "weight": 1.0e-3}
    assert isinstance(runtime["solver"]["output_directory"], str)
    assert isinstance(runtime["synthesis"]["line_ids"], list)
    json.dumps(runtime)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"unexpected": True}), "unknown keys"),
        (
            lambda value: value["training"]["physics"].update(
                {"unexpected_weight": 1.0}
            ),
            "unknown keys",
        ),
        (
            lambda value: value["observation"].pop("type"),
            "observation.type is required",
        ),
        (lambda value: value["solver"].update({"kind": "unsupported"}), "solver.kind"),
        (lambda value: value.update({"schema_version": 2}), "schema_version"),
    ],
)
def test_strict_schema_rejects_unknown_missing_and_unsupported_values(
    raw_hinode,
    mutate,
    message,
):
    document = deepcopy(raw_hinode)
    mutate(document)

    with pytest.raises(ConfigError, match=message):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_observation_and_instrument_types_must_be_compatible(raw_hinode):
    document = deepcopy(raw_hinode)
    document["instrument"] = {
        "type": "hmi_filter_profiles",
        "radial_velocity_correction_m_per_s": 0.0,
    }

    with pytest.raises(ConfigError, match="incompatible observation/instrument"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(path)


def test_environment_placeholders_are_strict():
    assert expand_environment("${ROOT:-relative}/data", environ={}) == "relative/data"
    assert expand_environment("${ROOT}/data", environ={"ROOT": "/srv"}) == "/srv/data"
    with pytest.raises(EnvironmentResolutionError, match="ROOT"):
        expand_environment("${ROOT}/data", environ={})


def test_public_configs_contain_no_unintended_home_directory_paths():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in CONFIG_PATHS)
    assert "/Users/" not in combined
    assert "/home/" not in combined


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("atmosphere", "network", "activation"), "relu", "activation"),
        (("atmosphere", "network", "encoding", "type"), "identity", "type"),
        (("atmosphere", "geometry", "tangent_margin_m"), 0.0, "tangent_margin_m"),
        (("atmosphere", "reference_atmosphere"), "other", "reference_atmosphere"),
    ],
)
def test_schema_rejects_values_the_runtime_cannot_construct(
    raw_hinode, path, value, message
):
    document = deepcopy(raw_hinode)
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_fourier_axes_must_match_time_dependent_coordinate_dimension(raw_hinode):
    document = deepcopy(raw_hinode)
    document["atmosphere"]["network"]["encoding"]["num_frequencies"] = [16] * 4
    document["atmosphere"]["network"]["encoding"]["max_frequencies"] = [16] * 4

    with pytest.raises(ConfigError, match="one entry per atmosphere coordinate"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_shell_must_bracket_reference_surface_when_depth_grid_crosses_zero(
    raw_hinode,
):
    document = deepcopy(raw_hinode)
    document["atmosphere"]["geometry"]["inner_height_megameter"] = 0.1

    with pytest.raises(ConfigError, match="crossing log_tau500=0"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("training", "physics", "collocation", "height_layers_per_step"),
            30,
            "divisible",
        ),
        (
            ("training", "physics", "collocation", "validation_height_layers"),
            1,
            "at least two validation",
        ),
        (
            ("diagnostics", "visualization", "slice_sampling", "layer_count"),
            1,
            "counts must be at least two",
        ),
        (("diagnostics", "visualization", "dpi"), 49, "dpi must be at least 50"),
    ],
)
def test_schema_enforces_runtime_sampling_constraints(raw_hinode, path, value, message):
    document = deepcopy(raw_hinode)
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ConfigError, match=message):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_force_balance_equations_are_mutually_exclusive(raw_hinode):
    document = deepcopy(raw_hinode)
    document["training"]["physics"]["equations"]["hydrostatic_equilibrium"] = {
        "enabled": True,
        "weight": 1.0e-3,
    }

    with pytest.raises(ConfigError, match="mutually exclusive"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_vector_equations_require_matching_spatial_basis(raw_hinode):
    document = deepcopy(raw_hinode)
    document["training"]["physics"]["vector_basis_matches_spatial_coordinates"] = False

    with pytest.raises(ConfigError, match="vector_basis_matches_spatial_coordinates"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_temporal_physics_requires_a_time_dependent_atmosphere(raw_hinode):
    document = deepcopy(raw_hinode)
    document["training"]["physics"]["equations"]["magnetohydrostatic_equilibrium"] = {
        "enabled": False,
        "weight": 0.0,
    }
    document["training"]["physics"]["equations"]["momentum"] = {
        "enabled": True,
        "weight": 1.0e-3,
    }

    with pytest.raises(ConfigError, match="Temporal LTE physics"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


@pytest.mark.parametrize(
    "path",
    [
        ("atmosphere", "parameters", "velocity", "maximum_m_per_s"),
        ("instrument", "radial_velocity_correction_m_per_s"),
    ],
)
def test_configured_velocity_limits_must_be_subluminal(raw_hinode, path):
    document = deepcopy(raw_hinode)
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = 299_792_458.0

    with pytest.raises(ConfigError, match="subluminal"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})
