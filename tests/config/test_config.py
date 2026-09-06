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
from prom3theus.config.schema import (
    VectorRegularizationConfig,
    VisualizationConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"
CONFIG_PATHS = sorted(CONFIG_DIRECTORY.glob("*.yaml"))


@pytest.fixture(scope="module")
def raw_hinode() -> dict:
    return yaml.safe_load((CONFIG_DIRECTORY / "hinode_lte_mhs.yaml").read_text())


def test_all_public_configs_load_with_explicit_discriminators():
    assert [path.name for path in CONFIG_PATHS] == [
        "hinode_lte_mhs.yaml",
        "hinode_lte_mhs_extrapolation.yaml",
        "hmi_lte_dynamic.yaml",
        "hmi_lte_mhs.yaml",
    ]
    environ = {
        "PROM3THEUS_RUNS_DIR": "/must/not/redirect/maintained/configs",
        "PROM3THEUS_WORK_DIR": "/must/not/redirect/maintained/configs",
        "PROM3THEUS_DATA_DIR": "/must/not/redirect/maintained/configs",
        "PROM3THEUS_CALIBRATION_DIR": "/must/not/redirect/maintained/configs",
    }

    hinode = load_config(CONFIG_DIRECTORY / "hinode_lte_mhs.yaml", environ=environ)
    extrapolation = load_config(
        CONFIG_DIRECTORY / "hinode_lte_mhs_extrapolation.yaml", environ=environ
    )
    hmi = load_config(CONFIG_DIRECTORY / "hmi_lte_dynamic.yaml", environ=environ)
    hmi_mhs = load_config(CONFIG_DIRECTORY / "hmi_lte_mhs.yaml", environ=environ)

    assert hinode.schema_version == hmi.schema_version == 2
    assert extrapolation.atmosphere.geometry.outer_height_megameter == 20.0
    assert extrapolation.atmosphere.upper_atmosphere.coronal_temperature_k == 1.0e6
    assert (
        extrapolation.atmosphere.geometry.line_formation_outer_height_megameter == 1.5
    )
    assert hinode.solver.kind == hmi.solver.kind == "lte"
    assert isinstance(hinode.observation, HinodeObservationConfig)
    assert isinstance(hmi.observation, HMIObservationConfig)
    assert hinode.observation.type == hinode.instrument.type == "hinode_sp"
    assert hmi.observation.type == "hmi_stokes"
    assert hmi.instrument.type == "hmi_filter_profiles"
    assert hmi.atmosphere.geometry.time_dependent is True
    assert hmi.atmosphere.geometry.outer_height_megameter == 50.0
    assert hmi.atmosphere.geometry.line_formation_outer_height_megameter == 1.5
    assert hmi.atmosphere.upper_atmosphere.coronal_temperature_k == 1.0e6
    assert hmi.observation.selection.acquisition_indices is None
    assert not hmi.training.physics.equations.adiabatic_pressure.enabled
    assert hmi.training.physics.equations.adiabatic_pressure.weight == 0.0
    assert hmi.instrument.optimize_line_of_sight_velocity_correction is True
    assert hmi.training.physics.equations.momentum.weight == pytest.approx(3.0e-3)
    assert hmi.training.physics.equations.magnetic_divergence.weight == pytest.approx(
        1.0
    )
    assert hmi.training.physics.equations.induction.weight == pytest.approx(3.0e-2)
    assert hmi.training.physics.equations.continuity.weight == pytest.approx(1.0e-2)
    assert hmi.training.physics.equations.upper_boundary_open_velocity.weight == (
        pytest.approx(1.0e-2)
    )
    assert hmi.training.physics.equations.side_boundary_open_velocity.weight == (
        pytest.approx(1.0e-3)
    )
    assert hmi.training.physics.equations.side_boundary_current_free.weight == (
        pytest.approx(1.0)
    )
    assert hmi.training.physics.collocation.side_boundary_points_per_step == 512
    assert hmi.training.physics.collocation.side_height_layers_per_step == 32
    assert hmi.training.physics.equations.upper_domain_temperature_prior.weight == (
        pytest.approx(1.0e-3)
    )
    assert hmi.training.physics.equations.upper_domain_microturbulence_prior.weight == (
        pytest.approx(1.0e-3)
    )
    assert hmi.training.physics.equations.upper_boundary_current_free.enabled is True
    assert hmi.training.physics.equations.upper_boundary_current_free.weight == (
        pytest.approx(1.0)
    )
    assert not hmi.training.physics.equations.radial_magnetic_energy_gradient.enabled
    assert hmi.training.physics.equations.radial_magnetic_energy_gradient.weight == 0.0
    assert hmi.training.physics.equations.upper_boundary_gas_pressure_prior.weight == (
        pytest.approx(1.0e-2)
    )
    assert not (
        extrapolation.training.physics.equations.radial_magnetic_energy_gradient.enabled
    )
    assert hmi.training.physics.collocation.upper_height_layers_per_step == 32
    assert hmi.training.physics.adiabatic_index == pytest.approx(5.0 / 3.0)
    assert hmi_mhs.atmosphere.geometry.time_dependent is False
    assert hmi_mhs.observation.selection.acquisition_indices == (10,)
    assert hmi_mhs.observation.selection.validation_raster == 10
    assert hmi.diagnostics.visualization.meridional_slice.enabled
    assert hmi.diagnostics.visualization.meridional_slice.longitude_deg == 215.0
    assert hmi_mhs.diagnostics.visualization.meridional_slice.enabled
    assert hmi_mhs.diagnostics.visualization.meridional_slice.longitude_deg == 215.0
    assert hmi_mhs.training.physics.equations.magnetohydrostatic_equilibrium.enabled
    assert not hmi_mhs.instrument.optimize_line_of_sight_velocity_correction
    assert not hinode.instrument.optimize_line_of_sight_velocity_correction
    assert not hmi_mhs.training.physics.equations.adiabatic_pressure.enabled
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
        "/glade/work/rjarolim/lte/hmi_subframe_dynamic_extrapolation_v03"
    )
    assert hmi.solver.work_directory == Path(
        "/glade/derecho/scratch/rjarolim/lte/hmi_subframe_dynamic_extrapolation_v03"
    )
    assert hmi.observation.directory == Path(
        "/glade/work/rjarolim/data/hmi_stokes/20240323_720s_subframe"
    )
    assert hmi.observation.calibration.transmission_profile_directory == Path(
        "/glade/work/rjarolim/data/hmi_calibration/20240323"
    )


def test_hmi_server_paths_are_literal_and_environment_independent():
    config = load_config(CONFIG_DIRECTORY / "hmi_lte_dynamic.yaml", environ={})

    assert config.solver.output_directory == Path(
        "/glade/work/rjarolim/lte/hmi_subframe_dynamic_extrapolation_v03"
    )
    assert config.resources.bundle == "packaged"
    assert config.observation.directory == Path(
        "/glade/work/rjarolim/data/hmi_stokes/20240323_720s_subframe"
    )


def test_to_dict_returns_a_plain_runtime_mapping():
    config = load_config(CONFIG_DIRECTORY / "hinode_lte_mhs.yaml", environ={})
    runtime = config.to_dict()

    assert runtime["schema_version"] == 2
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
    ] == {"enabled": True, "weight": 1.0e-1}
    assert isinstance(runtime["solver"]["output_directory"], str)
    assert isinstance(runtime["synthesis"]["line_ids"], list)
    json.dumps(runtime)


def test_validation_step_interval_is_optional(raw_hinode):
    document = deepcopy(raw_hinode)
    document["runtime"].pop("validation_check_interval_steps", None)

    config = parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})

    assert config.runtime.validation_check_interval_steps is None


def test_explicit_validation_step_interval_must_remain_positive(raw_hinode):
    document = deepcopy(raw_hinode)
    document["runtime"]["validation_check_interval_steps"] = 0

    with pytest.raises(ConfigError, match="validation_check_interval_steps"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_non_extrapolation_defaults_are_optional_and_neutral(raw_hinode):
    document = deepcopy(raw_hinode)
    geometry = document["atmosphere"]["geometry"]
    expected_line_formation_top = geometry["outer_height_megameter"]
    geometry.pop("line_formation_outer_height_megameter")

    physics = document["training"]["physics"]
    physics.pop("upper_boundary_current_free_ramp_steps")
    collocation = physics["collocation"]
    for name in (
        "upper_volume_points_per_step",
        "upper_height_layers_per_step",
        "validation_upper_height_layers",
        "side_boundary_points_per_step",
        "side_height_layers_per_step",
    ):
        collocation.pop(name)
    equations = physics["equations"]
    for name in (
        "upper_boundary_open_velocity",
        "side_boundary_open_velocity",
        "side_boundary_current_free",
        "upper_domain_microturbulence_prior",
        "upper_domain_temperature_prior",
        "upper_boundary_current_free",
        "radial_magnetic_energy_gradient",
    ):
        equations.pop(name, None)

    config = parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})

    assert (
        config.atmosphere.geometry.line_formation_outer_height_megameter
        == expected_line_formation_top
    )
    assert config.training.physics.upper_boundary_current_free_ramp_steps == 0
    assert config.training.physics.collocation.upper_volume_points_per_step == 0
    assert config.training.physics.collocation.upper_height_layers_per_step == 0
    assert config.training.physics.collocation.validation_upper_height_layers == 0
    assert not config.training.physics.equations.upper_boundary_open_velocity.enabled
    assert not config.training.physics.equations.side_boundary_open_velocity.enabled
    assert not config.training.physics.equations.side_boundary_current_free.enabled
    assert (
        not config.training.physics.equations.upper_domain_microturbulence_prior.enabled
    )
    assert not config.training.physics.equations.upper_domain_temperature_prior.enabled
    assert not config.training.physics.equations.radial_magnetic_energy_gradient.enabled
    assert not config.training.physics.equations.upper_boundary_current_free.enabled


def test_operational_defaults_are_optional(raw_hinode):
    document = deepcopy(raw_hinode)
    loader = document["observation"]["loader"]
    for name in (
        "validation_batch_size",
        "validation_stride",
        "preparation_workers",
        "workers",
        "pin_memory",
        "progress",
    ):
        loader.pop(name)
    document["observation"]["calibration"].pop("stokes_reference_angle_deg")
    document["atmosphere"]["network"]["encoding"].pop("include_input")
    document["synthesis"].pop("excluded_wavelength_windows_angstrom")
    document["instrument"].pop("line_of_sight_velocity_correction_m_per_s", None)
    document["training"].pop("vector_regularization")
    document["diagnostics"].pop("validation_every_n_epochs")
    document["diagnostics"]["visualization"] = {}
    document["runtime"].pop("log_every_n_steps")
    document["runtime"].pop("gradient_clip_norm")
    document["logging"].pop("tags")

    config = parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})

    assert config.observation.loader.validation_batch_size is None
    assert config.observation.loader.validation_stride == 1
    assert config.observation.loader.preparation_workers == 1
    assert config.observation.loader.workers == 0
    assert config.observation.loader.pin_memory is False
    assert config.observation.loader.progress is True
    assert config.observation.calibration.stokes_reference_angle_deg == 0.0
    assert config.atmosphere.network.encoding.include_input is True
    assert config.synthesis.excluded_wavelength_windows_angstrom == ()
    assert config.instrument.line_of_sight_velocity_correction_m_per_s == 0.0
    assert config.training.vector_regularization == VectorRegularizationConfig()
    assert config.diagnostics.validation_every_n_epochs == 1
    assert config.diagnostics.visualization == VisualizationConfig()
    assert config.runtime.log_every_n_steps == 50
    assert config.runtime.gradient_clip_norm == pytest.approx(0.1)
    assert config.logging.tags == ()


def test_gradient_clipping_can_be_disabled_explicitly(raw_hinode):
    document = deepcopy(raw_hinode)
    document["runtime"]["gradient_clip_norm"] = None

    config = parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})

    assert config.runtime.gradient_clip_norm is None


def test_hmi_validation_raster_defaults_to_first_acquisition():
    document = yaml.safe_load(
        (CONFIG_DIRECTORY / "hmi_lte_dynamic.yaml").read_text(encoding="utf-8")
    )
    document["observation"]["selection"].pop("validation_raster")

    config = parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})

    assert config.observation.selection.validation_raster == 0


def test_induction_requires_an_optimized_instrument_velocity_correction():
    document = yaml.safe_load(
        (CONFIG_DIRECTORY / "hmi_lte_dynamic.yaml").read_text(encoding="utf-8")
    )
    document["instrument"]["optimize_line_of_sight_velocity_correction"] = False
    with pytest.raises(
        ConfigError, match="optimize_line_of_sight_velocity_correction=true"
    ):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_static_inversions_reject_los_zero_point_fitting(raw_hinode):
    document = deepcopy(raw_hinode)
    document["instrument"]["optimize_line_of_sight_velocity_correction"] = True

    with pytest.raises(ConfigError, match="Static inversions"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_hmi_acquisition_selection_must_include_validation_raster():
    document = yaml.safe_load(
        (CONFIG_DIRECTORY / "hmi_lte_dynamic.yaml").read_text(encoding="utf-8")
    )
    document["observation"]["selection"] = {
        "acquisition_indices": [0],
        "validation_raster": 10,
    }

    with pytest.raises(ConfigError, match="selected acquisition_indices"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_disabled_features_do_not_require_unused_options(raw_hinode):
    document = deepcopy(raw_hinode)
    document["training"]["depth_sampling"]["coarse_to_fine"] = {"enabled": False}
    document["logging"] = {"type": "disabled"}

    config = parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})

    refinement = config.training.depth_sampling.coarse_to_fine
    assert refinement.fine_sample_count == 1
    assert refinement.uniform_weight_floor == 0.0
    assert config.logging.project is None
    assert config.logging.run_name is None
    assert config.logging.tags == ()


@pytest.mark.parametrize("missing", ["fine_sample_count", "uniform_weight_floor"])
def test_enabled_coarse_to_fine_requires_refinement_options(raw_hinode, missing):
    document = deepcopy(raw_hinode)
    document["training"]["depth_sampling"]["coarse_to_fine"].pop(missing)

    with pytest.raises(ConfigError, match="enabled coarse-to-fine"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


def test_wandb_logging_requires_names(raw_hinode):
    document = deepcopy(raw_hinode)
    document["logging"].pop("project")

    with pytest.raises(ConfigError, match="W&B logging requires"):
        parse_config(document, base_directory=CONFIG_DIRECTORY, environ={})


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
        (lambda value: value.update({"schema_version": 1}), "schema_version"),
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
        "line_of_sight_velocity_correction_m_per_s": 0.0,
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
            "layer counts must be at least three",
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
        ("instrument", "line_of_sight_velocity_correction_m_per_s"),
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
