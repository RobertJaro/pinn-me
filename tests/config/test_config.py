"""One strict stream schema for every maintained inversion."""
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from prom3theus.config import (
    ConfigError,
    load_config,
    parse_config,
    expand_environment,
    EnvironmentResolutionError,
)
from prom3theus.muram import config as _register_muram_schema

ROOT = Path(__file__).resolve().parents[2]


def test_vector_potential_scale_reaches_model_options():
    from prom3theus.application.configuration import atmosphere_options

    document = load_config(ROOT / "configs/hinode/mhs.yaml").to_dict()
    document["atmosphere"]["parameters"]["magnetic_field"]["scale_vector_potential"] = 0.01
    config = parse_config(document, base_directory=ROOT)
    assert atmosphere_options(config.atmosphere)["scale_vector_potential"] == 0.01
    del document["atmosphere"]["parameters"]["magnetic_field"]["scale_vector_potential"]
    default = parse_config(document, base_directory=ROOT)
    assert atmosphere_options(default.atmosphere)["scale_vector_potential"] == 1.0


@pytest.mark.parametrize("scale", [0.0, -0.01, float("nan"), float("inf")])
def test_vector_potential_scale_must_be_positive_and_finite(scale):
    document = load_config(ROOT / "configs/muram/hinode/mhs.yaml").to_dict()
    document["atmosphere"]["parameters"]["magnetic_field"]["scale_vector_potential"] = scale
    with pytest.raises(ConfigError, match="scale_vector_potential"):
        parse_config(document, base_directory=ROOT)


@pytest.mark.parametrize("factor", [-0.1, 1.1, float("nan"), float("inf")])
def test_current_penalty_final_factor_is_bounded(factor):
    document = load_config(ROOT / "configs/sdo/hmi_aia_dynamic.yaml").to_dict()
    document["physics"]["magnetic_current_free_final_factor"] = factor
    with pytest.raises(ConfigError, match="magnetic_current_free_final_factor"):
        parse_config(document, base_directory=ROOT)


def test_current_penalty_final_factor_requires_an_enabled_schedule():
    document = load_config(ROOT / "configs/sdo/hmi_aia_dynamic.yaml").to_dict()
    document["physics"]["magnetic_current_free_final_factor"] = 0.1
    document["physics"]["magnetic_current_free_steps"] = 0
    document["physics"]["equations"]["magnetic_current_free"]["enabled"] = False
    document["physics"]["equations"]["magnetic_current_free"]["weight"] = 0.0
    with pytest.raises(ConfigError, match="magnetic_current_free_final_factor"):
        parse_config(document, base_directory=ROOT)
    document["physics"].pop("magnetic_current_free_final_factor")
    config = parse_config(document, base_directory=ROOT)
    assert config.physics.magnetic_current_free_final_factor == 0.0


@pytest.mark.parametrize(
    "path", sorted(path for path in (ROOT / "configs").rglob("*.yaml")
                   if path.name not in {"synthesis.yaml", "surrogate.yaml"}),
    ids=lambda path: path.relative_to(ROOT / "configs").as_posix(),
)
def test_all_runs_use_stream_schema(path):
    config = load_config(path)
    assert config.schema_version == 5
    assert config.solver.kind == "joint"
    assert config.training is not None
    assert config.streams
    assert parse_config(config.to_dict(), base_directory=path.parent) == config
    assert config.solver.output_directory != config.solver.work_directory


def test_stokes_only_preset_omits_physics():
    config = load_config(ROOT / "configs/muram/hinode/stokes_only.yaml")
    assert config.physics is None
    assert config.atmosphere_regularization == ()


def test_omitted_physics_equations_are_disabled():
    document = load_config(ROOT / "configs/muram/hinode/mhs.yaml").to_dict()
    document["physics"].pop("equations")
    config = parse_config(document, base_directory=ROOT)
    assert all(not equation["enabled"] and equation["weight"] == 0
               for equation in config.physics.equations.to_dict().values())


def test_sparse_physics_equations_only_enable_explicit_residuals():
    document = load_config(ROOT / "configs/muram/hinode/mhs.yaml").to_dict()
    document["physics"]["equations"] = {"hydrostatic_equilibrium": {"enabled": True, "weight": .1}}
    config = parse_config(document, base_directory=ROOT)
    active = [name for name, equation in config.physics.equations.to_dict().items() if equation["enabled"]]
    assert active == ["hydrostatic_equilibrium"]


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_removed_schemas_are_rejected(version):
    document = yaml.safe_load((ROOT / "configs/sdo/hmi_aia_dynamic.yaml").read_text())
    document["schema_version"] = version
    with pytest.raises(ConfigError, match="schema_version"):
        parse_config(document, base_directory=ROOT)


@pytest.mark.parametrize(
    "mutation", ["unknown", "missing", "incompatible", "reference"]
)
def test_invalid_stream_contract_is_rejected(mutation):
    document = yaml.safe_load((ROOT / "configs/sdo/hmi_aia_dynamic.yaml").read_text())
    if mutation == "unknown":
        document["legacy"] = True
    if mutation == "missing":
        del document["streams"]
    if mutation == "incompatible":
        document["streams"][0]["observation"]["type"] = "aia_euv"
    if mutation == "reference":
        document["training"]["reference_stream"] = "absent"
    with pytest.raises(ConfigError):
        parse_config(document, base_directory=ROOT)


def test_duplicate_keys_and_missing_environment_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 4\nschema_version: 4\n")
    with pytest.raises(ConfigError):
        load_config(path)
    with pytest.raises(EnvironmentResolutionError):
        expand_environment("${MISSING}", environ={})
