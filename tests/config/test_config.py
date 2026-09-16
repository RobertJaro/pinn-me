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

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("factor", [-0.1, 1.1, float("nan"), float("inf")])
def test_current_penalty_final_factor_is_bounded(factor):
    document = load_config(ROOT / "configs/hmi_lte_dynamic.yaml").to_dict()
    document["physics"]["magnetic_current_free_final_factor"] = factor
    with pytest.raises(ConfigError, match="magnetic_current_free_final_factor"):
        parse_config(document, base_directory=ROOT)


def test_current_penalty_final_factor_requires_an_enabled_schedule():
    document = load_config(ROOT / "configs/hmi_lte_dynamic.yaml").to_dict()
    document["physics"]["magnetic_current_free_steps"] = 0
    document["physics"]["equations"]["magnetic_current_free"]["enabled"] = False
    document["physics"]["equations"]["magnetic_current_free"]["weight"] = 0.0
    with pytest.raises(ConfigError, match="magnetic_current_free_final_factor"):
        parse_config(document, base_directory=ROOT)
    document["physics"].pop("magnetic_current_free_final_factor")
    config = parse_config(document, base_directory=ROOT)
    assert config.physics.magnetic_current_free_final_factor == 0.0


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda path: path.stem
)
def test_all_runs_use_stream_schema(path):
    config = load_config(path)
    assert config.schema_version == 3
    assert config.solver.kind == "joint"
    assert config.training is not None
    assert config.streams
    assert parse_config(config.to_dict(), base_directory=path.parent) == config
    assert config.solver.output_directory != config.solver.work_directory


@pytest.mark.parametrize("version", [1, 2, 4])
def test_removed_schemas_are_rejected(version):
    document = yaml.safe_load((ROOT / "configs/hmi_lte_dynamic.yaml").read_text())
    document["schema_version"] = version
    with pytest.raises(ConfigError, match="schema_version"):
        parse_config(document, base_directory=ROOT)


@pytest.mark.parametrize(
    "mutation", ["unknown", "missing", "incompatible", "reference"]
)
def test_invalid_stream_contract_is_rejected(mutation):
    document = yaml.safe_load((ROOT / "configs/hmi_lte_dynamic.yaml").read_text())
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
    path.write_text("schema_version: 3\nschema_version: 3\n")
    with pytest.raises(ConfigError):
        load_config(path)
    with pytest.raises(EnvironmentResolutionError):
        expand_environment("${MISSING}", environ={})
