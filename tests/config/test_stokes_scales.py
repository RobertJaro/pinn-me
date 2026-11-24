from pathlib import Path

import pytest

from prom3theus.config import ConfigError, load_config, parse_config
from prom3theus.muram import config as _register_muram_schema

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path", [p for p in sorted((ROOT / "configs").rglob("*.yaml")) if p.name not in {"synthesis.yaml", "surrogate.yaml"}])
def test_stokes_presets_use_configured_weights_and_linear_scales(path):
    config = load_config(path)
    for stream in config.streams:
        if stream.data_term.type != "lte_stokes":
            continue
        objective = stream.data_term.objective
        assert objective.type == "mse"
        assert objective.continuum_normalization
        assert objective.continuum_quantile == .9
        assert objective.continuum_floor is None
        assert objective.stokes_scales.to_dict() == {"I": 1., "Q": .1, "U": .1, "V": .1}
        assert objective.stokes_weights.to_dict() == {"i": 1., "q": 1., "u": 1., "v": 1.}
    assert parse_config(config.to_dict(), base_directory=ROOT) == config


@pytest.mark.parametrize("value", [0., -.1, float("nan"), float("inf"), True])
def test_invalid_component_scale_rejected(value):
    document = load_config(ROOT / "configs/hinode/mhs.yaml").to_dict()
    document["streams"][0]["data_term"]["objective"]["stokes_scales"]["Q"] = value
    with pytest.raises(ConfigError):
        parse_config(document, base_directory=ROOT)


def test_schema_rejects_combining_asinh_and_linear_scales():
    document = load_config(ROOT / "configs/hinode/mhs.yaml").to_dict()
    document["streams"][0]["data_term"]["objective"]["type"] = "asinh_mse"
    with pytest.raises(ConfigError, match="either"):
        parse_config(document, base_directory=ROOT)


@pytest.mark.parametrize("quantile", [0., 1.1, float("nan"), True])
def test_invalid_continuum_quantile_rejected(quantile):
    document = load_config(ROOT / "configs/hinode/mhs.yaml").to_dict()
    document["streams"][0]["data_term"]["objective"]["continuum_quantile"] = quantile
    with pytest.raises(ConfigError, match="continuum_quantile"):
        parse_config(document, base_directory=ROOT)
