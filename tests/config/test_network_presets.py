from pathlib import Path

import pytest

from prom3theus.config import ConfigError, load_config, parse_config
from prom3theus.muram import config as _register_muram_schema

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path", [p for p in sorted((ROOT / "configs").rglob("*.yaml")) if p.name != "synthesis.yaml"])
def test_inversion_presets_use_siren(path):
    config = load_config(path)
    network = config.atmosphere.network
    assert network.type == "siren"
    assert network.activation is None and network.encoding is None
    expected_first_omega = 5. if path == ROOT / "configs/muram/hinode/mhs.yaml" else 30.
    assert network.first_omega_0 == expected_first_omega and network.hidden_omega_0 == 1.
    assert network.radial_weighting is None
    assert parse_config(config.to_dict(), base_directory=ROOT) == config


def test_schema_rejects_wrong_number_of_frequency_blocks():
    document = load_config(ROOT / "configs/hinode/mhs.yaml").to_dict()
    document["atmosphere"]["network"].update(type="mlp", activation="silu", encoding={
        "type": "positional_isotropic", "num_frequencies": [32, 8], "max_frequencies": [8., 2.]})
    with pytest.raises(ConfigError, match="encoding blocks"):
        parse_config(document, base_directory=ROOT)
