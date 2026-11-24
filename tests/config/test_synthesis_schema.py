from pathlib import Path

import pytest

from prom3theus.config import ConfigError, load_config, parse_config

ROOT = Path(__file__).resolve().parents[2]


def test_formal_synthesis_roundtrip_and_loss_balancing():
    config = load_config(ROOT / "configs/sdo/hmi_aia_dynamic.yaml")
    document = config.to_dict()
    document["training"]["loss_balance"]["enabled"] = True
    parsed = parse_config(document, base_directory=ROOT)
    assert parsed.synthesis.windows[0].id == parsed.streams[0].data_term.synthesis_window
    assert parse_config(parsed.to_dict(), base_directory=ROOT) == parsed


@pytest.mark.parametrize("key,value", [("depth_count", 1), ("wavelength_chunk_size", 64),
    ("jitter_fraction", -.01), ("jitter_fraction", 1.01), ("jitter_fraction", float("nan")),
    ("jitter_fraction", float("inf")), ("jitter_fraction", True),
    ("transfer_precision", "float16"), ("incoming_temperature_k", -1),
    ("collocation_count", 32), ("rt_weight", .1), ("hidden_dimension", 128)])
def test_invalid_numerics_and_removed_learned_solver_options_are_rejected(key, value):
    document = load_config(ROOT / "configs/sdo/hmi_aia_dynamic.yaml").to_dict()
    document["synthesis"]["windows"][0][key] = value
    with pytest.raises(ConfigError):
        parse_config(document, base_directory=ROOT)


def test_learned_radiation_configuration_requires_a_new_run():
    document = load_config(ROOT / "configs/sdo/hmi_aia_dynamic.yaml").to_dict()
    document["schema_version"] = 4
    with pytest.raises(ConfigError, match="schema_version"):
        parse_config(document, base_directory=ROOT)


@pytest.mark.parametrize("value", [0., .5, 1.])
def test_ray_jitter_configuration_roundtrip(value):
    document = load_config(ROOT / "configs/sdo/hmi_aia_dynamic.yaml").to_dict()
    document["synthesis"]["windows"][0]["jitter_fraction"] = value
    parsed = parse_config(document, base_directory=ROOT)
    assert parsed.synthesis.windows[0].jitter_fraction == value
    assert parse_config(parsed.to_dict(), base_directory=ROOT) == parsed


def test_missing_ray_jitter_preserves_old_uniform_sampling():
    document = load_config(ROOT / "configs/sdo/hmi_aia_dynamic.yaml").to_dict()
    document["synthesis"]["windows"][0].pop("jitter_fraction")
    assert parse_config(document, base_directory=ROOT).synthesis.windows[0].jitter_fraction == 0


@pytest.mark.parametrize("path", sorted((ROOT / "configs").rglob("*.yaml")))
def test_inversion_presets_enable_ray_jitter(path):
    from prom3theus.muram import config as _register_muram_schema

    if path.name in {"synthesis.yaml", "surrogate.yaml"}:
        return  # Only inversion presets configure ray sampling.
    if path.parent == ROOT / "configs/muram/hinode":
        return  # These runs share a deterministic depth grid with the target synthesis.
    assert all(window.jitter_fraction > 0 for window in load_config(path).synthesis.windows)


@pytest.mark.parametrize("preset", ["mhs", "stokes_only"])
def test_muram_hinode_uses_fixed_shell_grid_and_4096_rays(preset):
    from prom3theus.muram import config as _register_muram_schema
    config = load_config(ROOT / f"configs/muram/hinode/{preset}.yaml")
    window = config.synthesis.windows[0]
    assert window.depth_count == 32
    assert not window.coarse_to_fine.enabled
    assert window.jitter_fraction == 0
    assert window.shell_height_bounds_megameter == (-1., 1.)
    assert config.streams[0].observation.loader.batch_size * 4 == 4096
    assert parse_config(config.to_dict(), base_directory=ROOT) == config


@pytest.mark.parametrize("preset", ["mhs", "stokes_only"])
def test_muram_hinode_shell_contains_ar11158_corner_rays(preset):
    import torch
    from prom3theus.muram import config as _register_muram_schema
    from prom3theus.muram.geometry import SOLAR_RADIUS_M, build_disk_center_geometry
    from prom3theus.rt.ray_geometry import shell_characteristic

    window = load_config(ROOT / f"configs/muram/hinode/{preset}.yaml").synthesis.windows[0]
    n, spacing_m = 1024, 192_000
    corners = build_disk_center_geometry(n, n, spacing_m, spacing_m, stride=n - 1)
    half_width = n * spacing_m / 2e6 + window.lateral_halo_megameter
    shell_characteristic(
        corners.coordinates, corners.ray_direction,
        scene_basis=corners.stokes_basis[0, 0], solar_radius_m=SOLAR_RADIUS_M,
        height_bounds_m=torch.tensor(window.shell_height_bounds_megameter) * 1e6,
        chart_bounds_megameter=torch.tensor([[-half_width] * 2, [half_width] * 2]),
    )
