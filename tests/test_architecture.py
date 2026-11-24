"""Executable boundaries for the modular observation-stream pipeline."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "prom3theus"


def test_repository_contains_only_the_public_yaml_runs():
    runtime_roots = {".git", ".idea", "data", "runs"}
    yaml_paths = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for pattern in ("*.yaml", "*.yml")
        for path in PROJECT_ROOT.rglob(pattern)
        if not runtime_roots.intersection(path.relative_to(PROJECT_ROOT).parts)
    )
    import yaml

    assert yaml_paths
    assert all(name.startswith("configs/") for name in yaml_paths)
    for name in yaml_paths:
        document = yaml.safe_load((PROJECT_ROOT / name).read_text())
        if name == "configs/muram/hinode/synthesis.yaml":
            assert "instrument" in document  # explicit numerical fixture config
        elif name == "configs/muram/hinode/surrogate.yaml":
            assert set(document) == {
                "line_ids", "wavelength_bounds_vacuum_angstrom", "knots", "wavelength_count",
                "workers", "batch_size", "steps", "validate_every",
                "architecture", "width", "hidden_layers", "jitter", "wavelength_jitter_fraction",
                "replay_fraction", "replay_buffer_size",
                "optimizer", "learning_rate", "final_learning_rate", "learning_rate_decay_steps",
                "source_path", "iteration", "instrument_config", "output_dir",
            }
        else:
            assert document["schema_version"] == 5, name


def test_source_tree_contains_only_the_current_package():
    packages = sorted(
        path.name
        for path in (PROJECT_ROOT / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    assert packages == ["prom3theus"]


def test_preprocessing_lives_in_the_installed_package():
    assert not list((PROJECT_ROOT / "scripts").rglob("*.py"))
    for name in ("hmi", "aia"):
        module = SOURCE_ROOT / "preprocess" / f"{name}.py"
        assert module.is_file()
        assert "/glade/" not in module.read_text()


def _imports(path: Path) -> set[str]:
    """Resolve both import forms, including relative package imports."""
    package = "prom3theus." + ".".join(path.relative_to(SOURCE_ROOT).parts[:-1])
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                from importlib.util import resolve_name

                module = resolve_name("." * node.level + module, package)
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)
    return imported


def _assert_dependency_boundary(package: str, forbidden: set[str]) -> None:
    violations = []
    for path in (SOURCE_ROOT / package).rglob("*.py"):
        overlap = {
            name
            for name in _imports(path)
            if any(
                name == prefix or name.startswith(prefix + ".") for prefix in forbidden
            )
        }
        if overlap:
            violations.append((str(path.relative_to(SOURCE_ROOT)), sorted(overlap)))
    assert violations == []


def test_radiative_transfer_has_no_application_layer_dependencies():
    _assert_dependency_boundary(
        "rt",
        {
            "prom3theus.application",
            "prom3theus.artifacts",
            "prom3theus.cli",
            "prom3theus.config",
            "prom3theus.diagnostics",
            "prom3theus.instruments",
            "prom3theus.observations",
            "prom3theus.training",
            "prom3theus.validation",
            "pytorch_lightning",
            "astropy",
            "sunpy",
            "matplotlib",
            "wandb",
        },
    )


def test_inversion_has_no_training_or_application_dependencies():
    _assert_dependency_boundary(
        "inversion",
        {
            "prom3theus.application",
            "prom3theus.artifacts",
            "prom3theus.cli",
            "prom3theus.config",
            "prom3theus.diagnostics",
            "prom3theus.training",
            "prom3theus.validation",
            "pytorch_lightning",
        },
    )


def test_packaging_exposes_one_new_command_and_only_the_new_namespace():
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["name"] == "PROM3THEUS"
    assert project["project"]["requires-python"] == ">=3.11"
    assert project["project"]["scripts"] == {"prom3theus": "prom3theus.cli.main:main"}
    assert project["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"],
        "include": ["prom3theus*"],
        "namespaces": False,
    }
    assert project["tool"]["setuptools"]["package-data"] == {
        "prom3theus.resources": [
            "data/*.json",
            "data/common/*.json",
            "data/hinode_sp/*.json",
            "data/hmi_stokes/*.json",
            "sets/euv/aia_euv_v1/*.json",
            "sets/plasma/plasma_v1/*.json",
        ]
    }
    config_files = project["tool"]["setuptools"]["data-files"]
    assert set(config_files) == {
        "share/prom3theus/configs/hinode", "share/prom3theus/configs/sdo",
        "share/prom3theus/configs/muram/hinode",
    }
    assert {name for names in config_files.values() for name in names} == {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "configs").rglob("*.yaml")
    }
    assert (
        "sunpy[map]>=5" in project["project"]["optional-dependencies"]["observations"]
    )
    assert (
        "sunpy[map]>=5"
        in project["project"]["optional-dependencies"]["hmi-preparation"]
    )
    assert project["project"]["optional-dependencies"]["aia-preparation"] == [
        "aiapy==0.12.1",
        "astropy>=7.1.1",
        "drms>=0.6.2",
        "python-dateutil>=2.8.2",
        "sunpy[image,map,net]>=7.0",
        "tqdm>=4.61.2",
    ]


def test_root_import_does_not_load_optional_scientific_stacks():
    code = f"""
import json, sys
sys.path.insert(0, {str(PROJECT_ROOT / "src")!r})
import prom3theus
blocked = ('aiapy', 'astropy', 'matplotlib', 'pytorch_lightning', 'sunpy', 'wandb')
print(json.dumps([name for name in blocked if name in sys.modules]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"


def test_retired_lifecycle_modules_cannot_return():
    retired = (
        "application/runner.py",
        "training/lightning.py",
        "training/callbacks.py",
        "training/multimodal_callbacks.py",
        "training/batch.py",
        "artifacts/model.py",
        "rt/synthesis.py",
        "rt/transfer.py",
        "inversion/forward.py",
        "inversion/radiation.py",
        "inversion/rt_collocation.py",
        "rt/radiation_field.py",
        "rt/radiation_geometry.py",
        "training/radiation.py",
        "application/radiation_support.py",
        "config/radiation_schema.py",
        "diagnostics/radiation.py",
    )
    assert all(not (SOURCE_ROOT / name).exists() for name in retired)
    forbidden = {"prom3theus." + name[:-3].replace("/", ".") for name in retired}
    for package in ("application", "training", "artifacts", "cli"):
        _assert_dependency_boundary(package, forbidden)


def test_reference_solvers_are_explicit_offline_validation_tools():
    for module in ("reference_synthesis", "reference_transfer", "reference_forward"):
        assert (SOURCE_ROOT / "validation" / f"{module}.py").is_file()
    for package in ("rt", "inversion", "training", "components", "artifacts"):
        _assert_dependency_boundary(package, {"prom3theus.validation"})


def test_aia_preparation_is_independent_of_optimization():
    _assert_dependency_boundary(
        "instruments/aia_euv",
        {
            "prom3theus.inversion",
            "prom3theus.training",
            "pytorch_lightning",
        },
    )


def test_objective_coordinator_is_observation_neutral():
    imported = _imports(SOURCE_ROOT / "inversion/joint.py")
    assert not any(
        "instruments" in name or name.endswith(("stokes", "aia_euv"))
        for name in imported
    )


def test_snapshot_reconstruction_does_not_import_training():
    _assert_dependency_boundary(
        "artifacts",
        {
            "prom3theus.training",
            "pytorch_lightning",
        },
    )
