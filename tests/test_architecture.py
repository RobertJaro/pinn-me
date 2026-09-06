"""Executable boundaries for the clean LTE-only repository."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tomllib


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
    assert yaml_paths == [
        "configs/hinode_lte_mhs.yaml",
        "configs/hinode_lte_mhs_extrapolation.yaml",
        "configs/hmi_lte_dynamic.yaml",
        "configs/hmi_lte_mhs.yaml",
    ]


def test_source_tree_contains_only_the_current_package():
    packages = sorted(
        path.name
        for path in (PROJECT_ROOT / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    assert packages == ["prom3theus"]


def test_radiative_transfer_has_no_application_layer_dependencies():
    forbidden = {
        "prom3theus.artifacts",
        "prom3theus.cli",
        "prom3theus.config",
        "prom3theus.diagnostics",
        "prom3theus.instruments",
        "prom3theus.observations",
        "prom3theus.training",
    }
    violations = []
    for path in (SOURCE_ROOT / "rt").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        overlap = forbidden.intersection(imported)
        if overlap:
            violations.append((path.name, sorted(overlap)))
    assert violations == []


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
        ]
    }
    assert project["tool"]["setuptools"]["data-files"] == {
        "share/prom3theus/configs": [
            "configs/hinode_lte_mhs.yaml",
            "configs/hinode_lte_mhs_extrapolation.yaml",
            "configs/hmi_lte_mhs.yaml",
            "configs/hmi_lte_dynamic.yaml",
        ]
    }
    assert (
        "sunpy[map]>=5" in project["project"]["optional-dependencies"]["observations"]
    )
    assert (
        "sunpy[map]>=5"
        in project["project"]["optional-dependencies"]["hmi-preparation"]
    )


def test_root_import_does_not_load_optional_scientific_stacks():
    code = f"""
import json, sys
sys.path.insert(0, {str(PROJECT_ROOT / "src")!r})
import prom3theus
blocked = ('astropy', 'matplotlib', 'pytorch_lightning', 'sunpy', 'wandb')
print(json.dumps([name for name in blocked if name in sys.modules]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"
