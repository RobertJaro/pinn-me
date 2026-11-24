"""Build-level checks for the deliberately narrow distribution boundary."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIGS = {
    "sdo/hmi_aia_dynamic.yaml",
    "sdo/hmi_aia_energy.yaml",
    "hinode/mhs.yaml",
    "hinode/mhs_extrapolation.yaml",
    "muram/hinode/mhs.yaml",
    "muram/hinode/stokes_only.yaml",
    "muram/hinode/surrogate.yaml",
    "muram/hinode/synthesis.yaml",
}
RESOURCE_FILES = {
    "bundle.json",
    "common/abundances.json",
    "common/chianti_thermodynamic_table.json",
    "common/falc_reference_atmosphere.json",
    "common/lines.json",
    "common/stic_continuum_table.json",
    "hinode_sp/blend_inventory.json",
    "hinode_sp/instrument_hinode_sp.json",
    "hinode_sp/solar_reference_630nm.json",
    "hmi_stokes/instrument_hmi.json",
    "hmi_stokes/solar_reference_617nm.json",
    "sources.json",
}
AIA_RESOURCE_FILES = {
    "aia_temperature_response.json",
    "instrument_aia_euv.json",
    "manifest.json",
    "sources.json",
}
PLASMA_RESOURCE_FILES = {"coronal_cooling_v1.json", "manifest.json", "sources.json"}


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build from a clean, minimal source copy without repository-path effects."""

    root = tmp_path_factory.mktemp("distribution-source")
    for filename in ("LICENSE", "MANIFEST.in", "README.md", "pyproject.toml"):
        shutil.copy2(PROJECT_ROOT / filename, root / filename)
    shutil.copytree(PROJECT_ROOT / "configs", root / "configs")
    shutil.copytree(PROJECT_ROOT / "docs", root / "docs")
    shutil.copytree(PROJECT_ROOT / "resource_builder", root / "resource_builder")
    shutil.copytree(PROJECT_ROOT / "scripts", root / "scripts")
    shutil.copytree(PROJECT_ROOT / "src" / "prom3theus", root / "src" / "prom3theus")

    output = root / "dist"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools.build_meta import build_sdist, build_wheel; "
            "output = sys.argv[1]; build_sdist(output); build_wheel(output)",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(output.glob("*.whl"))
    sdists = list(output.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    return wheels[0], sdists[0]


def test_wheel_contains_only_the_public_package_configs_and_metadata(
    built_distributions: tuple[Path, Path],
):
    wheel, _ = built_distributions
    with zipfile.ZipFile(wheel) as archive:
        names = {name.rstrip("/") for name in archive.namelist() if name.rstrip("/")}
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        top_level_name = next(
            name for name in names if name.endswith(".dist-info/top_level.txt")
        )
        entry_points = archive.read(entry_points_name).decode("utf-8")
        top_level = archive.read(top_level_name).decode("utf-8")

    assert entry_points == (
        "[console_scripts]\nprom3theus = prom3theus.cli.main:main\n"
    )
    assert top_level == "prom3theus\n"

    python_members = {
        name.removeprefix("prom3theus/")
        for name in names
        if name.startswith("prom3theus/") and name.endswith(".py")
    }
    source_python = {
        path.relative_to(PROJECT_ROOT / "src" / "prom3theus").as_posix()
        for path in (PROJECT_ROOT / "src" / "prom3theus").rglob("*.py")
    }
    assert python_members == source_python
    assert "rt/local_coefficients.py" in python_members
    assert "rt/formal_transfer.py" in python_members
    assert "inversion/lte.py" in python_members
    assert "validation/reference_synthesis.py" in python_members
    assert "validation/reference_transfer.py" in python_members
    assert not {"rt/synthesis.py", "rt/transfer.py", "inversion/forward.py"} & python_members

    resource_prefix = "prom3theus/resources/data/"
    resource_members = {
        name.removeprefix(resource_prefix)
        for name in names
        if name.startswith(resource_prefix)
    }
    assert resource_members == RESOURCE_FILES

    aia_resource_prefix = "prom3theus/resources/sets/euv/aia_euv_v1/"
    aia_resource_members = {
        name.removeprefix(aia_resource_prefix)
        for name in names
        if name.startswith(aia_resource_prefix)
    }
    assert aia_resource_members == AIA_RESOURCE_FILES

    plasma_prefix = "prom3theus/resources/sets/plasma/plasma_v1/"
    assert {name.removeprefix(plasma_prefix) for name in names if name.startswith(plasma_prefix)} == PLASMA_RESOURCE_FILES

    config_members = {
        name.split(".data/data/share/prom3theus/configs/", 1)[1]
        for name in names
        if ".data/data/share/prom3theus/configs/" in name
    }
    assert config_members == RUN_CONFIGS
    assert all(
        name.startswith("prom3theus/")
        or ".dist-info/" in name
        or ".data/data/share/prom3theus/configs/" in name
        for name in names
    )


def test_sdist_excludes_repository_tests_and_nonpackage_trees(
    built_distributions: tuple[Path, Path],
):
    _, sdist = built_distributions
    with tarfile.open(sdist, "r:gz") as archive:
        members = {
            Path(name).as_posix().split("/", 1)[1]
            for name in archive.getnames()
            if "/" in name
        }

    top_level = {name.split("/", 1)[0] for name in members}
    assert top_level == {
        "LICENSE",
        "MANIFEST.in",
        "PKG-INFO",
        "README.md",
        "configs",
        "docs",
        "pyproject.toml",
        "resource_builder",
        "scripts",
        "setup.cfg",
        "src",
    }
    assert {
        name.removeprefix("configs/")
        for name in members
        if name.startswith("configs/") and name.endswith((".yaml", ".yml"))
    } == RUN_CONFIGS
    assert not any(name.startswith("tests/") for name in members)
    assert {
        name.removeprefix("docs/")
        for name in members
        if name.startswith("docs/") and name.endswith(".md")
    } == {path.relative_to(PROJECT_ROOT / "docs").as_posix()
          for path in (PROJECT_ROOT / "docs").rglob("*.md")}
    assert {
        name.removeprefix("scripts/")
        for name in members
        if name.startswith("scripts/") and name.endswith(".sh")
    } == {path.relative_to(PROJECT_ROOT / "scripts").as_posix() for path in (PROJECT_ROOT / "scripts").rglob("*.sh")}
    assert not any(name.startswith("scripts/") and name.endswith(".py") for name in members)
    assert not any(
        name.startswith("scripts/") and name.endswith(".json") for name in members
    )
    assert {
        name.removeprefix("resource_builder/")
        for name in members
        if name.startswith("resource_builder/")
    } == {
        "README.md",
        "__init__.py",
        "_shared.py",
        "aia_euv.py",
        "coronal_cooling.py",
        "build.py",
        "common_atomic.py",
        "hinode_sp.py",
        "hmi_stokes.py",
        "requirements.txt",
    }
    assert not any("/__pycache__/" in f"/{name}/" for name in members)

    package_root = "src/prom3theus/"
    assert {
        name.removeprefix(package_root)
        for name in members
        if name.startswith(package_root) and name.endswith(".py")
    } == {
        path.relative_to(PROJECT_ROOT / "src" / "prom3theus").as_posix()
        for path in (PROJECT_ROOT / "src" / "prom3theus").rglob("*.py")
    }
    assert {
        name.removeprefix("src/prom3theus/resources/data/")
        for name in members
        if name.startswith("src/prom3theus/resources/data/") and name.endswith(".json")
    } == RESOURCE_FILES
    assert {
        name.removeprefix("src/prom3theus/resources/sets/euv/aia_euv_v1/")
        for name in members
        if name.startswith("src/prom3theus/resources/sets/euv/aia_euv_v1/")
        and name.endswith(".json")
    } == AIA_RESOURCE_FILES
    assert {
        name.removeprefix("src/prom3theus/resources/sets/plasma/plasma_v1/")
        for name in members
        if name.startswith("src/prom3theus/resources/sets/plasma/plasma_v1/") and name.endswith(".json")
    } == PLASMA_RESOURCE_FILES


def test_wheel_runs_from_an_isolated_installation(
    built_distributions: tuple[Path, Path],
    tmp_path: Path,
):
    wheel, _ = built_distributions
    installation = tmp_path / "installation"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installation),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = {**os.environ, "PYTHONPATH": str(installation)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import pkgutil
from pathlib import Path
import prom3theus
from prom3theus.config import load_config
from prom3theus.resources import validate_resource_bundle

installation = Path.cwd() / "installation"
assert Path(prom3theus.__file__).resolve().is_relative_to(installation)
modules = [
    importlib.import_module(item.name)
    for item in pkgutil.walk_packages(
        prom3theus.__path__, prefix="prom3theus."
    )
]
for module in [prom3theus, *modules]:
    assert Path(module.__file__).resolve().is_relative_to(installation)
    assert all(hasattr(module, name) for name in getattr(module, "__all__", ()))
configs = installation / "share" / "prom3theus" / "configs"
assert {path.relative_to(configs).as_posix() for path in configs.rglob("*.yaml")} == {
    "hinode/mhs.yaml",
    "hinode/mhs_extrapolation.yaml",
    "sdo/hmi_aia_dynamic.yaml",
    "sdo/hmi_aia_energy.yaml",
    "muram/hinode/mhs.yaml",
    "muram/hinode/stokes_only.yaml",
    "muram/hinode/surrogate.yaml",
    "muram/hinode/synthesis.yaml",
}
from prom3theus.muram import config as muram_config_registration
for path in configs.rglob("*.yaml"):
    if path.name in {"synthesis.yaml", "surrogate.yaml"}:
        continue
    config = load_config(path)
    assert config.schema_version == 5 and config.solver.kind == "joint"
import prom3theus.rt as rt
assert not hasattr(rt, "LTESynthesizer")
from prom3theus.validation.reference_synthesis import LTESynthesizer
resources = validate_resource_bundle()
assert resources["supported_instruments"] == ["hinode_sp", "hmi_stokes"]
print(prom3theus.__version__)
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.1.0"

    command = installation / "bin" / "prom3theus"
    version = subprocess.run(
        [str(command), "--version"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.strip() == "prom3theus 0.1.0"
