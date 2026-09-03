from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER_ROOT = ROOT / "resource_builder"


def test_resource_generator_is_separate_from_runtime_package():
    assert BUILDER_ROOT.is_dir()
    assert not BUILDER_ROOT.is_relative_to(ROOT / "src")
    assert {path.name for path in BUILDER_ROOT.iterdir() if path.is_file()} == {
        "README.md",
        "__init__.py",
        "_shared.py",
        "build.py",
        "common_atomic.py",
        "hinode_sp.py",
        "hmi_stokes.py",
        "requirements.txt",
    }
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in BUILDER_ROOT.glob("*.py")
    )
    assert "import prom3theus" not in source
    assert "from prom3theus" not in source


def test_resource_generator_cli_is_inspectable_without_network_access():
    result = subprocess.run(
        [sys.executable, "-m", "resource_builder.build", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output-directory" in result.stdout
    assert "--cache-directory" in result.stdout


def test_flat_builders_own_the_complete_manifest_inventory():
    manifest = json.loads(
        (
            ROOT / "src" / "prom3theus" / "resources" / "data" / "sources.json"
        ).read_text(encoding="utf-8")
    )
    code = (
        "import json; "
        "from resource_builder import common_atomic, hinode_sp, hmi_stokes; "
        "from resource_builder.build import INSTRUMENT_BUILDERS; "
        "builders=(common_atomic,*INSTRUMENT_BUILDERS); "
        "print(json.dumps({'instruments':[m.__name__ for m in INSTRUMENT_BUILDERS],"
        "'reviewed':[r for m in builders for r in m.REVIEWED_RESOURCES],"
        "'generated':[r for m in builders for r in m.GENERATED_RESOURCES]}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ownership = json.loads(result.stdout)

    assert ownership["instruments"] == [
        "resource_builder.hinode_sp",
        "resource_builder.hmi_stokes",
    ]
    assert len(ownership["reviewed"]) == len(set(ownership["reviewed"]))
    assert set(ownership["reviewed"]) == set(manifest["reviewed_files"])
    assert len(ownership["generated"]) == len(set(ownership["generated"]))
    assert set(ownership["generated"]) == set(manifest["generated_files"])


def test_resource_rebuild_script_has_one_static_generation_command():
    script = (ROOT / "scripts" / "resources" / "rebuild.sh").read_text(encoding="utf-8")
    assert "python -m resource_builder.build" in script
    assert "build/reproduced-lte-resources" in script
    assert "if " not in script
