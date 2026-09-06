from pathlib import Path
import importlib
import json
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_ROOT = ROOT / "resource_builder"
sys.path.insert(0, str(ROOT))
_shared = importlib.import_module("resource_builder._shared")
_parse_chianti_ionization_equilibrium = _shared._parse_chianti_ionization_equilibrium
_parse_falc_atmosphere = _shared._parse_falc_atmosphere
sys.path.pop(0)


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
        (ROOT / "src" / "prom3theus" / "resources" / "data" / "sources.json").read_text(
            encoding="utf-8"
        )
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


def _synthetic_chianti_ioneq() -> bytes:
    temperature_count = 101
    element_count = 30
    lines = [
        f"{temperature_count:3d}{element_count:3d}",
        "".join(f"{4.0 + 0.05 * index:6.2f}" for index in range(temperature_count)),
    ]
    for atomic_number in range(1, element_count + 1):
        for stage_number in range(1, atomic_number + 2):
            values = [0.0] * temperature_count
            if atomic_number == 1:
                value = 0.5001 if stage_number == 1 else 0.4998
                values = [value] * temperature_count
            elif stage_number == 1:
                values = [1.0] * temperature_count
            if atomic_number == 30 and stage_number == 20:
                values[22] = 1.866e-109
            lines.append(
                f"{atomic_number:3d}{stage_number:3d}"
                + "".join(f"{value:10.3e}" for value in values)
            )
    lines.extend(
        (
            " -1",
            "%filename:  chianti.ioneq",
            "%comment:",
            "  Prepared for the release of CHIANTI 10.1.",
            " -1",
        )
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def test_chianti_parser_handles_fixed_width_adjacent_exponents_and_normalizes():
    payload = _synthetic_chianti_ioneq()
    assert b"0.000e+001.866e-109" in payload

    log_temperature, fractions = _parse_chianti_ionization_equilibrium(payload)

    assert log_temperature.tolist() == pytest.approx(
        [4.0 + 0.05 * index for index in range(101)]
    )
    assert fractions.shape == (30, 31, 101)
    assert fractions[0, :, 0].sum() == pytest.approx(1.0)
    assert fractions[29, :, 22].sum() == pytest.approx(1.0)


def test_chianti_parser_rejects_missing_sentinel_and_out_of_range_fraction():
    payload = _synthetic_chianti_ioneq()
    without_sentinel = payload.replace(b" -1\n", b"", 1)
    with pytest.raises(RuntimeError, match="incomplete"):
        _parse_chianti_ionization_equilibrium(without_sentinel)

    out_of_range = payload.replace(b" 5.001e-01", b" 1.100e+00", 1)
    with pytest.raises(RuntimeError, match="invalid row"):
        _parse_chianti_ionization_equilibrium(out_of_range)


def _synthetic_falc_atmosphere() -> bytes:
    lines = [
        "* Ndep",
        "82",
        "* log g",
        "4.44",
        "* lg column Mass     Temperature     Vlos     Bln     Vturb",
    ]
    lines.extend(
        f"{-5.0 + 0.1 * index:.8f} {5000.0 + index:.1f} "
        f"{index:.1f} {0.01 * index:.2f} {1.0 + 0.01 * index:.2f}"
        for index in range(82)
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def test_falc_parser_preserves_all_82_native_thermodynamic_rows():
    log_gravity, atmosphere = _parse_falc_atmosphere(_synthetic_falc_atmosphere())

    assert log_gravity == pytest.approx(4.44)
    assert atmosphere.shape == (82, 5)
    assert atmosphere[0].tolist() == pytest.approx([-5.0, 5000.0, 0.0, 0.0, 1.0])
    assert atmosphere[-1].tolist() == pytest.approx([3.1, 5081.0, 81.0, 0.81, 1.81])


def test_falc_parser_rejects_truncated_or_nonmonotone_native_profiles():
    payload = _synthetic_falc_atmosphere()
    with pytest.raises(RuntimeError, match="declares 82 depths but yielded 81"):
        _parse_falc_atmosphere(payload.rsplit(b"\n", 2)[0] + b"\n")

    nonmonotone = payload.replace(b"-4.90000000 5001.0", b"-5.10000000 5001.0")
    with pytest.raises(RuntimeError, match="invalid native atmosphere rows"):
        _parse_falc_atmosphere(nonmonotone)
