"""The checkout scripts are static, instrument-specific LTE commands."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HMI_DOWNLOAD = PROJECT_ROOT / "scripts" / "hmi" / "download.sh"
HMI_PREPARE = PROJECT_ROOT / "scripts" / "hmi" / "prepare.sh"
HMI_RUN = PROJECT_ROOT / "scripts" / "hmi" / "run.sh"
HINODE_RUN = PROJECT_ROOT / "scripts" / "hinode" / "run.sh"
HINODE_RESOURCES = PROJECT_ROOT / "scripts" / "hinode" / "load_resources.sh"
SCRIPTS = (HMI_DOWNLOAD, HMI_PREPARE, HMI_RUN, HINODE_RESOURCES, HINODE_RUN)


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir(parents=True)
    log = tmp_path / "commands.log"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "{ printf '%s\\0' \"$#\"; printf '%s\\0' \"$@\"; } "
        '>> "${PROM3THEUS_TEST_LOG:?}"\n'
    )
    executable.chmod(0o755)
    return executable, log


def _run(script: Path, tmp_path: Path) -> list[list[str]]:
    executable, log = _fake_python(tmp_path)
    subprocess.run(
        ["bash", str(script)],
        cwd="/",
        env={
            **os.environ,
            "PATH": f"{executable.parent}:{os.environ['PATH']}",
            "PROM3THEUS_TEST_LOG": str(log),
            "JSOC_EMAIL": "scientist@example.test",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    fields = log.read_bytes().split(b"\0")
    assert fields.pop() == b""
    commands = []
    cursor = 0
    while cursor < len(fields):
        count = int(fields[cursor])
        cursor += 1
        commands.append([value.decode() for value in fields[cursor : cursor + count]])
        cursor += count
    return commands


def test_scripts_are_executable_static_bash_without_dispatch_or_overwrite():
    assert sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "scripts").rglob("*.sh")
    ) == [
        "scripts/hinode/load_resources.sh",
        "scripts/hinode/run.sh",
        "scripts/hmi/download.sh",
        "scripts/hmi/prepare.sh",
        "scripts/hmi/run.sh",
        "scripts/resources/rebuild.sh",
    ]
    for script in SCRIPTS:
        assert os.access(script, os.X_OK)
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        assert "if " not in text
        assert "case " not in text
        assert "--overwrite" not in text
        assert "PROM3THEUS_" not in text
        assert "pme" not in text.lower()
        assert "PYTHONPATH=src python -m prom3theus.cli.main" in text


def test_run_scripts_preserve_the_pbs_job_header():
    common = {
        "#PBS -A P22100000",
        "#PBS -q main",
        "#PBS -l job_priority=economy",
        "#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb",
        "#PBS -l walltime=12:00:00",
    }
    for script, job_name in (
        (HINODE_RUN, "#PBS -N prom3theus-hinode"),
        (HMI_RUN, "#PBS -N prom3theus-hmi"),
    ):
        contents = script.read_text()
        lines = set(contents.splitlines())
        assert contents.startswith("#!/bin/bash -l\n")
        assert common <= lines
        assert job_name in lines


def test_scripts_reject_all_arguments(tmp_path):
    executable, _ = _fake_python(tmp_path)
    environment = {
        **os.environ,
        "PATH": f"{executable.parent}:{os.environ['PATH']}",
        "PROM3THEUS_TEST_LOG": str(tmp_path / "commands.log"),
        "JSOC_EMAIL": "scientist@example.test",
    }
    for script in SCRIPTS:
        result = subprocess.run(
            ["bash", str(script), "--overwrite"],
            cwd="/",
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


def test_hmi_download_command_is_fixed(tmp_path):
    assert _run(HMI_DOWNLOAD, tmp_path) == [
        [
            "-m",
            "prom3theus.cli.main",
            "download",
            "hmi-stokes",
            "--output",
            "data/hmi/2024-03-23/full_disk",
            "--email",
            "scientist@example.test",
            "--start",
            "2024-03-23T22:12:00",
            "--end",
            "2024-03-24T02:12:00",
        ]
    ]


def test_hmi_preparation_commands_are_fixed(tmp_path):
    assert _run(HMI_PREPARE, tmp_path) == [
        [
            "-m",
            "prom3theus.cli.main",
            "prepare",
            "hmi-subframes",
            "data/hmi/2024-03-23/full_disk",
            "--output",
            "data/hmi/2024-03-23/subframe",
            "--longitude-deg",
            "215",
            "--latitude-deg",
            "-12",
            "--width-pixels",
            "1024",
            "--height-pixels",
            "512",
        ],
        [
            "-m",
            "prom3theus.cli.main",
            "prepare",
            "hmi-responses",
            "data/hmi/2024-03-23/subframe",
            "--output",
            "data/calibration/hmi/2024-03-23",
            "--email",
            "scientist@example.test",
            "--phase-map-fsn",
            "230562565",
        ],
    ]


def test_run_commands_are_separate_and_fixed(tmp_path):
    assert _run(HMI_RUN, tmp_path / "hmi") == [
        ["-m", "prom3theus.cli.main", "invert", "configs/hmi_lte_subframe.yaml"]
    ]
    assert _run(HINODE_RUN, tmp_path / "hinode") == [
        [
            "-m",
            "prom3theus.cli.main",
            "invert",
            "configs/hinode_lte_mhs.yaml",
        ]
    ]


def test_hinode_resource_command_is_fixed(tmp_path):
    assert _run(HINODE_RESOURCES, tmp_path) == [
        [
            "-m",
            "prom3theus.cli.main",
            "resources",
            "validate",
            "--instrument",
            "hinode_sp",
        ]
    ]
