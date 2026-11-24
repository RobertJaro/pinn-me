"""The checkout scripts are static, instrument-specific LTE commands."""

from __future__ import annotations

import os
import re
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_hinode_stokes_launcher_starts_one_rank_per_allocated_gpu():
    script = PROJECT_ROOT / "scripts/muram_hinode/run_stokes_only.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    allocated = int(re.search(r"#PBS -l select=1:.*ngpus=(\d+)", text)[1])
    from prom3theus.config import load_config
    from prom3theus.muram import config as _register_muram_schema
    config = load_config(PROJECT_ROOT / "configs/muram/hinode/stokes_only.yaml").to_dict()
    assert allocated == 4
    assert config['training']['devices'] == 'auto'
    assert config['training']['device'] == 'cuda'
    assert 'torchrun' not in text and 'torch.distributed.run' not in text
    assert "python -m prom3theus.muram.cli invert configs/muram/hinode/stokes_only.yaml" in text


def test_shell_scripts_have_no_conditionals():
    for path in (PROJECT_ROOT / "scripts").rglob("*.sh"):
        assert not re.search(r"^\s*(if|elif|else|fi)\b", path.read_text(), re.MULTILINE), path

HMI_DOWNLOAD = PROJECT_ROOT / "scripts" / "hmi" / "download.sh"
HMI_PREPARE = PROJECT_ROOT / "scripts" / "hmi" / "prepare.sh"
HMI_RUN = PROJECT_ROOT / "scripts" / "hmi" / "run.sh"
HINODE_RUN = PROJECT_ROOT / "scripts" / "hinode" / "run.sh"
HINODE_RESOURCES = PROJECT_ROOT / "scripts" / "hinode" / "load_resources.sh"
SDO_DOWNLOAD = PROJECT_ROOT / "scripts" / "sdo" / "download.sh"
SDO_PREPARE = PROJECT_ROOT / "scripts" / "sdo" / "prepare.sh"
SDO_RUN = PROJECT_ROOT / "scripts" / "sdo" / "run.sh"
SCRIPTS = (HMI_DOWNLOAD, HMI_PREPARE, HMI_RUN, HINODE_RESOURCES, HINODE_RUN)
MURAM_UNPACK = PROJECT_ROOT / "scripts" / "muram" / "unpack_snapshots.sh"
MURAM_SYNTHESIZE = PROJECT_ROOT / "scripts" / "muram" / "synthesize_snapshots.sh"
MURAM_RUN = PROJECT_ROOT / "scripts" / "muram" / "run_inversion.sh"
MURAM_SCRIPTS = (MURAM_SYNTHESIZE, MURAM_RUN)


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
        "scripts/muram/run_inversion.sh",
        "scripts/muram/synthesize_rays.sh",
        "scripts/muram/synthesize_snapshots.sh",
        "scripts/muram/train_surrogate.sh",
        "scripts/muram/unpack_snapshots.sh",
        "scripts/muram_hinode/run.sh",
        "scripts/muram_hinode/run_stokes_only.sh",
        "scripts/muram_hinode/synthesize.sh",
        "scripts/resources/rebuild.sh",
        "scripts/sdo/download.sh",
        "scripts/sdo/prepare.sh",
        "scripts/sdo/run.sh",
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

    for script in (SDO_DOWNLOAD, SDO_PREPARE, SDO_RUN):
        assert os.access(script, os.X_OK)
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text(encoding="utf-8")
        assert "/glade/work/rjarolim/data/sdo" not in text
        assert "sample_" not in text
        assert "<<" not in text
        assert "--overwrite" not in text

    for script in MURAM_SCRIPTS:
        assert os.access(script, os.X_OK)
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        assert "if " not in text
        assert "case " not in text
        assert "--overwrite" not in text
        assert "PROM3THEUS_" not in text
        assert "pme" not in text.lower()
        assert "PYTHONPATH=src python -m prom3theus.muram.cli" in text

    assert os.access(MURAM_UNPACK, os.X_OK)


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
    for script in (*SCRIPTS, SDO_DOWNLOAD, SDO_PREPARE, SDO_RUN, *MURAM_SCRIPTS, MURAM_UNPACK):
        result = subprocess.run(
            ["bash", str(script), "--overwrite"],
            cwd="/",
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


def test_sdo_commands_are_fixed_and_minimal(tmp_path):
    script = SDO_DOWNLOAD.read_text()
    assert "readonly" not in script
    assert "env PYTHONPATH" not in script
    assert "exec " not in script
    assert "--channels 171 193 211" in script
    assert "--cadence-seconds 720" in script
    base = "/glade/work/rjarolim/data/prom3theus/2011_02"
    assert _run(SDO_DOWNLOAD, tmp_path / "download") == [
        [
            "-m", "prom3theus.cli.main", "download", "hmi-stokes",
            "--output", base + "/hmi/full_disk", "--email", "robert.jarolim@uni-graz.at",
            "--start", "2011-02-14T00:00:00", "--end", "2011-02-15T00:00:00",
        ],
        [
            "-m", "prom3theus.cli.main", "download", "aia-euv",
            "--output", base + "/aia/level1",
            "--email", "robert.jarolim@uni-graz.at", "--start", "2011-02-14T00:00:00",
            "--end", "2011-02-15T00:00:00", "--cadence-seconds", "720",
            "--channels", "171", "193", "211",
        ],
    ]
    base = "/glade/work/rjarolim/data/prom3theus/2011_02"
    assert _run(SDO_PREPARE, tmp_path / "prepare") == [
        [
            "-m", "prom3theus.cli.main", "prepare", "hmi", base + "/hmi/full_disk",
            "--output", base + "/hmi/prepared", "--longitude-deg", "35.3",
            "--latitude-deg", "-20", "--width-pixels", "500", "--height-pixels", "500",
        ],
        [
            "-m", "prom3theus.cli.main", "prepare", "hmi-responses", base + "/hmi/prepared",
            "--output", base + "/hmi/responses", "--email", "robert.jarolim@uni-graz.at",
        ],
        [
            "-m", "prom3theus.cli.main", "prepare", "aia", base + "/aia/level1",
            "--output", base + "/aia/prepared", "--calibration-directory", base + "/aia/calibration",
            "--longitude-deg", "35.3", "--latitude-deg", "-20",
            "--width-pixels", "500", "--height-pixels", "500",
        ],
    ]
    assert _run(SDO_RUN, tmp_path / "run") == [
        ["-m", "prom3theus.cli.main", "invert", "configs/sdo/hmi_aia_dynamic.yaml"],
    ]


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
            "/glade/work/rjarolim/data/hmi_stokes/20240323_720s_subframe",
            "--output",
            "/glade/work/rjarolim/data/hmi_calibration/20240323",
            "--email",
            "robert.jarolim@uni-graz.at",
        ],
    ]


def test_run_commands_are_separate_and_fixed(tmp_path):
    assert _run(HMI_RUN, tmp_path / "hmi") == [
        ["-m", "prom3theus.cli.main", "invert", "configs/sdo/hmi_aia_dynamic.yaml"],
        ["-m", "prom3theus.cli.main", "invert", "configs/sdo/hmi_aia_energy.yaml"],
        [
            "-m",
            "prom3theus.cli.main",
            "compare-hmi",
            "/glade/work/rjarolim/lte/hmi_subframe_dynamic_extrapolation_v07/state.p3s",
            "/glade/work/rjarolim/data/hmi_stokes/test_2024_03_24",
            "--output",
            "/glade/work/rjarolim/lte/hmi_comparison",
        ],
    ]
    assert _run(HINODE_RUN, tmp_path / "hinode") == [
        [
            "-m",
            "prom3theus.cli.main",
            "invert",
            "configs/hinode/mhs.yaml",
        ],
        [
            "-m",
            "prom3theus.cli.main",
            "invert",
            "configs/hinode/mhs_extrapolation.yaml",
        ],
    ]


def test_muram_inversion_launches_configured_runs(tmp_path):
    assert _run(MURAM_RUN, tmp_path) == [
        ["-m", "prom3theus.muram.cli", "invert", "configs/muram/hinode/stokes_only.yaml"],
    ]


def test_muram_hinode_synthesis_mapper_and_inversion_share_artifacts(tmp_path):
    import yaml

    script = PROJECT_ROOT / "scripts/muram/train_surrogate.sh"
    assert os.access(script, os.X_OK)
    (training,) = _run(script, tmp_path / "training")
    assert training == ["-m", "prom3theus.surrogate", "--config", "configs/muram/hinode/surrogate.yaml"]
    (synthesis,) = _run(PROJECT_ROOT / "scripts/muram/synthesize_ar11158.pbs", tmp_path / "synthesis")
    assert synthesis[:3] == ["-m", "prom3theus.muram.cli", "synthesize"]
    assert synthesis[synthesis.index("--inversion-depth-grid") + 1] == "configs/muram/hinode/mhs.yaml"
    assert float(synthesis[synthesis.index("--photosphere-height-megameters") + 1]) == 7.5
    assert synthesis[synthesis.index("--output-format") + 1] == "store"
    from prom3theus.surrogate.cli import parse_options

    window_config = PROJECT_ROOT / training[-1]
    run = parse_options(["--config", str(window_config)])
    assert run.source_path == Path(synthesis[synthesis.index("--source-path") + 1])
    assert run.iteration == int(synthesis[synthesis.index("--iterations") + 1])
    assert run.instrument_config == PROJECT_ROOT / synthesis[synthesis.index("--instrument-config") + 1]
    detector = yaml.safe_load(run.instrument_config.read_text())
    surrogate = yaml.safe_load(window_config.read_text())
    for preset in (window_config.with_name("stokes_only.yaml"), window_config.with_name("mhs.yaml")):
        config = yaml.safe_load(preset.read_text())
        window, = config["synthesis"]["windows"]
        stream, = config["streams"]
        assert window["coefficient_backend"] == "exact_lte"
        assert window.get("mapper_checkpoint") is None
        assert window["line_ids"] == surrogate["line_ids"] == detector["line_ids"]
        assert window["wavelength_bounds_vacuum_angstrom"] == surrogate["wavelength_bounds_vacuum_angstrom"]
        assert stream["data_term"]["instrument"] == detector["instrument"]
        assert stream["observation"]["directory"] == synthesis[synthesis.index("--out") + 1]


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
