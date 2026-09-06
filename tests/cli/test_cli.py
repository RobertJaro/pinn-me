"""Public command-line surface for the LTE framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prom3theus.cli.main import _build_parser, main


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_command_tree_is_small_and_explicit():
    parser = _build_parser()
    command_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(command_action.choices) == {
        "download",
        "export",
        "invert",
        "prepare",
        "recovery",
        "resources",
        "validate-config",
    }

    prepare = command_action.choices["prepare"]
    prepare_action = next(
        action for action in prepare._actions if action.dest == "prepare_command"
    )
    assert set(prepare_action.choices) == {"hmi-responses", "hmi-subframes"}

    download = command_action.choices["download"]
    download_action = next(
        action for action in download._actions if action.dest == "download_command"
    )
    assert set(download_action.choices) == {"hmi-stokes"}


def test_hmi_download_requires_an_explicit_closed_interval():
    parser = _build_parser()
    arguments = parser.parse_args(
        [
            "download",
            "hmi-stokes",
            "--output",
            "data/hmi",
            "--email",
            "scientist@example.org",
            "--start",
            "2024-03-23T22:15:00",
            "--end",
            "2024-03-24T02:15:00",
        ]
    )
    assert arguments.download_command == "hmi-stokes"
    assert arguments.start == "2024-03-23T22:15:00"

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            [
                "download",
                "hmi-stokes",
                "--output",
                "data/hmi",
                "--email",
                "scientist@example.org",
                "--start",
                "2024-03-23T22:15:00",
            ]
        )


def test_version_has_program_name(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert capsys.readouterr().out.strip().startswith("prom3theus ")


def test_hmi_preparation_resolves_phase_map_without_an_fsn_argument():
    parser = _build_parser()
    arguments = parser.parse_args(
        [
            "prepare",
            "hmi-responses",
            "input.I0.fits",
            "--output",
            "responses",
            "--email",
            "scientist@example.org",
        ]
    )
    assert not hasattr(arguments, "phase_map_fsn")

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            [
                "prepare",
                "hmi-responses",
                "input.I0.fits",
                "--output",
                "responses",
                "--phase-map-fsn",
                "4242",
            ]
        )


def test_hmi_subframe_preparation_requires_explicit_physical_selection():
    parser = _build_parser()
    arguments = parser.parse_args(
        [
            "prepare",
            "hmi-subframes",
            "data/hmi/full_disk",
            "--output",
            "data/hmi/subframe",
            "--longitude-deg",
            "215",
            "--latitude-deg",
            "-12",
            "--width-pixels",
            "1024",
            "--height-pixels",
            "512",
        ]
    )
    assert arguments.prepare_command == "hmi-subframes"
    assert arguments.longitude_deg == 215.0
    assert arguments.latitude_deg == -12.0
    assert arguments.width_pixels == 1024
    assert arguments.height_pixels == 512


def test_validate_config_prints_resolved_schema(capsys):
    main(
        [
            "validate-config",
            str(PROJECT_ROOT / "configs" / "hmi_lte_dynamic.yaml"),
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == 2
    assert document["solver"]["kind"] == "lte"
    assert document["resources"] == {"bundle": "packaged"}
    assert document["observation"]["type"] == "hmi_stokes"
