"""Public command-line surface for the LTE framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prom3theus.cli.main import _build_parser, main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_invert_dispatches_joint_training_with_active_aia(monkeypatch):
    from prom3theus.application import joint_training

    calls = []
    monkeypatch.setattr(joint_training, "run_joint_inversion", lambda config, **kw: calls.append((config, kw)))
    main(["invert", str(PROJECT_ROOT / "configs/hmi_aia_dynamic.yaml")])
    assert len(calls) == 1
    config, options = calls[0]
    assert config.training is not None
    assert options == {"rebuild_observations": False}
    assert next(stream for stream in config.streams if stream.id == "coronal_euv").data_term.weight == 0.1


def test_invert_can_record_a_bounded_training_profile(monkeypatch, tmp_path):
    from prom3theus.application import joint_training
    from pytorch_lightning.profilers import PyTorchProfiler

    calls = []
    monkeypatch.setattr(
        joint_training, "run_joint_inversion",
        lambda config, **kw: calls.append(kw),
    )
    main([
        "invert", str(PROJECT_ROOT / "configs/hmi_aia_dynamic.yaml"),
        "--profile", str(tmp_path),
    ])
    profiler = calls[0]["profiler"]
    assert isinstance(profiler, PyTorchProfiler)
    assert profiler.dirpath == str(tmp_path)


def test_hmi_comparison_download_dispatch(tmp_path, monkeypatch, capsys):
    from prom3theus.download import hmi

    calls = []
    monkeypatch.setenv("JSOC_EMAIL", "test@example.org")
    monkeypatch.setattr(
        hmi, "download_hmi_comparison",
        lambda **kwargs: calls.append(kwargs) or ["field.fits"],
    )
    main([
        "download", "hmi-comparison", "--time", "2011-02-14T01:00:00",
        "--output", str(tmp_path),
    ])
    assert calls == [{
        "output_directory": tmp_path, "email": "test@example.org",
        "time": "2011-02-14T01:00:00",
    }]
    assert json.loads(capsys.readouterr().out) == ["field.fits"]


def test_command_tree_is_small_and_explicit():
    parser = _build_parser()
    command_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(command_action.choices) == {
        "compare-hmi",
        "download",
        "export",
        "invert",
        "prepare",
        "recovery",
        "resources",
        "time-series",
        "validate-config",
    }

    prepare = command_action.choices["prepare"]
    prepare_action = next(
        action for action in prepare._actions if action.dest == "prepare_command"
    )
    assert set(prepare_action.choices) == {
        "aia-calibration",
        "aia",
        "hmi",
        "aia-euv",
        "hmi-responses",
        "hmi-subframes",
    }

    download = command_action.choices["download"]
    download_action = next(
        action for action in download._actions if action.dest == "download_command"
    )
    assert set(download_action.choices) == {
        "aia-euv",
        "hmi-comparison",
        "hmi-stokes",
    }


@pytest.mark.parametrize("command", ["hmi-stokes", "aia-euv"])
def test_download_interval_flags_are_uniform(command):
    parser = _build_parser()
    downloads = next(action for action in parser._actions if action.dest == "command").choices["download"]
    instrument = next(
        action for action in downloads._actions if action.dest == "download_command"
    ).choices[command]
    options = {option for action in instrument._actions for option in action.option_strings}
    assert {"--start", "--end"} <= options
    assert "--overwrite" not in options
    assert "--reuse-existing" not in options
    assert "--calibration-output" not in options
    assert options.isdisjoint({"--start-utc", "--end-utc"})


def test_aia_download_dispatch_is_direct(monkeypatch, capsys):
    from prom3theus.download import aia
    captured = {}
    def download(**kwargs):
        captured.update(kwargs)
        return ["/tmp/aia/image.fits"]
    monkeypatch.setattr(aia, "download_aia_observations", download)
    main(["download", "aia-euv", "--output", "/tmp/aia", "--email", "test@example.org",
          "--start", "2011-02-14", "--end", "2011-02-15", "--channels", "171", "193"])
    assert captured == {
        "output_directory": Path("/tmp/aia"), "email": "test@example.org",
        "start": "2011-02-14", "end": "2011-02-15",
        "cadence_seconds": 720, "channels": [171, 193],
    }
    assert json.loads(capsys.readouterr().out) == ["/tmp/aia/image.fits"]


def test_dry_run_command_is_removed(capsys):
    assert "dry-run" not in _build_parser().format_help()
    with pytest.raises(SystemExit) as exc:
        main(["dry-run", "configs/joint.yaml", "--rebuild-observations"])
    assert exc.value.code == 2
    assert "invalid choice: 'dry-run'" in capsys.readouterr().err


def test_hmi_comparison_has_explicit_p3s_reference_and_output():
    args = _build_parser().parse_args(
        [
            "compare-hmi",
            "run/state.p3s",
            "data/hmi-vector",
            "--output",
            "run/hmi-comparison",
            "--height-km",
            "100",
            "--disambig-bit",
            "2",
        ]
    )
    assert args.save_state == Path("run/state.p3s")
    assert args.hmi_directory == Path("data/hmi-vector")
    assert args.height_km == 100.0
    assert args.disambig_bit == 2


def test_time_series_has_explicit_p3s_reference_and_sampling():
    args = _build_parser().parse_args(
        [
            "time-series",
            "run/state.p3s",
            "--output",
            "run/time-series",
            "--height-km",
            "100",
            "--current-height-samples",
            "24",
        ]
    )
    assert args.save_state == Path("run/state.p3s")
    assert args.output == Path("run/time-series")
    assert args.height_km == 100.0
    assert args.current_height_samples == 24


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


def test_aia_calibration_download_requires_an_explicit_bounded_interval():
    parser = _build_parser()
    arguments = parser.parse_args(
        [
            "prepare",
            "aia-calibration",
            "--output",
            "data/sdo/calibration/aia",
            "--start",
            "2024-03-23T21:00:00Z",
            "--end",
            "2024-03-24T00:00:00Z",
        ]
    )
    assert arguments.prepare_command == "aia-calibration"
    assert arguments.output == Path("data/sdo/calibration/aia")
    assert arguments.start == "2024-03-23T21:00:00Z"
    assert arguments.end == "2024-03-24T00:00:00Z"

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            [
                "prepare",
                "aia-calibration",
                "--output",
                "data/sdo/calibration/aia",
                "--start",
                "2024-03-23T21:00:00Z",
            ]
        )


def test_aia_calibration_download_dispatch_is_lazy_and_exact(monkeypatch, capsys):
    from prom3theus.instruments.aia_euv import calibration_download

    captured = {}

    def download_calibration(**options):
        captured.update(options)
        return Path("/tmp/aia-calibration/manifest.json")

    monkeypatch.setattr(
        calibration_download,
        "download_aia_preprocessing_calibration",
        download_calibration,
    )
    main(
        [
            "prepare",
            "aia-calibration",
            "--output",
            "data/sdo/calibration/aia",
            "--start",
            "2024-03-23T21:00:00Z",
            "--end",
            "2024-03-24T00:00:00Z",
        ]
    )

    assert captured == {
        "output_directory": Path("data/sdo/calibration/aia"),
        "start_utc": "2024-03-23T21:00:00Z",
        "end_utc": "2024-03-24T00:00:00Z",
        "overwrite": False,
    }
    assert capsys.readouterr().out.strip() == ("/tmp/aia-calibration/manifest.json")


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


def test_aia_preparation_uses_explicit_carrington_degrees_and_local_tables():
    parser = _build_parser()
    arguments = parser.parse_args(
        [
            "prepare",
            "aia-euv",
            "data/sdo/raw/aia",
            "--output",
            "data/sdo/prepared/aia",
            "--correction-table",
            "data/sdo/calibration/aia/aia_correction.ecsv",
            "--pointing-table",
            "data/sdo/calibration/aia/aia_pointing.ecsv",
            "--longitude-deg",
            "215",
            "--latitude-deg",
            "-12",
            "--width-deg",
            "8",
            "--height-deg",
            "4",
        ]
    )

    assert arguments.prepare_command == "aia-euv"
    assert arguments.acquisition == Path("data/sdo/raw/aia")
    assert arguments.output == Path("data/sdo/prepared/aia")
    assert arguments.longitude_deg == 215.0
    assert arguments.latitude_deg == -12.0
    assert arguments.width_deg == 8.0
    assert arguments.height_deg == 4.0
    assert arguments.boundary_samples_per_axis == 33
    assert arguments.observation_id == "aia_euv"
    assert not hasattr(arguments, "shot_variance_per_abs_dn")

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            [
                "prepare",
                "aia-euv",
                "data/sdo/raw/aia",
                "--output",
                "data/sdo/prepared/aia",
                "--longitude-deg",
                "215",
                "--latitude-deg",
                "-12",
                "--width-deg",
                "8",
                "--height-deg",
                "4",
            ]
        )


def test_aia_preparation_dispatch_builds_exact_backend_inputs(monkeypatch, capsys):
    from prom3theus.instruments.aia_euv import aiapy_preparation

    captured = {}

    def prepare_store(
        acquisition_directory,
        output_directory,
        **options,
    ):
        captured["acquisition_directory"] = acquisition_directory
        captured["output_directory"] = output_directory
        captured.update(options)
        return Path("/tmp/prepared-aia")

    monkeypatch.setattr(
        aiapy_preparation,
        "prepare_aiapy_aia_observation_store",
        prepare_store,
    )
    main(
        [
            "prepare",
            "aia-euv",
            "data/sdo/raw/aia",
            "--output",
            "data/sdo/prepared/aia",
            "--correction-table",
            "calibration/aia_correction.ecsv",
            "--pointing-table",
            "calibration/aia_pointing.ecsv",
            "--longitude-deg",
            "575",
            "--latitude-deg",
            "-12",
            "--width-deg",
            "8",
            "--height-deg",
            "4",
            "--boundary-samples-per-axis",
            "41",
            "--observation-id",
            "aia_sample",
        ]
    )

    assert captured["acquisition_directory"] == Path("data/sdo/raw/aia")
    assert captured["output_directory"] == Path("data/sdo/prepared/aia")
    assert captured["correction_table_path"] == Path("calibration/aia_correction.ecsv")
    assert captured["pointing_table_path"] == Path("calibration/aia_pointing.ecsv")
    assert captured["observation_id"] == "aia_sample"
    footprint = captured["footprint"]
    assert footprint.longitude_deg == 215.0
    assert footprint.latitude_deg == -12.0
    assert footprint.width_deg == 8.0
    assert footprint.height_deg == 4.0
    assert footprint.boundary_samples_per_axis == 41
    assert "uncertainty_model" not in captured
    assert capsys.readouterr().out.strip() == "/tmp/prepared-aia"


def test_validate_config_prints_resolved_schema(capsys):
    main(
        [
            "validate-config",
            str(PROJECT_ROOT / "configs" / "hmi_lte_dynamic.yaml"),
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == 3
    assert document["solver"]["kind"] == "joint"
    assert len(document["streams"]) == 1
    assert document["streams"][0]["observation"]["type"] == "hmi_stokes"


def test_export_dispatches_p3s_and_preserves_sampling_options(monkeypatch, capsys):
    import prom3theus.artifacts.export as exporter

    captured = {}

    def export(save_state, output, **options):
        captured.update(save_state=save_state, output=output, **options)
        return output

    monkeypatch.setattr(exporter, "export_save_state", export)
    main(
        [
            "export",
            "run/state.p3s",
            "atmosphere.npz",
            "--depth-samples",
            "17",
            "--include-stokes",
            "--include-full-shell",
            "--full-shell-samples",
            "23",
            "--device",
            "cpu",
        ]
    )
    assert captured["save_state"] == Path("run/state.p3s")
    assert captured["depth_samples"] == 17
    assert captured["full_shell_samples"] == 23
    assert captured["include_stokes"] is True
    assert captured["include_full_shell"] is True
    assert "atmosphere.npz" in capsys.readouterr().out
