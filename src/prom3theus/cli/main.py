"""One command-line interface for stream inversions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from prom3theus import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prom3theus",
        description="Train, inspect, and export multi-observation inversions.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    invert = commands.add_parser("invert", help="run a stream inversion")
    invert.add_argument("config", type=Path, help="Hinode, HMI, or joint HMI/AIA YAML configuration")
    invert.add_argument(
        "--profile", type=Path, metavar="DIRECTORY",
        help="record a bounded CPU/CUDA training trace in this directory",
    )
    invert.add_argument(
        "--rebuild-observations",
        action="store_true",
        help="replace the prepared observation cache before training",
    )

    export = commands.add_parser("export", help="export a trained P3S save state")
    export.add_argument("save_state", type=Path, help="state.p3s path")
    export.add_argument("output", type=Path, help="output NPZ path")
    export.add_argument("--stream", default=None, help="stream ID for raster-based export (default: scene reference)")
    export.add_argument("--depth-samples", type=int, default=101)
    export.add_argument("--batch-size", type=int, default=4096)
    export.add_argument("--include-stokes", action="store_true")
    export.add_argument("--stokes-batch-size", type=int, default=16)
    export.add_argument(
        "--include-full-shell",
        action="store_true",
        help=(
            "also export radial Carrington columns over the complete configured "
            "geometric-height shell"
        ),
    )
    export.add_argument(
        "--full-shell-samples",
        type=int,
        default=101,
        help="outer-to-inner geometric-height samples for --include-full-shell",
    )
    export.add_argument("--device", default="auto")
    export.add_argument(
        "--storage-dtype", choices=("float32", "float64"), default="float32"
    )

    compare_hmi = commands.add_parser(
        "compare-hmi",
        help="compare a P3S save state with a matching native HMI B_720s record",
    )
    compare_hmi.add_argument("save_state", type=Path, help="state.p3s path")
    compare_hmi.add_argument(
        "hmi_directory", type=Path, help="directory containing B_720s FITS segments"
    )
    compare_hmi.add_argument("--output", type=Path, required=True)
    compare_hmi.add_argument("--height-km", type=float, default=0.0)
    compare_hmi.add_argument("--disambig-bit", type=int, choices=(0, 1, 2), default=0)
    compare_hmi.add_argument("--minimum-transverse-gauss", type=float, default=200.0)
    compare_hmi.add_argument("--time-tolerance-seconds", type=float, default=2.0)
    compare_hmi.add_argument("--alignment-tolerance-pixels", type=float, default=0.1)
    compare_hmi.add_argument("--batch-size", type=int, default=4096)
    compare_hmi.add_argument("--device", default="auto")
    compare_hmi.add_argument("--dpi", type=int, default=180)

    time_series = commands.add_parser(
        "time-series",
        help="render spherical atmosphere evolution from a P3S save state",
    )
    time_series.add_argument("save_state", type=Path, help="state.p3s path")
    time_series.add_argument("--output", type=Path, required=True)
    time_series.add_argument("--height-km", type=float, default=0.0)
    time_series.add_argument("--longitude-points", type=int, default=96)
    time_series.add_argument("--latitude-points", type=int, default=96)
    time_series.add_argument("--current-height-samples", type=int, default=48)
    time_series.add_argument("--batch-size", type=int, default=4096)
    time_series.add_argument("--device", default="auto")
    time_series.add_argument("--dpi", type=int, default=120)

    validate_config = commands.add_parser(
        "validate-config",
        help="strictly validate and display a resolved YAML configuration",
    )
    validate_config.add_argument("config", type=Path)

    resources = commands.add_parser("resources", help="inspect pinned LTE resources")
    resource_commands = resources.add_subparsers(dest="resource_command", required=True)
    validate_resources = resource_commands.add_parser(
        "validate", help="validate the packaged production-bundle checksums"
    )
    validate_resources.add_argument(
        "--instrument",
        choices=("hinode_sp", "hmi_stokes"),
        help="display the complete atomic and instrument contract for one instrument",
    )

    download = commands.add_parser("download", help="download observation inputs")
    download_commands = download.add_subparsers(dest="download_command", required=True)
    hmi_download = download_commands.add_parser(
        "hmi-stokes",
        help="download complete native-cadence hmi.S_720s Stokes acquisitions",
    )
    hmi_download.add_argument("--output", type=Path, required=True)
    hmi_download.add_argument("--email", required=True)
    hmi_download.add_argument(
        "--start", required=True, help="inclusive start; unzoned dates use TAI"
    )
    hmi_download.add_argument(
        "--end", required=True, help="exclusive end; unzoned dates use TAI"
    )
    hmi_comparison = download_commands.add_parser(
        "hmi-comparison", help="download one HMI B_720s vector record for compare-hmi"
    )
    hmi_comparison.add_argument("--output", type=Path, required=True)
    hmi_comparison.add_argument("--email", default=os.environ.get("JSOC_EMAIL"))
    hmi_comparison.add_argument(
        "--time", required=True, help="record time; unzoned dates use TAI"
    )
    aia_download = download_commands.add_parser(
        "aia-euv",
        help="download AIA EUV Level-1 files with DRMS",
    )
    aia_download.add_argument("--output", type=Path, required=True)
    aia_download.add_argument("--email", required=True)
    aia_download.add_argument("--start", required=True, help="inclusive start; unzoned dates use UTC")
    aia_download.add_argument("--end", required=True, help="exclusive end; unzoned dates use UTC")
    aia_download.add_argument("--cadence-seconds", type=float, default=720)
    aia_download.add_argument(
        "--channels", dest="channels_angstrom", type=int,
        nargs="+", default=(171, 193, 211),
    )
    prepare = commands.add_parser("prepare", help="prepare local observation inputs")
    prepare_commands = prepare.add_subparsers(dest="prepare_command", required=True)
    aia_calibration = prepare_commands.add_parser(
        "aia-calibration",
        help=(
            "cache V10 correction and event-specific master-pointing tables"
        ),
    )
    aia_calibration.add_argument("--output", type=Path, required=True)
    aia_calibration.add_argument(
        "--overwrite", action="store_true",
        help="download again and replace existing data (default: skip matching data)",
    )
    aia_calibration.add_argument(
        "--start",
        required=True,
        help=(
            "inclusive start (unzoned dates use UTC) for the pointing "
            "support interval"
        ),
    )
    aia_calibration.add_argument(
        "--end",
        required=True,
        help="exclusive end (unzoned dates use UTC)",
    )

    hmi_workflow = prepare_commands.add_parser(
        "hmi", help="check and crop a full HMI Stokes time series"
    )
    hmi_workflow.add_argument("input_directory", type=Path)
    hmi_workflow.add_argument("--output", type=Path, required=True)
    hmi_workflow.add_argument("--longitude-deg", type=float, required=True)
    hmi_workflow.add_argument("--latitude-deg", type=float, required=True)
    hmi_workflow.add_argument("--width-pixels", type=int, required=True)
    hmi_workflow.add_argument("--height-pixels", type=int, required=True)
    aia_workflow = prepare_commands.add_parser(
        "aia", help="preprocess AIA observations with a local calibration bundle and crop"
    )
    aia_workflow.add_argument("input_directory", type=Path)
    aia_workflow.add_argument("--output", type=Path, required=True)
    aia_workflow.add_argument("--calibration-directory", type=Path, required=True)
    aia_workflow.add_argument("--longitude-deg", type=float, required=True)
    aia_workflow.add_argument("--latitude-deg", type=float, required=True)
    aia_workflow.add_argument("--width-pixels", type=int, required=True, help="HMI reference pixels")
    aia_workflow.add_argument("--height-pixels", type=int, required=True, help="HMI reference pixels")
    hmi_subframes = prepare_commands.add_parser(
        "hmi-subframes",
        help="crop complete full-disk HMI acquisitions around a Carrington location",
    )
    hmi_subframes.add_argument("inputs", nargs="+")
    hmi_subframes.add_argument("--output", type=Path, required=True)
    hmi_subframes.add_argument("--longitude-deg", type=float, required=True)
    hmi_subframes.add_argument("--latitude-deg", type=float, required=True)
    hmi_subframes.add_argument("--width-pixels", type=int, required=True)
    hmi_subframes.add_argument("--height-pixels", type=int, required=True)
    hmi_subframes.add_argument("--overwrite", action="store_true")
    hmi = prepare_commands.add_parser(
        "hmi-responses",
        help=(
            "resolve assigned phase maps from JSOC and prepare offline HMI filter "
            "responses for FITS inputs"
        ),
    )
    hmi.add_argument("inputs", nargs="+")
    hmi.add_argument("--output", type=Path, required=True)
    hmi.add_argument("--email", default=os.environ.get("JSOC_EMAIL"))
    hmi.add_argument("--overwrite", action="store_true")
    aia = prepare_commands.add_parser(
        "aia-euv",
        help=(
            "register an immutable AIA acquisition and build a native-grid "
            "image-observation store"
        ),
    )
    aia.add_argument(
        "acquisition",
        type=Path,
        help="immutable AIA acquisition directory containing manifest.json",
    )
    aia.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new immutable ImageObservationStore directory",
    )
    aia.add_argument(
        "--correction-table",
        type=Path,
        required=True,
        help="explicit local aiapy V10 degradation-correction ECSV file",
    )
    aia.add_argument(
        "--pointing-table",
        type=Path,
        required=True,
        help="explicit local aiapy master-pointing ECSV file",
    )
    aia.add_argument(
        "--longitude-deg",
        type=float,
        required=True,
        help="Heliographic Carrington longitude of the cutout center [deg]",
    )
    aia.add_argument(
        "--latitude-deg",
        type=float,
        required=True,
        help="Heliographic Carrington latitude of the cutout center [deg]",
    )
    aia.add_argument(
        "--width-deg",
        type=float,
        required=True,
        help=(
            "full Carrington surface-longitude extent [deg], not pixels, arcsec, or Mm"
        ),
    )
    aia.add_argument(
        "--height-deg",
        type=float,
        required=True,
        help=(
            "full Carrington surface-latitude extent [deg], not pixels, arcsec, or Mm"
        ),
    )
    aia.add_argument(
        "--boundary-samples-per-axis",
        type=int,
        default=33,
        help="footprint-boundary samples per axis (default: 33; minimum: 5)",
    )
    aia.add_argument(
        "--observation-id",
        default="aia_euv",
        help="observation identifier stored in the prepared dataset",
    )
    recovery = commands.add_parser(
        "recovery", help="run the synthetic LTE recovery check"
    )
    recovery.add_argument("--steps", type=int, default=500)
    recovery.add_argument("--device", default="cpu")
    recovery.add_argument("--check", action="store_true")
    return parser


def _config_mapping(path: Path) -> tuple[object, dict]:
    from prom3theus.config import load_config

    config = load_config(path)
    mapping = config.to_dict()
    return config, mapping


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch a CLI command while keeping optional dependencies lazy."""

    args = _build_parser().parse_args(argv)
    if args.command == "validate-config":
        _, mapping = _config_mapping(args.config)
        print(json.dumps(mapping, indent=2, sort_keys=True))
        return

    if args.command == "invert":
        config, _ = _config_mapping(args.config)
        from prom3theus.application.joint_training import run_joint_inversion

        if config.training is None:
            raise SystemExit("Inversion requires a training section.")
        options = {"rebuild_observations": args.rebuild_observations}
        if args.profile is not None:
            import torch
            from pytorch_lightning.profilers import PyTorchProfiler

            options["profiler"] = PyTorchProfiler(
                dirpath=str(args.profile),
                filename="training",
                schedule=torch.profiler.schedule(wait=5, warmup=2, active=3, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
        run_joint_inversion(config, **options)
        return

    if args.command == "export":
        from prom3theus.artifacts.export import export_save_state

        output = export_save_state(
            args.save_state,
            args.output,
            stream_id=args.stream,
            depth_samples=args.depth_samples,
            batch_size=args.batch_size,
            include_stokes=args.include_stokes,
            stokes_batch_size=args.stokes_batch_size,
            include_full_shell=args.include_full_shell,
            full_shell_samples=args.full_shell_samples,
            storage_dtype=args.storage_dtype,
            device=args.device,
        )
        print(output)
        return

    if args.command == "compare-hmi":
        from prom3theus.diagnostics.hmi_comparison import compare_hmi_save_state

        result = compare_hmi_save_state(
            args.save_state,
            args.hmi_directory,
            args.output,
            height_km=args.height_km,
            disambig_bit=args.disambig_bit,
            minimum_transverse_gauss=args.minimum_transverse_gauss,
            time_tolerance_seconds=args.time_tolerance_seconds,
            alignment_tolerance_pixels=args.alignment_tolerance_pixels,
            batch_size=args.batch_size,
            device=args.device,
            dpi=args.dpi,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return

    if args.command == "time-series":
        from prom3theus.diagnostics.time_series import evaluate_p3s_time_series

        result = evaluate_p3s_time_series(
            args.save_state,
            args.output,
            height_km=args.height_km,
            longitude_points=args.longitude_points,
            latitude_points=args.latitude_points,
            current_height_samples=args.current_height_samples,
            batch_size=args.batch_size,
            device=args.device,
            dpi=args.dpi,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return

    if args.command == "resources":
        from prom3theus.resources import (
            validate_instrument_resource_bundle,
            validate_resource_bundle,
        )

        metadata = (
            validate_resource_bundle()
            if args.instrument is None
            else validate_instrument_resource_bundle(args.instrument)
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    if args.command == "download" and args.download_command == "hmi-stokes":
        from prom3theus.download.hmi import download_hmi_stokes

        downloaded = download_hmi_stokes(
            output_directory=args.output,
            email=args.email,
            start=args.start,
            end=args.end,
        )
        print(json.dumps([str(path) for path in downloaded], indent=2))
        return

    if args.command == "download" and args.download_command == "hmi-comparison":
        from prom3theus.download.hmi import download_hmi_comparison

        if not args.email:
            raise ValueError("Set JSOC_EMAIL or pass --email with your registered JSOC email.")
        downloaded = download_hmi_comparison(
            output_directory=args.output, email=args.email, time=args.time,
        )
        print(json.dumps(downloaded, indent=2))
        return

    if args.command == "download" and args.download_command == "aia-euv":
        from prom3theus.download.aia import download_aia_observations

        downloaded = download_aia_observations(
            output_directory=args.output, email=args.email,
            start=args.start, end=args.end,
            cadence_seconds=args.cadence_seconds, channels=args.channels_angstrom,
        )
        print(json.dumps([str(path) for path in downloaded], indent=2))
        return

    if args.command == "prepare" and args.prepare_command == "aia-calibration":
        from prom3theus.instruments.aia_euv.calibration_download import (
            download_aia_preprocessing_calibration,
        )

        manifest = download_aia_preprocessing_calibration(
            output_directory=args.output,
            start_utc=args.start,
            end_utc=args.end,
            overwrite=args.overwrite,
        )
        print(manifest)
        return

    if args.command == "prepare" and args.prepare_command == "hmi":
        from prom3theus.preprocess.hmi import preprocess_hmi_time_series

        result = preprocess_hmi_time_series(
            input_directory=args.input_directory,
            output_directory=args.output,
            longitude_deg=args.longitude_deg,
            latitude_deg=args.latitude_deg,
            width_pixels=args.width_pixels,
            height_pixels=args.height_pixels,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "prepare" and args.prepare_command == "aia":
        from prom3theus.preprocess.aia import preprocess_aia

        result = preprocess_aia(
            input_directory=args.input_directory,
            output_directory=args.output,
            calibration_directory=args.calibration_directory,
            longitude_deg=args.longitude_deg,
            latitude_deg=args.latitude_deg,
            width_pixels=args.width_pixels,
            height_pixels=args.height_pixels,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "prepare" and args.prepare_command == "hmi-subframes":
        from prom3theus.instruments.hmi.subframe import prepare_hmi_subframes

        prepared = prepare_hmi_subframes(
            inputs=args.inputs,
            output_directory=args.output,
            longitude_deg=args.longitude_deg,
            latitude_deg=args.latitude_deg,
            width_pixels=args.width_pixels,
            height_pixels=args.height_pixels,
            overwrite=args.overwrite,
        )
        print(json.dumps([str(path) for path in prepared], indent=2))
        return

    if args.command == "prepare" and args.prepare_command == "hmi-responses":
        if not args.email:
            raise SystemExit("--email or JSOC_EMAIL is required for HMI preparation")
        from prom3theus.instruments.hmi.preparation import (
            prepare_hmi_response_directory,
        )

        result = prepare_hmi_response_directory(
            inputs=args.inputs,
            output_directory=args.output,
            email=args.email,
            overwrite=args.overwrite,
        )
        print(result)
        return

    if args.command == "prepare" and args.prepare_command == "aia-euv":
        from prom3theus.instruments.aia_euv.aiapy_preparation import (
            CarringtonCutout,
            prepare_aiapy_aia_observation_store,
        )

        footprint = CarringtonCutout(
            longitude_deg=args.longitude_deg,
            latitude_deg=args.latitude_deg,
            width_deg=args.width_deg,
            height_deg=args.height_deg,
            boundary_samples_per_axis=args.boundary_samples_per_axis,
        )
        result = prepare_aiapy_aia_observation_store(
            args.acquisition,
            args.output,
            correction_table_path=args.correction_table,
            pointing_table_path=args.pointing_table,
            footprint=footprint,
            observation_id=args.observation_id,
        )
        print(result)
        return

    if args.command == "recovery":
        from prom3theus.diagnostics.recovery import run_recovery

        result = run_recovery(steps=args.steps, device=args.device)
        print(json.dumps(result, indent=2))
        if args.check and (
            result["final_loss"] >= 5.0e-8 or result["max_scaled_error"] >= 0.025
        ):
            raise SystemExit(
                "synthetic recovery did not reach the validation tolerance"
            )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
