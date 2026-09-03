"""One discoverable command-line interface for the LTE framework."""

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
        description="Train, inspect, and export LTE spectropolarimetric inversions.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    invert = commands.add_parser("invert", help="run an LTE inversion")
    invert.add_argument("config", type=Path, help="Hinode or HMI YAML configuration")
    invert.add_argument(
        "--rebuild-observations",
        action="store_true",
        help="replace the prepared observation cache before training",
    )

    export = commands.add_parser("export", help="export a trained inversion artifact")
    export.add_argument("artifact", type=Path, help="artifact directory")
    export.add_argument("output", type=Path, help="output NPZ path")
    export.add_argument("--depth-samples", type=int, default=101)
    export.add_argument("--batch-size", type=int, default=4096)
    export.add_argument("--include-stokes", action="store_true")
    export.add_argument("--stokes-batch-size", type=int, default=16)
    export.add_argument("--device", default="auto")
    export.add_argument(
        "--storage-dtype", choices=("float32", "float64"), default="float32"
    )

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
        "--start", required=True, help="inclusive start timestamp in TAI"
    )
    hmi_download.add_argument(
        "--end", required=True, help="exclusive end timestamp in TAI"
    )

    prepare = commands.add_parser("prepare", help="prepare local observation inputs")
    prepare_commands = prepare.add_subparsers(dest="prepare_command", required=True)
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
        "hmi-responses", help="prepare offline HMI filter responses for FITS inputs"
    )
    hmi.add_argument("inputs", nargs="+")
    hmi.add_argument("--output", type=Path, required=True)
    hmi.add_argument("--email", default=os.environ.get("JSOC_EMAIL"))
    hmi.add_argument(
        "--phase-map-fsn",
        type=int,
        required=True,
        help="authoritative hmi.phasemaps_extended FSN for these acquisitions",
    )
    hmi.add_argument("--overwrite", action="store_true")

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
        from prom3theus.inversion.runner import run_inversion

        run_inversion(config, rebuild_observations=args.rebuild_observations)
        return

    if args.command == "export":
        from prom3theus.artifacts.export import export_artifact

        output = export_artifact(
            args.artifact,
            args.output,
            depth_samples=args.depth_samples,
            batch_size=args.batch_size,
            include_stokes=args.include_stokes,
            stokes_batch_size=args.stokes_batch_size,
            storage_dtype=args.storage_dtype,
            device=args.device,
        )
        print(output)
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
        from prom3theus.instruments.hmi.download import download_hmi_stokes

        downloaded = download_hmi_stokes(
            output_directory=args.output,
            email=args.email,
            start=args.start,
            end=args.end,
        )
        print(json.dumps([str(path) for path in downloaded], indent=2))
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
            phase_map_fsn=args.phase_map_fsn,
            overwrite=args.overwrite,
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
