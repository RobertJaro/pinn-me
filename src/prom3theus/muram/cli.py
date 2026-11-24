"""``python -m prom3theus.muram.cli ...``: the normal prom3theus CLI, with the
``muram_stored`` observation type registered, plus a ``synthesize`` command
that forward-renders packaged-instrument observables from MURaM snapshots.

The schema extension in :mod:`prom3theus.muram.config` must be imported
before a yaml config using ``observation: {type: muram_stored, ...}`` is
parsed; this is the supported way to get that ordering right from the
command line for stored synthetic observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from astropy.io import fits

from prom3theus.cli.main import main as delegate_main
from prom3theus.config import load_config
from prom3theus.muram import config as _register_muram_schema  # noqa: F401
from prom3theus.observations.store import ObservationStore

from . import sdo
from . import synthesize as synthesis
from .atmosphere_builder import DEFAULT_STOKES_HEIGHT_MEGAMETERS, UniformShellGrid
from .checkpoint import (
    completed_store_iterations,
    prepare_synthesis_output,
    save_completed_raster,
    write_status,
)
from .columns import close_snapshot
from .configuration import validate_execution, validate_options
from .diagnostics import ensure_diagnostics
from .fits import read_metadata, write_hmi_transmission_profiles
from .reader import MURaMSnapshot, _FIELD_IDS, _REQUIRED_FOR_SYNTHESIS
from .reference import verify_native_reference
from .reporting import LOG_LEVELS, cli_logging, progress_bar, timed_stage

logger = logging.getLogger(__name__)


def _source_file_metadata(paths):
    """Identify native inputs without a second full read of the simulation cubes."""
    records = []
    for filename in paths:
        path = Path(filename).resolve()
        stat = path.stat()
        records.append({
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        })
    return records


def _synthesize(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="prom3theus.muram.cli synthesize",
        description=(
            "Render MURaM snapshots into HMI+AIA FITS and training inputs; "
            "optionally write single-instrument stores."
        ),
    )
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the output directory; otherwise resume and skip completed snapshots.",
    )
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output-format",
        choices=("sdo-fits", "store"),
        default="sdo-fits",
        help="SDO HMI+AIA FITS (default), or single-instrument stores/NPZ.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Select every Nth native column along both horizontal axes before synthesis; omitted means all columns.",
    )
    parser.add_argument(
        "--binning", "-binning", type=int, default=None,
        help="Optional lossy NxN output averaging; default preserves native pixels. Prefer observation.binning during loading.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars and periodic progress messages; stage logs remain."
    )
    parser.add_argument("--vertical-axis", type=int, choices=(0, 1, 2), default=None,
                        help="Vertical storage/header axis; default is the unique finest-spaced axis. ")
    parser.add_argument("--component-order", type=int, nargs=3, choices=(0, 1, 2),
                        default=(0, 1, 2), help="Raw vector component indices aligned with original header axes 0/1/2; default 0 1 2.")
    parser.add_argument("--batch-size", type=int, default=256)
    depth = parser.add_mutually_exclusive_group()
    depth.add_argument("--inversion-depth-grid", type=Path, metavar="CONFIG",
                       help="Use the uniform ray distances and shell bounds of a single inversion window.")
    parser.add_argument("--photosphere-height-megameters", type=float,
                        help="Native height at the nominal solar radius for --inversion-depth-grid.")
    depth.add_argument("--stokes-height-megameters", type=float,
                        help="Stokes upper height in megameters above the bottom native cell center (default: 3). AIA retains full height.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA if available, otherwise CPU), cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Transfer precision; float32 is the default, float64 is available for reference checks.",
    )
    parser.add_argument(
        "--instrument-config",
        type=Path,
        help="YAML containing instrument, observed_wavelength_angstrom, line_ids, and optional response",
    )
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    args = parser.parse_args(argv)
    with cli_logging(args.log_level):
        if args.batch_size < 1:
            parser.error("batch-size must be a positive integer")
        if args.stride is not None and args.stride < 1:
            parser.error("stride must be a positive integer")
        if args.binning is not None and args.binning < 1:
            parser.error("binning must be a positive integer")
        if sorted(args.component_order) != [0, 1, 2]:
            parser.error("component-order must be a permutation of 0 1 2")
        if len(set(args.iterations)) != len(args.iterations):
            parser.error("iterations must be unique")
        options = {}
        if args.instrument_config:
            with args.instrument_config.open() as handle:
                options = yaml.safe_load(handle)
            if not isinstance(options, dict):
                parser.error("instrument-config must contain a YAML mapping")
        aia = options.get("instrument", {}).get("type") == "aia_euv"
        if aia:
            options = dict(options)
            instrument = dict(options.pop("instrument"))
            instrument.pop("type")
            options.update(instrument)
        device = synthesis.resolve_synthesis_device(args.device)
        dtype = getattr(torch, args.dtype)
        logger.info("Synthesis device=%s, dtype=%s, batch_size=%d", device, args.dtype, args.batch_size)
        if device.type == "cuda":
            logger.info("GPU: %s", torch.cuda.get_device_name(device))
        synthesize = (
            synthesis.synthesize_aia
            if aia
            else (synthesis.synthesize_stokes if args.instrument_config else synthesis.synthesize_hmi)
        )

        try:
            validate_options(synthesize, options, kind="aia" if aia else "stokes")
            shell = None
            if (args.inversion_depth_grid is None) != (args.photosphere_height_megameters is None):
                raise ValueError("--inversion-depth-grid and --photosphere-height-megameters must be supplied together.")
            if args.inversion_depth_grid is not None:
                if synthesize is not synthesis.synthesize_stokes or args.output_format != "store":
                    raise ValueError("--inversion-depth-grid requires a Stokes instrument config and store output.")
                windows = load_config(args.inversion_depth_grid).synthesis.windows
                if len(windows) != 1:
                    raise ValueError("--inversion-depth-grid requires one synthesis window.")
                window = windows[0]
                if window.coarse_to_fine.enabled or window.jitter_fraction or window.shell_height_bounds_megameter is None:
                    raise ValueError("The inversion window must specify fixed shell bounds, zero jitter and no refinement.")
                shell = UniformShellGrid(window.depth_count, window.shell_height_bounds_megameter,
                                         args.photosphere_height_megameters)
                options["uniform_shell_grid"] = asdict(shell)
                args.stokes_height_megameters = shell.native_height_range_m[1] / 1e6
            elif args.stokes_height_megameters is None:
                args.stokes_height_megameters = DEFAULT_STOKES_HEIGHT_MEGAMETERS
            validate_execution(args, uniform_shell_grid=shell)
        except (TypeError, ValueError, KeyError, OSError) as error:
            parser.error(str(error))

        if args.output_format == "sdo-fits":
            if args.instrument_config:
                parser.error(
                    "--instrument-config requires --output-format store; sdo-fits synthesizes HMI plus AIA 171/193/211."
                )
            sdo.synthesize_sdo_sequence(args, options, device=device, dtype=dtype)
            return
        previous = prepare_synthesis_output(
            args, options, instrument_types=("aia_euv",) if aia else None,
        )
        completed = completed_store_iterations(args.out, aia=aia)
        run_metadata = {
            "source": "prom3theus.muram synthetic forward-render",
            "muram_source_path": str(args.source_path.resolve()),
            "requested_iterations": list(args.iterations),
            "stride": args.stride,
            "binning": args.binning,
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "output_format": args.output_format,
            "source_signature_method": "file_metadata_v1",
            "depth_sampling": "uniform_shell_ray_distance" if shell else "native_cells",
            **({"uniform_shell_grid": asdict(shell)} if shell else {}),
            "stokes_height_megameters": None if aia else args.stokes_height_megameters,
            "generator_provenance": previous["generator_provenance"],
        }
        digest = hashlib.sha256(
            json.dumps(
                {
                    **run_metadata,
                    "options": options,
                    "contract_version": 4,
                    "vertical_axis": args.vertical_axis,
                    "component_order": args.component_order,
                },
                sort_keys=True,
            ).encode()
        )
        if completed and not aia:
            digest.update(ObservationStore.manifest(args.out)["source_signature"].encode())
        references = dict(previous.get("native_references", {}))
        if not aia and completed:
            # Raster metadata survives even if a process died before status publication.
            for record in ObservationStore.manifest(args.out)["rasters"]:
                reference = record.get("metadata", {}).get("muram_synthesis", {}).get("native_reference")
                if reference:
                    references[str(int(record["name"].removeprefix("muram_")))] = reference
        if aia:
            for done in completed:
                with np.load(args.out / f"muram_{done:06d}.npz", allow_pickle=False) as archive:
                    if "metadata_json" in archive:
                        reference = json.loads(str(archive["metadata_json"].item())).get("native_reference")
                        if reference:
                            references[str(done)] = reference
        for done in completed:
            reference = references.get(str(done))
            if not reference:
                raise ValueError(f"Completed iteration {done} has no native source reference; use --overwrite.")
            verify_native_reference(reference)
        iteration = None

        def status(state, **details):
            write_status(
                args.out,
                state=state,
                current_iteration=iteration,
                completed_iterations=completed,
                native_references=references,
                **run_metadata,
                **details,
            )

        status("starting")
        logger.info("Store output: %s (%d requested, %d already complete)", args.out.resolve(),
                    len(args.iterations), sum(i in completed for i in args.iterations))
        snapshot = None
        try:
            with progress_bar(args.iterations, description="Snapshots", unit="snapshot",
                              enabled=not args.no_progress) as iterations:
                for iteration in iterations:
                    if iteration in completed:
                        ensure_diagnostics(args.out, iteration, references[str(iteration)],
                                           batch_size=args.batch_size, progress=not args.no_progress)
                        logger.info("Skipping completed iteration %s", iteration)
                        continue
                    with timed_stage(logger, "Iteration %s: reading snapshot header", iteration):
                        snapshot = MURaMSnapshot(
                            str(args.source_path), iteration, vertical_axis=args.vertical_axis,
                            component_order=tuple(args.component_order),
                        )
                    status("reading_source_metadata")

                    paths = [snapshot._field_path("Header")]
                    paths += [
                        snapshot._field_path(_FIELD_IDS[name])
                        for name in _REQUIRED_FOR_SYNTHESIS
                    ]
                    source_files = _source_file_metadata(paths)
                    digest.update(json.dumps(source_files, sort_keys=True).encode())
                    status("synthesizing")
                    logger.info("Iteration %s: synthesizing from %s", iteration, args.source_path)
                    result = synthesize(
                        snapshot,
                        stride=args.stride,
                        binning=args.binning,
                        batch_size=args.batch_size,
                        progress=not args.no_progress,
                        device=device,
                        dtype=dtype,
                        **({} if aia else {"stokes_height_megameters": args.stokes_height_megameters}),
                        **options,
                    )
                    if result.native_reference:
                        verify_native_reference(result.native_reference)
                        references[str(iteration)] = result.native_reference
                    status("saving")
                    with timed_stage(logger, "Iteration %s: saving instrument products", iteration):
                        if aia:
                            output = args.out / f"muram_{iteration:06d}.npz"
                            temporary = output.with_suffix(".tmp")
                            try:
                                with temporary.open("wb") as handle:
                                    np.savez_compressed(
                                        handle,
                                        intensity=result.intensity.numpy(),
                                        channels_angstrom=result.channels_angstrom,
                                        dx_m=result.dx_m,
                                        dy_m=result.dy_m,
                                        time_s=snapshot.time_s,
                                        intensity_unit="DN s^-1 reference_AIA_pixel^-1",
                                        response_resource=result.response_resource,
                                        metadata_json=np.array(json.dumps({
                                            "native_reference": result.native_reference,
                                            "generator_provenance": previous["generator_provenance"],
                                        }, sort_keys=True)),
                                    )
                                os.replace(temporary, output)
                            finally:
                                temporary.unlink(missing_ok=True)
                        else:
                            raster = result.to_observation_raster()
                            digest.update(
                                json.dumps(
                                    dict(raster.metadata["muram_synthesis"]), sort_keys=True
                                ).encode()
                            )
                            save_completed_raster(
                                args.out,
                                raster,
                                name=f"muram_{iteration:06d}",
                                source_signature=digest.hexdigest(),
                                metadata={
                                    **run_metadata,
                                    "latest_snapshot_source_files": source_files,
                                    "muram_iterations": [*completed, iteration],
                                    "complete": set(args.iterations).issubset([*completed, iteration]),
                                },
                            )
                            del raster
                    completed.append(iteration)
                    status("snapshot_saved")
                    ensure_diagnostics(args.out, iteration, result.native_reference, result=result.diagnostics,
                                       batch_size=args.batch_size, progress=not args.no_progress)
                    logger.info("Iteration %s complete: %s", iteration, args.out)
                    # Release native snapshot arrays and raw profiles before loading
                    # the next snapshot. Only small manifest records accumulate.
                    del result
                    close_snapshot(snapshot)
                    snapshot = None
            status("complete")
        except BaseException as error:
            logger.error("Synthesis stopped at iteration %s: %s", iteration, error)
            status("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            close_snapshot(snapshot)
        logger.info("All %d requested snapshots complete: %s", len(args.iterations), args.out)


def _diagnostics(argv):
    """Repair diagnostics even when the instrument generator has since changed."""

    parser = argparse.ArgumentParser(description="Create native diagnostic maps or repair overviews without spectral synthesis.")
    parser.add_argument("--out", type=Path, required=True, help="Existing synthesis output root")
    parser.add_argument("--iterations", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    args = parser.parse_args(argv)
    with cli_logging(args.log_level):
        if not args.out.is_dir() or args.batch_size < 1:
            parser.error("An existing output directory and positive batch size are required.")
        references = {}
        status = args.out / "synthesis_status.json"
        if status.is_file():
            references.update(json.loads(status.read_text()).get("native_references", {}))
        # Recover references from durable instrument products when status was lost.
        manifest = args.out / "manifest.json"
        if manifest.is_file():
            for raster in json.loads(manifest.read_text())["rasters"]:
                reference = raster.get("metadata", {}).get("muram_synthesis", {}).get("native_reference")
                if reference:
                    references[str(reference["iteration"])] = reference
        for path in args.out.glob("hmi/prepared/*.I0.fits"):
            with fits.open(path) as hdus:
                reference = read_metadata(hdus).get("muram_synthesis", {}).get("native_reference")
            if reference:
                references[str(reference["iteration"])] = reference
        for path in args.out.glob("muram_*.npz"):
            with np.load(path, allow_pickle=False) as archive:
                reference = json.loads(str(archive["metadata_json"].item())).get("native_reference") if "metadata_json" in archive else None
            if reference:
                references[str(reference["iteration"])] = reference
        available = set(map(int, references)) | {int(path.stem.removeprefix("muram_"))
                                              for path in (args.out / "diagnostics").glob("muram_*.npz")}
        iterations = args.iterations if args.iterations is not None else sorted(available)
        if not iterations or len(set(iterations)) != len(iterations) or not set(iterations) <= available:
            parser.error("Requested iterations must be unique and have saved maps or a native source reference.")
        with progress_bar(iterations, description="Diagnostics", unit="snapshot",
                          enabled=not args.no_progress) as snapshots:
            for iteration in snapshots:
                paths = ensure_diagnostics(args.out, iteration, references.get(str(iteration)),
                                           batch_size=args.batch_size, progress=not args.no_progress)
                logger.info("Diagnostics %s: %s and %s", iteration, paths[0], paths[1])


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "diagnostics":
        _diagnostics(argv[1:])
        return
    if argv and argv[0] == "prepare-responses":
        parser = argparse.ArgumentParser(
            description="Export exact synthetic HMI transmission profiles from existing I0 FITS."
        )
        parser.add_argument("--source-path", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--no-progress", action="store_true")
        parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
        args = parser.parse_args(argv[1:])
        with cli_logging(args.log_level):
            files = sorted(args.source_path.glob("*.I0.fits"))
            if not files:
                parser.error(f"No synthetic I0 FITS found in {args.source_path}")
            with progress_bar(files, description="Transmission profiles", unit="file",
                              enabled=not args.no_progress) as inputs:
                for path in inputs:
                    output = write_hmi_transmission_profiles(args.out, path)
                    logger.info("Saved transmission profile: %s", output)
        return
    if argv and argv[0] == "synthesize":
        _synthesize(argv[1:])
        return

    delegate_main(argv)


if __name__ == "__main__":
    main()
