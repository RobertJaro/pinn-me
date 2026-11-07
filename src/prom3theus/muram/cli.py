"""``python -m prom3theus.muram.cli ...``: the normal prom3theus CLI, with the
``muram_stored`` observation type registered, plus a ``synthesize`` command
that forward-renders packaged-instrument observables from MURaM snapshots.

The schema extension in :mod:`prom3theus.muram.config` must be imported
before a yaml config using ``observation: {type: muram_stored, ...}`` is
parsed; this is the supported way to get that ordering right from the
command line for stored synthetic observations. ``configs/muram/sdo/joint.yaml`` uses
the ordinary HMI and AIA observation schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from prom3theus.muram import config as _register_muram_schema  # noqa: F401


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
    import torch
    from .reader import MURaMSnapshot
    from .columns import close_snapshot
    from .synthesize import (
        synthesize_hmi,
        synthesize_stokes,
        synthesize_aia,
        resolve_synthesis_device,
    )
    from .checkpoint import (
        completed_store_iterations, prepare_synthesis_output,
        save_completed_raster, write_status,
    )

    parser = argparse.ArgumentParser(
        prog="prom3theus.muram.cli synthesize",
        description=(
            "Render MURaM snapshots into HMI+AIA FITS and training inputs; "
            "optionally write legacy single-instrument stores."
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
        help="SDO HMI+AIA FITS and JPEGs (default), or legacy single-instrument stores/NPZ.",
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
        "--no-progress", action="store_true", help="Disable synthesis progress bars."
    )
    parser.add_argument("--vertical-axis", type=int, choices=(0, 1, 2), default=None,
                        help="Vertical storage/header axis; default is the unique finest-spaced axis. ")
    parser.add_argument("--component-order", type=int, nargs=3, choices=(0, 1, 2),
                        default=(0, 1, 2), help="Raw vector component indices aligned with original header axes 0/1/2; default 0 1 2.")
    parser.add_argument("--batch-size", type=int, default=256)
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
        help="YAML containing instrument, observed_wavelength_angstrom, line_ids, and optional response; synthesis uses all native vertical layers",
    )
    args = parser.parse_args(argv)
    if args.binning is not None and args.binning < 1:
        parser.error("binning must be a positive integer")
    if sorted(args.component_order) != [0, 1, 2]:
        parser.error("component-order must be a permutation of 0 1 2")
    if len(set(args.iterations)) != len(args.iterations):
        parser.error("iterations must be unique")
    options = {}
    if args.instrument_config:
        import yaml

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
    device = resolve_synthesis_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(
        f"Synthesis device={device}, dtype={args.dtype}, batch_size={args.batch_size}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    synthesize = (
        synthesize_aia
        if aia
        else (synthesize_stokes if args.instrument_config else synthesize_hmi)
    )

    if args.output_format == "sdo-fits":
        if args.instrument_config:
            parser.error(
                "--instrument-config requires --output-format store; sdo-fits synthesizes HMI plus AIA 171/193/211."
            )
        from .sdo import synthesize_sdo_sequence

        synthesize_sdo_sequence(args, options, device=device, dtype=dtype)
        return
    prepare_synthesis_output(args, options)
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
        "depth_sampling": "native_cells",
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
        from prom3theus.observations.store import ObservationStore
        digest.update(ObservationStore.manifest(args.out)["source_signature"].encode())
    iteration = None

    def status(state, **details):
        write_status(
            args.out,
            state=state,
            current_iteration=iteration,
            completed_iterations=completed,
            **run_metadata,
            **details,
        )

    status("starting")
    print(
        f"Output: {args.out.resolve()}",
        flush=True,
    )
    snapshot = None
    try:
        for iteration in args.iterations:
            if iteration in completed:
                print(f"Skipping completed iteration {iteration}", flush=True)
                continue
            snapshot = MURaMSnapshot(
                str(args.source_path), iteration, vertical_axis=args.vertical_axis,
                component_order=tuple(args.component_order),
            )
            status("reading_source_metadata")
            from .reader import _FIELD_IDS, _REQUIRED_FOR_SYNTHESIS

            paths = [snapshot._field_path("Header")]
            paths += [
                snapshot._field_path(_FIELD_IDS[name])
                for name in _REQUIRED_FOR_SYNTHESIS
            ]
            source_files = _source_file_metadata(paths)
            digest.update(json.dumps(source_files, sort_keys=True).encode())
            status("synthesizing")
            print(
                f"Synthesizing iteration {iteration} from {args.source_path} ...",
                flush=True,
            )
            result = synthesize(
                snapshot,
                stride=args.stride,
                binning=args.binning,
                batch_size=args.batch_size,
                progress=not args.no_progress,
                device=device,
                dtype=dtype,
                **options,
            )
            status("saving")
            if aia:
                import numpy as np
                import os

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
                            response_resource=options.get(
                                "response_resource",
                                "aia_euv_v1:aia_temperature_response",
                            ),
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
            print(
                f"Saved iteration {iteration} ({len(completed)}/{len(args.iterations)}) to {args.out}",
                flush=True,
            )
            # Release native snapshot arrays and raw profiles before loading
            # the next snapshot. Only small manifest records accumulate.
            del result
            close_snapshot(snapshot)
            snapshot = None
        status("complete")
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close_snapshot(snapshot)
    print(f"Wrote {len(completed)} snapshots to {args.out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "prepare-responses":
        import argparse
        from pathlib import Path
        from .fits import write_hmi_transmission_profiles

        parser = argparse.ArgumentParser(
            description="Export exact synthetic HMI transmission profiles from existing I0 FITS."
        )
        parser.add_argument("--source-path", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        files = sorted(args.source_path.glob("*.I0.fits"))
        if not files:
            parser.error(f"No synthetic I0 FITS found in {args.source_path}")
        for path in files:
            output = write_hmi_transmission_profiles(args.out, path)
            print(f"Saved transmission profile: {output}", flush=True)
        return
    if argv and argv[0] == "synthesize":
        _synthesize(argv[1:])
        return
    from prom3theus.cli.main import main as delegate_main

    delegate_main(argv)


if __name__ == "__main__":
    main()
