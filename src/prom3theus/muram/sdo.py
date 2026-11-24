"""Save one MURaM snapshot at a time as synthetic SDO training products."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from astropy.io import fits

from .checkpoint import prepare_synthesis_output, write_status
from .columns import close_snapshot
from .configuration import validate_execution, validate_options
from .diagnostics import ensure_diagnostics
from .fits import (
    prepare_aia_fits,
    read_metadata,
    write_aia_fits,
    write_hmi_fits,
    write_hmi_transmission_profiles,
)
from .reader import MURaMSnapshot
from .reference import verify_native_reference
from .synthesize import resolve_synthesis_device, synthesize_hmi, synthesize_sdo
from .reporting import progress_bar, timed_stage
from .atmosphere_builder import DEFAULT_STOKES_HEIGHT_MEGAMETERS

logger = logging.getLogger(__name__)


def _completed_products(root, previous):
    products = previous.get("snapshot_products", {})
    if not products and previous.get("completed_iterations"):
        # Recover product lists from outputs written before resume support.

        grouped = {}
        for directory in (root / "hmi" / "prepared", root / "aia" / "fits"):
            for path in directory.glob("*.fits"):
                iteration = str(fits.getheader(path)["MURAMIT"])
                grouped.setdefault(iteration, []).append(path)
                if path.name.endswith(".I0.fits"):
                    grouped[iteration].append(
                        root / "hmi" / "responses"
                        / f"{path.name.removesuffix('.I0.fits')}.response.npz"
                    )
        for iteration in previous["completed_iterations"]:
            paths = grouped.get(str(iteration), [])
            if len(paths) != 28:  # 24 HMI, one response, three AIA
                continue
            products[str(iteration)] = [str(path.relative_to(root)) for path in paths]
    # Older runs also listed JPEG diagnostics; only instrument products are required.
    products = {key: [path for path in paths if not path.endswith(".jpg")]
                for key, paths in products.items()}
    return {key: paths for key, paths in products.items()
            if len(paths) == 28 and all((root / path).is_file() for path in paths)}


def _publish_products(staging, root, protected):
    paths = []
    for source in sorted(staging.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(staging)
        if str(relative) in protected:
            raise FileExistsError(f"Product belongs to a completed snapshot: {root / relative}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        paths.append(str(relative))
    return paths


def synthesize_sdo_sequence(args, options, *, device, dtype):
    device = resolve_synthesis_device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Synthesis dtype must be torch.float32 or torch.float64.")
    args = SimpleNamespace(**({
        "output_format": "sdo-fits", "binning": None,
        "component_order": (0, 1, 2), "overwrite": False,
        "dtype": str(dtype).removeprefix("torch."),
        "stokes_height_megameters": DEFAULT_STOKES_HEIGHT_MEGAMETERS,
    } | vars(args)))
    validate_execution(args)
    validate_options(synthesize_hmi, options, kind="stokes")
    root = args.out.resolve()
    previous = prepare_synthesis_output(args, options)
    products = _completed_products(root, previous)
    references = dict(previous.get("native_references", {}))
    for key, paths in products.items():
        # The FITS contract is durable even if status publication was interrupted.
        if not references.get(key):
            first = next(root / path for path in paths if path.endswith(".I0.fits"))
            with fits.open(first) as hdus:
                reference = read_metadata(hdus).get("muram_synthesis", {}).get("native_reference")
            if reference:
                references[key] = reference
        if not references.get(key):
            raise ValueError(f"Completed iteration {key} has no native source reference; use --overwrite.")
        verify_native_reference(references[key])
    completed = [int(key) for key in products]
    protected = {path for paths in products.values() for path in paths}
    if (previous.get("state") == "complete"
            and set(args.iterations).issubset(completed)
            and (root / "aia" / "prepared" / "manifest.json").is_file()):
        with progress_bar(args.iterations, description="Snapshots", unit="snapshot",
                          enabled=not args.no_progress) as iterations:
            for iteration in iterations:
                ensure_diagnostics(root, iteration, references[str(iteration)],
                                   batch_size=args.batch_size, progress=not args.no_progress)
        logger.info("All %d requested snapshots complete: %s", len(args.iterations), root)
        return
    iteration = None

    def status(state, **extra):
        write_status(
            root,
            state=state,
            current_iteration=iteration,
            completed_iterations=completed,
            snapshot_products=products,
            requested_iterations=args.iterations,
            output_format="sdo-fits",
            depth_sampling="native_cells",
            stokes_height_megameters=args.stokes_height_megameters,
            generator_provenance=previous["generator_provenance"],
            native_references=references,
            stride=args.stride,
            binning=getattr(args, "binning", None),
            **extra,
        )

    status("starting")
    logger.info("SDO output: %s (%d requested, %d already complete)", root, len(args.iterations),
                sum(i in completed for i in args.iterations))
    snapshot = None
    try:
        with progress_bar(args.iterations, description="Snapshots", unit="snapshot",
                          enabled=not args.no_progress) as iterations:
            for iteration in iterations:
                if iteration in completed:
                    ensure_diagnostics(root, iteration, references[str(iteration)],
                                       batch_size=args.batch_size, progress=not args.no_progress)
                    logger.info("Skipping completed iteration %s", iteration)
                    continue
                status("reading_snapshot")
                with timed_stage(logger, "Iteration %s: reading snapshot header", iteration):
                    snapshot = MURaMSnapshot(
                        str(args.source_path), iteration, vertical_axis=args.vertical_axis,
                        component_order=tuple(getattr(args, "component_order", (0, 1, 2))),
                    )
                common = dict(
                    stride=args.stride,
                    binning=getattr(args, "binning", None),
                    batch_size=args.batch_size,
                    progress=not args.no_progress,
                    device=device,
                    dtype=dtype,
                    stokes_height_megameters=args.stokes_height_megameters,
                )
                status("synthesizing_hmi_aia")
                hmi, aia = synthesize_sdo(snapshot, **common, **options)
                if hmi.native_reference:
                    verify_native_reference(hmi.native_reference)
                    references[str(iteration)] = hmi.native_reference
                status("saving_hmi_fits")
                # Stage each instrument before replacing an interrupted snapshot's
                # files. Completed snapshots are protected from replacement.
                with tempfile.TemporaryDirectory(prefix=".hmi-", dir=root) as temporary:
                    staging = Path(temporary)
                    hmi_files = write_hmi_fits(
                        staging / "hmi" / "prepared", hmi,
                        time_s=snapshot.time_s, iteration=iteration, progress=not args.no_progress,
                    )
                    write_hmi_transmission_profiles(staging / "hmi" / "responses", hmi_files[0])
                    snapshot_files = _publish_products(staging, root, protected)
                geometry = hmi.geometry
                diagnostics = hmi.diagnostics
                del hmi
                logger.info("Iteration %s: saved 24 HMI segments and transmission profile", iteration)
                status("saving_aia_fits")
                with tempfile.TemporaryDirectory(prefix=".aia-", dir=root) as temporary:
                    staging = Path(temporary)
                    write_aia_fits(
                        staging / "aia" / "fits", aia, geometry,
                        time_s=snapshot.time_s, iteration=iteration, progress=not args.no_progress,
                    )
                    snapshot_files += _publish_products(staging, root, protected)
                products[str(iteration)] = snapshot_files
                protected.update(snapshot_files)
                completed.append(iteration)
                status("snapshot_saved")
                ensure_diagnostics(root, iteration, aia.native_reference, result=diagnostics,
                                   batch_size=args.batch_size, progress=not args.no_progress)
                logger.info("Iteration %s complete: HMI and AIA products with diagnostics", iteration)
                del aia, geometry, diagnostics
                close_snapshot(snapshot)
                snapshot = None
        status("preparing_aia_training_store")
        aia_files = sorted(root / path for paths in products.values() for path in paths
                           if path.startswith("aia/fits/"))
        prepared = root / "aia" / "prepared"
        if prepared.exists():
            shutil.rmtree(prepared)
        with timed_stage(logger, "Preparing AIA training store from %d saved FITS files", len(aia_files)):
            prepare_aia_fits(aia_files, prepared, progress=not args.no_progress)
        status("complete")
        logger.info("All %d requested snapshots complete: %s", len(args.iterations), root)
    except BaseException as error:
        logger.error("SDO sequence stopped at iteration %s: %s", iteration, error)
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close_snapshot(snapshot)
