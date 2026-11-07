"""Save one MURaM snapshot at a time as synthetic SDO training products."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import os
import shutil
import tempfile

from .checkpoint import prepare_synthesis_output, write_status
from .columns import close_snapshot, snapshot_columns, strided_columns
from .fits import (
    prepare_aia_fits, write_aia_fits, write_hmi_fits,
    write_hmi_transmission_profiles,
)
from .reader import MURaMSnapshot
from .synthesize import AIAAccumulator, synthesize_hmi


def _completed_products(root, previous):
    products = previous.get("snapshot_products", {})
    if not products and previous.get("completed_iterations"):
        # Recover product lists from outputs written before resume support.
        from astropy.io import fits

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
    args = SimpleNamespace(**({
        "output_format": "sdo-fits", "binning": None,
        "component_order": (0, 1, 2), "overwrite": False,
        "dtype": str(dtype).removeprefix("torch."),
    } | vars(args)))
    root = args.out.resolve()
    previous = prepare_synthesis_output(args, options)
    products = _completed_products(root, previous)
    completed = [int(key) for key in products]
    protected = {path for paths in products.values() for path in paths}
    if (previous.get("state") == "complete"
            and set(args.iterations).issubset(completed)
            and (root / "aia" / "prepared" / "manifest.json").is_file()):
        print(f"All requested snapshots are complete: {root}", flush=True)
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
            stride=args.stride,
            binning=getattr(args, "binning", None),
            **extra,
        )

    status("starting")
    print(f"SDO FITS: {root.resolve()}", flush=True)
    snapshot = None
    try:
        for iteration in args.iterations:
            if iteration in completed:
                print(f"Skipping completed iteration {iteration}", flush=True)
                continue
            status("reading_snapshot")
            print(f"Reading MURaM {iteration} ...", flush=True)
            snapshot = MURaMSnapshot(
                str(args.source_path), iteration, vertical_axis=args.vertical_axis,
                component_order=tuple(getattr(args, "component_order", (0, 1, 2))),
            )
            quantities = snapshot_columns(snapshot)
            selected, step = strided_columns(quantities, args.stride)
            aia_accumulator = AIAAccumulator(
                selected, stride=step, binning=args.binning, device=device, dtype=dtype,
            )

            common = dict(
                stride=args.stride,
                binning=getattr(args, "binning", None),
                batch_size=args.batch_size,
                progress=not args.no_progress,
                device=device,
                dtype=dtype,
            )
            status("synthesizing_hmi")
            hmi = synthesize_hmi(snapshot, column_callback=aia_accumulator, **common, **options)
            aia = aia_accumulator.result()
            del aia_accumulator
            status("saving_hmi_fits")
            # Stage each instrument before replacing an interrupted snapshot's
            # files. Completed snapshots are protected from replacement.
            with tempfile.TemporaryDirectory(prefix=".hmi-", dir=root) as temporary:
                staging = Path(temporary)
                hmi_files = write_hmi_fits(
                    staging / "hmi" / "prepared", hmi,
                    time_s=snapshot.time_s, iteration=iteration,
                )
                write_hmi_transmission_profiles(staging / "hmi" / "responses", hmi_files[0])
                snapshot_files = _publish_products(staging, root, protected)
            geometry = hmi.geometry
            del hmi
            print(f"Saved HMI {iteration}: 24 FITS segments and transmission profile", flush=True)
            status("saving_aia_fits")
            with tempfile.TemporaryDirectory(prefix=".aia-", dir=root) as temporary:
                staging = Path(temporary)
                write_aia_fits(
                    staging / "aia" / "fits", aia, geometry,
                    time_s=snapshot.time_s, iteration=iteration,
                )
                snapshot_files += _publish_products(staging, root, protected)
            products[str(iteration)] = snapshot_files
            protected.update(snapshot_files)
            completed.append(iteration)
            status("snapshot_saved")
            print(
                f"Saved MURaM {iteration}: HMI + AIA 171/193/211 FITS",
                flush=True,
            )
            del aia, geometry, quantities, selected
            close_snapshot(snapshot)
            snapshot = None
        status("preparing_aia_training_store")
        print(
            "Preparing the standard AIA training store from saved FITS ...", flush=True
        )
        aia_files = sorted(root / path for paths in products.values() for path in paths
                           if path.startswith("aia/fits/"))
        prepared = root / "aia" / "prepared"
        if prepared.exists():
            shutil.rmtree(prepared)
        prepare_aia_fits(aia_files, prepared)
        status("complete")
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close_snapshot(snapshot)
