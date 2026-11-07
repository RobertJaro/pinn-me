"""Publish completed synthetic rasters without retaining the entire sequence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from prom3theus.observations.store import MANIFEST_FILENAME, ObservationStore


def prepare_synthesis_output(args, options):
    """Open a resumable output, or explicitly replace it, without touching inputs."""
    destination = args.out.resolve()
    source = args.source_path.resolve()
    if destination == source or destination in source.parents:
        raise ValueError("Synthesis output must not be the source directory or its ancestor.")
    if args.out.is_symlink():
        raise ValueError("Synthesis output must not be a symbolic link.")
    if args.out.exists() and not args.out.is_dir():
        raise NotADirectoryError(args.out)
    contract = {
        "muram_source_path": str(source),
        "output_format": args.output_format,
        "stride": args.stride,
        "binning": args.binning,
        "dtype": args.dtype,
        "vertical_axis": args.vertical_axis,
        "component_order": list(args.component_order),
        "options": options,
        "depth_sampling": "native_cells",
    }
    # The CLI supplies lists; the Python SDO entry point also accepts NumPy grids.
    contract = json.loads(json.dumps(
        contract, default=lambda value: value.tolist(), allow_nan=False,
    ))
    if args.overwrite and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "synthesis_run.json"
    status_path = args.out / "synthesis_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if contract_path.exists():
        recorded = json.loads(contract_path.read_text())
    else:
        # Earlier runs recorded only these fields. Keep their completed products
        # and validate every setting available in their status/manifest metadata.
        recorded = dict(previous)
        if (args.out / MANIFEST_FILENAME).exists():
            recorded.update(ObservationStore.manifest(args.out)["metadata"])
            recorded["output_format"] = "store"
        elif list(args.out.glob("muram_*.npz")):
            recorded["output_format"] = "store"
        if not recorded and any(args.out.iterdir()):
            raise ValueError(f"Unrecognized synthesis output: {args.out}; use --overwrite to replace it.")
    if recorded:
        recorded.setdefault("depth_sampling", "legacy_log_tau")
    mismatches = [key for key in contract if key in recorded and recorded[key] != contract[key]]
    if mismatches:
        raise ValueError(
            f"Synthesis settings changed ({', '.join(mismatches)}); use a new --out or --overwrite."
        )
    # Publish the contract before the first expensive step so failed first
    # snapshots can resume too. Batch size and device may change on resume.
    with tempfile.NamedTemporaryFile(mode="w", dir=args.out, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(contract, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            os.replace(temporary, contract_path)
        finally:
            temporary.unlink(missing_ok=True)
    return previous


def completed_store_iterations(destination, *, aia=False):
    destination = Path(destination)
    if aia:
        return sorted(int(path.stem.removeprefix("muram_"))
                      for path in destination.glob("muram_*.npz"))
    if not (destination / MANIFEST_FILENAME).exists():
        return []
    manifest = ObservationStore.manifest(destination)
    completed = []
    for raster in manifest["rasters"]:
        for record in raster["arrays"].values():
            if not (destination / record["file"]).is_file():
                raise ValueError(f"Incomplete published raster {raster['name']}; use --overwrite.")
        completed.append(int(raster["name"].removeprefix("muram_")))
    return completed


def write_status(destination: Path, **status) -> None:
    """Atomically expose progress, including before the first raster completes."""
    with tempfile.NamedTemporaryFile(mode="w", dir=destination, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(status, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            os.replace(temporary, destination / "synthesis_status.json")
        finally:
            temporary.unlink(missing_ok=True)


def save_completed_raster(destination, raster, *, name, source_signature, metadata):
    """Atomically extend the manifest after publishing a completed raster's arrays.

    Previous manifest entries and arrays remain untouched. If writing fails,
    readers can still load the last completed sequence. Unreferenced arrays
    from an interrupted publication are never interpreted as complete rasters.
    """
    destination = Path(destination)
    previous = (
        ObservationStore.manifest(destination)
        if (destination / MANIFEST_FILENAME).exists()
        else None
    )
    if previous and any(record["name"] == name for record in previous["rasters"]):
        raise FileExistsError(f"Raster already published: {name}")
    with tempfile.TemporaryDirectory(prefix=".raster-", dir=destination) as temporary:
        staging = Path(temporary) / "store"
        ObservationStore.save_sequence(
            staging,
            [raster],
            raster_names=[name],
            source_signature=source_signature,
            metadata=metadata,
        )
        manifest = ObservationStore.manifest(staging)
        for record in manifest["rasters"][0]["arrays"].values():
            original = staging / record["file"]
            filename = f"{name}_{record['file']}"
            target = destination / filename
            # An interrupted publication may leave arrays without a manifest
            # entry. Only those unpublished arrays may be replaced on retry.
            if previous and any(
                array["file"] == filename
                for raster_record in previous["rasters"]
                for array in raster_record["arrays"].values()
            ):
                raise FileExistsError(f"Array already published: {target}")
            os.replace(original, target)
            record["file"] = filename
        manifest["rasters"] = (
            [] if previous is None else previous["rasters"]
        ) + manifest["rasters"]
        path = staging / MANIFEST_FILENAME
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(path, destination / MANIFEST_FILENAME)
