"""Independent eager-volume, straight-ray Stokes synthesis command.

Run with ``python -m prom3theus.muram.synthesize_rays --help``. Native Cartesian
volumes are rigidly placed in solar geometry, never reprojected onto a sphere.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from prom3theus.core import carrington_rotation_velocity_cartesian
from prom3theus.instruments.registry import build_instrument
from .configuration import validate_options
from .diagnostic_maps import atomic_path
from .hmi_filter import build_idealized_hmi_filter
from .provenance import synthesis_run_provenance
from .ray_geometry import BOUNDARY_TYPES, BoxPlacement, build_ray_image, sample_ray_batch
from .ray_transfer import RayStokesRenderer
from .ray_volume import DEFAULT_VOLUME_HEIGHT_MEGAMETERS, EagerVolume, volume_depth_count
from .reader import MURaMSnapshot
from .reference import verify_native_reference
from .reporting import LOG_LEVELS, cli_logging, progress_bar, timed_stage
from .synthesize import MURAM_HMI_RADIANCE_SCALE_W_M3_SR, resolve_synthesis_device, synthesize_hmi, synthesize_stokes

logger = logging.getLogger(__name__)
SCHEMA = "muram_ray_stokes_v1"


class RayDataset(Dataset):
    """Flat output-ray indices; worker processes never receive atmosphere cubes."""

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return index


def instrument_options(options=None):
    """Use the same HMI/Hinode spectral responses as column synthesis."""
    if options is not None and not isinstance(options, dict):
        raise ValueError("Instrument configuration must be a YAML mapping")
    if options:
        validate_options(synthesize_stokes, options, kind="stokes")
        return json.loads(json.dumps(options, default=lambda value: value.tolist(), allow_nan=False))
    validate_options(synthesize_hmi, {}, kind="stokes")
    operator, observed, weights, continuum = build_idealized_hmi_filter()
    return dict(instrument={"type": "hmi_filter_profiles",
                            "quadrature_wavelength_angstrom": operator.quadrature_wavelength_angstrom.tolist(),
                            "inner_half_width_angstrom": operator.inner_half_width_angstrom,
                            "magnetic_azimuth_offset_deg": 90.},
                observed_wavelength_angstrom=observed.tolist(), line_ids=["FeI_6173.3352"],
                response={"spectral_weights": weights.tolist(), "continuum_weights": continuum.tolist()})


def _write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_ray_product(path):
    """Open saved Stokes/geometry arrays lazily for evaluation, without native cubes."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"Not a ray-synthesis product: {path}")
    required = {"stokes", "instrument_stokes", "valid_mask", "boundary_type", "path_length_m",
                "distinct_depth_samples", "ray_origin_local_m", "ray_direction_local", "wavelength_angstrom",
                "observed_wavelength_angstrom", "detector_basis", "synthesis_stokes_basis", "image_x_m", "image_y_m"}
    if not required.issubset(metadata.get("arrays", {})):
        raise ValueError("Ray-product manifest is missing required arrays")
    arrays = {}
    for name, entry in metadata["arrays"].items():
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Ray-product array must be a local filename.")
        array = np.load(path / entry["file"], mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry["shape"] or array.dtype.str != entry["dtype"]:
            raise ValueError(f"Ray-product array contract mismatch: {name}")
        arrays[name] = array
    shape = arrays["valid_mask"].shape
    if len(shape) != 2 or arrays["valid_mask"].dtype != np.bool_:
        raise ValueError("Invalid ray-product image mask")
    expected = {"stokes": (*shape, 4, arrays["wavelength_angstrom"].size),
                "instrument_stokes": (*shape, 4, arrays["observed_wavelength_angstrom"].size),
                "ray_origin_local_m": (*shape, 3), "ray_direction_local": (*shape, 3),
                "detector_basis": (3, 3), "synthesis_stokes_basis": (3, 3),
                "image_x_m": (shape[1],), "image_y_m": (shape[0],)}
    expected.update({name: shape for name in ("boundary_type", "path_length_m", "distinct_depth_samples")})
    if any(arrays[name].shape != dimensions for name, dimensions in expected.items()):
        raise ValueError("Inconsistent ray-product array dimensions")
    return arrays, metadata


def _project_fields(fields, samples, placement, detector_basis, operator, device, dtype):
    """Native vectors -> global solar frame -> one observer basis per ray."""
    values = {name: torch.as_tensor(fields[name], device=device, dtype=dtype)
              for name in ("temperature", "pressure")}
    basis = torch.as_tensor(detector_basis, device=device, dtype=dtype)
    for names in (("bx", "by", "bz"), ("vx", "vy", "vz")):
        local = np.stack([fields[name] for name in names], axis=-1)
        global_vector = torch.as_tensor(placement.vectors_to_global(local), device=device, dtype=dtype)
        if names[0] == "vx":
            positions = torch.as_tensor(samples.global_position_m, device=device, dtype=torch.float64)
            global_vector = global_vector + carrington_rotation_velocity_cartesian(positions, torch).to(dtype)
        projected = global_vector @ basis.T
        if names[0] == "bx":
            projected = operator.polarization_convention.to_synthesis_frame(projected)
        for i, name in enumerate(names):
            values[name] = projected[..., i]
    return values


@torch.no_grad()
def synthesize_ray_volume(volume, destination, *, options=None, bottom_height_megameters,
                          longitude_deg=0., latitude_deg=0., observer_direction=(1., 0., 0.),
                          shell_height_bounds_megameters=None, pixel_size_m=None, depth_samples=128,
                          batch_size=32, device="auto", dtype=torch.float32, workers=4, progress=True, generator_provenance=None):
    """Write one snapshot in bounded ray batches; spectra stream into NPY arrays."""
    if type(depth_samples) is not int or depth_samples < 2:
        raise ValueError("depth_samples must be an integer >= 2")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    device = resolve_synthesis_device(device)
    options = instrument_options(options)
    if generator_provenance is None:
        generator_provenance = synthesis_run_provenance((options["instrument"]["type"],))
    placement = BoxPlacement.from_bounds(volume.bounds_m, bottom_height_megameters=bottom_height_megameters,
                                          longitude_deg=longitude_deg, latitude_deg=latitude_deg)
    if pixel_size_m is None:
        pixel_size_m = tuple(float(delta) * volume.metadata["horizontal_stride"]
                             for delta in volume.metadata["native_spacing_m"][:2])
    image = build_ray_image(placement, pixel_size_m=pixel_size_m, observer_direction=observer_direction)
    operator = build_instrument(options["instrument"]).double()
    observed = torch.as_tensor(options["observed_wavelength_angstrom"], dtype=torch.float64)
    wavelength = operator.synthesis_grid(observed)
    if callable(getattr(operator, "prepare", None)):
        operator.prepare(wavelength, observed)
    renderer = RayStokesRenderer(line_ids=options["line_ids"], wavelength_angstrom=wavelength,
                                 device=device, dtype=dtype, radiance_scale=MURAM_HMI_RADIANCE_SCALE_W_M3_SR)
    operator.to(device=device, dtype=dtype)
    response = {name: torch.as_tensor(value, device=device, dtype=dtype)
                for name, value in (options.get("response") or {}).items()}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    shape = image.spatial_shape
    count = int(np.prod(shape))
    output_dtype = np.float32 if dtype == torch.float32 else np.float64
    arrays = {}

    def allocate(name, tail, dtype):
        array = np.lib.format.open_memmap(destination / f"{name}.npy", mode="w+", dtype=dtype, shape=(*shape, *tail))
        arrays[name] = array
        return array.reshape(count, *tail)

    raw = allocate("stokes", (4, wavelength.numel()), output_dtype)
    filtered = allocate("instrument_stokes", (4, observed.numel()), output_dtype)
    valid = allocate("valid_mask", (), np.bool_)
    boundary = allocate("boundary_type", (), np.uint8)
    path_length = allocate("path_length_m", (), np.float64)
    effective_depth = allocate("distinct_depth_samples", (), np.int32)
    origins = allocate("ray_origin_local_m", (3,), np.float64)
    directions = allocate("ray_direction_local", (3,), np.float64)
    origins[:] = image.origins_local_m
    directions[:] = image.directions_local
    loader = DataLoader(RayDataset(count), batch_size=batch_size, num_workers=workers,
                        shuffle=False, **({"prefetch_factor": 1, "multiprocessing_context": "spawn"} if workers else {}))
    started = last_report = perf_counter()
    completed = hits = thermal = 0
    logger.info("Ray synthesis: %d x %d pixels, %d nominal shell samples; batch_size=%d, workers=%d, device=%s",
                shape[1], shape[0], depth_samples, batch_size, workers, device)
    try:
        with progress_bar(total=count, description="Ray synthesis", unit="ray", enabled=progress) as bar:
            for indices in loader:
                indices = indices.numpy()
                samples = sample_ray_batch(placement, image.origins_local_m[indices], image.directions_local[indices],
                                           depth_samples=depth_samples, shell_height_bounds_megameters=shell_height_bounds_megameters)
                valid[indices] = samples.valid_mask
                boundary[indices] = samples.boundary_type
                raw[indices] = 0
                filtered[indices] = 0
                path_length[indices] = samples.distance_m[:, -1]
                effective_depth[indices] = samples.interval_mask.numpy().sum(axis=-1) + samples.valid_mask.numpy().astype(int)
                selected = np.flatnonzero(samples.valid_mask.numpy())
                if selected.size:
                    hit_samples = samples.select(selected)
                    fields, inside = volume.sample(hit_samples.local_position_m)
                    if not inside.all():
                        raise ValueError("Ray geometry produced interpolation positions outside the volume")
                    fields = _project_fields(fields, hit_samples, placement, image.stokes_basis, operator, device, dtype)
                    profiles = renderer(fields, hit_samples.distance_m, hit_samples.bottom_boundary_mask)
                    signal = operator(profiles, wavelength.to(device), observed.to(device),
                                      **{name: value.expand(len(selected), *value.shape) for name, value in response.items()})
                    raw[indices[selected]] = profiles.cpu().numpy() * MURAM_HMI_RADIANCE_SCALE_W_M3_SR
                    filtered[indices[selected]] = signal.cpu().numpy() * MURAM_HMI_RADIANCE_SCALE_W_M3_SR
                    hits += len(selected)
                    thermal += int(hit_samples.bottom_boundary_mask.sum())
                completed += len(indices)
                bar.update(len(indices))
                now = perf_counter()
                if progress and bar.disable and now - last_report >= 30:
                    rate = completed / (now - started)
                    logger.info("Rays: %d/%d, %.1f rays/s, ETA %.0f s", completed, count, rate, (count-completed)/rate)
                    last_report = now
        with timed_stage(logger, "Flushing ray spectra and saving geometry"):
            for array in arrays.values():
                array.flush()
            angle = np.deg2rad(operator.polarization_convention.offset_deg)
            c, s = np.cos(angle), np.sin(angle)
            synthesis_basis = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ image.stokes_basis.numpy()
            for name, value in {"wavelength_angstrom": wavelength.cpu().numpy(),
                                "observed_wavelength_angstrom": observed.cpu().numpy(),
                                "detector_basis": image.stokes_basis,
                                "synthesis_stokes_basis": synthesis_basis,
                                "image_x_m": image.image_x_m, "image_y_m": image.image_y_m,
                                **{f"response_{name}": value.cpu().numpy() for name, value in response.items()}}.items():
                np.save(destination / f"{name}.npy", value, allow_pickle=False)
                arrays[name] = np.asarray(value)
            metadata = dict(schema=SCHEMA, generator_provenance=generator_provenance, volume=volume.metadata, native_reference=volume.native_reference,
                            placement=placement.metadata(), image=image.metadata(),
                            shell_height_bounds_megameters=list(placement.shell_height_bounds_megameters if shell_height_bounds_megameters is None else shell_height_bounds_megameters),
                            depth_samples=depth_samples, sampling="uniform shell ray-distance nodes clipped to box; repeated endpoints have zero interval",
                            interpolation="trilinear on retained native Cartesian coordinates",
                            boundary_codes={str(code): label for code, label in BOUNDARY_TYPES.items()},
                            boundary_condition="Planck at Cartesian bottom only; zero incoming radiation at every other boundary",
                            units={"stokes": "W m^-3 sr^-1", "instrument_stokes": "W m^-3 sr^-1", "wavelength": "air Angstrom"},
                            stokes_order=["I", "Q", "U", "V"], stokes_frame="synthesis_stokes_basis; detector basis plus recorded instrument azimuth convention",
                            instrument_options=options,
                            velocity_frame="native co-rotating flow plus local Carrington rotation; stationary observer",
                            polarization_convention=operator.polarization_convention.metadata(),
                            counts=dict(rays=count, intersecting=hits, thermal_bottom=thermal),
                            arrays={name: dict(file=f"{name}.npy", shape=list(array.shape), dtype=array.dtype.str)
                                    for name, array in arrays.items()})
            _write_json(destination / "metadata.json", metadata)
        logger.info("Ray synthesis finished: %d/%d intersecting rays (%d thermal bottom), %.1f s", hits, count, thermal, perf_counter()-started)
    finally:
        # Output spectra are disk-backed; no whole-map conversion or NPZ compression.
        arrays.clear()
    return destination


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=1, help="Horizontal volume stride; original coordinates and endpoints retained")
    parser.add_argument("--vertical-stride", type=int, default=1)
    parser.add_argument("--volume-height-megameters", type=float, default=DEFAULT_VOLUME_HEIGHT_MEGAMETERS,
                        help="Load only this height above the bottom native cell center (default: 3), before striding")
    parser.add_argument("--bottom-height-megameters", type=float, required=True, help="Height of bottom native cell center above nominal solar radius; explicit registration")
    parser.add_argument("--longitude-deg", type=float, default=0.)
    parser.add_argument("--latitude-deg", type=float, default=0.)
    parser.add_argument("--observer-direction", type=float, nargs=3, default=(1., 0., 0.), help="Global direction from Sun toward distant observer")
    parser.add_argument("--shell-height-megameters", type=float, nargs=2, metavar=("MIN", "MAX"), help="Default encloses the loaded, height-cropped Cartesian volume")
    parser.add_argument("--pixel-size-km", type=float, nargs=2, metavar=("DX", "DY"))
    parser.add_argument("--depth-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vertical-axis", type=int, choices=(0, 1, 2))
    parser.add_argument("--component-order", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--instrument-config", type=Path, help="Existing Hinode/Stokes instrument YAML; default idealized HMI")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    with cli_logging(args.log_level):
        if any(getattr(args, name) < 1 for name in ("stride", "vertical_stride", "batch_size")) or args.depth_samples < 2 or args.workers < 0:
            parser.error("strides/batch-size must be positive; depth-samples >= 2; workers >= 0")
        if len(set(args.iterations)) != len(args.iterations):
            parser.error("iterations must be unique")
        if sorted(args.component_order) != [0, 1, 2]:
            parser.error("component-order must be a permutation of 0 1 2")
        source, output = args.source_path.resolve(), args.out.resolve()
        if source == output or source in output.parents or output in source.parents:
            parser.error("output must be separate from the source directory")
        if args.out.is_symlink():
            parser.error("output must not be a symbolic link")
        options = instrument_options(yaml.safe_load(args.instrument_config.read_text()) if args.instrument_config else None)
        device = resolve_synthesis_device(args.device)
        dtype = getattr(torch, args.dtype)
        pixel_size = tuple(value * 1e3 for value in args.pixel_size_km) if args.pixel_size_km else None
        settings = dict(bottom_height_megameters=args.bottom_height_megameters, longitude_deg=args.longitude_deg,
                        latitude_deg=args.latitude_deg, observer_direction=args.observer_direction,
                        shell_height_bounds_megameters=args.shell_height_megameters, pixel_size_m=pixel_size,
                        depth_samples=args.depth_samples)
        provenance = synthesis_run_provenance((options["instrument"]["type"],))
        contract = dict(schema=SCHEMA, source_path=str(source), stride=args.stride, vertical_stride=args.vertical_stride,
                        volume_height_megameters=args.volume_height_megameters,
                        vertical_axis=args.vertical_axis, component_order=args.component_order, dtype=args.dtype,
                        instrument_options=options, settings=settings, generator_provenance=provenance)
        contract = json.loads(json.dumps(contract, allow_nan=False))
        # Validate headers, identity and geometry before any requested replacement.
        references = {}
        for iteration in args.iterations:
            snapshot = MURaMSnapshot(str(source), iteration, vertical_axis=args.vertical_axis,
                                     component_order=tuple(args.component_order))
            try:
                reference = snapshot.native_reference()
                references[str(iteration)] = reference
                geom = reference["native_geometry"]
                nx, ny, nz = geom["oriented_shape_xyz"]
                expected_bytes = nx * ny * nz * np.dtype(np.float32).itemsize
                for name, entry in reference["files"].items():
                    if name != "header" and entry["size_bytes"] != expected_bytes:
                        raise ValueError(f"Native field {name} has invalid file size: {entry['path']}")
                dx, dy, dz = geom["oriented_spacing_m"]
                nz = volume_depth_count(nz, dz, args.volume_height_megameters)
                bounds = np.array([[-(nx-1)*dx/2, -(ny-1)*dy/2, 0.], [(nx-1)*dx/2, (ny-1)*dy/2, (nz-1)*dz]])
                placement = BoxPlacement.from_bounds(bounds, bottom_height_megameters=args.bottom_height_megameters,
                                                      longitude_deg=args.longitude_deg, latitude_deg=args.latitude_deg)
                image = build_ray_image(placement, pixel_size_m=pixel_size or (dx*args.stride, dy*args.stride),
                                        observer_direction=args.observer_direction)
                sample_ray_batch(placement, image.origins_local_m[:1], image.directions_local[:1],
                                 depth_samples=args.depth_samples, shell_height_bounds_megameters=args.shell_height_megameters)
            finally:
                snapshot.close()
        if args.overwrite and output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        contract_path = output / "ray_synthesis_run.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != contract:
                raise ValueError("Ray synthesis settings changed; use a new --out or --overwrite")
        elif any(output.iterdir()):
            raise ValueError("Unrecognized output directory; use a new --out or --overwrite")
        _write_json(contract_path, contract)
        status_path = output / "ray_synthesis_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"snapshots": {}}
        with progress_bar(args.iterations, description="Snapshots", unit="snapshot", enabled=not args.no_progress) as iterations:
            for iteration in iterations:
                key = str(iteration)
                target = output / f"muram_{iteration:06d}"
                reference = references[key]
                if target.exists():
                    metadata_path = target / "metadata.json"
                    if metadata_path.exists():
                        existing = json.loads(metadata_path.read_text())
                        if existing["native_reference"] != reference:
                            raise ValueError(f"Native source changed for iteration {iteration}")
                        try:
                            saved, _ = load_ray_product(target)
                        except (OSError, ValueError):
                            logger.info("Iteration %s: incomplete arrays; regenerating", iteration)
                        else:
                            del saved
                            status["snapshots"][key] = "complete"
                            _write_json(status_path, status)
                            logger.info("Skipping completed iteration %s", iteration)
                            continue
                status["snapshots"][key] = "running"
                _write_json(status_path, status)
                snapshot = MURaMSnapshot(str(source), iteration, vertical_axis=args.vertical_axis,
                                         component_order=tuple(args.component_order))
                try:
                    with timed_stage(logger, "Iteration %s: loading Cartesian volume", iteration):
                        volume = EagerVolume.from_snapshot(snapshot, stride=args.stride, vertical_stride=args.vertical_stride,
                                                           volume_height_megameters=args.volume_height_megameters,
                                                           progress=not args.no_progress)
                    verify_native_reference(reference)
                    with tempfile.TemporaryDirectory(prefix=f".ray-{iteration}-", dir=output) as staging:
                        product = synthesize_ray_volume(volume, Path(staging) / "product", options=options,
                                                        **settings, batch_size=args.batch_size, device=device, dtype=dtype,
                                                        workers=args.workers, progress=not args.no_progress, generator_provenance=provenance)
                        del volume
                        verify_native_reference(reference)
                        if target.exists():
                            shutil.rmtree(target)
                        os.replace(product, target)
                    status["snapshots"][key] = "complete"
                    _write_json(status_path, status)
                except BaseException as error:
                    status["snapshots"][key] = f"failed: {type(error).__name__}: {error}"
                    _write_json(status_path, status)
                    raise
                finally:
                    snapshot.close()
        logger.info("All %d requested ray-synthesis snapshots complete: %s", len(args.iterations), output)


if __name__ == "__main__":
    main()
