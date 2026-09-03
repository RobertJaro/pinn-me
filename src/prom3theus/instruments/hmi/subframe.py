"""Prepare detector-registered Carrington cutouts from full-disk HMI Stokes.

The preparation boundary is deliberately narrower than the runtime reader.  It
accepts only complete native ``hmi.S_720s`` acquisitions on the physical HMI
CCD, computes the requested Carrington centre independently at every
acquisition time, and publishes one aligned 24-segment cutout per acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from astropy.io import fits

from .acquisition import (
    SEGMENT_KEYS,
    hmi_observation_wcs_header,
    resolve_acquisition_groups,
)
from .constants import HMI_CAMERA, HMI_CCD_SIZE

_ALIGNMENT_KEYS = (
    "DATE-OBS",
    "T_OBS",
    "T_REC",
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "TELESCOP",
    "INSTRUME",
    "CAMERA",
    "HCAMID",
    "CDELT1",
    "CDELT2",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CROTA2",
    "DSUN_OBS",
    "RSUN_REF",
    "RSUN_OBS",
    "CRLN_OBS",
    "CRLT_OBS",
)


@dataclass(frozen=True, slots=True)
class _CropPlan:
    sources: tuple[Path, ...]
    destinations: tuple[Path, ...]
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int


def _pixel_count(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    if value > HMI_CCD_SIZE:
        raise ValueError(f"{name} cannot exceed the {HMI_CCD_SIZE}-pixel HMI CCD.")
    return value


def _angle(value: float, name: str, *, latitude: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite angle in degrees.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite angle in degrees.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite angle in degrees.")
    if latitude and not -90.0 <= result <= 90.0:
        raise ValueError("latitude_deg must lie between -90 and 90 degrees.")
    return result


def _read_full_disk_header(path: Path) -> fits.Header:
    """Read and validate one primary full-detector image without copying it."""

    with fits.open(
        path,
        mode="readonly",
        memmap=True,
        do_not_scale_image_data=True,
        lazy_load_hdus=True,
    ) as hdul:
        if not hdul or hdul[0].data is None or hdul[0].data.ndim != 2:
            raise ValueError(f"HMI segment must contain one primary 2D image: {path}.")
        shape = tuple(map(int, hdul[0].data.shape))
        if shape != (HMI_CCD_SIZE, HMI_CCD_SIZE):
            raise ValueError(
                "HMI subframe preparation requires exact full-disk "
                f"{HMI_CCD_SIZE}x{HMI_CCD_SIZE} inputs; {path} has shape {shape}."
            )
        header = hdul[0].header.copy()

    missing = [key for key in _ALIGNMENT_KEYS if key not in header]
    if missing:
        raise ValueError(
            f"HMI segment {path.name} is missing required identity/WCS keys {missing}."
        )
    camera = header["CAMERA"]
    if type(camera) is not int or camera != HMI_CAMERA:
        raise ValueError(
            f"HMI segment {path.name} requires CAMERA={HMI_CAMERA}; got {camera!r}."
        )
    hcamid = header["HCAMID"]
    if type(hcamid) is not int or hcamid not in {2, 3}:
        raise ValueError(f"HMI segment {path.name} has unsupported HCAMID={hcamid!r}.")
    try:
        carrington_longitude = float(header["CRLN_OBS"])
        carrington_latitude = float(header["CRLT_OBS"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"HMI segment {path.name} has invalid CRLN_OBS/CRLT_OBS."
        ) from error
    if (
        not math.isfinite(carrington_longitude)
        or not math.isfinite(carrington_latitude)
        or not -90.0 <= carrington_latitude <= 90.0
    ):
        raise ValueError(f"HMI segment {path.name} has non-physical CRLN_OBS/CRLT_OBS.")
    for key in ("CCD_X0", "CCD_Y0"):
        if key in header:
            try:
                value = float(header[key])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"HMI segment {path.name} has invalid {key}."
                ) from error
            if not math.isfinite(value) or value != 0.0:
                raise ValueError(
                    f"Full-disk HMI segment {path.name} must have {key}=0 when present."
                )
    for key in ("CCD_NX", "CCD_NY"):
        if key in header:
            try:
                value = int(header[key])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"HMI segment {path.name} has invalid {key}."
                ) from error
            if value != HMI_CCD_SIZE:
                raise ValueError(
                    f"Full-disk HMI segment {path.name} must have "
                    f"{key}={HMI_CCD_SIZE} when present."
                )
    return header


def _alignment_signature(header: fits.Header) -> tuple[object, ...]:
    return tuple(header[key] for key in _ALIGNMENT_KEYS)


def _carrington_center_pixel(
    header: fits.Header,
    longitude_deg: float,
    latitude_deg: float,
) -> tuple[float, float]:
    """Project a solar-surface Carrington location through one acquisition WCS."""

    # SunPy is optional for package import and is needed only by this explicit
    # observation-preparation operation.
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import frames
        from sunpy.map import Map
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HMI subframe preparation requires the optional HMI preparation "
            "stack; install prom3theus[hmi-preparation]."
        ) from error

    metadata = hmi_observation_wcs_header(header)
    # The image values are irrelevant to coordinate transforms.  A one-pixel
    # placeholder keeps this operation independent of a 4096-square data load;
    # the original reference pixel and WCS remain in the supplied header.
    source_map = Map(np.zeros((1, 1), dtype=np.uint8), metadata)
    carrington_frame = frames.HeliographicCarrington(
        observer=source_map.observer_coordinate,
        obstime=source_map.reference_date,
    )
    centre = SkyCoord(
        lon=longitude_deg * u.deg,
        lat=latitude_deg * u.deg,
        radius=source_map.rsun_meters,
        frame=carrington_frame,
    )
    projected = centre.transform_to(source_map.coordinate_frame)
    if not bool(projected.is_visible(tolerance=0 * u.m)):
        raise ValueError(
            "The requested Carrington centre is on the far side of the Sun "
            "at this HMI acquisition time."
        )
    pixel = source_map.world_to_pixel(projected)
    x_pixel = float(pixel.x.to_value(u.pix))
    y_pixel = float(pixel.y.to_value(u.pix))
    if not math.isfinite(x_pixel) or not math.isfinite(y_pixel):
        raise ValueError(
            "The requested Carrington centre does not project to a finite HMI pixel."
        )
    return x_pixel, y_pixel


def _crop_bounds(center: float, size: int, axis: str) -> tuple[int, int]:
    # Place the geometric centre of the exact-size integer pixel window as near
    # as possible to the requested coordinate.  Half-pixel ties choose the
    # larger detector index deterministically.
    start = math.floor(center - (size - 1) / 2.0 + 0.5)
    stop = start + size
    if start < 0 or stop > HMI_CCD_SIZE:
        raise ValueError(
            f"Requested HMI {axis}-crop [{start}, {stop}) lies outside the "
            f"physical detector [0, {HMI_CCD_SIZE})."
        )
    return start, stop


def _updated_header(
    header: fits.Header,
    *,
    x_start: int,
    y_start: int,
) -> fits.Header:
    result = header.copy()
    result["CRPIX1"] = (float(header["CRPIX1"]) - x_start, "cropped WCS ref pixel")
    result["CRPIX2"] = (float(header["CRPIX2"]) - y_start, "cropped WCS ref pixel")
    result["CCD_X0"] = (x_start, "zero-based x origin on full HMI CCD")
    result["CCD_Y0"] = (y_start, "zero-based y origin on full HMI CCD")
    result["CCD_NX"] = (HMI_CCD_SIZE, "physical HMI CCD width")
    result["CCD_NY"] = (HMI_CCD_SIZE, "physical HMI CCD height")
    result.add_history("PROM3THEUS Carrington-centred HMI detector crop")
    return result


def _write_cropped_segment(source: Path, target: Path, plan: _CropPlan) -> None:
    """Write one cropped primary HDU while holding only that crop in memory."""

    with fits.open(
        source,
        mode="readonly",
        memmap=True,
        do_not_scale_image_data=True,
        lazy_load_hdus=True,
    ) as hdul:
        source_data = hdul[0].data
        if source_data is None or tuple(source_data.shape) != (
            HMI_CCD_SIZE,
            HMI_CCD_SIZE,
        ):
            raise RuntimeError(f"HMI source changed during preparation: {source}.")
        data = np.array(
            source_data[plan.y_start : plan.y_stop, plan.x_start : plan.x_stop],
            copy=True,
            order="C",
        )
        header = _updated_header(
            hdul[0].header,
            x_start=plan.x_start,
            y_start=plan.y_start,
        )
    scaling_cards = {
        key: (header[key], header.comments[key])
        for key in ("BSCALE", "BZERO", "BLANK")
        if key in header
    }
    cropped_hdu = fits.PrimaryHDU(data=data, header=header)
    # PrimaryHDU removes scaling cards when attaching raw integer data.  Restore
    # them after construction so the unchanged storage values decode to the
    # same physical values as the source image (including integer blanks).
    for key, card in scaling_cards.items():
        cropped_hdu.header[key] = card
    cropped_hdu.writeto(
        target,
        overwrite=False,
        checksum=True,
        output_verify="exception",
    )


def _publish_output_directory(
    staging_directory: Path,
    output_directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically publish a complete dedicated output directory."""

    if not output_directory.exists():
        os.replace(staging_directory, output_directory)
        return
    if not overwrite:
        # Validation guarantees that a pre-existing non-colliding output is
        # empty, which POSIX directory rename can replace atomically.
        os.replace(staging_directory, output_directory)
        return

    backup_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.hmi-subframe-backup-",
            dir=output_directory.parent,
        )
    )
    backup_directory.rmdir()
    os.replace(output_directory, backup_directory)
    try:
        os.replace(staging_directory, output_directory)
    except BaseException:
        os.replace(backup_directory, output_directory)
        raise
    shutil.rmtree(backup_directory)


def prepare_hmi_subframes(
    *,
    inputs,
    output_directory: str | os.PathLike[str],
    longitude_deg: float,
    latitude_deg: float,
    width_pixels: int,
    height_pixels: int,
    overwrite: bool = False,
) -> list[Path]:
    """Create exact-size HMI cutouts centred on one Carrington coordinate.

    Each acquisition is transformed independently because a fixed Carrington
    location moves across the detector with observation time.  All sources and
    destinations are validated before any FITS output is staged.  The function
    returns output paths in acquisition order and canonical ``I0...V5`` order.
    """

    longitude_deg = _angle(longitude_deg, "longitude_deg")
    latitude_deg = _angle(latitude_deg, "latitude_deg", latitude=True)
    width_pixels = _pixel_count(width_pixels, "width_pixels")
    height_pixels = _pixel_count(height_pixels, "height_pixels")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean.")
    if not isinstance(output_directory, (str, os.PathLike)):
        raise TypeError("output_directory must be a filesystem path.")

    output = Path(output_directory).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"HMI subframe output is not a directory: {output}.")

    groups = resolve_acquisition_groups(files=inputs)
    sources = [path.resolve() for _, paths in groups for path in paths]
    if len(sources) != len(set(sources)):
        raise ValueError("HMI subframe inputs contain duplicate segment paths.")
    destinations = [output / source.name for source in sources]
    destination_identities = [destination.resolve() for destination in destinations]
    if len(destination_identities) != len(set(destination_identities)):
        raise ValueError("HMI input filenames collide in the output directory.")
    for source, destination in zip(sources, destination_identities, strict=True):
        if source == destination:
            raise ValueError(
                "HMI subframe output must not overwrite an input file, even with "
                f"overwrite=True: {source}."
            )

    expected_names = {path.name for path in destinations}
    if output.exists():
        unrelated = sorted(
            path for path in output.iterdir() if path.name not in expected_names
        )
        if unrelated:
            raise FileExistsError(
                "Refusing unrelated entries in the dedicated HMI subframe output "
                "directory, even with overwrite=True: "
                + ", ".join(str(path) for path in unrelated)
            )

    collisions = [path for path in destinations if os.path.lexists(path)]
    invalid_collisions = [
        path for path in collisions if path.is_symlink() or not path.is_file()
    ]
    if invalid_collisions:
        raise FileExistsError(
            "Refusing non-file HMI subframe output collisions: "
            + ", ".join(str(path) for path in invalid_collisions)
        )
    if collisions and not overwrite:
        qualifier = (
            "partial/colliding" if len(collisions) < len(destinations) else "colliding"
        )
        raise FileExistsError(
            f"Refusing {qualifier} HMI subframe outputs without overwrite=True: "
            + ", ".join(str(path) for path in collisions)
        )

    plans: list[_CropPlan] = []
    destination_offset = 0
    for acquisition_name, acquisition_paths in groups:
        if not acquisition_name.startswith("hmi.S_720s."):
            raise ValueError(
                "HMI subframe preparation supports only hmi.S_720s acquisitions; "
                f"received {acquisition_name!r}."
            )
        if len(acquisition_paths) != len(SEGMENT_KEYS):
            raise ValueError(
                f"HMI acquisition {acquisition_name!r} is not a complete "
                f"{len(SEGMENT_KEYS)}-segment set."
            )
        headers = [_read_full_disk_header(path) for path in acquisition_paths]
        reference_signature = _alignment_signature(headers[0])
        for key, header in zip(SEGMENT_KEYS[1:], headers[1:], strict=True):
            if _alignment_signature(header) != reference_signature:
                raise ValueError(
                    f"HMI segment {key} in {acquisition_name!r} is not aligned with I0."
                )

        center_x, center_y = _carrington_center_pixel(
            headers[0], longitude_deg, latitude_deg
        )
        x_start, x_stop = _crop_bounds(center_x, width_pixels, "x")
        y_start, y_stop = _crop_bounds(center_y, height_pixels, "y")
        group_destinations = tuple(
            destinations[
                destination_offset : destination_offset + len(acquisition_paths)
            ]
        )
        destination_offset += len(acquisition_paths)
        plans.append(
            _CropPlan(
                sources=tuple(path.resolve() for path in acquisition_paths),
                destinations=group_destinations,
                x_start=x_start,
                x_stop=x_stop,
                y_start=y_start,
                y_stop=y_stop,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.hmi-subframe-stage-",
            dir=output.parent,
        )
    )
    try:
        for plan in plans:
            for source, destination in zip(
                plan.sources, plan.destinations, strict=True
            ):
                staged_path = staging_directory / destination.name
                _write_cropped_segment(source, staged_path, plan)
        _publish_output_directory(
            staging_directory,
            output,
            overwrite=overwrite,
        )
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)

    return [path.resolve() for path in destinations]


__all__ = ["HMI_CCD_SIZE", "prepare_hmi_subframes"]
