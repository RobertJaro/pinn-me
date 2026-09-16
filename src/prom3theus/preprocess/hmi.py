"""Create Carrington-centered HMI subframes for every acquisition on disk."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Sequence

from astropy.io import fits
from prom3theus.core.parallel import parallel_map

from prom3theus.instruments.hmi import prepare_hmi_subframes
from prom3theus.instruments.hmi.acquisition import (
    read_hmi_image_header,
    resolve_acquisition_groups,
)
from prom3theus.instruments.hmi.subframe import hmi_subframe_bounds

logger = logging.getLogger(__name__)


def scan_hmi_time_series(
    directory: str | os.PathLike[str],
) -> list[tuple[str, list[Path]]]:
    """Return complete supported acquisitions, rejecting nonzero QUALITY when present."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HMI input directory not found: {root}.")
    candidates = []
    for path in sorted(root.glob("*.fits")):
        candidates.append(path.resolve())
    if not candidates:
        raise FileNotFoundError(f"No HMI Stokes files in {root}.")
    groups = resolve_acquisition_groups(files=candidates)
    resolved = {path.resolve() for _, paths in groups for path in paths}
    unexpected = sorted(set(candidates) - resolved)
    if unexpected:
        raise ValueError(
            "Unrecognized HMI FITS files in the input directory: "
            + ", ".join(str(path) for path in unexpected)
        )
    def check_quality(group):
        name, paths = group
        try:
            for path in paths:
                header = read_hmi_image_header(path)
                if "QUALITY" not in header:
                    continue
                quality = header["QUALITY"]
                if type(quality) is not int:
                    raise ValueError(f"Invalid QUALITY={quality!r} in {path.name}; require an integer.")
                if quality != 0:
                    raise ValueError(f"QUALITY={quality:#x} in {path.name}; require QUALITY=0.")
        except (OSError, ValueError, TypeError, KeyError, EOFError) as error:
            logger.warning("Skipping HMI acquisition %s: %s", name, error)
            return None
        return name, paths

    accepted = [group for group in parallel_map(check_quality, groups, description="HMI quality checks")
                if group is not None]
    if not accepted:
        raise ValueError(f"No complete supported QUALITY=0 HMI acquisitions in {root}.")
    return accepted


def _validate_subframes(
    raw_groups: Sequence[tuple[str, Sequence[Path]]],
    directory: Path,
    *,
    longitude_deg: float,
    latitude_deg: float,
    width_pixels: int,
    height_pixels: int,
) -> list[tuple[str, list[Path]]]:
    prepared_groups = scan_hmi_time_series(directory)
    raw_names = [path.name for _, paths in raw_groups for path in paths]
    prepared_names = [path.name for _, paths in prepared_groups for path in paths]
    if prepared_names != raw_names:
        raise ValueError("Existing HMI subframes do not match the raw time series.")
    for (_, raw_paths), (_, prepared_paths) in zip(
        raw_groups, prepared_groups, strict=True
    ):
        x_start, _, y_start, _ = hmi_subframe_bounds(
            read_hmi_image_header(raw_paths[0]),
            longitude_deg=longitude_deg,
            latitude_deg=latitude_deg,
            width_pixels=width_pixels,
            height_pixels=height_pixels,
        )
        for path in prepared_paths:
            header = fits.getheader(path, 0)
            if (
                header.get("NAXIS1"),
                header.get("NAXIS2"),
                header.get("CCD_X0"),
                header.get("CCD_Y0"),
            ) != (
                width_pixels,
                height_pixels,
                x_start,
                y_start,
            ):
                raise ValueError(
                    f"Existing HMI subframe does not match the requested crop: {path}."
                )
    return prepared_groups


def preprocess_hmi_time_series(
    *,
    input_directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    longitude_deg: float,
    latitude_deg: float,
    width_pixels: int,
    height_pixels: int,
) -> dict:
    """Prepare every HMI slot using the caller's Carrington subframe settings."""

    input_directory = Path(input_directory).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    raw_groups = scan_hmi_time_series(input_directory)

    if output_directory.exists():
        _validate_subframes(
            raw_groups,
            output_directory,
            longitude_deg=longitude_deg,
            latitude_deg=latitude_deg,
            width_pixels=width_pixels,
            height_pixels=height_pixels,
        )
        subframe_status = "reused"
    else:
        prepare_hmi_subframes(
            inputs=[path for _, paths in raw_groups for path in paths],
            output_directory=output_directory,
            longitude_deg=longitude_deg,
            latitude_deg=latitude_deg,
            width_pixels=width_pixels,
            height_pixels=height_pixels,
        )
        subframe_status = "prepared"

    return {
        "acquisition_count": len(raw_groups),
        "segment_count": sum(len(paths) for _, paths in raw_groups),
        "subframes": {
            "directory": str(output_directory),
            "status": subframe_status,
        },
    }


__all__ = ["scan_hmi_time_series", "preprocess_hmi_time_series"]
