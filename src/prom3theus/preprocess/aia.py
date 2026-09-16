"""Prepare calibrated native-grid AIA images with an explicit Carrington subframe."""

from __future__ import annotations

import os
import logging
from datetime import timedelta, timezone
from pathlib import Path

from prom3theus.core.parallel import parallel_map

from astropy.io import fits

from prom3theus.instruments.aia_euv.acquisition import (
    SelectedAIARecord, parse_aia_target_time, validate_aia_level1_metadata,
)
from prom3theus.instruments.aia_euv.aiapy_preparation import (
    CarringtonCutout,
    prepare_aiapy_aia_observation_store,
)
from prom3theus.instruments.aia_euv.calibration_download import (
    AIA_CORRECTION_ECSV,
    AIA_POINTING_ECSV,
    download_aia_preprocessing_calibration,
)
from prom3theus.instruments.aia_euv.download import (
    _content_digest, _sha256_file,
)
from prom3theus.observations.image_store import ImageObservationStore
from prom3theus.observations.pixel_footprint import CenteredPixelCutout

logger = logging.getLogger(__name__)


def _scan_aia_files(directory: Path) -> dict:
    """Build preparation metadata from plain DRMS FITS files, without a download manifest."""
    groups = {}
    def read_file(path):
        try:
            with fits.open(path) as hdus:
                index = next((i for i, hdu in enumerate(hdus) if hdu.header.get("NAXIS") == 2), None)
                if index is None:
                    raise ValueError("No two-dimensional image HDU.")
                header = hdus[index].header
                identity = validate_aia_level1_metadata(header, expected_channel=int(header["WAVELNTH"]))
                if identity.quality != 0:
                    raise ValueError(f"QUALITY={identity.quality}; require QUALITY=0.")
                if identity.channel_angstrom not in (171, 193, 211):
                    raise ValueError(f"Unsupported channel {identity.channel_angstrom}.")
                slot = round(float(identity.record_time.tai.to_value("unix_tai")) / 12)
                return slot, (path, identity, index, hdus[index].name, header.copy())
        except (OSError, ValueError, TypeError, KeyError, EOFError) as error:
            logger.warning("Skipping AIA file %s: %s", path, error)
            return None

    for result in parallel_map(read_file, sorted(directory.glob("*.fits")), description="AIA file checks"):
        if result is not None:
            slot, entry = result
            groups.setdefault(slot, []).append(entry)
    if not groups:
        raise FileNotFoundError(f"No usable AIA FITS files in {directory}")
    targets, records = [], []
    for slot, group in sorted(groups.items()):
        group.sort(key=lambda item: item[1].channel_angstrom)
        channels = [item[1].channel_angstrom for item in group]
        if channels != [171, 193, 211]:
            logger.warning("Skipping AIA exposure group %s: require one 171/193/211 image; found %s",
                           slot, channels)
            continue
        target_index = len(targets)
        target_time = group[0][1].observation_time
        target_records = []
        for path, identity, index, name, header in group:
            selected = SelectedAIARecord(
                target_index=target_index, target_time=target_time,
                channel_angstrom=identity.channel_angstrom, record_id=f"fits:{path.name}",
                query="", identity=identity,
                signed_offset_seconds=float((identity.observation_time - target_time).to_value("sec")),
            )
            record = selected.manifest_metadata()
            record.update(
                file=path.name, sha256=None,
                image_hdu_index=index, image_hdu_name=name,
                export_query=None, export_filename=path.name,
                fits_checksum=header.get("CHECKSUM"), fits_datasum=header.get("DATASUM"),
                fits_observation_time_utc=record["observation_time_utc"],
            )
            target_records.append(record)
        records.extend(target_records)
        targets.append({"records": target_records})
    if not records:
        raise ValueError(f"No complete valid 171/193/211 exposure groups in {directory}")
    return {"targets": targets, "record_count": len(records), "content_sha256": None}


def _calibration_interval(acquisition_manifest: dict) -> tuple[str, str]:
    """Cover actual downloaded exposures, not requested observation times."""
    times = [
        parse_aia_target_time(record["fits_observation_time_utc"]).to_datetime(timezone=timezone.utc)
        for target in acquisition_manifest["targets"]
        for record in target["records"]
    ]
    if not times:
        raise ValueError("AIA preprocessing requires downloaded exposures.")
    first, last = min(times), max(times)
    start = first.replace(hour=first.hour - first.hour % 3, minute=0, second=0, microsecond=0)
    end = last.replace(hour=last.hour - last.hour % 3, minute=0, second=0, microsecond=0) + timedelta(hours=3)
    return start.isoformat(), end.isoformat()


def _validate_prepared_store(
    output_directory: Path,
    acquisition_manifest: dict,
    footprint: CarringtonCutout | CenteredPixelCutout,
) -> None:
    if acquisition_manifest["content_sha256"] is None:
        records = [record for target in acquisition_manifest["targets"] for record in target["records"]]
        paths = [Path(acquisition_manifest["input_directory"]) / record["file"] for record in records]
        for record, digest in zip(records, parallel_map(_sha256_file, paths, description="AIA reuse checksums")):
            record["sha256"] = digest
        acquisition_manifest["content_sha256"] = _content_digest(records)
    manifest = ImageObservationStore.manifest(output_directory)
    metadata = manifest["metadata"]
    if len(manifest["rasters"]) != acquisition_manifest["record_count"]:
        raise ValueError("Existing AIA preparation does not match the input records.")
    if metadata.get("source_files_sha256") != acquisition_manifest["content_sha256"]:
        raise ValueError("Existing AIA preparation does not match the input checksum.")
    dependencies = metadata.get("preparation", {}).get("preparation_dependencies", {})
    if dependencies.get("cutout") != footprint.metadata():
        raise ValueError("Existing AIA preparation does not match the requested crop.")


def preprocess_aia(
    *,
    input_directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    calibration_directory: str | os.PathLike[str],
    longitude_deg: float,
    latitude_deg: float,
    width_pixels: int,
    height_pixels: int,
) -> dict:
    """Acquire/reuse calibration for downloaded exposures, then prepare images."""

    footprint = CenteredPixelCutout(
        longitude_deg=longitude_deg,
        latitude_deg=latitude_deg,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
    )
    input_directory = Path(input_directory).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    calibration_directory = Path(calibration_directory).expanduser().resolve()
    directories = (input_directory, output_directory, calibration_directory)
    if any(left == right or left in right.parents or right in left.parents
           for index, left in enumerate(directories) for right in directories[index + 1:]):
        raise ValueError("AIA input, output, and calibration require separate, non-nested directories.")
    acquisition_manifest = _scan_aia_files(input_directory)
    start, end = _calibration_interval(acquisition_manifest)
    acquisition_manifest["input_directory"] = str(input_directory)
    download_aia_preprocessing_calibration(
        output_directory=calibration_directory, start_utc=start, end_utc=end,
    )

    if output_directory.exists():
        _validate_prepared_store(output_directory, acquisition_manifest, footprint)
        status = "reused"
    else:
        prepare_aiapy_aia_observation_store(
            input_directory,
            output_directory,
            correction_table_path=calibration_directory / AIA_CORRECTION_ECSV,
            pointing_table_path=calibration_directory / AIA_POINTING_ECSV,
            footprint=footprint,
            acquisition_manifest=acquisition_manifest,
        )
        status = "prepared"

    return {
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "record_count": acquisition_manifest["record_count"],
        "status": status,
    }


__all__ = ["preprocess_aia"]
