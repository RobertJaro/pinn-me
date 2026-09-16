"""Immutable acquisition bundles for selected AIA Level-1 EUV images.

Metadata selection is completed before export.  Each exact selected record is
then exported with its ``image`` segment only, validated as FITS, hashed, and
published as one atomic directory. DRMS must be installed for downloads;
the import stays inside the download path to keep offline readers independent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy as np

from prom3theus.core.downloads import download_fits_export, publish_download

from .acquisition import (
    AIA_EUV_CHANNELS_ANGSTROM,
    AIA_IMAGE_SEGMENT,
    AIA_LEVEL1_SERIES,
    AIA_METADATA_KEYS,
    SelectedAIARecord,
    normalize_aia_target_times,
    select_aia_level1_records,
    validate_aia_channels,
    validate_aia_level1_metadata,
    validate_max_offset_seconds,
)


AIA_ACQUISITION_FORMAT = "prom3theus.aia_euv.level1_acquisition"
AIA_ACQUISITION_VERSION = 1
AIA_ACQUISITION_MANIFEST = "manifest.json"
AIA_CCD_SIZE = 4096
AIA_SMALL_SAMPLE_TARGET_UTC = "2024-03-23T22:11:21.388Z"
AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS = 6.0
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _validate_email(email: str) -> str:
    if (
        type(email) is not str
        or email.strip() != email
        or email.count("@") != 1
        or any(character.isspace() for character in email)
    ):
        raise ValueError("email must be an explicit, non-empty JSOC email address.")
    local, domain = email.split("@")
    if not local or not domain or "." not in domain:
        raise ValueError("email must be an explicit, non-empty JSOC email address.")
    return email


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _downloaded_entries(result: Any, directory: Path) -> list[tuple[str, Path]]:
    if result is None:
        raise RuntimeError("DRMS returned no AIA download result.")
    try:
        values = list(result["download"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "DRMS download result is missing the required 'download' column."
        ) from error
    try:
        record_values = list(result["record"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "DRMS download result is missing the required 'record' column."
        ) from error
    if not values:
        raise RuntimeError("DRMS returned no AIA FITS file.")
    if len(record_values) != len(values):
        raise RuntimeError("DRMS download record and file counts disagree.")
    root = directory.resolve()
    paths: list[Path] = []
    records: list[str] = []
    for record_value, value in zip(record_values, values, strict=True):
        if (
            not isinstance(record_value, str)
            or not record_value
            or record_value.strip() != record_value
        ):
            raise RuntimeError("DRMS returned an invalid AIA record identity.")
        if not isinstance(value, (str, os.PathLike)):
            raise RuntimeError("DRMS returned a non-path AIA download entry.")
        raw = Path(value)
        path = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        if path.parent != root:
            raise RuntimeError(
                f"DRMS reported an AIA file outside its staging directory: {path}."
            )
        if not path.is_file():
            raise RuntimeError(f"DRMS did not create the reported AIA file: {path}.")
        if path.suffix.lower() not in {".fits", ".fit", ".fts"}:
            raise RuntimeError(f"DRMS returned a non-FITS AIA file: {path.name}.")
        records.append(record_value)
        paths.append(path)
    if len(set(paths)) != len(paths) or len({path.name for path in paths}) != len(
        paths
    ):
        raise RuntimeError("DRMS returned duplicate AIA download paths.")
    if len(set(records)) != len(records):
        raise RuntimeError("DRMS returned duplicate AIA record identities.")
    return list(zip(records, paths, strict=True))


def _same_time(left: Time, right: Time) -> bool:
    return abs(float((left - right).to_value("sec"))) <= 1.0e-6


def _canonical_time(value: Time, *, scale: str) -> str:
    converted = getattr(value, scale).copy(format="isot")
    converted.precision = 9
    suffix = "Z" if scale == "utc" else " TAI"
    return f"{converted.value}{suffix}"


def _verify_selected_fits(
    path: Path,
    selected: SelectedAIARecord,
) -> dict[str, Any]:
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            do_not_scale_image_data=True,
            lazy_load_hdus=True,
        ) as hdul:
            hdul.verify("exception")
            image_hdus = [
                (index, hdu)
                for index, hdu in enumerate(hdul)
                if hdu.data is not None and getattr(hdu.data, "ndim", None) == 2
            ]
            if len(image_hdus) != 1:
                raise ValueError(
                    "the FITS file must contain exactly one two-dimensional image HDU"
                )
            image_hdu_index, image_hdu = image_hdus[0]
            data = image_hdu.data
            if tuple(data.shape) != (AIA_CCD_SIZE, AIA_CCD_SIZE):
                raise ValueError(
                    f"image shape is {tuple(data.shape)}, expected "
                    f"{AIA_CCD_SIZE}x{AIA_CCD_SIZE}"
                )
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError("image data are not numeric")
            data[-1, -1]
            header = image_hdu.header.copy()
            checksum_present = "CHECKSUM" in header
            datasum_present = "DATASUM" in header
            if checksum_present != datasum_present:
                raise ValueError(
                    "the image HDU must provide CHECKSUM and DATASUM together"
                )
            if checksum_present and image_hdu.verify_checksum() != 1:
                raise ValueError("the image-HDU checksum is invalid")
            if datasum_present and image_hdu.verify_datasum() != 1:
                raise ValueError("the image-HDU data checksum is invalid")
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError(
            f"Invalid AIA Level-1 FITS image {path}: {error}."
        ) from error

    try:
        identity = validate_aia_level1_metadata(
            header,
            expected_channel=selected.channel_angstrom,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"AIA FITS header does not satisfy its Level-1 identity: {path.name}."
        ) from error
    expected = selected.identity
    if (
        identity.quality != 0
        or not _same_time(identity.observation_time, expected.observation_time)
        or not math.isclose(
            identity.exposure_seconds,
            expected.exposure_seconds,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or identity.level != expected.level
        or identity.telescope != expected.telescope
        or identity.instrument != expected.instrument
    ):
        raise RuntimeError(
            f"Downloaded AIA FITS identity does not match {selected.record_id!r}."
        )
    for key, expected_value in expected.wcs.items():
        actual_value = identity.wcs[key]
        if isinstance(expected_value, str):
            equal = str(actual_value).strip().lower() == expected_value.strip().lower()
        else:
            equal = math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=1.0e-12,
                abs_tol=1.0e-10,
            )
        if not equal:
            raise RuntimeError(
                f"Downloaded AIA FITS {key} does not match discovered metadata."
            )
    return {
        "image_hdu_index": image_hdu_index,
        "image_hdu_name": str(image_hdu.name),
        "fits_record_time_utc": _canonical_time(identity.record_time, scale="utc"),
        "fits_record_time_tai": _canonical_time(identity.record_time, scale="tai"),
        "fits_observation_time_utc": _canonical_time(
            identity.observation_time, scale="utc"
        ),
        "fits_observation_time_tai": _canonical_time(
            identity.observation_time, scale="tai"
        ),
        "fits_checksum": (str(header["CHECKSUM"]) if "CHECKSUM" in header else None),
        "fits_datasum": str(header["DATASUM"]) if "DATASUM" in header else None,
    }


def _export_one(
    client: Any,
    selected: SelectedAIARecord,
    *,
    download_directory: Path,
) -> tuple[Path, str, str]:
    export_query = f"{selected.record_id}{{{AIA_IMAGE_SEGMENT}}}"
    request = client.export(export_query, method="url", protocol="fits")
    result = download_fits_export(request, download_directory)
    entries = _downloaded_entries(result, download_directory)
    if len(entries) != 1:
        raise RuntimeError(
            "Exact AIA record export returned "
            f"{len(entries)} files; expected one image."
        )
    exported_record, path = entries[0]
    if exported_record not in {selected.record_id, export_query}:
        raise RuntimeError(
            "DRMS downloaded an AIA record other than the exact selected record: "
            f"{exported_record!r}."
        )
    return path, export_query, exported_record


def _content_digest(records: Sequence[Mapping[str, Any]]) -> str:
    identities = [
        {
            "record_id": record["record_id"],
            "target_time_utc": record["target_time_utc"],
            "signed_offset_seconds": record["signed_offset_seconds"],
            "file": record["file"],
            "sha256": record["sha256"],
        }
        for record in records
    ]
    payload = json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    selected_records: Sequence[SelectedAIARecord],
    downloaded: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[Time],
    channels: Sequence[int],
    max_offset_seconds: float,
) -> dict[str, Any]:
    records_by_target: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(targets))
    }
    flat_records: list[dict[str, Any]] = []
    for selected, file_metadata in zip(selected_records, downloaded, strict=True):
        record = {**selected.manifest_metadata(), **dict(file_metadata)}
        records_by_target[selected.target_index].append(record)
        flat_records.append(record)
    targets_manifest = []
    for target_index, target in enumerate(targets):
        records = records_by_target[target_index]
        if [record["channel_angstrom"] for record in records] != list(channels):
            raise RuntimeError(
                f"AIA target {target_index} is missing its exact channel sequence."
            )
        targets_manifest.append(
            {
                "target_index": target_index,
                "target_time_utc": selected_records[
                    target_index * len(channels)
                ].manifest_metadata()["target_time_utc"],
                "records": records,
            }
        )
    return {
        "format": AIA_ACQUISITION_FORMAT,
        "version": AIA_ACQUISITION_VERSION,
        "series": AIA_LEVEL1_SERIES,
        "segment": AIA_IMAGE_SEGMENT,
        "channels_angstrom": list(channels),
        "max_offset_seconds": max_offset_seconds,
        "selection": {
            "time_keyword": "T_OBS",
            "algorithm": "unique nearest QUALITY=0 record",
            "signed_offset_convention": "selected_T_OBS_minus_target",
            "metadata_keys": list(AIA_METADATA_KEYS),
        },
        "target_count": len(targets),
        "record_count": len(flat_records),
        "content_sha256": _content_digest(flat_records),
        "targets": targets_manifest,
    }


def _validate_manifest_paths(root: Path, manifest: Mapping[str, Any]) -> None:
    expected_top = {
        "format",
        "version",
        "series",
        "segment",
        "channels_angstrom",
        "max_offset_seconds",
        "selection",
        "target_count",
        "record_count",
        "content_sha256",
        "targets",
    }
    if set(manifest) != expected_top:
        raise ValueError("AIA acquisition manifest has an invalid top-level schema.")
    if (
        manifest["format"] != AIA_ACQUISITION_FORMAT
        or manifest["version"] != AIA_ACQUISITION_VERSION
        or manifest["series"] != AIA_LEVEL1_SERIES
        or manifest["segment"] != AIA_IMAGE_SEGMENT
    ):
        raise ValueError("AIA acquisition manifest identity is unsupported.")
    validate_aia_channels(tuple(manifest["channels_angstrom"]))
    validate_max_offset_seconds(manifest["max_offset_seconds"])
    if not isinstance(manifest["targets"], list) or (
        len(manifest["targets"]) != manifest["target_count"]
    ):
        raise ValueError("AIA acquisition manifest target count is inconsistent.")
    flat_records = []
    seen_files: set[str] = set()
    seen_records: set[str] = set()
    for target_index, target in enumerate(manifest["targets"]):
        if not isinstance(target, Mapping) or set(target) != {
            "target_index",
            "target_time_utc",
            "records",
        }:
            raise ValueError("AIA acquisition target entry has an invalid schema.")
        if target["target_index"] != target_index:
            raise ValueError("AIA acquisition target indices are not ordered.")
        normalize_aia_target_times(target["target_time_utc"])
        records = target["records"]
        if not isinstance(records, list) or [
            record.get("channel_angstrom")
            for record in records
            if isinstance(record, Mapping)
        ] != manifest["channels_angstrom"]:
            raise ValueError("AIA acquisition target has an invalid channel group.")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("AIA acquisition record entry must be an object.")
            relative = record.get("file")
            digest = record.get("sha256")
            record_id = record.get("record_id")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or Path(relative).parts[:1] != ("records",)
                or ".." in Path(relative).parts
                or not isinstance(digest, str)
                or _SHA256_PATTERN.fullmatch(digest) is None
                or not isinstance(record_id, str)
                or not record_id
            ):
                raise ValueError("AIA acquisition record file identity is invalid.")
            if relative in seen_files or record_id in seen_records:
                raise ValueError("AIA acquisition manifest contains duplicate records.")
            seen_files.add(relative)
            seen_records.add(record_id)
            path = (root / relative).resolve()
            if path.parent != (root / "records").resolve() or not path.is_file():
                raise FileNotFoundError(
                    f"AIA acquisition manifest file is missing: {relative}."
                )
            if _sha256_file(path) != digest:
                raise ValueError(f"AIA acquisition file checksum mismatch: {relative}.")
            flat_records.append(record)
    if len(flat_records) != manifest["record_count"]:
        raise ValueError("AIA acquisition manifest record count is inconsistent.")
    if manifest["content_sha256"] != _content_digest(flat_records):
        raise ValueError("AIA acquisition content digest is inconsistent.")


def load_aia_acquisition_manifest(
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and checksum-validate one immutable AIA acquisition bundle."""

    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("directory must be a filesystem path.")
    root = Path(directory).expanduser().resolve()
    path = root / AIA_ACQUISITION_MANIFEST
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"AIA acquisition manifest not found: {path}."
        ) from error
    if not isinstance(value, Mapping):
        raise TypeError("AIA acquisition manifest must contain one JSON object.")
    manifest = dict(value)
    _validate_manifest_paths(root, manifest)
    return manifest


def download_aia_euv(
    *,
    target_times: Time | str | Sequence[Time | str],
    output_directory: str | os.PathLike[str],
    email: str,
    channels: Sequence[int] = AIA_EUV_CHANNELS_ANGSTROM,
    max_offset_seconds: float | TimeDelta = 12.0,
    client: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Discover, export, validate, and atomically publish AIA Level-1 images.

    Existing matching acquisitions are validated and skipped by default.
    Overwrite downloads a replacement first, retaining the previous bundle as
    a sibling backup. Failed downloads leave the existing bundle unchanged.
    """

    targets = normalize_aia_target_times(target_times)
    ordered_channels = validate_aia_channels(channels)
    tolerance = validate_max_offset_seconds(max_offset_seconds)
    email = _validate_email(email)
    if not isinstance(output_directory, (str, os.PathLike)):
        raise TypeError("output_directory must be a filesystem path.")
    output = Path(output_directory).expanduser().resolve()
    if os.path.lexists(output) and not overwrite:
        manifest = load_aia_acquisition_manifest(output)
        if (
            tuple(manifest["channels_angstrom"]) != ordered_channels
            or manifest["max_offset_seconds"] != tolerance
            or tuple(entry["target_time_utc"] for entry in manifest["targets"])
            != tuple(_canonical_time(target, scale="utc") for target in targets)
        ):
            raise ValueError("Existing AIA request differs; use overwrite=True to download again.")
        return output / AIA_ACQUISITION_MANIFEST
    output.parent.mkdir(parents=True, exist_ok=True)
    if client is None:
        import drms

        client = drms.Client(email=email)
    selected = select_aia_level1_records(
        client,
        target_times=targets,
        channels=ordered_channels,
        max_offset_seconds=tolerance,
    )

    with tempfile.TemporaryDirectory(
        prefix=".aia-lev1-euv-", dir=output.parent
    ) as temporary_directory:
        staging = Path(temporary_directory)
        records_directory = staging / "records"
        records_directory.mkdir()
        downloaded_metadata: list[dict[str, Any]] = []
        for index, record in enumerate(selected):
            transfer_directory = staging / f".transfer-{index:05d}"
            transfer_directory.mkdir()
            downloaded, export_query, exported_record = _export_one(
                client,
                record,
                download_directory=transfer_directory,
            )
            original_filename = downloaded.name
            fits_identity = _verify_selected_fits(downloaded, record)
            relative = Path("records") / (
                f"target_{record.target_index:04d}_aia_"
                f"{record.channel_angstrom:03d}.fits"
            )
            destination = staging / relative
            if destination.exists():  # guarded by unique record/channel selection
                raise RuntimeError(f"Duplicate staged AIA destination: {relative}.")
            os.replace(downloaded, destination)
            shutil.rmtree(transfer_directory)
            downloaded_metadata.append(
                {
                    "file": relative.as_posix(),
                    "sha256": _sha256_file(destination),
                    "export_query": export_query,
                    "export_record_id": exported_record,
                    "export_filename": original_filename,
                    **fits_identity,
                }
            )

        manifest = _manifest(
            selected,
            downloaded_metadata,
            targets=targets,
            channels=ordered_channels,
            max_offset_seconds=tolerance,
        )
        manifest_path = staging / AIA_ACQUISITION_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_aia_acquisition_manifest(staging)
        publish_download(staging, output, overwrite=overwrite)
    return (output / AIA_ACQUISITION_MANIFEST).resolve()


def download_aia_euv_small_sample(
    *,
    output_directory: str | os.PathLike[str],
    email: str,
    client: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Fetch the planned one-exposure-group HMI/AIA regression sample.

    The fixed UTC target is the physical instant corresponding to the sample
    HMI ``T_OBS``.  Selection remains metadata-driven: the immutable manifest
    records whichever unique nearest quality-good 171/193/211 images satisfy
    the six-second tolerance, rather than assuming filenames in advance.
    """

    return download_aia_euv(
        target_times=(AIA_SMALL_SAMPLE_TARGET_UTC,),
        output_directory=output_directory,
        email=email,
        channels=AIA_EUV_CHANNELS_ANGSTROM,
        max_offset_seconds=AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS,
        client=client,
        overwrite=overwrite,
    )


__all__ = [
    "AIA_ACQUISITION_FORMAT",
    "AIA_ACQUISITION_MANIFEST",
    "AIA_ACQUISITION_VERSION",
    "AIA_CCD_SIZE",
    "AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS",
    "AIA_SMALL_SAMPLE_TARGET_UTC",
    "download_aia_euv",
    "download_aia_euv_small_sample",
    "load_aia_acquisition_manifest",
]
