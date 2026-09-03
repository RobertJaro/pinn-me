"""Strict download support for native-cadence HMI Stokes acquisitions.

The only supported product is ``hmi.S_720s`` with all 24 ``I0`` through
``V5`` segments.  DRMS remains an optional dependency and is imported only
when a caller does not provide a client explicitly.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from astropy.io import fits

from .acquisition import SEGMENT_KEYS, format_jsoc_time, resolve_acquisition_groups
from .constants import HMI_CAMERA, HMI_CCD_SIZE


HMI_STOKES_SERIES = "hmi.S_720s"
HMI_NATIVE_CADENCE_SECONDS = 720
HMI_STOKES_SEGMENTS = SEGMENT_KEYS
_ACQUISITION_PATTERN = re.compile(r"hmi\.S_720s\.(?P<time>\d{8}_\d{6}_TAI)\.3")
_ALIGNMENT_KEYS = (
    "T_REC",
    "T_OBS",
    "DATE-OBS",
    "CAMERA",
    "HCAMID",
    "QUALITY",
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
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
    "OBS_VR",
    "OBS_VW",
    "OBS_VN",
    "TELESCOP",
    "INSTRUME",
)
_FINITE_KEYS = (
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
    "OBS_VR",
    "OBS_VW",
    "OBS_VN",
)


def _tai_datetime(value: str | datetime, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif type(value) is str and value.strip() == value and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                f"{field} must be an ISO-8601 timestamp interpreted as TAI."
            ) from error
    else:
        raise TypeError(f"{field} must be a non-empty ISO-8601 string or datetime.")
    if parsed.tzinfo is not None:
        raise ValueError(
            f"{field} must not include a timezone; HMI record times are expressed in TAI."
        )
    if parsed.microsecond != 0:
        raise ValueError(f"{field} must have whole-second precision.")
    return parsed


def _validated_interval(
    start: str | datetime, end: str | datetime
) -> tuple[datetime, datetime, int]:
    start_time = _tai_datetime(start, field="start")
    end_time = _tai_datetime(end, field="end")
    for field, value in (("start", start_time), ("end", end_time)):
        seconds_since_midnight = value.hour * 3600 + value.minute * 60 + value.second
        if seconds_since_midnight % HMI_NATIVE_CADENCE_SECONDS != 0:
            raise ValueError(
                f"{field} must lie on the native hmi.S_720s 12-minute TAI slot grid."
            )
    duration = (end_time - start_time).total_seconds()
    if duration <= 0:
        raise ValueError("end must be later than start.")
    duration_seconds = int(duration)
    if duration != duration_seconds:
        raise ValueError("The HMI download interval must span whole seconds.")
    if duration_seconds % HMI_NATIVE_CADENCE_SECONDS != 0:
        raise ValueError(
            "The HMI download interval must be an integer multiple of the native "
            "720-second cadence."
        )
    return start_time, end_time, duration_seconds


def build_hmi_stokes_query(*, start: str | datetime, end: str | datetime) -> str:
    """Build one immutable native-cadence ``hmi.S_720s`` DRMS query."""

    start_time, _, duration_seconds = _validated_interval(start, end)

    start_key = start_time.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    segments = ",".join(HMI_STOKES_SEGMENTS)
    return (
        f"{HMI_STOKES_SERIES}[{start_key}/{duration_seconds}s@"
        f"{HMI_NATIVE_CADENCE_SECONDS}s][{HMI_CAMERA}]{{{segments}}}"
    )


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


def _drms_client(email: str):
    try:
        import drms
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HMI downloading requires the optional DRMS dependency; install "
            "prom3theus[hmi-preparation]."
        ) from error
    return drms.Client(email=email)


def _downloaded_paths(result: Any, staging_directory: Path) -> list[Path]:
    if result is None:
        raise RuntimeError("DRMS returned no HMI download result.")
    try:
        values = list(result["download"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "DRMS download result is missing the required 'download' column."
        ) from error
    if not values:
        raise RuntimeError("DRMS returned no HMI FITS files.")

    staging_root = staging_directory.resolve()
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, (str, os.PathLike)):
            raise RuntimeError("DRMS returned a non-path download entry.")
        path = Path(value).resolve()
        if path.parent != staging_root:
            raise RuntimeError(
                f"DRMS reported a download outside the flat staging directory: {path}."
            )
        if not path.is_file():
            raise RuntimeError(f"DRMS did not create the reported FITS file: {path}.")
        if not path.name.startswith(f"{HMI_STOKES_SERIES}."):
            raise RuntimeError(
                f"DRMS returned a file outside {HMI_STOKES_SERIES}: {path.name}."
            )
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise RuntimeError("DRMS returned duplicate HMI download paths.")
    if len({path.name for path in paths}) != len(paths):
        raise RuntimeError("DRMS returned duplicate HMI FITS filenames.")
    return sorted(paths)


def _read_full_disk_identity(path: Path) -> tuple[object, ...]:
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            do_not_scale_image_data=True,
            lazy_load_hdus=True,
        ) as hdul:
            hdul.verify("exception")
            if not hdul or hdul[0].data is None or hdul[0].data.ndim != 2:
                raise ValueError("the primary HDU is not a two-dimensional image")
            data = hdul[0].data
            if tuple(data.shape) != (HMI_CCD_SIZE, HMI_CCD_SIZE):
                raise ValueError(
                    f"image shape is {tuple(data.shape)}, expected "
                    f"{HMI_CCD_SIZE}x{HMI_CCD_SIZE}"
                )
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError("image data are not numeric")
            # Touch the final stored value so a truncated image cannot pass from
            # its header shape alone while retaining memory-mapped validation.
            data[-1, -1]
            header = hdul[0].header.copy()
            if "CHECKSUM" in header and hdul[0].verify_checksum() != 1:
                raise ValueError("the primary-HDU checksum is invalid")
            if "DATASUM" in header and hdul[0].verify_datasum() != 1:
                raise ValueError("the primary-HDU data checksum is invalid")
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError(
            f"Invalid full-disk HMI FITS segment {path}: {error}."
        ) from error

    missing = [key for key in _ALIGNMENT_KEYS if key not in header]
    if missing:
        raise RuntimeError(
            f"HMI FITS segment {path.name} is missing required keys {missing}."
        )
    try:
        finite = np.asarray([float(header[key]) for key in _FINITE_KEYS])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"HMI FITS segment {path.name} has invalid numeric metadata."
        ) from error
    if not np.isfinite(finite).all():
        raise RuntimeError(
            f"HMI FITS segment {path.name} has non-finite numeric metadata."
        )
    if type(header["CAMERA"]) is not int or header["CAMERA"] != HMI_CAMERA:
        raise RuntimeError(
            f"HMI FITS segment {path.name} must have CAMERA={HMI_CAMERA}."
        )
    if type(header["HCAMID"]) is not int or header["HCAMID"] not in {2, 3}:
        raise RuntimeError(f"HMI FITS segment {path.name} has unsupported HCAMID.")
    if (
        str(header["CTYPE1"]).strip().upper() != "HPLN-TAN"
        or str(header["CTYPE2"]).strip().upper() != "HPLT-TAN"
        or str(header["CUNIT1"]).strip().lower() != "arcsec"
        or str(header["CUNIT2"]).strip().lower() != "arcsec"
        or float(header["CDELT1"]) <= 0
        or float(header["CDELT2"]) <= 0
        or float(header["DSUN_OBS"]) <= float(header["RSUN_REF"])
        or float(header["RSUN_REF"]) <= 0
        or float(header["RSUN_OBS"]) <= 0
        or not -90 <= float(header["CRLT_OBS"]) <= 90
    ):
        raise RuntimeError(
            f"HMI FITS segment {path.name} has non-physical full-disk WCS metadata."
        )
    for key in ("T_REC", "T_OBS"):
        try:
            format_jsoc_time(str(header[key]))
        except ValueError as error:
            raise RuntimeError(
                f"HMI FITS segment {path.name} has invalid {key}."
            ) from error
    return tuple(header[key] for key in _ALIGNMENT_KEYS)


def _validate_complete_acquisitions(
    paths: list[Path], *, start: datetime, end: datetime
) -> None:
    try:
        acquisitions = resolve_acquisition_groups(files=paths)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            "Downloaded HMI data do not contain complete I0 through V5 acquisitions."
        ) from error
    resolved_paths = [path for _, group in acquisitions for path in group]
    if len(resolved_paths) != len(paths) or set(resolved_paths) != set(paths):
        raise RuntimeError(
            "Downloaded HMI data contain unexpected files outside complete "
            "I0 through V5 acquisitions."
        )

    expected_count = int((end - start).total_seconds() // HMI_NATIVE_CADENCE_SECONDS)
    if len(acquisitions) != expected_count:
        raise RuntimeError(
            "Downloaded HMI sequence has "
            f"{len(acquisitions)} acquisitions; expected exactly {expected_count}."
        )
    expected_times = [
        start + timedelta(seconds=index * HMI_NATIVE_CADENCE_SECONDS)
        for index in range(expected_count)
    ]
    for (acquisition_name, acquisition_paths), expected_time in zip(
        acquisitions, expected_times, strict=True
    ):
        match = _ACQUISITION_PATTERN.fullmatch(acquisition_name)
        if match is None:
            raise RuntimeError(
                f"Downloaded HMI acquisition has invalid identity {acquisition_name!r}."
            )
        filename_time = datetime.strptime(match.group("time"), "%Y%m%d_%H%M%S_TAI")
        if filename_time != expected_time:
            raise RuntimeError(
                "Downloaded HMI sequence is not the exact requested native-cadence "
                f"grid: expected {expected_time.isoformat()}, got "
                f"{filename_time.isoformat()}."
            )
        expected_t_rec = expected_time.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        reference_identity = None
        for segment, path in zip(HMI_STOKES_SEGMENTS, acquisition_paths, strict=True):
            identity = _read_full_disk_identity(path)
            if format_jsoc_time(str(identity[0])) != expected_t_rec:
                raise RuntimeError(
                    f"HMI segment {segment} filename and T_REC disagree: {path.name}."
                )
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise RuntimeError(
                    f"HMI segment {segment} in {acquisition_name!r} is not aligned "
                    "with I0."
                )


def download_hmi_stokes(
    *,
    output_directory: str | os.PathLike[str],
    email: str,
    start: str | datetime,
    end: str | datetime,
    client=None,
) -> list[Path]:
    """Download and validate complete native-cadence HMI Stokes acquisitions.

    Downloads are staged beside the requested output directory.  Files are
    published only after every acquisition has exactly one copy of all 24
    Stokes/filter segments; existing destination files are never overwritten.
    """

    start_time, end_time, _ = _validated_interval(start, end)
    query = build_hmi_stokes_query(start=start_time, end=end_time)
    email = _validate_email(email)
    if not isinstance(output_directory, (str, os.PathLike)):
        raise TypeError("output_directory must be a filesystem path.")
    output = Path(output_directory).expanduser().resolve()
    if os.path.lexists(output):
        raise FileExistsError(
            "HMI output is an immutable sequence directory and must not already "
            f"exist: {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    drms_client = _drms_client(email) if client is None else client
    with tempfile.TemporaryDirectory(
        prefix=".hmi-S_720s-", dir=output.parent
    ) as temporary_directory:
        staging_directory = Path(temporary_directory)
        request = drms_client.export(query, protocol="fits")
        if request is None or not callable(getattr(request, "wait", None)):
            raise RuntimeError("DRMS did not return a valid export request.")
        request.wait()
        if not callable(getattr(request, "download", None)):
            raise RuntimeError("DRMS export request cannot download files.")
        result = request.download(str(staging_directory), fname_from_rec=True)
        staged_paths = _downloaded_paths(result, staging_directory)
        _validate_complete_acquisitions(staged_paths, start=start_time, end=end_time)
        if os.path.lexists(output):
            raise FileExistsError(
                f"HMI output appeared while downloading; refusing publish: {output}."
            )
        os.replace(staging_directory, output)
    return [(output / path.name).resolve() for path in staged_paths]


__all__ = [
    "HMI_NATIVE_CADENCE_SECONDS",
    "HMI_CAMERA",
    "HMI_STOKES_SEGMENTS",
    "HMI_STOKES_SERIES",
    "build_hmi_stokes_query",
    "download_hmi_stokes",
]
