"""Deterministic discovery and FITS reading for HMI Stokes acquisitions."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import re
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.time import Time
from dateutil.parser import parse

from .constants import HMI_CAMERA


SEGMENT_KEYS = tuple(
    f"{component}{index}" for component in "IQUV" for index in range(6)
)
_SEGMENT_PATTERN = re.compile(r"\.(?P<stokes>[IQUV])(?P<filter>[0-5])\.fits$")
# JSOC export filenames spell the series prefix as ``hmi.s_720s`` even though
# the DRMS series name is conventionally written ``hmi.S_720s``. Accept only
# those two exact spellings; the remaining acquisition contract stays strict.
_ACQUISITION_PATTERN = re.compile(r"hmi\.[Ss]_720s\.\d{8}_\d{6}_TAI\.3")
_IDENTITY_TEXT_KEYS = (
    "DATE-OBS",
    "T_OBS",
    "T_REC",
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "TELESCOP",
    "INSTRUME",
)
_IDENTITY_INTEGER_KEYS = ("CAMERA", "HCAMID")
_IDENTITY_FLOAT_KEYS = (
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


def format_jsoc_time(value: str | datetime) -> str:
    """Normalize an observation timestamp to the JSOC TAI key representation."""

    if isinstance(value, str) and value.endswith("_TAI"):
        raw = value.removesuffix("_TAI")
        formats = ("%Y.%m.%d_%H:%M:%S", "%Y.%m.%d_%H:%M:%S.%f")
        if not any(
            _matches_datetime_format(raw, date_format) for date_format in formats
        ):
            raise ValueError(f"Invalid JSOC TAI timestamp {value!r}.")
        tai = Time(
            next(
                datetime.strptime(raw, time_format)
                for time_format in formats
                if _matches_datetime_format(raw, time_format)
            ),
            scale="tai",
        )
    else:
        if isinstance(value, str):
            parsed = parse(value)
        elif isinstance(value, datetime):
            parsed = value
        else:
            raise TypeError("HMI timestamp must be a string or datetime.")
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            tai = Time(parsed).tai
        else:
            tai = Time(parsed, scale="tai")
    timestamp = tai.to_datetime().strftime("%Y.%m.%d_%H:%M:%S.%f").rstrip("0")
    return f"{timestamp.rstrip('.')}_TAI"


def _matches_datetime_format(value: str, date_format: str) -> bool:
    try:
        datetime.strptime(value, date_format)
    except ValueError:
        return False
    return True


def parse_hmi_tai_time(value: str | datetime) -> Time:
    """Parse an HMI/JSOC timestamp as an explicit TAI instant."""

    if isinstance(value, str) and not value.endswith("_TAI"):
        raise ValueError(f"HMI timestamp must declare the TAI scale: {value!r}.")
    normalized = format_jsoc_time(value)
    for time_format in (
        "%Y.%m.%d_%H:%M:%S_TAI",
        "%Y.%m.%d_%H:%M:%S.%f_TAI",
    ):
        try:
            parsed = datetime.strptime(normalized, time_format)
        except ValueError:
            continue
        return Time(parsed, scale="tai")
    raise ValueError(f"Invalid HMI TAI timestamp {value!r}.")


def hmi_observation_wcs_header(header: fits.Header) -> fits.Header:
    """Return WCS metadata dated at HMI's nominal physical observation time."""

    if "T_OBS" not in header:
        raise KeyError("HMI FITS header is missing canonical T_OBS.")
    parse_hmi_tai_time(str(header["T_OBS"]))
    observation_time = format_jsoc_time(str(header["T_OBS"]))
    metadata = header.copy()
    metadata["DATE-OBS"] = observation_time
    metadata["TIMESYS"] = "TAI"
    metadata.pop("T_OBS", None)
    for key in ("DATE-AVG", "MJD-OBS", "MJDOBS", "MJD-AVG"):
        metadata.pop(key, None)
    return metadata


def hmi_acquisition_key(record_time: str | datetime, hcamid: int) -> str:
    """Return the immutable record-slot and optical-camera acquisition key."""

    if type(hcamid) is not int or hcamid not in {2, 3}:
        raise ValueError("HCAMID must be integer camera 2 or 3.")
    return f"{format_jsoc_time(record_time)}|HCAMID={hcamid}"


def _resolve_input_paths(inputs) -> list[Path]:
    if inputs is None:
        raise ValueError("HMI FITS inputs must be provided.")
    patterns = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    paths: list[Path] = []
    for pattern in patterns:
        value = os.fspath(pattern)
        if os.path.isdir(value):
            paths.extend(
                Path(path) for path in glob.glob(os.path.join(value, "*.fits"))
            )
        else:
            paths.extend(Path(path) for path in glob.glob(value))
    return sorted(dict.fromkeys(paths))


def resolve_segment_files(files) -> dict[str, Path]:
    """Resolve exactly one complete 24-segment HMI Stokes acquisition."""

    paths = _resolve_input_paths(files)
    if not paths:
        raise FileNotFoundError(f"No HMI FITS segments match {files!r}.")
    segments: dict[str, Path] = {}
    prefixes = set()
    for path in paths:
        match = _SEGMENT_PATTERN.search(path.name)
        if match is None:
            continue
        key = f"{match.group('stokes')}{match.group('filter')}"
        if key in segments:
            raise ValueError(
                f"Multiple HMI acquisitions were selected for segment {key}."
            )
        segments[key] = path
        prefixes.add(path.name[: match.start()])
    missing = set(SEGMENT_KEYS) - set(segments)
    if missing or len(prefixes) != 1:
        raise ValueError(
            "HMI LTE loading requires exactly one complete hmi.S_720s acquisition "
            f"with I0...V5; missing={sorted(missing)}, acquisitions={sorted(prefixes)}."
        )
    acquisition = next(iter(prefixes))
    if _ACQUISITION_PATTERN.fullmatch(acquisition) is None:
        raise ValueError(
            "HMI LTE loading requires exact hmi.S_720s CAMERA=3 filenames; "
            f"received {acquisition!r}."
        )
    return segments


def resolve_acquisition_groups(
    files=None, directory=None
) -> list[tuple[str, list[Path]]]:
    """Resolve a deterministic sequence of complete 24-segment acquisitions."""

    if files is not None and directory is not None:
        raise ValueError("Specify either HMI files or directory, not both.")
    source = directory if directory is not None else files
    if source is None:
        raise ValueError("HMI data require files or directory.")
    paths = _resolve_input_paths(source)
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        match = _SEGMENT_PATTERN.search(path.name)
        if match is not None:
            grouped.setdefault(path.name[: match.start()], []).append(path)
    if not grouped:
        raise FileNotFoundError(f"No HMI FITS segments match {source!r}.")
    resolved = []
    for name, acquisition_paths in sorted(grouped.items()):
        segments = resolve_segment_files(acquisition_paths)
        resolved.append((name, [segments[key] for key in SEGMENT_KEYS]))
    return resolved


def read_acquisition_header(path: Path) -> dict:
    """Read the acquisition identity and timestamp from one reference segment."""

    header = fits.getheader(path, 0)
    try:
        date = parse_hmi_tai_time(str(header["T_OBS"]))
        observation_time = format_jsoc_time(str(header["T_OBS"]))
        parse_hmi_tai_time(str(header["T_REC"]))
        record_time = format_jsoc_time(str(header["T_REC"]))
        camera = header["CAMERA"]
        hcamid = header["HCAMID"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"HMI FITS acquisition has invalid T_OBS/T_REC/CAMERA/HCAMID: {path}."
        ) from error
    if type(camera) is not int or camera != HMI_CAMERA:
        raise ValueError(
            f"HMI FITS acquisition requires CAMERA={HMI_CAMERA}; got {camera!r}: "
            f"{path}."
        )
    if type(hcamid) is not int or hcamid not in {2, 3}:
        raise ValueError(
            f"HMI FITS acquisition has unsupported HCAMID={hcamid}: {path}."
        )
    return {
        "date": date,
        "path": str(path.resolve()),
        "observation_time": observation_time,
        "record_time": record_time,
        "hcamid": hcamid,
        "acquisition_key": hmi_acquisition_key(record_time, hcamid),
    }


def read_stokes_cube(
    segments: dict[str, Path],
    *,
    require_quality_zero: bool,
) -> tuple[list[fits.Header], np.ndarray]:
    """Stream one acquisition's 24 segments into ``[y, x, Stokes, filter]``."""

    headers: list[fits.Header] = []
    reference = None
    stokes = None
    for key in SEGMENT_KEYS:
        path = segments[key]
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header.copy()
            data = hdul[0].data
            if (
                data is None
                or data.ndim != 2
                or min(data.shape) < 1
                or not np.issubdtype(data.dtype, np.number)
            ):
                raise ValueError(
                    f"HMI segment {path.name} must contain one non-empty 2D image."
                )
            required = {
                *_IDENTITY_TEXT_KEYS,
                *_IDENTITY_INTEGER_KEYS,
                *_IDENTITY_FLOAT_KEYS,
                "QUALITY",
            }
            missing = sorted(required - set(header))
            if missing:
                raise ValueError(
                    f"HMI segment {path.name} is missing required FITS keywords {missing}."
                )
            if ("CCD_X0" in header) != ("CCD_Y0" in header):
                raise ValueError(
                    f"HMI segment {path.name} must provide CCD_X0 and CCD_Y0 together."
                )
            try:
                floating_values = {
                    name: float(header[name]) for name in _IDENTITY_FLOAT_KEYS
                }
                floating_identity = tuple(floating_values.values())
                integer_identity = tuple(
                    int(header[name]) for name in _IDENTITY_INTEGER_KEYS
                )
                quality = int(header["QUALITY"])
                detector_origin = (
                    None
                    if "CCD_X0" not in header
                    else (float(header["CCD_X0"]), float(header["CCD_Y0"]))
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"HMI segment {path.name} has invalid numeric FITS geometry."
                ) from error
            numeric_identity = np.asarray(
                (
                    *floating_identity,
                    *(() if detector_origin is None else detector_origin),
                ),
                dtype=np.float64,
            )
            if not np.isfinite(numeric_identity).all():
                raise ValueError(
                    f"HMI segment {path.name} has non-finite FITS geometry."
                )
            text_values = {
                name: str(header[name]).strip() for name in _IDENTITY_TEXT_KEYS
            }
            try:
                Time(text_values["DATE-OBS"])
                parse_hmi_tai_time(text_values["T_OBS"])
                parse_hmi_tai_time(text_values["T_REC"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"HMI segment {path.name} has invalid DATE-OBS/T_OBS/T_REC metadata."
                ) from error
            if (
                text_values["CTYPE1"].upper() != "HPLN-TAN"
                or text_values["CTYPE2"].upper() != "HPLT-TAN"
                or text_values["CUNIT1"].lower() != "arcsec"
                or text_values["CUNIT2"].lower() != "arcsec"
                or not text_values["TELESCOP"]
                or not text_values["INSTRUME"]
                or integer_identity[0] != HMI_CAMERA
                or integer_identity[1] not in {2, 3}
                or quality < 0
                or floating_values["CDELT1"] <= 0
                or floating_values["CDELT2"] <= 0
                or floating_values["RSUN_REF"] <= 0
                or floating_values["DSUN_OBS"] <= floating_values["RSUN_REF"]
                or floating_values["RSUN_OBS"] <= 0
                or not -90 <= floating_values["CRLT_OBS"] <= 90
            ):
                raise ValueError(
                    f"HMI segment {path.name} has non-physical FITS geometry metadata."
                )
            expected_solar_radius_arcsec = (
                np.degrees(
                    np.arcsin(floating_values["RSUN_REF"] / floating_values["DSUN_OBS"])
                )
                * 3600.0
            )
            if not np.isclose(
                floating_values["RSUN_OBS"],
                expected_solar_radius_arcsec,
                rtol=0.0,
                atol=1.0,
            ):
                raise ValueError(
                    f"HMI segment {path.name} has inconsistent solar-radius metadata."
                )
            identity = (
                data.shape,
                tuple(text_values.values()),
                integer_identity,
                floating_identity,
                detector_origin,
            )
            if reference is None:
                reference = identity
                stokes = np.empty((*data.shape, 4, 6), dtype=np.float32)
            elif identity != reference:
                raise ValueError(f"HMI segment {path.name} is not aligned with I0.")
            if require_quality_zero and quality != 0:
                raise ValueError(
                    f"Refusing HMI segment {path.name} with QUALITY={quality:#x}."
                )
            component_index = "IQUV".index(key[0])
            wavelength_index = 5 - int(key[1])
            stokes[..., component_index, wavelength_index] = data
        headers.append(header)
    if stokes is None:  # guarded by resolve_segment_files; keeps the type explicit
        raise RuntimeError("HMI acquisition contains no readable segments.")
    return headers, stokes


def discover_hmi_acquisitions(inputs) -> list[dict]:
    """Discover unique HMI acquisition identities from I0 FITS segments."""

    candidates = _resolve_input_paths(inputs)
    representatives = [path for path in candidates if path.name.endswith(".I0.fits")]
    if not representatives:
        raise ValueError(
            "No HMI I0 FITS segments found; expected filenames ending in '.I0.fits'."
        )
    acquisitions = {}
    for path in representatives:
        item = read_acquisition_header(path)
        acquisitions.setdefault(item["acquisition_key"], item)
    return [acquisitions[key] for key in sorted(acquisitions)]


__all__ = [
    "SEGMENT_KEYS",
    "discover_hmi_acquisitions",
    "format_jsoc_time",
    "hmi_acquisition_key",
    "hmi_observation_wcs_header",
    "parse_hmi_tai_time",
    "read_acquisition_header",
    "read_stokes_cube",
    "resolve_acquisition_groups",
    "resolve_segment_files",
]
