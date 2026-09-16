"""Strict, instrument-only discovery of AIA Level-1 EUV records.

This module performs metadata discovery and deterministic nearest-record
selection only.  It has no DRMS import and no dependency on another
instrument's acquisition or timing conventions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import numbers
import re
from types import MappingProxyType
from typing import Any

import numpy as np
from astropy.time import Time, TimeDelta
from dateutil.parser import parse


AIA_LEVEL1_SERIES = "aia.lev1_euv_12s"
AIA_IMAGE_SEGMENT = "image"
AIA_NATIVE_CADENCE_SECONDS = 12.0
AIA_EUV_CHANNELS_ANGSTROM = (171, 193, 211)
SUPPORTED_AIA_EUV_CHANNELS_ANGSTROM = (94, 131, 171, 193, 211, 304, 335)

AIA_METADATA_KEYS = (
    "T_REC",
    "T_OBS",
    "WAVELNTH",
    "EXPTIME",
    "QUALITY",
    "LVL_NUM",
    "TELESCOP",
    "INSTRUME",
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
)

_WCS_TEXT_KEYS = ("CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2")
_WCS_FLOAT_KEYS = (
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
_RECORD_PATTERN = re.compile(
    r"^aia\.lev1_euv_12s\[(?P<time>[^\]]+)\]\[(?P<channel>\d+)\]$"
)
_JSOC_TAI_PATTERN = re.compile(
    r"^(?P<date>\d{4}\.\d{2}\.\d{2})_"
    r"(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)_TAI$"
)


def _canonical_time(value: Time, *, scale: str) -> str:
    converted = getattr(value, scale)
    converted = converted.copy(format="isot")
    converted.precision = 9
    suffix = "Z" if scale == "utc" else " TAI"
    return f"{converted.value}{suffix}"


def _parse_utc_datetime(value: str | datetime, *, field: str) -> Time:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = parse(value.strip())
        except ValueError as error:
            raise ValueError(f"{field} must be a valid date/time.") from error
    else:
        raise TypeError(f"{field} must be an Astropy Time, datetime, or date string.")
    if parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return Time(parsed.astimezone(timezone.utc), scale="utc")


def parse_aia_target_time(value: Time | str | datetime, *, field: str = "target time") -> Time:
    """Parse flexible dates as UTC when unzoned; convert explicit offsets to UTC."""

    if isinstance(value, Time):
        if not value.isscalar:
            raise ValueError(f"{field} must be a scalar Astropy Time.")
        result = value.utc
    else:
        result = _parse_utc_datetime(value, field=field)
    if not np.isfinite(float(result.jd)):
        raise ValueError(f"{field} must be finite.")
    return result


def normalize_aia_target_times(
    values: Time | str | datetime | Sequence[Time | str | datetime],
) -> tuple[Time, ...]:
    """Normalize a non-empty, ordered collection of unique target instants."""

    if isinstance(values, Time):
        raw = (
            (values,)
            if values.isscalar
            else tuple(values[index] for index in range(len(values)))
        )
    elif isinstance(values, (str, datetime)):
        raw = (values,)
    elif isinstance(values, Sequence):
        raw = tuple(values)
    else:
        raise TypeError(
            "target_times must be an Astropy Time, UTC string, or sequence."
        )
    if not raw:
        raise ValueError("target_times must not be empty.")
    targets = tuple(
        parse_aia_target_time(value, field=f"target_times[{index}]")
        for index, value in enumerate(raw)
    )
    canonical = tuple(_canonical_time(value, scale="utc") for value in targets)
    if len(set(canonical)) != len(canonical):
        raise ValueError("target_times must contain unique instants.")
    return targets


def validate_aia_channels(channels: Sequence[int]) -> tuple[int, ...]:
    """Validate a non-empty selection of unique EUV channels, preserving order."""

    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise TypeError("channels must be an ordered integer sequence.")
    normalized = tuple(channels)
    if any(type(channel) is not int for channel in normalized):
        raise TypeError("AIA channels must be integers.")
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("AIA channels must be non-empty and unique.")
    if any(channel not in SUPPORTED_AIA_EUV_CHANNELS_ANGSTROM for channel in normalized):
        raise ValueError(
            f"AIA EUV channels must be selected from {SUPPORTED_AIA_EUV_CHANNELS_ANGSTROM}."
        )
    return normalized


def validate_max_offset_seconds(value: float | TimeDelta) -> float:
    """Return one finite positive record-matching tolerance in seconds."""

    if isinstance(value, TimeDelta):
        if not value.isscalar:
            raise ValueError("max_offset_seconds must be scalar.")
        result = float(value.to_value("sec"))
    elif isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("max_offset_seconds must be a number or Astropy TimeDelta.")
    else:
        result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("max_offset_seconds must be finite and strictly positive.")
    return result


def _query_utc(value: Time) -> str:
    utc = value.utc.copy(format="isot")
    utc.precision = 3
    return f"{utc.value}Z"


def build_aia_candidate_query(
    *,
    target_time: Time | str,
    channel_angstrom: int,
    max_offset_seconds: float | TimeDelta,
) -> str:
    """Build a bounded query containing both neighbors of a target instant."""

    target = parse_aia_target_time(target_time)
    if type(channel_angstrom) is not int or (
        channel_angstrom not in SUPPORTED_AIA_EUV_CHANNELS_ANGSTROM
    ):
        raise ValueError(
            f"channel_angstrom must be one of {SUPPORTED_AIA_EUV_CHANNELS_ANGSTROM}."
        )
    tolerance = validate_max_offset_seconds(max_offset_seconds)
    half_width = tolerance + AIA_NATIVE_CADENCE_SECONDS
    start = target - TimeDelta(half_width, format="sec")
    duration = 2.0 * half_width
    duration_text = f"{duration:.9f}".rstrip("0").rstrip(".")
    return (
        f"{AIA_LEVEL1_SERIES}[{_query_utc(start)}/{duration_text}s][{channel_angstrom}]"
    )


def parse_aia_jsoc_time(value: Any, *, field: str) -> Time:
    """Parse an explicitly scaled JSOC time or a UTC AIA FITS timestamp.

    JSOC may return ``_TAI`` record-key values or ``Z`` keyword values.  AIA
    FITS keyword timestamps are ISO-shaped and omit a scale suffix; those are
    UTC, consistent with the FITS time default and AIA's exported time
    convention. User target strings are parsed separately with dateutil and
    default to UTC when unzoned.
    """

    if isinstance(value, Time):
        if not value.isscalar:
            raise ValueError(f"{field} must be scalar.")
        result = value
    elif type(value) is str and value and value.strip() == value:
        match = _JSOC_TAI_PATTERN.fullmatch(value)
        if match is not None:
            iso = f"{match.group('date').replace('.', '-')}T{match.group('time')}"
            try:
                result = Time(iso, format="isot", scale="tai")
            except ValueError as error:
                raise ValueError(
                    f"{field} has an invalid JSOC TAI timestamp."
                ) from error
        else:
            try:
                result = _parse_utc_datetime(value, field=field)
            except ValueError:
                try:
                    result = Time(value, format="isot", scale="utc")
                except ValueError as error:
                    raise ValueError(
                        f"{field} has an invalid AIA timestamp."
                    ) from error
    else:
        raise TypeError(f"{field} must contain an explicit TAI or UTC timestamp.")
    if not result.isscalar or not np.isfinite(float(result.jd)):
        raise ValueError(f"{field} must contain one finite timestamp.")
    return result


def _strict_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field} must be an integer.")
    result = int(value)
    if not math.isfinite(float(value)) or float(value) != result:
        raise ValueError(f"{field} must be an integer.")
    return result


def _quality_integer(value: Any) -> int:
    """Parse QUALITY without accepting ambiguous or partial text."""

    if isinstance(value, str):
        if not value or value.strip() != value:
            raise ValueError("QUALITY must be an integer or hexadecimal word.")
        try:
            result = int(value, 0)
        except ValueError as error:
            raise ValueError(
                "QUALITY must be an integer or hexadecimal word."
            ) from error
        if str(result) != value and not re.fullmatch(r"0[xX][0-9a-fA-F]+", value):
            raise ValueError("QUALITY must be an integer or hexadecimal word.")
        return result
    return _strict_integer(value, field="QUALITY")


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class AIALevel1Identity:
    """Validated record metadata shared by discovery and FITS verification."""

    record_time: Time
    observation_time: Time
    channel_angstrom: int
    exposure_seconds: float
    quality: int
    level: float
    telescope: str
    instrument: str
    wcs: Mapping[str, str | float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "wcs", MappingProxyType(dict(self.wcs)))


def validate_aia_level1_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_channel: int,
) -> AIALevel1Identity:
    """Validate scientific identity and full-disk helioprojective WCS metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("AIA Level-1 metadata must be a mapping.")
    missing = sorted(set(AIA_METADATA_KEYS) - set(metadata))
    if missing:
        raise ValueError(f"AIA Level-1 metadata is missing required keys {missing}.")
    channel = _strict_integer(metadata["WAVELNTH"], field="WAVELNTH")
    if channel != expected_channel:
        raise ValueError(
            f"AIA record wavelength {channel} does not match requested "
            f"channel {expected_channel}."
        )
    quality = _quality_integer(metadata["QUALITY"])
    exposure = _finite_float(metadata["EXPTIME"], field="EXPTIME")
    if exposure <= 0:
        raise ValueError("AIA EXPTIME must be strictly positive.")
    level = _finite_float(metadata["LVL_NUM"], field="LVL_NUM")
    if level != 1.0:
        raise ValueError(f"AIA LVL_NUM must be exactly 1.0; got {level!r}.")
    telescope = _required_text(metadata["TELESCOP"], field="TELESCOP")
    instrument = _required_text(metadata["INSTRUME"], field="INSTRUME")
    if telescope.upper() != "SDO/AIA" or not instrument.upper().startswith("AIA_"):
        raise ValueError("AIA Level-1 metadata has the wrong telescope/instrument.")

    text_wcs = {key: _required_text(metadata[key], field=key) for key in _WCS_TEXT_KEYS}
    float_wcs = {
        key: _finite_float(metadata[key], field=key) for key in _WCS_FLOAT_KEYS
    }
    if (
        text_wcs["CTYPE1"].upper() != "HPLN-TAN"
        or text_wcs["CTYPE2"].upper() != "HPLT-TAN"
        or text_wcs["CUNIT1"].lower() != "arcsec"
        or text_wcs["CUNIT2"].lower() != "arcsec"
        or float_wcs["CDELT1"] <= 0
        or float_wcs["CDELT2"] <= 0
        or float_wcs["CRPIX1"] <= 0
        or float_wcs["CRPIX2"] <= 0
        or float_wcs["RSUN_REF"] <= 0
        or float_wcs["DSUN_OBS"] <= float_wcs["RSUN_REF"]
        or float_wcs["RSUN_OBS"] <= 0
        or not 0 <= float_wcs["CRLN_OBS"] < 360
        or not -90 <= float_wcs["CRLT_OBS"] <= 90
    ):
        raise ValueError("AIA Level-1 metadata has non-physical WCS values.")
    expected_radius_arcsec = (
        math.degrees(math.asin(float_wcs["RSUN_REF"] / float_wcs["DSUN_OBS"])) * 3600.0
    )
    if not math.isclose(
        float_wcs["RSUN_OBS"],
        expected_radius_arcsec,
        rel_tol=0.0,
        abs_tol=1.0,
    ):
        raise ValueError("AIA Level-1 metadata has inconsistent solar-radius WCS.")
    wcs: dict[str, str | float] = {**text_wcs, **float_wcs}
    return AIALevel1Identity(
        record_time=parse_aia_jsoc_time(metadata["T_REC"], field="T_REC"),
        observation_time=parse_aia_jsoc_time(metadata["T_OBS"], field="T_OBS"),
        channel_angstrom=channel,
        exposure_seconds=exposure,
        quality=quality,
        level=level,
        telescope=telescope,
        instrument=instrument,
        wcs=wcs,
    )


@dataclass(frozen=True, slots=True)
class SelectedAIARecord:
    """One exact AIA record selected for one target/channel pair."""

    target_index: int
    target_time: Time
    channel_angstrom: int
    record_id: str
    query: str
    identity: AIALevel1Identity
    signed_offset_seconds: float

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "target_index": self.target_index,
            "target_time_utc": _canonical_time(self.target_time, scale="utc"),
            "channel_angstrom": self.channel_angstrom,
            "record_id": self.record_id,
            "record_time_tai": _canonical_time(self.identity.record_time, scale="tai"),
            "record_time_utc": _canonical_time(self.identity.record_time, scale="utc"),
            "observation_time_utc": _canonical_time(
                self.identity.observation_time, scale="utc"
            ),
            "observation_time_tai": _canonical_time(
                self.identity.observation_time, scale="tai"
            ),
            "signed_offset_seconds": self.signed_offset_seconds,
            "quality": self.identity.quality,
            "exposure_seconds": self.identity.exposure_seconds,
            "level": self.identity.level,
            "telescope": self.identity.telescope,
            "instrument": self.identity.instrument,
            "wcs": dict(self.identity.wcs),
            "discovery_query": self.query,
        }


def _query_rows(result: Any) -> list[tuple[str | None, Mapping[str, Any]]]:
    if result is None:
        return []
    if callable(getattr(result, "iterrows", None)):
        rows = []
        for index, row in result.iterrows():
            value = row.to_dict() if callable(getattr(row, "to_dict", None)) else row
            if not isinstance(value, Mapping):
                raise TypeError("DRMS AIA query rows must be mappings.")
            rows.append((str(index), dict(value)))
        return rows
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        rows = []
        for value in result:
            if not isinstance(value, Mapping):
                raise TypeError("DRMS AIA query rows must be mappings.")
            row = dict(value)
            record = row.pop("record_id", row.pop("record", None))
            rows.append((None if record is None else str(record), row))
        return rows
    raise TypeError("DRMS AIA query must return a table or sequence of mappings.")


def _record_label_time(value: str, *, field: str) -> Time:
    """Parse an AIA series-slot label without applying a scale conversion.

    The JSOC record index spells AIA slot labels with ``_TAI``, while the
    linked ``T_REC`` keyword can expose the same numeric label with ``Z``.
    Record-key agreement is therefore a comparison of slot labels.  Physical
    observation-time comparisons continue to use :func:`parse_aia_jsoc_time`.
    """

    match = _JSOC_TAI_PATTERN.fullmatch(value)
    if match is not None:
        iso = f"{match.group('date').replace('.', '-')}T{match.group('time')}"
    else:
        if value.endswith("Z"):
            iso = value[:-1]
        elif value.endswith("+00:00"):
            iso = value[:-6]
        else:
            iso = value
    try:
        result = Time(iso, format="isot", scale="utc")
    except ValueError as error:
        raise ValueError(f"{field} has an invalid AIA record-slot label.") from error
    if not result.isscalar or not np.isfinite(float(result.jd)):
        raise ValueError(f"{field} must contain one finite record-slot label.")
    return result


def _record_id(
    reported: str | None,
    *,
    raw_record_time: Any,
    channel: int,
) -> str:
    if not isinstance(raw_record_time, str) or not raw_record_time.strip():
        raise ValueError("AIA T_REC must retain a textual time representation.")
    raw_label = _record_label_time(raw_record_time, field="T_REC")
    label = raw_label.copy(format="isot")
    label.precision = 9
    date, clock = label.value.split("T")
    clock = clock.rstrip("0").rstrip(".")
    record_key = f"{date.replace('-', '.')}_{clock}_TAI"
    canonical = f"{AIA_LEVEL1_SERIES}[{record_key}][{channel}]"
    if reported is None:
        return canonical
    match = _RECORD_PATTERN.fullmatch(reported)
    if match is None:
        raise ValueError(f"DRMS returned malformed AIA record identity {reported!r}.")
    reported_label = _record_label_time(match.group("time"), field="record identity")
    if (
        int(match.group("channel")) != channel
        or abs(float((reported_label - raw_label).to_value("sec"))) > 1.0e-9
    ):
        raise ValueError("DRMS AIA record identity disagrees with its prime keys.")
    return reported


def select_aia_level1_records(
    client: Any,
    *,
    target_times: Time | str | Sequence[Time | str],
    channels: Sequence[int] = AIA_EUV_CHANNELS_ANGSTROM,
    max_offset_seconds: float | TimeDelta = 12.0,
) -> tuple[SelectedAIARecord, ...]:
    """Select exactly one nearest quality-good record per target and channel."""

    if not callable(getattr(client, "query", None)):
        raise TypeError("client must provide the DRMS query() interface.")
    targets = normalize_aia_target_times(target_times)
    ordered_channels = validate_aia_channels(channels)
    tolerance = validate_max_offset_seconds(max_offset_seconds)
    selected: list[SelectedAIARecord] = []
    selected_record_ids: set[str] = set()

    for target_index, target in enumerate(targets):
        for channel in ordered_channels:
            query = build_aia_candidate_query(
                target_time=target,
                channel_angstrom=channel,
                max_offset_seconds=tolerance,
            )
            raw_result = client.query(
                query,
                key=list(AIA_METADATA_KEYS),
                rec_index=True,
            )
            rows = _query_rows(raw_result)
            candidates: list[tuple[float, str, AIALevel1Identity]] = []
            seen: set[str] = set()
            for reported_record, metadata in rows:
                core_missing = sorted(
                    {"T_REC", "T_OBS", "WAVELNTH", "QUALITY"} - set(metadata)
                )
                if core_missing:
                    raise ValueError(
                        "AIA candidate metadata is missing identity keys "
                        f"{core_missing}."
                    )
                row_channel = _strict_integer(metadata["WAVELNTH"], field="WAVELNTH")
                if row_channel != channel:
                    raise ValueError(
                        f"AIA query for {channel} Angstrom returned {row_channel}."
                    )
                parse_aia_jsoc_time(metadata["T_REC"], field="T_REC")
                record_id = _record_id(
                    reported_record,
                    raw_record_time=metadata["T_REC"],
                    channel=channel,
                )
                if record_id in seen:
                    raise ValueError(
                        f"DRMS returned duplicate AIA record {record_id!r}."
                    )
                seen.add(record_id)
                identity = validate_aia_level1_metadata(
                    metadata,
                    expected_channel=channel,
                )
                if identity.quality != 0:
                    continue
                offset = float((identity.observation_time - target).to_value("sec"))
                candidates.append((offset, record_id, identity))
            if not candidates:
                raise RuntimeError(
                    "No QUALITY=0 AIA Level-1 record with valid metadata exists "
                    f"for target {_canonical_time(target, scale='utc')} and "
                    f"channel {channel}."
                )
            nearest_distance = min(abs(item[0]) for item in candidates)
            nearest = [
                item
                for item in candidates
                if math.isclose(
                    abs(item[0]), nearest_distance, rel_tol=0.0, abs_tol=1.0e-9
                )
            ]
            if len(nearest) != 1:
                identities = sorted(item[1] for item in nearest)
                raise ValueError(
                    "AIA nearest-record selection is tied for target "
                    f"{_canonical_time(target, scale='utc')}, channel {channel}: "
                    f"{identities}."
                )
            offset, record_id, identity = nearest[0]
            if abs(offset) > tolerance + 1.0e-9:
                raise RuntimeError(
                    f"Nearest quality-good AIA {channel} record is offset by "
                    f"{offset:+.9f} s, exceeding {tolerance:.9f} s."
                )
            if record_id in selected_record_ids:
                raise ValueError(
                    f"AIA record {record_id!r} was selected for multiple targets."
                )
            selected_record_ids.add(record_id)
            selected.append(
                SelectedAIARecord(
                    target_index=target_index,
                    target_time=target,
                    channel_angstrom=channel,
                    record_id=record_id,
                    query=query,
                    identity=identity,
                    signed_offset_seconds=offset,
                )
            )
    expected_count = len(targets) * len(ordered_channels)
    if len(selected) != expected_count:  # pragma: no cover - loop is exhaustive
        raise RuntimeError(
            f"Selected {len(selected)} AIA records; expected {expected_count}."
        )
    return tuple(selected)


__all__ = [
    "AIA_EUV_CHANNELS_ANGSTROM",
    "SUPPORTED_AIA_EUV_CHANNELS_ANGSTROM",
    "AIA_IMAGE_SEGMENT",
    "AIA_LEVEL1_SERIES",
    "AIA_METADATA_KEYS",
    "AIA_NATIVE_CADENCE_SECONDS",
    "AIALevel1Identity",
    "SelectedAIARecord",
    "build_aia_candidate_query",
    "normalize_aia_target_times",
    "parse_aia_jsoc_time",
    "parse_aia_target_time",
    "select_aia_level1_records",
    "validate_aia_channels",
    "validate_aia_level1_metadata",
    "validate_max_offset_seconds",
]
