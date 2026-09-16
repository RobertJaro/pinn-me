"""Fetch and cache the two aiapy tables used during AIA preprocessing."""

from datetime import timezone
import json
from pathlib import Path

import numpy as np
from astropy.table import QTable
from astropy.time import Time

from .acquisition import parse_aia_target_time
from .download import _sha256_file

AIA_CALIBRATION_MANIFEST = "manifest.json"
AIA_CORRECTION_ECSV = "aia_correction.ecsv"
AIA_POINTING_ECSV = "aia_pointing.ecsv"


class AiapyCalibrationAcquisitionBackend:
    """Use aiapy's standard table retrieval functions."""

    def fetch_correction_table(self):
        from aiapy.calibrate.utils import get_correction_table
        return get_correction_table(source="SSW")

    def fetch_pointing_table(self, *, start_utc, end_utc):
        from aiapy.calibrate.utils import get_pointing_table
        return get_pointing_table(
            source="jsoc", time_range=(
                parse_aia_target_time(start_utc), parse_aia_target_time(end_utc),
            ),
        )


def _request_interval(start_utc, end_utc):
    start = parse_aia_target_time(start_utc).to_datetime(timezone=timezone.utc)
    end = parse_aia_target_time(end_utc).to_datetime(timezone=timezone.utc)
    if end <= start:
        raise ValueError("end_utc must be later than start_utc.")
    return start, end


def _pointing_covers(table, start, end):
    """Require continuous pointing coverage for the requested exposures."""
    intervals = sorted(zip(
        Time(table["T_START"]).to_datetime(timezone=timezone.utc),
        Time(table["T_STOP"]).to_datetime(timezone=timezone.utc),
    ))
    cursor = start
    for left, right in intervals:
        if left > cursor:
            break
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def load_aia_calibration_manifest(directory):
    """Check that the two cached tables still match their recorded checksums."""
    root = Path(directory)
    manifest = json.loads((root / AIA_CALIBRATION_MANIFEST).read_text())
    for name, filename in (("correction", AIA_CORRECTION_ECSV), ("pointing", AIA_POINTING_ECSV)):
        if _sha256_file(root / filename) != manifest["tables"][name]["sha256"]:
            raise ValueError(f"AIA cached {name} table changed; use overwrite=True.")
    return manifest


def download_aia_preprocessing_calibration(
    *, output_directory, start_utc, end_utc, backend=None, overwrite=False,
):
    """Cache correction and pointing tables; reuse them when they cover the request."""
    start, end = _request_interval(start_utc, end_utc)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and not overwrite:
        manifest = load_aia_calibration_manifest(output)
        existing_start, existing_end = _request_interval(
            manifest["request"]["pointing_start_utc"],
            manifest["request"]["pointing_end_utc"],
        )
        if start < existing_start or end > existing_end:
            raise ValueError("Existing AIA calibration does not cover the interval; use overwrite=True.")
        return output / AIA_CALIBRATION_MANIFEST

    backend = backend or AiapyCalibrationAcquisitionBackend()
    correction = backend.fetch_correction_table()
    pointing = backend.fetch_pointing_table(start_utc=start.isoformat(), end_utc=end.isoformat())
    if not np.any(np.asarray(correction["VER_NUM"]) == 10):
        raise ValueError("AIA correction table must contain response-compatible V10 calibration.")
    if not _pointing_covers(pointing, start, end):
        raise ValueError("AIA pointing table has a gap or does not cover the requested interval.")

    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, table, filename in (
        ("correction", correction, AIA_CORRECTION_ECSV),
        ("pointing", pointing, AIA_POINTING_ECSV),
    ):
        QTable(table).write(output / filename, format="ascii.ecsv", overwrite=overwrite)
        tables[name] = {"file": filename, "sha256": _sha256_file(output / filename)}
    manifest = {
        "request": {"pointing_start_utc": start.isoformat(), "pointing_end_utc": end.isoformat()},
        "tables": tables,
    }
    (output / AIA_CALIBRATION_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return output / AIA_CALIBRATION_MANIFEST
