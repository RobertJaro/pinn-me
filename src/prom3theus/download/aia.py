"""Download AIA EUV files directly with DRMS."""

from datetime import datetime, timezone
from pathlib import Path

import drms
from dateutil.parser import parse


def _utc(value: str | datetime) -> datetime:
    value = parse(value) if isinstance(value, str) else value
    return value.replace(tzinfo=timezone.utc) if value.utcoffset() is None else value.astimezone(timezone.utc)


def download_aia_observations(*, output_directory, email, start, end,
                              cadence_seconds=720, channels=(171, 193, 211)) -> list:
    """Download each channel over the requested interval; unzoned dates use UTC."""
    start, end = _utc(start), _utc(end)
    duration = (end - start).total_seconds()
    if duration <= 0 or cadence_seconds <= 0:
        raise ValueError("end must follow start and cadence_seconds must be positive")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = drms.Client(email=email)
    paths = []
    for channel in channels:
        query = f"aia.lev1_euv_12s[{start:%Y-%m-%dT%H:%M:%S}_UTC/{duration:g}s@{cadence_seconds:g}s][{channel}]{{image}}"
        request = client.export(query, method="url", protocol="fits")
        request.wait()
        paths.extend(request.download(str(output))["download"].tolist())
    return paths
