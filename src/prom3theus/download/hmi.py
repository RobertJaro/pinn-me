"""Download HMI full-disk Stokes and vector comparison files with DRMS."""

from datetime import datetime
from pathlib import Path

import drms
from astropy.time import Time
from dateutil.parser import parse

from prom3theus.core.downloads import download_fits_export


def _tai(value: str | datetime) -> datetime:
    value = parse(value) if isinstance(value, str) else value
    return Time(value).tai.to_datetime() if value.utcoffset() is not None else value


def download_hmi_comparison(*, output_directory, email, time) -> list[str]:
    """Download one B_720s vector record for compare-hmi; unzoned times use TAI."""
    timestamp = _tai(time)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    segments = ("field", "inclination", "azimuth", "disambig")
    query = (
        f"hmi.B_720s[{timestamp:%Y-%m-%dT%H:%M:%S}_TAI]"
        f"{{{','.join(segments)}}}"
    )
    request = drms.Client(email=email).export(query, method="url", protocol="fits")
    result = download_fits_export(request, output)
    files = result["download"].tolist()
    if len(files) != len(segments) or any(
        not isinstance(path, (str, Path)) or not Path(path).is_file() for path in files
    ):
        raise RuntimeError("HMI comparison download did not return all four FITS files.")
    for segment in segments:
        if sum(Path(path).name.endswith(f".{segment}.fits") for path in files) != 1:
            raise RuntimeError(f"HMI comparison download is missing segment {segment!r}.")
    return [str(path) for path in files]


def download_hmi_stokes(*, output_directory, email, start, end) -> list[str]:
    """Download through DRMS. Unzoned dates use TAI."""
    start, end = _tai(start), _tai(end)
    duration = (end - start).total_seconds()
    if duration <= 0:
        raise ValueError("end must be later than start")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    segments = ",".join(f"{component}{index}" for component in "IQUV" for index in range(6))
    query = f"hmi.S_720s[{start:%Y-%m-%dT%H:%M:%S}_TAI/{duration:g}s]{{{segments}}}"
    request = drms.Client(email=email).export(query, method="url", protocol="fits")
    request.wait()
    result = request.download(str(output))
    return result["download"].tolist()
