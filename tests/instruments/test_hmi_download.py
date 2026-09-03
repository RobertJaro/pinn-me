"""Strict, network-free tests for HMI S_720s downloads."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from prom3theus.instruments.hmi.acquisition import SEGMENT_KEYS
from prom3theus.instruments.hmi import download as download_module
from prom3theus.instruments.hmi.download import (
    build_hmi_stokes_query,
    download_hmi_stokes,
)


def _tai_key(value: datetime) -> str:
    return value.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def _filename_time(value: datetime) -> str:
    return value.strftime("%Y%m%d_%H%M%S_TAI")


def _header(value: datetime) -> fits.Header:
    return fits.Header(
        {
            "T_REC": _tai_key(value),
            "T_OBS": _tai_key(value - timedelta(seconds=2)),
            "DATE-OBS": (value - timedelta(seconds=84)).isoformat(),
            "CAMERA": 3,
            "HCAMID": 2,
            "QUALITY": 0,
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CDELT1": 0.5,
            "CDELT2": 0.5,
            "CRPIX1": 2.5,
            "CRPIX2": 2.5,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CROTA2": 0.0,
            "DSUN_OBS": 1.496e11,
            "RSUN_REF": 6.957e8,
            "RSUN_OBS": 959.2,
            "CRLN_OBS": 215.0,
            "CRLT_OBS": -7.0,
            "OBS_VR": 100.0,
            "OBS_VW": -20.0,
            "OBS_VN": 5.0,
            "TELESCOP": "SDO/HMI",
            "INSTRUME": "HMI_SIDE1",
        }
    )


class _DownloadResult:
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __getitem__(self, key: str):
        if key != "download":
            raise KeyError(key)
        return self.paths


class _ExportRequest:
    def __init__(
        self,
        times: tuple[datetime, ...],
        *,
        segment_keys: tuple[str, ...] = SEGMENT_KEYS,
        mutate_header=None,
        unreadable_segment: str | None = None,
        image_shape: tuple[int, int] = (4, 4),
    ):
        self.times = times
        self.segment_keys = segment_keys
        self.mutate_header = mutate_header
        self.unreadable_segment = unreadable_segment
        self.image_shape = image_shape
        self.waited = False
        self.fname_from_rec = None

    def wait(self):
        self.waited = True

    def download(self, directory: str, *, fname_from_rec: bool):
        assert self.waited
        self.fname_from_rec = fname_from_rec
        root = Path(directory)
        paths = []
        for acquisition_time in self.times:
            for segment_index, segment in enumerate(self.segment_keys):
                path = root / (
                    f"hmi.S_720s.{_filename_time(acquisition_time)}.3.{segment}.fits"
                )
                if segment == self.unreadable_segment:
                    path.touch()
                else:
                    header = _header(acquisition_time)
                    if self.mutate_header is not None:
                        self.mutate_header(acquisition_time, segment, header)
                    data = np.full(self.image_shape, segment_index, dtype=np.int16)
                    fits.PrimaryHDU(data=data, header=header).writeto(
                        path, checksum=True
                    )
                paths.append(path)
        return _DownloadResult(paths)


class _Client:
    def __init__(self, request: _ExportRequest):
        self.request = request
        self.calls: list[tuple[str, str]] = []

    def export(self, query: str, *, protocol: str):
        self.calls.append((query, protocol))
        return self.request


@pytest.fixture(autouse=True)
def _tiny_ccd(monkeypatch):
    monkeypatch.setattr(download_module, "HMI_CCD_SIZE", 4)


def test_query_is_fixed_to_camera_three_complete_native_cadence_s720s():
    query = build_hmi_stokes_query(
        start="2024-03-23T22:12:00",
        end="2024-03-24T02:12:00",
    )

    segments = ",".join(SEGMENT_KEYS)
    assert query == (
        f"hmi.S_720s[2024.03.23_22:12:00_TAI/14400s@720s][3]{{{segments}}}"
    )
    assert "S_90s" not in query


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2024-03-24T00:00:00", "2024-03-23T23:48:00", "later"),
        ("2024-03-23T00:00:00", "2024-03-23T00:10:00", "slot grid"),
        ("2024-03-23T00:01:00", "2024-03-23T00:13:00", "slot grid"),
        (
            "2024-03-23T00:00:00+00:00",
            "2024-03-23T00:12:00+00:00",
            "timezone",
        ),
    ],
)
def test_query_rejects_ambiguous_or_non_native_intervals(start, end, message):
    with pytest.raises(ValueError, match=message):
        build_hmi_stokes_query(start=start, end=end)


def test_download_validates_real_fits_and_atomically_publishes_sequence(tmp_path):
    first = datetime(2024, 3, 23, 22, 12)
    second = first + timedelta(seconds=720)
    request = _ExportRequest((first, second))
    client = _Client(request)
    output = tmp_path / "hmi"

    paths = download_hmi_stokes(
        output_directory=output,
        email="scientist@example.org",
        start=first,
        end=second + timedelta(seconds=720),
        client=client,
    )

    assert client.calls == [
        (
            build_hmi_stokes_query(start=first, end=second + timedelta(seconds=720)),
            "fits",
        )
    ]
    assert request.fname_from_rec is True
    assert len(paths) == 2 * len(SEGMENT_KEYS)
    assert all(path.is_file() and path.parent == output.resolve() for path in paths)
    assert all(fits.getdata(path).shape == (4, 4) for path in paths)


def test_incomplete_segment_set_is_rejected_without_publishing(tmp_path):
    start = datetime(2024, 3, 23, 22, 12)
    request = _ExportRequest((start,), segment_keys=SEGMENT_KEYS[:-1])

    with pytest.raises(RuntimeError, match="complete I0 through V5"):
        download_hmi_stokes(
            output_directory=tmp_path / "hmi",
            email="scientist@example.org",
            start=start,
            end=start + timedelta(seconds=720),
            client=_Client(request),
        )

    assert not (tmp_path / "hmi").exists()


def test_missing_native_slot_is_rejected_without_publishing(tmp_path):
    start = datetime(2024, 3, 23, 22, 12)
    request = _ExportRequest((start,))

    with pytest.raises(RuntimeError, match="expected exactly 2"):
        download_hmi_stokes(
            output_directory=tmp_path / "hmi",
            email="scientist@example.org",
            start=start,
            end=start + timedelta(seconds=1440),
            client=_Client(request),
        )

    assert not (tmp_path / "hmi").exists()


def test_wrong_native_slot_is_rejected_without_publishing(tmp_path):
    start = datetime(2024, 3, 23, 22, 12)
    wrong = start + timedelta(seconds=1440)
    request = _ExportRequest((start, wrong))

    with pytest.raises(RuntimeError, match="exact requested native-cadence grid"):
        download_hmi_stokes(
            output_directory=tmp_path / "hmi",
            email="scientist@example.org",
            start=start,
            end=start + timedelta(seconds=1440),
            client=_Client(request),
        )

    assert not (tmp_path / "hmi").exists()


def test_unreadable_or_wrong_size_fits_is_rejected(tmp_path):
    start = datetime(2024, 3, 23, 22, 12)
    unreadable = _ExportRequest((start,), unreadable_segment="Q2")
    with pytest.raises(RuntimeError, match="Invalid full-disk HMI FITS segment"):
        download_hmi_stokes(
            output_directory=tmp_path / "unreadable",
            email="scientist@example.org",
            start=start,
            end=start + timedelta(seconds=720),
            client=_Client(unreadable),
        )

    wrong_size = _ExportRequest((start,), image_shape=(3, 4))
    with pytest.raises(RuntimeError, match="image shape"):
        download_hmi_stokes(
            output_directory=tmp_path / "wrong-size",
            email="scientist@example.org",
            start=start,
            end=start + timedelta(seconds=720),
            client=_Client(wrong_size),
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda _, segment, header: (
                header.__setitem__("CAMERA", 2) if segment == "U3" else None
            ),
            "CAMERA=3",
        ),
        (
            lambda value, segment, header: (
                header.__setitem__("T_REC", _tai_key(value + timedelta(seconds=720)))
                if segment == "V1"
                else None
            ),
            "filename and T_REC disagree",
        ),
        (
            lambda _, segment, header: (
                header.__setitem__("CRPIX1", 3.5) if segment == "Q4" else None
            ),
            "not aligned with I0",
        ),
    ],
)
def test_header_identity_and_segment_alignment_are_strict(tmp_path, mutator, message):
    start = datetime(2024, 3, 23, 22, 12)
    request = _ExportRequest((start,), mutate_header=mutator)

    with pytest.raises(RuntimeError, match=message):
        download_hmi_stokes(
            output_directory=tmp_path / "hmi",
            email="scientist@example.org",
            start=start,
            end=start + timedelta(seconds=720),
            client=_Client(request),
        )

    assert not (tmp_path / "hmi").exists()


def test_existing_output_directory_is_refused_before_network_access(tmp_path):
    output = tmp_path / "hmi"
    output.mkdir()
    stale = output / "stale.fits"
    stale.write_bytes(b"must remain untouched")
    start = datetime(2024, 3, 23, 22, 12)
    client = _Client(_ExportRequest((start,)))

    with pytest.raises(FileExistsError, match="immutable sequence"):
        download_hmi_stokes(
            output_directory=output,
            email="scientist@example.org",
            start=start,
            end=start + timedelta(seconds=720),
            client=client,
        )

    assert client.calls == []
    assert stale.read_bytes() == b"must remain untouched"


@pytest.mark.parametrize("email", ["", "name", " name@example.org", "a@localhost"])
def test_download_requires_explicit_valid_email(tmp_path, email):
    start = datetime(2024, 3, 23, 22, 12)
    with pytest.raises(ValueError, match="email"):
        download_hmi_stokes(
            output_directory=tmp_path / "hmi",
            email=email,
            start=start,
            end=start + timedelta(seconds=720),
            client=_Client(_ExportRequest((start,))),
        )
