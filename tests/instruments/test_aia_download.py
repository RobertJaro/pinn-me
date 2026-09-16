"""Network-free record selection and acquisition tests for AIA Level 1."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy as np
import pytest

from prom3theus.instruments.aia_euv import download as download_module
from prom3theus.instruments.aia_euv.acquisition import (
    AIA_EUV_CHANNELS_ANGSTROM,
    AIA_LEVEL1_SERIES,
    AIA_METADATA_KEYS,
    build_aia_candidate_query,
    select_aia_level1_records,
)
from prom3theus.instruments.aia_euv.download import (
    AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS,
    AIA_SMALL_SAMPLE_TARGET_UTC,
    download_aia_euv,
    download_aia_euv_small_sample,
    load_aia_acquisition_manifest,
)


_CHANNEL_PATTERN = re.compile(r"\[(\d+)\]$")


def _jsoc_tai(value: Time) -> str:
    text = value.tai.copy(format="isot")
    text.precision = 3
    date, clock = text.value.split("T")
    return f"{date.replace('-', '.')}_{clock}_TAI"


def _metadata(
    target: Time,
    channel: int,
    offset_seconds: float,
    *,
    quality: int = 0,
) -> dict:
    observation = target + TimeDelta(offset_seconds, format="sec")
    t_rec = _jsoc_tai(observation)
    rsun_ref = 6.957e8
    dsun_obs = 1.496e11
    rsun_obs = math.degrees(math.asin(rsun_ref / dsun_obs)) * 3600.0
    metadata = {
        "record_id": f"{AIA_LEVEL1_SERIES}[{t_rec}][{channel}]",
        "T_REC": t_rec,
        "T_OBS": t_rec,
        "WAVELNTH": channel,
        "EXPTIME": 1.9,
        "QUALITY": quality,
        "LVL_NUM": 1.0,
        "TELESCOP": "SDO/AIA",
        "INSTRUME": "AIA_3",
        "CTYPE1": "HPLN-TAN",
        "CTYPE2": "HPLT-TAN",
        "CUNIT1": "arcsec",
        "CUNIT2": "arcsec",
        "CDELT1": 0.6,
        "CDELT2": 0.6,
        "CRPIX1": 2.5,
        "CRPIX2": 2.5,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CROTA2": 0.1,
        "DSUN_OBS": dsun_obs,
        "RSUN_REF": rsun_ref,
        "RSUN_OBS": rsun_obs,
        "CRLN_OBS": 215.0,
        "CRLT_OBS": -7.0,
    }
    assert set(AIA_METADATA_KEYS) <= set(metadata)
    return metadata


class _DownloadResult:
    def __init__(self, path: Path, record: str):
        self.path = path
        self.record = record

    def __getitem__(self, key: str):
        if key == "download":
            return [self.path]
        if key == "record":
            return [self.record]
        raise KeyError(key)


class _ExportRequest:
    def __init__(
        self,
        metadata: dict,
        *,
        mutate_header=None,
        returned_record: str | None = None,
    ):
        self.metadata = metadata
        self.mutate_header = mutate_header
        self.returned_record = returned_record
        self.waited = False
        self.fname_from_rec = None

    def wait(self):
        self.waited = True

    def download(self, directory: str, *, fname_from_rec: bool):
        self.wait()
        self.fname_from_rec = fname_from_rec
        header = fits.Header(
            {key: value for key, value in self.metadata.items() if key != "record_id"}
        )
        if self.mutate_header is not None:
            self.mutate_header(header)
        path = Path(directory) / (
            f"aia.lev1_euv_12s.{self.metadata['WAVELNTH']}.image.fits"
        )
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.CompImageHDU(
                    data=np.arange(16, dtype=np.int16).reshape(4, 4),
                    header=header,
                ),
            ]
        ).writeto(path)
        record = self.returned_record or f"{self.metadata['record_id']}{{image}}"
        return _DownloadResult(path, record)


class _Client:
    def __init__(
        self,
        rows_by_channel,
        *,
        mutate_header=None,
        returned_record: str | None = None,
    ):
        self.rows_by_channel = rows_by_channel
        self.mutate_header = mutate_header
        self.returned_record = returned_record
        self.query_calls = []
        self.export_calls = []
        self.requests = []
        self.by_record = {
            row["record_id"]: row
            for rows in rows_by_channel.values()
            for row in rows
        }

    def query(self, query, *, key, rec_index):
        self.query_calls.append((query, key, rec_index))
        match = _CHANNEL_PATTERN.search(query)
        assert match is not None
        return [dict(row) for row in self.rows_by_channel[int(match.group(1))]]

    def export(self, query, *, method, protocol):
        assert method == "url"
        self.export_calls.append((query, protocol))
        assert query.endswith("{image}")
        record_id = query.removesuffix("{image}")
        request = _ExportRequest(
            self.by_record[record_id],
            mutate_header=self.mutate_header,
            returned_record=self.returned_record,
        )
        self.requests.append(request)
        request.urls = {
            "record": [query],
            "filename": [f"aia.lev1_euv_12s.{self.by_record[record_id]['WAVELNTH']}.image.fits"],
        }
        return request


@pytest.fixture(autouse=True)
def _tiny_aia_ccd(monkeypatch):
    monkeypatch.setattr(download_module, "AIA_CCD_SIZE", 4)


@pytest.fixture
def target() -> Time:
    return Time("2024-03-23T22:11:21.388", format="isot", scale="utc")


def _client(target: Time, *, good_offset: float = 2.0) -> _Client:
    return _Client(
        {
            channel: [
                _metadata(target, channel, -3.0),
                _metadata(target, channel, 1.0, quality=1),
                _metadata(target, channel, good_offset),
            ]
            for channel in AIA_EUV_CHANNELS_ANGSTROM
        }
    )


def _utc_metadata(target: Time, channel: int, offset_seconds: float) -> dict:
    metadata = _metadata(target, channel, offset_seconds)
    observation = target + TimeDelta(offset_seconds, format="sec")
    utc = observation.utc.copy(format="isot")
    utc.precision = 3
    metadata["T_REC"] = f"{utc.value}Z"
    metadata["T_OBS"] = f"{utc.value}Z"
    date, clock = utc.value.split("T")
    metadata["record_id"] = (
        f"{AIA_LEVEL1_SERIES}[{date.replace('-', '.')}_{clock}_TAI][{channel}]"
    )
    return metadata


def test_download_constructs_drms_client_directly(tmp_path, target, monkeypatch):
    client = _client(target)
    emails = []

    def make_client(*, email):
        emails.append(email)
        return client

    monkeypatch.setitem(sys.modules, "drms", SimpleNamespace(Client=make_client))
    manifest_path = download_aia_euv(
        output_directory=tmp_path / "aia", email="test@example.org",
        target_times=[target], max_offset_seconds=6,
    )
    assert emails == ["test@example.org"]
    assert len(client.export_calls) == 3
    assert manifest_path.is_file()


def test_candidate_query_has_native_guard_band_and_explicit_utc(target):
    query = build_aia_candidate_query(
        target_time=target,
        channel_angstrom=171,
        max_offset_seconds=6,
    )
    assert query == (
        "aia.lev1_euv_12s[2024-03-23T22:11:03.388Z/36s][171]"
    )


def test_small_sample_route_has_fixed_target_channels_and_tolerance(tmp_path):
    target = Time(AIA_SMALL_SAMPLE_TARGET_UTC, scale="utc")
    client = _client(target)

    manifest_path = download_aia_euv_small_sample(
        output_directory=tmp_path / "sample",
        email="scientist@example.org",
        client=client,
    )

    manifest = load_aia_acquisition_manifest(manifest_path.parent)
    assert AIA_SMALL_SAMPLE_MAX_OFFSET_SECONDS == 6.0
    assert manifest["max_offset_seconds"] == 6.0
    assert manifest["targets"][0]["target_time_utc"].startswith(
        "2024-03-23T22:11:21.388"
    )
    assert [
        record["channel_angstrom"]
        for record in manifest["targets"][0]["records"]
    ] == [171, 193, 211]

@pytest.mark.parametrize("channels", [(193,), (211, 171), (335, 94, 131, 304)])
def test_download_preserves_requested_channel_selection(tmp_path, target, channels):
    client = _Client({channel: [_metadata(target, channel, 2.0)] for channel in channels})
    output = tmp_path / "aia"
    download_aia_euv(
        target_times=[target], channels=channels, max_offset_seconds=6,
        output_directory=output, email="scientist@example.org", client=client,
    )
    manifest = load_aia_acquisition_manifest(output)
    assert manifest["channels_angstrom"] == list(channels)
    assert [record["channel_angstrom"] for record in manifest["targets"][0]["records"]] == list(channels)


def test_download_selects_nearest_good_records_and_publishes_manifest(
    tmp_path, target
):
    client = _client(target)
    output = tmp_path / "aia"

    manifest_path = download_aia_euv(
        target_times=[target],
        channels=(171, 193, 211),
        max_offset_seconds=6.0,
        output_directory=output,
        email="scientist@example.org",
        client=client,
    )

    assert manifest_path == (output / "manifest.json").resolve()
    manifest = load_aia_acquisition_manifest(output)
    json.dumps(manifest, allow_nan=False)
    assert manifest["series"] == "aia.lev1_euv_12s"
    assert manifest["segment"] == "image"
    assert manifest["record_count"] == 3
    assert len(client.query_calls) == 3
    assert all(call[1] == list(AIA_METADATA_KEYS) for call in client.query_calls)
    assert all(call[2] is True for call in client.query_calls)
    records = manifest["targets"][0]["records"]
    assert [record["channel_angstrom"] for record in records] == [171, 193, 211]
    assert all(
        record["signed_offset_seconds"] == pytest.approx(2.0)
        for record in records
    )
    assert client.export_calls == [
        (f"{record['record_id']}{{image}}", "fits") for record in records
    ]
    assert all(request.fname_from_rec is False for request in client.requests)
    for record in records:
        path = output / record["file"]
        assert path.is_file()
        assert record["sha256"] == download_module._sha256_file(path)
        assert record["image_hdu_index"] == 1
        assert record["image_hdu_name"] == "COMPRESSED_IMAGE"
        assert record["fits_checksum"] is None
        assert record["fits_datasum"] is None
        assert record["export_record_id"] == f"{record['record_id']}{{image}}"


def test_real_export_shape_uses_extension_one_and_linked_fits_record_time(
    tmp_path, target
):
    rows = {
        channel: [_utc_metadata(target, channel, 2.0)]
        for channel in AIA_EUV_CHANNELS_ANGSTROM
    }

    def mutate_header(header):
        header["T_OBS"] = header["T_OBS"].removesuffix("Z")
        linked_time = Time(header["T_REC"].removesuffix("Z"), scale="utc")
        linked_time += TimeDelta(header["WAVELNTH"] / 100.0, format="sec")
        linked_time.format = "isot"
        linked_time.precision = 3
        header["T_REC"] = linked_time.value

    output = tmp_path / "extension-one"
    download_aia_euv(
        target_times=[target],
        max_offset_seconds=6,
        output_directory=output,
        email="scientist@example.org",
        client=_Client(rows, mutate_header=mutate_header),
    )

    records = load_aia_acquisition_manifest(output)["targets"][0]["records"]
    assert all(record["image_hdu_index"] == 1 for record in records)
    assert all(
        record["fits_record_time_utc"] != record["record_time_utc"]
        for record in records
    )
    assert all(
        record["fits_observation_time_utc"] == record["observation_time_utc"]
        for record in records
    )


def test_selection_rejects_ties_duplicates_and_time_tolerance(target):
    tied = _Client(
        {
            channel: [
                _metadata(target, channel, -2.0),
                _metadata(target, channel, 2.0),
            ]
            for channel in AIA_EUV_CHANNELS_ANGSTROM
        }
    )
    with pytest.raises(ValueError, match="tied"):
        select_aia_level1_records(
            tied,
            target_times=[target],
            max_offset_seconds=6,
        )

    duplicate = _metadata(target, 171, 1.0)
    rows = {171: [duplicate, dict(duplicate)], 193: [], 211: []}
    with pytest.raises(ValueError, match="duplicate"):
        select_aia_level1_records(
            _Client(rows),
            target_times=[target],
            max_offset_seconds=6,
        )

    with pytest.raises(RuntimeError, match="exceeding"):
        select_aia_level1_records(
            _client(target, good_offset=7.0),
            target_times=[target],
            max_offset_seconds=2.5,
        )


def test_selection_rejects_wrong_channel_and_malformed_good_wcs(target):
    wrong = _metadata(target, 193, 1.0)
    with pytest.raises(ValueError, match="query for 171"):
        select_aia_level1_records(
            _Client({171: [wrong], 193: [], 211: []}),
            target_times=[target],
            max_offset_seconds=6,
        )

    malformed = _metadata(target, 171, 1.0)
    malformed["CDELT1"] = float("nan")
    with pytest.raises(ValueError, match="CDELT1 must be finite"):
        select_aia_level1_records(
            _Client({171: [malformed], 193: [], 211: []}),
            target_times=[target],
            max_offset_seconds=6,
        )


def test_download_rejects_malformed_inputs_and_existing_output_before_query(
    tmp_path, target
):
    client = _client(target)
    with pytest.raises(ValueError, match="unique"):
        download_aia_euv(
            target_times=[target],
            channels=(193, 171, 193),
            output_directory=tmp_path / "wrong-order",
            email="scientist@example.org",
            client=client,
        )
    with pytest.raises(ValueError, match="valid date/time"):
        download_aia_euv(
            target_times=["not a date"],
            output_directory=tmp_path / "invalid-time",
            email="scientist@example.org",
            client=client,
        )
    output = tmp_path / "exists"
    output.mkdir()
    stale = output / "stale.fits"
    stale.write_bytes(b"preserve")
    with pytest.raises(FileNotFoundError):
        download_aia_euv(
            target_times=[target],
            output_directory=output,
            email="scientist@example.org",
            client=client,
        )
    assert client.query_calls == []
    assert stale.read_bytes() == b"preserve"


def test_aia_skips_matching_download_and_overwrites_only_when_requested(tmp_path, target):
    client = _client(target)
    output = tmp_path / "aia"
    options = dict(target_times=[target], max_offset_seconds=6,
                   output_directory=output, email="test@example.org", client=client)
    manifest = download_aia_euv(**options)
    before = manifest.read_bytes()
    calls = len(client.export_calls)
    assert download_aia_euv(**options) == manifest
    assert len(client.export_calls) == calls
    download_aia_euv(**options, overwrite=True)
    assert len(client.export_calls) == 2 * calls
    assert next(tmp_path.glob(".aia.previous-*/aia/manifest.json")).read_bytes() == before
    before = manifest.read_bytes()
    client.mutate_header = lambda header: header.__setitem__("WAVELNTH", 193)
    with pytest.raises(RuntimeError, match="Level-1 identity"):
        download_aia_euv(**options, overwrite=True)
    assert manifest.read_bytes() == before
    load_aia_acquisition_manifest(output)


def test_fits_identity_failure_is_atomic(tmp_path, target):
    client = _client(target)
    client.mutate_header = lambda header: header.__setitem__("WAVELNTH", 193)
    output = tmp_path / "aia"

    with pytest.raises(RuntimeError, match="Level-1 identity"):
        download_aia_euv(
            target_times=[target],
            max_offset_seconds=6,
            output_directory=output,
            email="scientist@example.org",
            client=client,
        )

    assert not output.exists()


def test_export_record_identity_failure_is_atomic(tmp_path, target):
    output = tmp_path / "aia"
    client = _client(target)
    client.returned_record = (
        "aia.lev1_euv_12s[2024.03.23_00:00:00_TAI][171]{image}"
    )

    with pytest.raises(RuntimeError, match="other than the exact selected record"):
        download_aia_euv(
            target_times=[target],
            max_offset_seconds=6,
            output_directory=output,
            email="scientist@example.org",
            client=client,
        )

    assert not output.exists()


def test_image_hdu_fits_checksum_is_verified(tmp_path, target):
    client = _client(target)
    selected = select_aia_level1_records(
        client,
        target_times=[target],
        max_offset_seconds=6,
    )[0]
    header = fits.Header(
        {
            key: value
            for key, value in client.by_record[selected.record_id].items()
            if key != "record_id"
        }
    )
    path = tmp_path / "invalid-checksum.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.arange(16, dtype=np.int16).reshape(4, 4),
                header=header,
            ),
        ]
    ).writeto(path, checksum=True)
    with fits.open(path, mode="update", checksum=False) as hdul:
        hdul[1].header["CHECKSUM"] = "0000000000000000"

    with pytest.raises(RuntimeError, match="checksum is invalid"):
        download_module._verify_selected_fits(path, selected)


def test_manifest_hash_validation_detects_post_publish_changes(tmp_path, target):
    output = tmp_path / "aia"
    download_aia_euv(
        target_times=[target],
        max_offset_seconds=6,
        output_directory=output,
        email="scientist@example.org",
        client=_client(target),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    path = output / manifest["targets"][0]["records"][0]["file"]
    with path.open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_aia_acquisition_manifest(output)
