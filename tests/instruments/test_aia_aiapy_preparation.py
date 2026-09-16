"""Network-free tests for the optional pinned aiapy preparation boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import astropy.units as u
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time

from prom3theus.instruments.aia_euv import aiapy_preparation as module
from prom3theus.instruments.aia_euv.acquisition import AIA_METADATA_KEYS
from prom3theus.instruments.aia_euv.acquisition import validate_aia_level1_metadata
from prom3theus.instruments.aia_euv.aiapy_preparation import (
    AIAPY_PIPELINE_ORDER,
    AIACarringtonGeometry,
    AiapyPreparationBackend,
    CarringtonCutout,
    prepare_aiapy_aia_observation_store,
)
from prom3theus.instruments.aia_euv.download import _content_digest
from prom3theus.observations import ImageObservationStore


SOLAR_RADIUS_M = 6.957e8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    root: Path,
    channel: int,
    *,
    target_index: int = 0,
) -> dict:
    relative = Path("records") / f"target_{target_index:04d}_aia_{channel}.fits"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"synthetic AIA {target_index} {channel}".encode())
    record_time = "2024-03-23T22:12:00.000000000 TAI"
    return {
        "target_index": target_index,
        "target_time_utc": "2024-03-23T22:11:23.000000000Z",
        "channel_angstrom": channel,
        "record_id": (
            "aia.lev1_euv_12s[2024.03.23_22:12:00_TAI]"
            f"[{channel}]"
        ),
        "record_time_tai": record_time,
        "record_time_utc": "2024-03-23T22:11:23.000000000Z",
        "observation_time_utc": "2024-03-23T22:11:23.000000000Z",
        "observation_time_tai": record_time,
        "signed_offset_seconds": 0.0,
        "quality": 0,
        "exposure_seconds": 2.0,
        "level": 1.0,
        "telescope": "SDO/AIA",
        "instrument": "AIA_3",
        "wcs": {},
        "discovery_query": "synthetic",
        "file": relative.as_posix(),
        "sha256": _sha256(path),
        "export_query": "synthetic{image}",
        "export_record_id": (
            "aia.lev1_euv_12s[2024.03.23_22:12:00_TAI]"
            f"[{channel}]{{image}}"
        ),
        "export_filename": path.name,
        "image_hdu_index": 1,
        "image_hdu_name": "COMPRESSED_IMAGE",
        "fits_record_time_utc": "2024-03-23T22:11:23.000000000Z",
        "fits_record_time_tai": record_time,
        "fits_observation_time_utc": "2024-03-23T22:11:23.000000000Z",
        "fits_observation_time_tai": record_time,
        "fits_checksum": None,
        "fits_datasum": None,
    }


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "aia-acquisition"
    records = [_record(root, channel) for channel in (171, 193, 211)]
    manifest = {
        "format": "prom3theus.aia_euv.level1_acquisition",
        "version": 1,
        "series": "aia.lev1_euv_12s",
        "segment": "image",
        "channels_angstrom": [171, 193, 211],
        "max_offset_seconds": 12.0,
        "selection": {
            "time_keyword": "T_OBS",
            "algorithm": "unique nearest QUALITY=0 record",
            "signed_offset_convention": "selected_T_OBS_minus_target",
            "metadata_keys": list(AIA_METADATA_KEYS),
        },
        "target_count": 1,
        "record_count": 3,
        "content_sha256": _content_digest(records),
        "targets": [
            {
                "target_index": 0,
                "target_time_utc": "2024-03-23T22:11:23.000000000Z",
                "records": records,
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _tables(tmp_path: Path) -> tuple[Path, Path]:
    correction = tmp_path / "aia-correction.ecsv"
    pointing = tmp_path / "aia-pointing.ecsv"
    correction.write_text("# synthetic correction ECSV\n", encoding="utf-8")
    pointing.write_text("# synthetic pointing ECSV\n", encoding="utf-8")
    return correction, pointing


class _Map:
    def __init__(self, data, channel, label):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.channel = channel
        self.label = label


class _InjectedBackend:
    """Small in-memory backend proving orchestration does not import aiapy."""

    def __init__(self):
        self.calls = []

    def dependency_metadata(self):
        self.calls.append(("dependency_metadata",))
        return {"backend": "synthetic", "network_access": False}

    def read_pointing_table(self, path):
        self.calls.append(("read_pointing_table", path.name))
        return {"kind": "pointing"}

    def read_correction_table(self, path):
        self.calls.append(("read_correction_table", path.name))
        return {"kind": "correction"}

    def load_level1_map(self, path, record):
        self.calls.append(("load_level1_map", record["channel_angstrom"]))
        return _Map(
            [[-4.0, 8.0], [np.nan, 16.0]],
            record["channel_angstrom"],
            "level1",
        )

    def data(self, image_map):
        return image_map.data

    def with_data(self, image_map, data):
        return _Map(data, image_map.channel, "auxiliary")

    def update_pointing(self, image_map, pointing_table):
        assert pointing_table == {"kind": "pointing"}
        self.calls.append(("update_pointing", image_map.channel))
        return _Map(image_map.data, image_map.channel, "pointed")

    def register(self, image_map, *, missing, order):
        self.calls.append(("register", image_map.channel, order))
        data = image_map.data.copy()
        if order == 0:
            data[~np.isfinite(data)] = missing
        return _Map(data, image_map.channel, f"registered-{order}")

    def crop(self, image_map, footprint):
        assert footprint.longitude_deg == 210.0
        self.calls.append(("crop", image_map.channel, image_map.label))
        return _Map(image_map.data, image_map.channel, image_map.label)

    def degradation_factor(
        self, image_map, *, channel_angstrom, correction_table
    ):
        assert correction_table == {"kind": "correction"}
        assert channel_angstrom == image_map.channel
        self.calls.append(("degradation_factor", channel_angstrom))
        return 2.0

    def carrington_geometry(self, image_map, footprint):
        self.calls.append(("carrington_geometry", image_map.channel))
        ray = np.zeros((2, 2, 3), dtype=np.float64)
        ray[..., 2] = -1.0
        surface = np.zeros_like(ray)
        surface[..., 2] = SOLAR_RADIUS_M
        return AIACarringtonGeometry(
            surface_position_m=surface,
            ray_direction=ray,
            on_disk_mask=np.ones((2, 2), dtype=bool),
            footprint_mask=np.array([[True, True], [True, False]]),
            solar_radius_m=SOLAR_RADIUS_M,
            absolute_tai_seconds=1_700_000_000.125 + image_map.channel,
            provenance={
                "native_registered_grid_retained": True,
                "hmi_reprojection_performed": False,
            },
        )


def test_injected_backend_prepares_store_in_documented_order_without_aiapy(
    tmp_path,
):
    bundle = _bundle(tmp_path)
    correction, pointing = _tables(tmp_path)
    backend = _InjectedBackend()

    output = prepare_aiapy_aia_observation_store(
        bundle,
        tmp_path / "aia.image",
        correction_table_path=correction,
        pointing_table_path=pointing,
        footprint=CarringtonCutout(210.0, -5.0, 12.0, 8.0),
        backend=backend,
        observation_id="aia-synthetic",
    )

    rasters, names, metadata = ImageObservationStore.load_sequence(
        output, mmap=False
    )
    assert names == [
        "target-0000-aia-171",
        "target-0000-aia-193",
        "target-0000-aia-211",
    ]
    assert metadata["preparation"]["preparation_dependencies"][
        "pipeline_order"
    ] == list(AIAPY_PIPELINE_ORDER)
    assert metadata["preparation"]["preparation_dependencies"][
        "degradation_application_owner"
    ] == "prepare_aia_observation_store"
    raster = rasters[0]
    assert raster.valid_mask.tolist() == [[True, True], [False, False]]
    assert raster.intensity[0, 0].item() == pytest.approx(-1.0)
    assert raster.intensity[0, 0] < 0
    assert torch.isfinite(raster.intensity).all()
    assert raster.uncertainty is None
    assert raster.metadata["calibration"]["degradation_factor"] == 2.0
    assert raster.metadata["calibration"][
        "input_correction_applications"
    ] == 0
    assert raster.metadata["calibration"][
        "correction_performed_by_preparation"
    ] is True
    assert raster.metadata["calibration"]["total_correction_applications"] == 1
    provenance = raster.metadata["provenance"]
    assert provenance["degradation"]["applied_by_backend"] is False
    assert provenance["degradation"]["application_owner"] == (
        "prepare_aia_observation_store"
    )
    assert provenance["geometry"]["hmi_reprojection_performed"] is False

    first_channel_calls = [
        call[0]
        for call in backend.calls
        if len(call) == 1 or (len(call) > 1 and call[1] == 171)
    ]
    assert first_channel_calls.index("update_pointing") < first_channel_calls.index(
        "register"
    )
    assert [
        call[2]
        for call in backend.calls
        if call[:2] == ("register", 171)
    ] == [3, 0]


def test_aia_threads_share_tables_loaded_once(tmp_path, monkeypatch):
    from threading import Barrier, get_ident

    monkeypatch.setenv("PROM3THEUS_PREP_WORKERS", "3")
    barrier = Barrier(3, timeout=10)
    threads, pointing_ids, correction_ids = set(), set(), set()

    class SharedBackend(_InjectedBackend):
        def load_level1_map(self, path, record):
            threads.add(get_ident())
            barrier.wait()
            return super().load_level1_map(path, record)

        def update_pointing(self, image_map, pointing_table):
            pointing_ids.add(id(pointing_table))
            return super().update_pointing(image_map, pointing_table)

        def degradation_factor(self, image_map, *, channel_angstrom, correction_table):
            correction_ids.add(id(correction_table))
            return super().degradation_factor(
                image_map, channel_angstrom=channel_angstrom, correction_table=correction_table,
            )

    backend = SharedBackend()
    correction, pointing = _tables(tmp_path)
    prepare_aiapy_aia_observation_store(
        _bundle(tmp_path), tmp_path / "parallel.image",
        correction_table_path=correction, pointing_table_path=pointing,
        footprint=CarringtonCutout(210, -5, 12, 8), backend=backend,
    )
    assert len(threads) == 3
    assert len(pointing_ids) == len(correction_ids) == 1
    assert sum(call[0] == "read_pointing_table" for call in backend.calls) == 1
    assert sum(call[0] == "read_correction_table" for call in backend.calls) == 1


def test_each_worker_writes_and_releases_image_before_next_read(tmp_path, monkeypatch):
    import weakref
    from threading import get_ident

    monkeypatch.setenv("PROM3THEUS_PREP_WORKERS", "1")
    references, reads, writes = [], [], []
    original_prepare = module._prepare_record
    original_write = ImageObservationStore.write_raster

    def prepare_record(**kwargs):
        raster = original_prepare(**kwargs)
        references.append(weakref.ref(raster.intensity))
        return raster

    def write_raster(directory, index, name, raster):
        assert directory == tmp_path / "streamed.image"
        entry = original_write(directory, index, name, raster)
        assert all((directory / info["file"]).is_file() for info in entry["arrays"].values())
        writes.append((raster.channel_angstrom, get_ident()))
        return entry

    class Backend(_InjectedBackend):
        def load_level1_map(self, path, record):
            assert len(writes) == len(reads)
            assert all(reference() is None for reference in references)
            reads.append((record["channel_angstrom"], get_ident()))
            return super().load_level1_map(path, record)

    monkeypatch.setattr(module, "_prepare_record", prepare_record)
    monkeypatch.setattr(ImageObservationStore, "write_raster", write_raster)
    correction, pointing = _tables(tmp_path)
    prepare_aiapy_aia_observation_store(
        _bundle(tmp_path), tmp_path / "streamed.image",
        correction_table_path=correction, pointing_table_path=pointing,
        footprint=CarringtonCutout(210, -5, 12, 8), backend=Backend(),
    )
    assert reads == writes
    assert len(reads) == 3
    assert all(reference() is None for reference in references)


def test_scan_reads_only_headers_not_arrays_or_checksums(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from prom3theus.preprocess import aia

    for channel in (171, 193, 211):
        (tmp_path / f"{channel}.fits").touch()

    def open_headers(path):
        header = _level1_header(int(path.stem))
        header["NAXIS"] = 2
        return nullcontext([SimpleNamespace(header=header, name="IMAGE")])

    def no_checksum(path):
        raise AssertionError("Header scan must not read file contents for hashing.")

    monkeypatch.setattr(aia.fits, "open", open_headers)
    monkeypatch.setattr(aia, "_sha256_file", no_checksum)
    manifest = aia._scan_aia_files(tmp_path)
    assert manifest["record_count"] == 3
    assert manifest["content_sha256"] is None
    assert all(record["sha256"] is None for record in manifest["targets"][0]["records"])


def test_manifest_checksum_is_validated_before_backend_or_optional_imports(tmp_path):
    bundle = _bundle(tmp_path)
    correction, pointing = _tables(tmp_path)
    (bundle / "records" / "target_0000_aia_171.fits").write_bytes(b"tampered")
    backend = _InjectedBackend()

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_aiapy_aia_observation_store(
            bundle,
            tmp_path / "never-created.image",
            correction_table_path=correction,
            pointing_table_path=pointing,
            footprint=CarringtonCutout(210, 0, 10, 10),
            backend=backend,
        )
    assert backend.calls == []


def test_only_explicit_local_ecsv_tables_are_accepted(tmp_path):
    bundle = _bundle(tmp_path)
    _, pointing = _tables(tmp_path)
    correction = tmp_path / "implicit.txt"
    correction.write_text("not ECSV", encoding="utf-8")
    backend = _InjectedBackend()

    with pytest.raises(ValueError, match=r"explicit \.ecsv"):
        prepare_aiapy_aia_observation_store(
            bundle,
            tmp_path / "never-created.image",
            correction_table_path=correction,
            pointing_table_path=pointing,
            footprint=CarringtonCutout(210, 0, 10, 10),
            backend=backend,
        )
    assert backend.calls == []


def test_real_backend_loads_optional_stack_once_on_first_use(monkeypatch):
    calls = []
    runtime = SimpleNamespace()
    monkeypatch.setattr(module, "_load_aiapy_runtime", lambda: calls.append("import") or runtime)
    backend = AiapyPreparationBackend()
    assert backend._runtime is None
    assert backend.runtime is runtime
    assert backend.runtime is runtime
    assert calls == ["import"]


def _level1_header(channel: int = 171) -> fits.Header:
    rsun_ref = 6.957e8
    dsun_obs = 1.496e11
    return fits.Header(
        {
            "T_REC": "2024-03-23T22:11:23.000",
            "T_OBS": "2024-03-23T22:11:23.000",
            "WAVELNTH": channel,
            "EXPTIME": 2.0,
            "QUALITY": 0,
            "LVL_NUM": 1.0,
            "TELESCOP": "SDO/AIA",
            "INSTRUME": "AIA_3",
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CDELT1": 0.6,
            "CDELT2": 0.6,
            "CRPIX1": 1.5,
            "CRPIX2": 1.5,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CROTA2": 0.0,
            "DSUN_OBS": dsun_obs,
            "RSUN_REF": rsun_ref,
            "RSUN_OBS": math.degrees(math.asin(rsun_ref / dsun_obs)) * 3600,
            "CRLN_OBS": 210.0,
            "CRLT_OBS": -5.0,
        }
    )


def test_plain_drms_fits_are_scanned_for_preparation(tmp_path):
    from prom3theus.preprocess.aia import _scan_aia_files

    for channel in (171, 193, 211):
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2)), header=_level1_header(channel)),
        ]).writeto(tmp_path / f"aia.{channel}.fits")
    manifest = _scan_aia_files(tmp_path)
    assert manifest["record_count"] == 3
    records = module._validated_records(tmp_path, manifest)
    assert [record[2]["channel_angstrom"] for record in records] == [171, 193, 211]
    assert all(path.parent == tmp_path for _, path, _ in records)
    assert {group for group, _, _ in records} == {"target-0000"}
    assert not (tmp_path / "manifest.json").exists()
    correction, pointing = _tables(tmp_path)
    output = prepare_aiapy_aia_observation_store(
        tmp_path, tmp_path / "prepared", acquisition_manifest=manifest,
        correction_table_path=correction, pointing_table_path=pointing,
        footprint=CarringtonCutout(210.0, -5.0, 12.0, 8.0), backend=_InjectedBackend(),
    )
    rasters, _, _ = ImageObservationStore.load_sequence(output, mmap=False)
    assert len(rasters) == 3


@pytest.mark.parametrize("invalid", ["exposure", "quality", "missing_quality", "no_image", "broken"])
def test_plain_fits_scanner_skips_invalid_files_and_incomplete_groups(tmp_path, caplog, invalid):
    from prom3theus.preprocess.aia import _scan_aia_files

    for channel in (171, 193, 211):
        fits.writeto(tmp_path / f"good.{channel}.fits", np.ones((2, 2)), _level1_header(channel))
        header = _level1_header(channel)
        header["T_REC"] = "2024-03-23T22:11:35.000"
        header["T_OBS"] = "2024-03-23T22:11:35.000"
        path = tmp_path / f"bad_group.{channel}.fits"
        data = np.ones((2, 2))
        if channel == 193:
            if invalid == "exposure":
                header["EXPTIME"] = 0
            elif invalid == "quality":
                header["QUALITY"] = "0x00000001"
            elif invalid == "missing_quality":
                del header["QUALITY"]
            elif invalid == "no_image":
                fits.PrimaryHDU(header=header).writeto(path)
                continue
            elif invalid == "broken":
                path.write_bytes(b"not a FITS file")
                continue
        fits.writeto(path, data, header)

    manifest = _scan_aia_files(tmp_path)
    assert manifest["record_count"] == 3
    assert len(manifest["targets"]) == 1
    records = manifest["targets"][0]["records"]
    assert all(record["file"].startswith("good.") for record in records)
    assert all(record["target_index"] == 0 for record in records)
    assert "Skipping AIA file" in caplog.text
    assert "Skipping AIA exposure group" in caplog.text
    assert len(list(tmp_path.glob("*.fits"))) == 6  # No files removed.


def test_plain_fits_scanner_rejects_no_complete_valid_groups(tmp_path):
    from prom3theus.preprocess.aia import _scan_aia_files

    fits.writeto(tmp_path / "aia.171.fits", np.ones((2, 2)), _level1_header())
    with pytest.raises(ValueError, match="No complete valid"):
        _scan_aia_files(tmp_path)


def test_real_hdu_loader_reads_only_the_header_selected_image_extension(
    tmp_path, monkeypatch
):
    header = _level1_header()
    path = tmp_path / "level1.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.array([[-3.0, 2.0], [4.0, 8.0]], dtype=np.float32),
                header=header,
                name="SCI",
            ),
        ]
    ).writeto(path)
    identity = validate_aia_level1_metadata(header, expected_channel=171)
    record = _record(tmp_path / "unused", 171)
    record.update(
        {
            "image_hdu_index": 1,
            "image_hdu_name": "SCI",
            "observation_time_tai": "2024-03-23T22:12:00.000000000 TAI",
            "exposure_seconds": 2.0,
            "wcs": dict(identity.wcs),
        }
    )

    class Map:
        def __init__(self, data, meta):
            self.data = np.asarray(data)
            self.meta = meta

    backend = AiapyPreparationBackend()
    backend._runtime = SimpleNamespace(fits=fits, Map=Map, Time=Time)
    loaded = backend.load_level1_map(path, record)
    assert loaded.data.tolist() == [[-3.0, 2.0], [4.0, 8.0]]

    ambiguous = tmp_path / "ambiguous.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(data=np.ones((2, 2))),
            fits.ImageHDU(data=np.ones((2, 2)), header=header, name="SCI"),
        ]
    ).writeto(ambiguous)
    loaded = backend.load_level1_map(ambiguous, record)
    np.testing.assert_array_equal(loaded.data, np.ones((2, 2)))
    fits.setval(path, "BUNIT", value="DN/s", ext=1)
    with pytest.raises(ValueError, match="already exposure normalized"):
        backend.load_level1_map(path, record)


def test_correction_ecsv_retains_units_and_requires_v10(
    tmp_path, monkeypatch
):
    table = QTable()
    table["DATE"] = ["2020-01-01T00:00:00"]
    table["VER_NUM"] = [10]
    table["WAVE_STR"] = ["171_THIN"]
    table["WAVELNTH"] = [171.0] * u.angstrom
    table["T_START"] = Time(["2010-01-01T00:00:00"], scale="utc")
    table["T_STOP"] = Time(["2020-01-01T00:00:00"], scale="utc")
    table["EFFA_P1"] = [0.0]
    table["EFFA_P2"] = [0.0]
    table["EFFA_P3"] = [0.0]
    table["EFF_AREA"] = [1.0] * u.cm**2
    table["EFF_WVLN"] = [171.0] * u.angstrom
    path = tmp_path / "correction.ecsv"
    table.write(path, format="ascii.ecsv")
    backend = AiapyPreparationBackend()
    backend._runtime = SimpleNamespace(QTable=QTable, Time=Time, u=u)

    loaded = backend.read_correction_table(path)
    assert len(loaded) == 1

    loaded["EFF_AREA"][0] = 2.0 * u.cm**2
    changed = tmp_path / "changed.ecsv"
    loaded.write(changed, format="ascii.ecsv")
    reread = backend.read_correction_table(changed)
    assert reread["EFF_AREA"][0] == 2.0 * u.cm**2
    assert isinstance(reread["T_START"], Time)
    reread["VER_NUM"][:] = 9
    reread.write(changed, format="ascii.ecsv", overwrite=True)
    with pytest.raises(ValueError, match="V10"):
        backend.read_correction_table(changed)


@pytest.mark.parametrize("supports_version_keyword", [False, True])
def test_degradation_uses_shared_v10_table_with_both_aiapy_signatures(
    tmp_path, supports_version_keyword,
):
    table = QTable({"VER_NUM": [9, 10, 10, 11]})
    table["T_START"] = Time(["2010-01-01"] * 4)
    table["T_STOP"] = Time(["2020-01-01"] * 4)
    path = tmp_path / "mixed_versions.ecsv"
    table.write(path, format="ascii.ecsv")
    backend = AiapyPreparationBackend()
    backend._runtime = SimpleNamespace(QTable=QTable, Time=Time, u=u)
    shared_table = backend.read_correction_table(path)
    np.testing.assert_array_equal(shared_table["VER_NUM"], [10, 10])
    calls = []

    def degradation(channel, obstime, *, correction_table):
        assert correction_table is shared_table
        calls.append((channel, obstime))
        return np.array([0.75]) * u.one

    def newer_degradation(channel, obstime, *, correction_table, calibration_version=10):
        assert calibration_version == 10
        return degradation(channel, obstime, correction_table=correction_table)

    backend._runtime.degradation = newer_degradation if supports_version_keyword else degradation
    image_map = SimpleNamespace(reference_date=Time("2011-02-14"))
    for channel in (171, 193, 211):
        assert backend.degradation_factor(
            image_map, channel_angstrom=channel, correction_table=shared_table,
        ) == pytest.approx(0.75)
    assert len(calls) == 3



@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"latitude_deg": 91}, "latitude_deg"),
        ({"width_deg": 181}, "width_deg"),
        ({"latitude_deg": 80, "height_deg": 30}, "within the poles"),
        ({"boundary_samples_per_axis": 4}, ">= 5"),
    ],
)
def test_carrington_cutout_rejects_ambiguous_or_nonlocal_rectangles(
    kwargs, message
):
    values = {
        "longitude_deg": 210,
        "latitude_deg": 0,
        "width_deg": 10,
        "height_deg": 10,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        CarringtonCutout(**values)
