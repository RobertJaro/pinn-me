"""Generic packaged preprocessing workflows and their CLI entry points."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from astropy.io import fits
import numpy as np


from prom3theus.cli.main import main


def _write_hmi_names(directory: Path, timestamp: str, *, complete: bool = True, camera=3):
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for component in "IQUV":
        for index in range(6):
            if not complete and component == "V" and index == 5:
                continue
            name = f"hmi.S_720s.{timestamp}_TAI.{camera}.{component}{index}.fits"
            fits.PrimaryHDU(data=np.zeros((2, 2)), header=fits.Header({"QUALITY": 0})).writeto(directory / name)
            names.append(name)
    return names


def test_hmi_preprocessor_scans_every_complete_time_slot(tmp_path):
    from prom3theus.preprocess import hmi as module

    first = _write_hmi_names(tmp_path, "20240323_221200")
    second = _write_hmi_names(tmp_path, "20240323_222400")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    groups = module.scan_hmi_time_series(tmp_path)

    assert [name for name, _ in groups] == [
        "hmi.S_720s.20240323_221200_TAI.3",
        "hmi.S_720s.20240323_222400_TAI.3",
    ]
    assert [path.name for _, paths in groups for path in paths] == first + second


@pytest.mark.parametrize("quality", [1, -2147483648, "bad", 0.5])
def test_hmi_scanner_skips_entire_bad_quality_acquisition(tmp_path, caplog, quality):
    from prom3theus.preprocess.hmi import scan_hmi_time_series

    good = _write_hmi_names(tmp_path, "20240323_221200")
    bad = _write_hmi_names(tmp_path, "20240323_222400")
    # Check even the last segment, not only the reference I0 header.
    with fits.open(tmp_path / bad[-1], mode="update") as hdus:
        if quality is None:
            del hdus[0].header["QUALITY"]
        else:
            hdus[0].header["QUALITY"] = quality
    groups = scan_hmi_time_series(tmp_path)
    assert [path.name for _, paths in groups for path in paths] == good
    assert "Skipping HMI acquisition" in caplog.text
    assert "QUALITY" in caplog.text
    assert len(list(tmp_path.glob("*.fits"))) == 48


def test_hmi_scanner_accepts_missing_quality(tmp_path):
    from prom3theus.preprocess.hmi import scan_hmi_time_series

    names = _write_hmi_names(tmp_path, "20110214_200000", camera=1)
    for name in names:
        with fits.open(tmp_path / name, mode="update") as hdus:
            del hdus[0].header["QUALITY"]
    groups = scan_hmi_time_series(tmp_path)
    assert len(groups) == 1
    assert len(groups[0][1]) == 24


def test_hmi_scanner_fails_clearly_if_all_quality_checks_fail(tmp_path):
    from prom3theus.preprocess.hmi import scan_hmi_time_series

    names = _write_hmi_names(tmp_path, "20240323_221200")
    fits.setval(tmp_path / names[0], "QUALITY", value=1)
    with pytest.raises(ValueError, match="No complete supported QUALITY=0"):
        scan_hmi_time_series(tmp_path)


def test_hmi_preprocessor_rejects_an_incomplete_time_slot(tmp_path):
    from prom3theus.preprocess import hmi as module

    _write_hmi_names(tmp_path, "20240323_221200")
    _write_hmi_names(tmp_path, "20240323_222400", complete=False)

    with pytest.raises(ValueError, match="missing=.*V5"):
        module.scan_hmi_time_series(tmp_path)


def test_hmi_preprocessor_passes_only_selected_files_to_instrument_apis(
    tmp_path,
    monkeypatch,
):
    from prom3theus.preprocess import hmi as module

    base = tmp_path / "data"
    raw = base / "raw_hmi"
    _write_hmi_names(raw, "20240323_221200")
    _write_hmi_names(raw, "20240323_222400")
    calls = []

    monkeypatch.setattr(
        module,
        "prepare_hmi_subframes",
        lambda **options: calls.append(("subframes", options)),
    )

    result = module.preprocess_hmi_time_series(
        input_directory=base / "raw_hmi",
        output_directory=base / "cutouts",
        longitude_deg=215,
        latitude_deg=-12,
        width_pixels=256,
        height_pixels=128,
    )

    assert result["acquisition_count"] == 2
    assert result["segment_count"] == 48
    assert calls == [
        (
            "subframes",
            {
                "inputs": [path for _, paths in module.scan_hmi_time_series(raw) for path in paths],
                "output_directory": base / "cutouts",
                "longitude_deg": 215.0,
                "latitude_deg": -12.0,
                "width_pixels": 256,
                "height_pixels": 128,
            },
        ),
    ]


@pytest.mark.parametrize("camera", [1, 2, 3])
def test_hmi_scanner_accepts_all_cameras(tmp_path, camera):
    from prom3theus.preprocess.hmi import scan_hmi_time_series

    names = _write_hmi_names(tmp_path, "20110214_000000", camera=camera)
    groups = scan_hmi_time_series(tmp_path)
    assert [p.name for _, paths in groups for p in paths] == names


def test_hmi_scanner_reports_empty_directory(tmp_path):
    from prom3theus.preprocess.hmi import scan_hmi_time_series

    with pytest.raises(FileNotFoundError, match="No HMI Stokes files"):
        scan_hmi_time_series(tmp_path)


def test_aia_preprocessor_keeps_only_the_required_footprint_explicit(
    tmp_path,
    monkeypatch,
):
    from prom3theus.preprocess import aia as module

    base = tmp_path / "data"
    acquisition = {
        "record_count": 3, "content_sha256": "a" * 64,
        "targets": [{"records": [
            {"fits_observation_time_utc": time}
            for time in ("2024-03-23T20:59:59Z", "2024-03-23T21:00:01Z", "2024-03-23T21:00:03Z")
        ]}],
    }
    monkeypatch.setattr(module, "_scan_aia_files", lambda directory: acquisition)
    calls = []
    calibration_calls = []

    monkeypatch.setattr(
        module,
        "download_aia_preprocessing_calibration",
        lambda **options: calibration_calls.append(options),
    )
    monkeypatch.setattr(
        module,
        "prepare_aiapy_aia_observation_store",
        lambda *args, **options: calls.append((args, options)),
    )

    result = module.preprocess_aia(
        input_directory=base / "source_images",
        output_directory=base / "image_store",
        calibration_directory=base / "tables",
        longitude_deg=215,
        latitude_deg=-12,
        width_pixels=256,
        height_pixels=128,
    )

    assert result["record_count"] == 3
    assert calibration_calls == [{
        "output_directory": base / "tables",
        "start_utc": "2024-03-23T18:00:00+00:00",
        "end_utc": "2024-03-24T00:00:00+00:00",
    }]
    assert len(calls) == 1
    args, options = calls[0]
    assert args == (base / "source_images", base / "image_store")
    assert options["correction_table_path"] == (base / "tables" / "aia_correction.ecsv")
    assert options["pointing_table_path"] == (base / "tables" / "aia_pointing.ecsv")
    assert options["footprint"].metadata() == {
        "kind": "centered_nominal_hmi_pixel_rectangle",
        "longitude_deg": 215.0,
        "latitude_deg": -12.0,
        "width_pixels": 256,
        "height_pixels": 128,
        "reference_pixel_scale_arcsec": 0.5,
        "grid_policy": "native_aiapy_registered_submap_no_reprojection",
    }
    assert "uncertainty_model" not in options


@pytest.mark.parametrize(
    "instrument, size_args, expected",
    [
        (
            "hmi",
            ["--width-pixels", "96", "--height-pixels", "48"],
            {"width_pixels": 96, "height_pixels": 48},
        ),
        (
            "aia",
            ["--width-pixels", "96", "--height-pixels", "48"],
            {"width_pixels": 96, "height_pixels": 48},
        ),
    ],
)
def test_preprocessor_cli_passes_explicit_subframe_configuration(
    monkeypatch, tmp_path, instrument, size_args, expected
):
    module = importlib.import_module(f"prom3theus.preprocess.{instrument}")
    calls = []
    function_name = (
        "preprocess_hmi_time_series" if instrument == "hmi" else "preprocess_aia"
    )
    monkeypatch.setattr(module, function_name, lambda *a, **kw: calls.append(kw) or {})
    arguments = [
        "prepare",
        instrument,
        str(tmp_path / "raw"),
        "--output",
        str(tmp_path / "processed"),
        "--longitude-deg",
        "125",
        "--latitude-deg",
        "15",
        *size_args,
    ]
    expected.update(
        input_directory=tmp_path / "raw", output_directory=tmp_path / "processed"
    )
    if instrument == "aia":
        arguments += ["--calibration-directory", str(tmp_path / "tables")]
        expected.update(calibration_directory=tmp_path / "tables")
    main(arguments)
    assert calls == [{"longitude_deg": 125.0, "latitude_deg": 15.0, **expected}]


@pytest.mark.parametrize(
    "longitude, width, valid", [(215, 256, True), (216, 256, False), (215, 128, False)]
)
def test_hmi_reuse_checks_crop_position_and_size(
    monkeypatch, tmp_path, longitude, width, valid
):
    from prom3theus.preprocess import hmi as module

    raw = [("slot", [tmp_path / "raw" / "I0.fits"])]
    prepared = [("slot", [tmp_path / "prepared" / "I0.fits"])]
    monkeypatch.setattr(module, "scan_hmi_time_series", lambda path: prepared)
    monkeypatch.setattr(module, "read_hmi_image_header", lambda path: {})
    monkeypatch.setattr(
        module.fits,
        "getheader",
        lambda *a: {"NAXIS1": 256, "NAXIS2": 128, "CCD_X0": 215, "CCD_Y0": 1},
    )
    monkeypatch.setattr(
        module,
        "hmi_subframe_bounds",
        lambda header, **kw: (kw["longitude_deg"], 500, 1, 129),
    )
    options = dict(
        longitude_deg=longitude, latitude_deg=-12, width_pixels=width, height_pixels=128
    )
    if valid:
        assert (
            module._validate_subframes(raw, tmp_path / "prepared", **options)
            == prepared
        )
    else:
        with pytest.raises(ValueError, match="requested crop"):
            module._validate_subframes(raw, tmp_path / "prepared", **options)


@pytest.mark.parametrize("longitude, valid", [(215, True), (216, False)])
def test_aia_reuse_checks_crop_configuration(monkeypatch, tmp_path, longitude, valid):
    from prom3theus.preprocess import aia as module

    original = module.CarringtonCutout(
        longitude_deg=215, latitude_deg=-12, width_deg=8, height_deg=4
    )
    requested = module.CarringtonCutout(
        longitude_deg=longitude, latitude_deg=-12, width_deg=8, height_deg=4
    )
    metadata = {
        "source_files_sha256": "abc",
        "preparation": {"preparation_dependencies": {"cutout": original.metadata()}},
    }
    monkeypatch.setattr(
        module.ImageObservationStore,
        "manifest",
        lambda *a, **kw: {"rasters": [None], "metadata": metadata},
    )
    acquisition = {"record_count": 1, "content_sha256": "abc"}
    if valid:
        module._validate_prepared_store(tmp_path, acquisition, requested)
    else:
        with pytest.raises(ValueError, match="requested crop"):
            module._validate_prepared_store(tmp_path, acquisition, requested)
