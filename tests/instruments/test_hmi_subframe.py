"""Focused contracts for Carrington-centred HMI subframe preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from prom3theus.instruments.hmi import subframe
from prom3theus.instruments.hmi.acquisition import SEGMENT_KEYS


def _header(
    *,
    date_obs: str,
    t_rec: str,
    size: int,
    t_obs: str | None = None,
) -> fits.Header:
    return fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CDELT1": 1.0,
            "CDELT2": 1.0,
            "CRPIX1": (size + 1) / 2,
            "CRPIX2": (size + 1) / 2,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CROTA2": 0.0,
            "DATE-OBS": date_obs,
            "MJD-OBS": float(Time(date_obs, format="isot", scale="utc").mjd),
            "T_OBS": t_rec if t_obs is None else t_obs,
            "T_REC": t_rec,
            "CAMERA": 3,
            "HCAMID": 3,
            "QUALITY": 0,
            "DSUN_OBS": 1.496e11,
            "RSUN_REF": 6.957e8,
            "RSUN_OBS": 959.2,
            "CRLN_OBS": 180.0,
            "CRLT_OBS": -7.0,
            "TELESCOP": "SDO/HMI",
            "INSTRUME": "HMI_SIDE1",
        }
    )


def _write_acquisition(
    directory: Path,
    timestamp: str,
    header: fits.Header,
    *,
    size: int,
) -> dict[str, np.ndarray]:
    arrays = {}
    pixels = np.arange(size * size, dtype=np.float32).reshape(size, size)
    for segment_index, segment in enumerate(SEGMENT_KEYS):
        data = pixels + segment_index * 100
        path = directory / f"hmi.S_720s.{timestamp}_TAI.3.{segment}.fits"
        fits.writeto(path, data, header)
        arrays[segment] = data
    return arrays


def _disk_center_carrington(header: fits.Header) -> tuple[float, float]:
    from sunpy.coordinates import frames
    from sunpy.map import Map
    from prom3theus.instruments.hmi.acquisition import (
        hmi_observation_wcs_header,
    )

    source_map = Map(
        np.zeros((1, 1), dtype=np.uint8), hmi_observation_wcs_header(header)
    )
    observer = source_map.observer_coordinate.transform_to(
        frames.HeliographicCarrington(observer="self")
    )
    return float(observer.lon.deg), float(observer.lat.deg)


@pytest.fixture
def tiny_ccd(monkeypatch):
    monkeypatch.setattr(subframe, "HMI_CCD_SIZE", 8)
    return 8


@pytest.mark.parametrize("camera", [1, 2, 3])
@pytest.mark.parametrize("layout", ["primary", "extension", "compressed"])
def test_prepare_hmi_subframes_preserves_pixels_wcs_and_detector_origin(
    tmp_path, tiny_ccd, camera, layout
):
    inputs = tmp_path / "full_disk"
    output = tmp_path / "subframes"
    inputs.mkdir()
    header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    header["CAMERA"] = camera
    arrays = _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    if layout != "primary":
        for path in inputs.glob("*.fits"):
            data, metadata = fits.getdata(path, header=True)
            cls = fits.ImageHDU if layout == "extension" else fits.CompImageHDU
            fits.HDUList([fits.PrimaryHDU(), cls(data=data, header=metadata)]).writeto(path, overwrite=True)
    longitude, latitude = _disk_center_carrington(header)

    outputs = subframe.prepare_hmi_subframes(
        inputs=inputs,
        output_directory=output,
        longitude_deg=longitude,
        latitude_deg=latitude,
        width_pixels=4,
        height_pixels=2,
    )

    assert len(outputs) == len(SEGMENT_KEYS)
    assert [path.name.rsplit(".", 2)[-2] for path in outputs] == list(SEGMENT_KEYS)
    result_data, result_header = fits.getdata(outputs[0], header=True)
    np.testing.assert_array_equal(result_data, arrays["I0"][3:5, 2:6])
    assert result_data.shape == (2, 4)
    assert result_header["CRPIX1"] == pytest.approx(header["CRPIX1"] - 2)
    assert result_header["CRPIX2"] == pytest.approx(header["CRPIX2"] - 3)
    assert result_header["CCD_X0"] == 2
    assert result_header["CCD_Y0"] == 3
    assert result_header["CCD_NX"] == tiny_ccd
    assert result_header["CCD_NY"] == tiny_ccd
    assert result_header["CRLN_OBS"] == header["CRLN_OBS"]
    assert result_header["CRLT_OBS"] == header["CRLT_OBS"]
    assert "HGLN_OBS" not in result_header
    assert "HGLT_OBS" not in result_header

    source_wcs = WCS(header)
    result_wcs = WCS(result_header)
    source_world = source_wcs.pixel_to_world_values(2.0, 3.0)
    result_world = result_wcs.pixel_to_world_values(0.0, 0.0)
    np.testing.assert_allclose(result_world, source_world, rtol=0.0, atol=1.0e-12)
    assert fits.getdata(inputs / outputs[0].name).shape == (tiny_ccd, tiny_ccd)


def test_workers_use_sunpy_submap_and_save_directly(tmp_path, tiny_ccd, monkeypatch):
    from sunpy.map import GenericMap

    inputs, output = tmp_path / "full_disk", tmp_path / "subframes"
    inputs.mkdir()
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    longitude, latitude = _disk_center_carrington(header)
    calls = []
    original_submap, original_save = GenericMap.submap, GenericMap.save

    def submap_spy(self, *args, **kwargs):
        calls.append("submap")
        return original_submap(self, *args, **kwargs)

    def save_spy(self, path, **kwargs):
        assert path.parent == output
        result = original_save(self, path, **kwargs)
        assert path.is_file()
        calls.append("save")
        return result

    monkeypatch.setattr(GenericMap, "submap", submap_spy)
    monkeypatch.setattr(GenericMap, "save", save_spy)
    paths = subframe.prepare_hmi_subframes(
        inputs=inputs, output_directory=output, longitude_deg=longitude,
        latitude_deg=latitude, width_pixels=4, height_pixels=2,
    )
    assert calls.count("submap") == calls.count("save") == 24
    assert len(paths) == 24
    assert set(tmp_path.iterdir()) == {inputs, output}
    with pytest.raises(FileExistsError, match="partial/colliding"):
        subframe.prepare_hmi_subframes(
            inputs=inputs, output_directory=output, longitude_deg=longitude,
            latitude_deg=latitude, width_pixels=4, height_pixels=2,
        )
    subframe.prepare_hmi_subframes(
        inputs=inputs, output_directory=output, longitude_deg=longitude,
        latitude_deg=latitude, width_pixels=2, height_pixels=2, overwrite=True,
    )
    assert fits.getdata(paths[0]).shape == (2, 2)


def test_scaled_fits_preserves_physical_values_and_blanks(tmp_path, tiny_ccd):
    inputs = tmp_path / "full_disk"
    inputs.mkdir()
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    path = inputs / "hmi.S_720s.20240324_010000_TAI.3.I0.fits"
    raw = np.arange(64, dtype=np.int16).reshape(8, 8)
    raw[3, 3] = -32768
    hdu = fits.PrimaryHDU(raw, header)
    hdu.header.update(BSCALE=0.25, BZERO=100., BLANK=-32768)
    hdu.writeto(path, overwrite=True)
    expected = fits.getdata(path)[3:5, 2:6]
    longitude, latitude = _disk_center_carrington(header)
    paths = subframe.prepare_hmi_subframes(
        inputs=inputs, output_directory=tmp_path / "subframes",
        longitude_deg=longitude, latitude_deg=latitude, width_pixels=4, height_pixels=2,
    )
    np.testing.assert_allclose(fits.getdata(paths[0]), expected, equal_nan=True)
    assert fits.getheader(paths[0])["T_OBS"] == header["T_OBS"]


def test_sunpy_bounds_match_written_detector_offsets(tmp_path, tiny_ccd):
    inputs = tmp_path / "full_disk"
    inputs.mkdir()
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    longitude, latitude = _disk_center_carrington(header)
    options = dict(longitude_deg=longitude, latitude_deg=latitude, width_pixels=4, height_pixels=2)
    reference_header = fits.getheader(inputs / "hmi.S_720s.20240324_010000_TAI.3.I0.fits")
    bounds = subframe.hmi_subframe_bounds(reference_header, **options)
    paths = subframe.prepare_hmi_subframes(
        inputs=inputs, output_directory=tmp_path / "subframes", **options,
    )
    crop_header = fits.getheader(paths[0])
    assert bounds == (crop_header["CCD_X0"], crop_header["CCD_X0"] + 4,
                      crop_header["CCD_Y0"], crop_header["CCD_Y0"] + 2)


@pytest.mark.parametrize("floating_origin", [False, True])
def test_nested_subframes_preserve_full_detector_origin_wcs_and_reuse(
    tmp_path, tiny_ccd, floating_origin,
):
    from prom3theus.preprocess.hmi import preprocess_hmi_time_series

    inputs = tmp_path / "full_disk"
    inputs.mkdir()
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    arrays = _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    longitude, latitude = _disk_center_carrington(header)
    options = dict(longitude_deg=longitude, latitude_deg=latitude)
    first = subframe.prepare_hmi_subframes(
        inputs=inputs, output_directory=tmp_path / "first", width_pixels=6,
        height_pixels=4, **options,
    )
    if floating_origin:
        # Historical prepared cutouts store integer-valued offsets as FITS floats.
        for path in first:
            with fits.open(path, mode="update") as hdul:
                for key in ("CCD_X0", "CCD_Y0"):
                    hdul[0].header[key] = float(hdul[0].header[key])
    nested_options = dict(input_directory=tmp_path / "first",
                          output_directory=tmp_path / "nested",
                          width_pixels=2, height_pixels=2, **options)
    assert preprocess_hmi_time_series(**nested_options)["subframes"]["status"] == "prepared"
    assert preprocess_hmi_time_series(**nested_options)["subframes"]["status"] == "reused"
    data, nested_header = fits.getdata(tmp_path / "nested" / first[0].name, header=True)
    np.testing.assert_array_equal(data, arrays["I0"][3:5, 3:5])
    assert (nested_header["CCD_X0"], nested_header["CCD_Y0"]) == (3, 3)
    assert (nested_header["CCD_NX"], nested_header["CCD_NY"]) == (tiny_ccd, tiny_ccd)
    assert subframe.hmi_subframe_bounds(
        fits.getheader(first[0]), width_pixels=2, height_pixels=2, **options,
    ) == (3, 5, 3, 5)
    np.testing.assert_allclose(
        WCS(nested_header).pixel_to_world_values(0., 0.),
        WCS(header).pixel_to_world_values(3., 3.), rtol=0., atol=1.e-12,
    )


def test_nested_scaled_fits_does_not_apply_storage_scaling_twice(tmp_path, tiny_ccd):
    inputs = tmp_path / "full_disk"
    inputs.mkdir()
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    source = inputs / "hmi.S_720s.20240324_010000_TAI.3.I0.fits"
    raw = np.arange(tiny_ccd ** 2, dtype=np.int16).reshape(tiny_ccd, tiny_ccd)
    raw[3, 3] = -32768
    hdu = fits.PrimaryHDU(raw, header)
    hdu.header.update(BSCALE=0.25, BZERO=100., BLANK=-32768)
    hdu.writeto(source, overwrite=True)
    expected = fits.getdata(source)[3:5, 3:5]
    longitude, latitude = _disk_center_carrington(header)
    options = dict(longitude_deg=longitude, latitude_deg=latitude)
    subframe.prepare_hmi_subframes(
        inputs=inputs, output_directory=tmp_path / "first", width_pixels=6,
        height_pixels=4, **options,
    )
    nested = subframe.prepare_hmi_subframes(
        inputs=tmp_path / "first", output_directory=tmp_path / "nested",
        width_pixels=2, height_pixels=2, **options,
    )
    actual, metadata = fits.getdata(nested[0], header=True)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert not any(key in metadata for key in ("BSCALE", "BZERO", "BLANK"))


@pytest.mark.parametrize("detector", [
    {},
    {"CCD_X0": 2, "CCD_Y0": 3},
    {"CCD_X0": 2, "CCD_Y0": 3, "CCD_NX": 4, "CCD_NY": 2},
    {"CCD_X0": 2.5, "CCD_Y0": 3, "CCD_NX": 8, "CCD_NY": 8},
    {"CCD_X0": -1, "CCD_Y0": 3, "CCD_NX": 8, "CCD_NY": 8},
    {"CCD_X0": 6, "CCD_Y0": 3, "CCD_NX": 8, "CCD_NY": 8},
])
def test_cutouts_require_trustworthy_detector_metadata(tmp_path, tiny_ccd, detector):
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    header.update(detector)
    path = tmp_path / "cutout.fits"
    fits.writeto(path, np.zeros((2, 4), dtype=np.float32), header)
    with pytest.raises(ValueError, match="HMI"):
        subframe._hmi_map(path)
    with pytest.raises(ValueError, match="HMI"):
        subframe.hmi_subframe_bounds(fits.getheader(path), longitude_deg=180,
                                    latitude_deg=-7, width_pixels=2, height_pixels=2)


def test_each_time_slot_uses_its_own_sunpy_wcs(tmp_path, tiny_ccd):
    inputs = tmp_path / "full_disk"
    inputs.mkdir()
    first = _header(date_obs="2024-03-24T00:59:20.000",
                    t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    second = _header(date_obs="2024-03-24T01:11:20.000",
                     t_rec="2024.03.24_01:12:00_TAI", size=tiny_ccd)
    second["CRPIX1"] += 1
    _write_acquisition(inputs, "20240324_010000", first, size=tiny_ccd)
    _write_acquisition(inputs, "20240324_011200", second, size=tiny_ccd)
    longitude, latitude = _disk_center_carrington(first)
    paths = subframe.prepare_hmi_subframes(
        inputs=inputs, output_directory=tmp_path / "subframes",
        longitude_deg=longitude, latitude_deg=latitude, width_pixels=4, height_pixels=2,
    )
    assert len(paths) == 48
    first_crop, second_crop = fits.getheader(paths[0]), fits.getheader(paths[24])
    assert second_crop["CCD_X0"] == first_crop["CCD_X0"] + 1
    assert first_crop["T_OBS"] == first["T_OBS"]
    assert second_crop["T_OBS"] == second["T_OBS"]


def test_input_protection_and_incomplete_sets(tmp_path, tiny_ccd):
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    _write_acquisition(tmp_path, "20240324_010000", header, size=tiny_ccd)
    with pytest.raises(ValueError, match="must not overwrite an input"):
        subframe.prepare_hmi_subframes(
            inputs=tmp_path, output_directory=tmp_path,
            longitude_deg=180, latitude_deg=-7, width_pixels=2, height_pixels=2, overwrite=True,
        )
    (tmp_path / "hmi.S_720s.20240324_010000_TAI.3.V5.fits").unlink()
    with pytest.raises(ValueError, match="missing=.*V5"):
        subframe.prepare_hmi_subframes(
            inputs=tmp_path, output_directory=tmp_path / "crop",
            longitude_deg=180, latitude_deg=-7, width_pixels=2, height_pixels=2,
        )


def test_direct_writes_keep_completed_files_when_worker_fails(tmp_path, tiny_ccd, monkeypatch):
    from sunpy.map import GenericMap

    inputs, output = tmp_path / "full_disk", tmp_path / "subframes"
    inputs.mkdir()
    header = _header(date_obs="2024-03-24T00:59:20.000",
                     t_rec="2024.03.24_01:00:00_TAI", size=tiny_ccd)
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    longitude, latitude = _disk_center_carrington(header)
    monkeypatch.setenv("PROM3THEUS_PREP_WORKERS", "1")
    save = GenericMap.save

    def fail_second(self, path, **kwargs):
        if path.name.endswith(".I1.fits"):
            raise OSError("write failed")
        return save(self, path, **kwargs)

    monkeypatch.setattr(GenericMap, "save", fail_second)
    with pytest.raises(OSError, match="write failed"):
        subframe.prepare_hmi_subframes(
            inputs=inputs, output_directory=output, longitude_deg=longitude,
            latitude_deg=latitude, width_pixels=2, height_pixels=2,
        )
    assert (output / "hmi.S_720s.20240324_010000_TAI.3.I0.fits").is_file()
    assert set(tmp_path.iterdir()) == {inputs, output}
