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


def test_prepare_hmi_subframes_preserves_pixels_wcs_and_detector_origin(
    tmp_path, tiny_ccd
):
    inputs = tmp_path / "full_disk"
    output = tmp_path / "subframes"
    inputs.mkdir()
    header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    arrays = _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
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


def test_carrington_center_is_recomputed_for_each_acquisition(
    tmp_path, tiny_ccd, monkeypatch
):
    inputs = tmp_path / "full_disk"
    inputs.mkdir()
    first_header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    second_header = _header(
        date_obs="2024-03-24T01:11:20.000",
        t_rec="2024.03.24_01:12:00_TAI",
        size=tiny_ccd,
    )
    first = _write_acquisition(inputs, "20240324_010000", first_header, size=tiny_ccd)
    second = _write_acquisition(inputs, "20240324_011200", second_header, size=tiny_ccd)
    calls = []

    def project(header, longitude_deg, latitude_deg):
        calls.append((header["T_REC"], longitude_deg, latitude_deg))
        if header["T_REC"] == first_header["T_REC"]:
            return 3.5, 3.5
        return 4.5, 2.5

    monkeypatch.setattr(subframe, "_carrington_center_pixel", project)
    outputs = subframe.prepare_hmi_subframes(
        inputs=inputs,
        output_directory=tmp_path / "subframes",
        longitude_deg=215.0,
        latitude_deg=-12.0,
        width_pixels=2,
        height_pixels=2,
    )

    assert calls == [
        (first_header["T_REC"], 215.0, -12.0),
        (second_header["T_REC"], 215.0, -12.0),
    ]
    np.testing.assert_array_equal(fits.getdata(outputs[0]), first["I0"][3:5, 3:5])
    np.testing.assert_array_equal(
        fits.getdata(outputs[len(SEGMENT_KEYS)]), second["I0"][2:4, 4:6]
    )


def test_carrington_projection_uses_t_obs_and_rejects_far_side(tiny_ccd):
    header = _header(
        date_obs="2020-03-24T00:59:20.000",
        t_obs="2024.03.24_00:59:59_TAI",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    longitude, latitude = _disk_center_carrington(header)

    x_pixel, y_pixel = subframe._carrington_center_pixel(
        header,
        longitude,
        latitude,
    )

    assert x_pixel == pytest.approx((tiny_ccd - 1) / 2)
    assert y_pixel == pytest.approx((tiny_ccd - 1) / 2)
    with pytest.raises(ValueError, match="far side"):
        subframe._carrington_center_pixel(
            header,
            (longitude + 180.0) % 360.0,
            latitude,
        )


def test_rejects_non_full_disk_or_misaligned_acquisition_before_writing(
    tmp_path, tiny_ccd, monkeypatch
):
    inputs = tmp_path / "full_disk"
    output = tmp_path / "subframes"
    inputs.mkdir()
    header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    monkeypatch.setattr(subframe, "_carrington_center_pixel", lambda *args: (3.5, 3.5))

    malformed = inputs / "hmi.S_720s.20240324_010000_TAI.3.V5.fits"
    fits.writeto(
        malformed,
        np.zeros((tiny_ccd - 1, tiny_ccd), dtype=np.float32),
        header,
        overwrite=True,
    )
    with pytest.raises(ValueError, match="exact full-disk"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=output,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=2,
            height_pixels=2,
        )
    assert not list(output.glob("*.fits"))

    fits.writeto(
        malformed,
        np.zeros((tiny_ccd, tiny_ccd), dtype=np.float32),
        header,
        overwrite=True,
    )
    fits.setval(malformed, "CRPIX1", value=header["CRPIX1"])
    fits.setval(malformed, "T_OBS", value="2024.03.24_01:00:01_TAI")
    with pytest.raises(ValueError, match="V5.*not aligned"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=output,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=2,
            height_pixels=2,
        )
    assert not list(output.glob("*.fits"))

    fits.writeto(
        malformed,
        np.zeros((tiny_ccd, tiny_ccd), dtype=np.float32),
        header,
        overwrite=True,
    )
    fits.setval(malformed, "CRPIX1", value=header["CRPIX1"] + 1)
    with pytest.raises(ValueError, match="V5.*not aligned"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=output,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=2,
            height_pixels=2,
        )
    assert not list(output.glob("*.fits"))

    fits.writeto(
        malformed,
        np.zeros((tiny_ccd, tiny_ccd), dtype=np.float32),
        header,
        overwrite=True,
    )
    fits.setval(malformed, "CAMERA", value=2)
    with pytest.raises(ValueError, match="CAMERA=3"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=output,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=2,
            height_pixels=2,
        )
    assert not list(output.glob("*.fits"))


def test_rejects_out_of_detector_output_input_and_partial_collision(
    tmp_path, tiny_ccd, monkeypatch
):
    inputs = tmp_path / "full_disk"
    output = tmp_path / "subframes"
    inputs.mkdir()
    header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)

    monkeypatch.setattr(subframe, "_carrington_center_pixel", lambda *args: (0.0, 0.0))
    with pytest.raises(ValueError, match="outside the physical detector"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=output,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=4,
            height_pixels=4,
        )
    assert not list(output.glob("*.fits"))

    with pytest.raises(ValueError, match="must not overwrite an input"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=inputs,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=2,
            height_pixels=2,
            overwrite=True,
        )

    output.mkdir(exist_ok=True)
    colliding = output / "hmi.S_720s.20240324_010000_TAI.3.I0.fits"
    colliding.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="partial/colliding"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=output,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=2,
            height_pixels=2,
        )
    assert colliding.read_bytes() == b"existing"
    assert len(list(output.iterdir())) == 1


def test_explicit_overwrite_replaces_a_complete_staged_set(
    tmp_path, tiny_ccd, monkeypatch
):
    inputs = tmp_path / "full_disk"
    output = tmp_path / "subframes"
    inputs.mkdir()
    output.mkdir()
    header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    arrays = _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    existing = output / "hmi.S_720s.20240324_010000_TAI.3.I0.fits"
    existing.write_bytes(b"old")
    monkeypatch.setattr(subframe, "_carrington_center_pixel", lambda *args: (3.5, 3.5))

    outputs = subframe.prepare_hmi_subframes(
        inputs=inputs,
        output_directory=output,
        longitude_deg=0.0,
        latitude_deg=0.0,
        width_pixels=2,
        height_pixels=2,
        overwrite=True,
    )

    assert len(outputs) == len(SEGMENT_KEYS)
    np.testing.assert_array_equal(fits.getdata(existing), arrays["I0"][3:5, 3:5])
    assert not any(path.name.startswith(".hmi-subframe-") for path in output.iterdir())


def test_scaled_fits_crop_preserves_storage_scaling_and_physical_values(
    tmp_path, tiny_ccd, monkeypatch
):
    inputs = tmp_path / "full_disk"
    inputs.mkdir()
    header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    source = inputs / "hmi.S_720s.20240324_010000_TAI.3.I0.fits"
    physical = (
        100.0
        + np.arange(tiny_ccd * tiny_ccd, dtype=np.float32).reshape(tiny_ccd, tiny_ccd)
        * 0.25
    )
    scaled = fits.PrimaryHDU(data=physical.copy(), header=header)
    scaled.scale("int16", bscale=0.25, bzero=100.0)
    scaled.writeto(source, overwrite=True)
    with fits.open(source, do_not_scale_image_data=True) as hdul:
        source_storage = np.array(hdul[0].data, copy=True)

    monkeypatch.setattr(subframe, "_carrington_center_pixel", lambda *args: (3.5, 3.5))
    outputs = subframe.prepare_hmi_subframes(
        inputs=inputs,
        output_directory=tmp_path / "subframes",
        longitude_deg=0.0,
        latitude_deg=0.0,
        width_pixels=4,
        height_pixels=2,
    )

    result = fits.getdata(outputs[0])
    result_header = fits.getheader(outputs[0])
    np.testing.assert_array_equal(result, physical[3:5, 2:6])
    assert result_header["BSCALE"] == pytest.approx(0.25)
    assert result_header["BZERO"] == pytest.approx(100.0)
    with fits.open(outputs[0], do_not_scale_image_data=True) as hdul:
        np.testing.assert_array_equal(hdul[0].data, source_storage[3:5, 2:6])


def test_overwrite_rejects_unrelated_stale_fits(tmp_path, tiny_ccd, monkeypatch):
    inputs = tmp_path / "full_disk"
    output = tmp_path / "subframes"
    inputs.mkdir()
    output.mkdir()
    header = _header(
        date_obs="2024-03-24T00:59:20.000",
        t_rec="2024.03.24_01:00:00_TAI",
        size=tiny_ccd,
    )
    _write_acquisition(inputs, "20240324_010000", header, size=tiny_ccd)
    stale = output / "hmi.S_720s.stale.3.I0.fits"
    stale.write_bytes(b"unrelated")
    monkeypatch.setattr(subframe, "_carrington_center_pixel", lambda *args: (3.5, 3.5))

    with pytest.raises(FileExistsError, match="unrelated.*overwrite=True"):
        subframe.prepare_hmi_subframes(
            inputs=inputs,
            output_directory=output,
            longitude_deg=0.0,
            latitude_deg=0.0,
            width_pixels=2,
            height_pixels=2,
            overwrite=True,
        )

    assert stale.read_bytes() == b"unrelated"
    assert list(output.iterdir()) == [stale]
