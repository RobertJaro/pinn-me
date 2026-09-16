"""Coordinate conversion for the interactive AIA bounds selector."""

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pytest
from sunpy.map import Map, make_fitswcs_header

from prom3theus.preprocess.select_subframe import selection_parameters, shell_settings


@pytest.fixture
def image_map():
    data = np.zeros((256, 256))
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame="helioprojective",
                      observer="earth", obstime="2024-03-23")
    return Map(data, make_fitswcs_header(data, center, scale=[10, 10] * u.arcsec / u.pix))


def test_center_and_pixel_dimensions(image_map):
    result = selection_parameters(image_map, (111.5, 143.5, 119.5, 135.5))
    assert result["width_pixels"] == 640
    assert result["height_pixels"] == 320
    assert result["longitude_deg"] == pytest.approx(image_map.carrington_longitude.deg)
    assert result["latitude_deg"] == pytest.approx(image_map.heliographic_latitude.deg)
    assert "width_pixels=640" in shell_settings(result)


def test_off_disk_selection_rejected(image_map):
    with pytest.raises(ValueError, match="solar disk"):
        selection_parameters(image_map, (0, 20, 0, 20))


def test_outside_image_selection_rejected(image_map):
    with pytest.raises(ValueError, match="inside the image"):
        selection_parameters(image_map, (-5, 10, 10, 20))


def test_interactive_accept_prints_shell_settings(tmp_path, image_map, monkeypatch, capsys):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    import matplotlib.widgets as widgets
    from astropy.io import fits
    from prom3theus.preprocess.select_subframe import main

    header = image_map.fits_header
    header["INSTRUME"] = "AIA"
    header["TELESCOP"] = "SDO/AIA"
    header["WAVELNTH"] = 193
    header["WAVEUNIT"] = "angstrom"
    path = tmp_path / "aia.fits"
    fits.writeto(path, image_map.data, header)
    selectors = []
    original = widgets.RectangleSelector

    def rectangle(*args, **kwargs):
        selector = original(*args, **kwargs)
        selectors.append(selector)
        return selector

    def show():
        selector = selectors[0]
        canvas = selector.ax.figure.canvas
        canvas.draw()
        for name, x, y in [("button_press_event", 111.6, 119.6),
                           ("motion_notify_event", 143.4, 135.4),
                           ("button_release_event", 143.4, 135.4)]:
            px, py = selector.ax.transData.transform((x, y))
            event = MouseEvent(name, canvas, px, py, button=1)
            event.xdata, event.ydata = x, y
            canvas.callbacks.process(name, event)
            if name == "motion_notify_event":
                assert selector.get_visible()
                np.testing.assert_allclose(selector.extents, [111.6, 143.4, 119.6, 135.4])
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="enter"))

    monkeypatch.setattr(widgets, "RectangleSelector", rectangle)
    monkeypatch.setattr(plt, "show", show)
    main([str(path)])
    output = capsys.readouterr().out
    assert "longitude_deg=" in output
    assert "width_pixels=640" in output
    assert "height_pixels=320" in output


def test_full_disk_preview_is_small_and_keeps_detector_coordinates(monkeypatch):
    import matplotlib.pyplot as plt
    import sunpy.map
    from prom3theus.preprocess.select_subframe import main

    data = np.broadcast_to(1., (4096, 4096))
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame="helioprojective",
                      observer="earth", obstime="2024-03-23")
    header = make_fitswcs_header(data, center, scale=[0.6, 0.6] * u.arcsec / u.pix,
                                instrument="AIA", telescope="SDO/AIA",
                                wavelength=193 * u.angstrom)
    source = Map(data, header)
    monkeypatch.setattr(sunpy.map, "Map", lambda path: source)

    def show():
        figure = plt.gcf()
        axes = figure.axes[0]
        artist = axes.images[0]
        assert artist.get_array().shape == (1024, 1024, 4)
        assert artist.get_array().dtype == np.uint8
        assert axes.get_xlim() == (-0.5, 4095.5)
        assert axes.get_ylim() == (-0.5, 4095.5)
        left, right, bottom, top = artist.get_extent()
        assert left + (right - left) / 2048 == 0  # First sample center is detector pixel 0.
        assert bottom + (top - bottom) / 2048 == 0
        assert right - (right - left) / 2048 == 4092  # Last sampled detector pixel.
        plt.close(figure)

    monkeypatch.setattr(plt, "show", show)
    main(["aia.fits"])
