"""Independent centered crops using nominal HMI pixel sizes."""

from types import SimpleNamespace

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pytest
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk

from prom3theus.instruments.aia_euv.aiapy_preparation import AiapyPreparationBackend
from prom3theus.observations.pixel_footprint import CenteredPixelCutout
from prom3theus.preprocess.select_subframe import selection_parameters


def solar_map(shape, scale, rotation=0, time="2024-03-23"):
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame="helioprojective",
                      observer="earth", obstime=time)
    data = np.zeros(shape)
    return Map(data, make_fitswcs_header(
        data, center, scale=[scale, scale] * u.arcsec / u.pix,
        rotation_angle=rotation * u.deg,
    ))


@pytest.mark.parametrize("rotation", [0, 23])
def test_aia_centered_native_crop(rotation):
    aia = solar_map((100, 100), 0.6, rotation=rotation)
    footprint = CenteredPixelCutout(
        aia.carrington_longitude.deg, aia.heliographic_latitude.deg, 40, 24,
    )
    backend = AiapyPreparationBackend()
    backend._runtime = SimpleNamespace(
        u=u, SkyCoord=SkyCoord, HeliographicCarrington=frames.HeliographicCarrington,
        all_coordinates_from_map=all_coordinates_from_map,
        coordinate_is_on_solar_disk=coordinate_is_on_solar_disk,
    )
    crop = backend.crop(aia, footprint)
    geometry = backend.carrington_geometry(crop, footprint)
    assert crop.data.shape == (20, 33)
    assert geometry.footprint_mask.all()
    assert crop.scale.axis1.to_value(u.arcsec / u.pix) == pytest.approx(0.6)
    center_x, center_y = crop.world_to_pixel(aia.center)
    assert abs(center_x.value - 16) <= 0.500001
    assert abs(center_y.value - 9.5) <= 0.500001
    with pytest.raises(ValueError, match="outside"):
        backend.crop(aia, CenteredPixelCutout(
            aia.carrington_longitude.deg, aia.heliographic_latitude.deg, 200, 200,
        ))


def test_2011_surface_center_is_not_rejected_by_roundoff():
    # Geometry from the user's 2011-02-14 AIA image; no large image allocation.
    header = {
        "DATE-OBS": "2011-02-14T00:00:07.844", "T_OBS": "2011-02-14T00:00:08.844",
        "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN", "CUNIT1": "arcsec", "CUNIT2": "arcsec",
        "CRPIX1": 2046.20996, "CRPIX2": 2041.08997, "CRVAL1": 0., "CRVAL2": 0.,
        "CDELT1": 0.600758016, "CDELT2": 0.600758016, "CROTA2": 0.05633853,
        "DSUN_OBS": 147696736313.876, "RSUN_REF": 696000000.,
        "HGLN_OBS": 0., "HGLT_OBS": -6.77550459,
        "INSTRUME": "AIA_3", "TELESCOP": "SDO/AIA", "WAVELNTH": 193, "WAVEUNIT": "angstrom",
    }
    aia = Map(np.broadcast_to(0., (4096, 4096)), header)
    backend = AiapyPreparationBackend()
    backend._runtime = SimpleNamespace(u=u, SkyCoord=SkyCoord,
                                       HeliographicCarrington=frames.HeliographicCarrington)
    crop = backend.crop(aia, CenteredPixelCutout(35.3, -20, 900, 800))
    assert crop.data.shape == (666, 749)
    with pytest.raises(ValueError, match="far side.*2011-02-14"):
        backend.crop(aia, CenteredPixelCutout(215.3, -20, 900, 800))


def test_selector_converts_aia_pixels_to_hmi_pixels():
    aia = solar_map((100, 100), 0.6)
    result = selection_parameters(aia, (39.5, 59.5, 44.5, 54.5))
    assert result["width_pixels"] == 24
    assert result["height_pixels"] == 12


@pytest.mark.parametrize("width", [0, -1, 1.5, 4097])
def test_invalid_pixel_sizes(width):
    with pytest.raises(ValueError, match="dimensions"):
        CenteredPixelCutout(10, 20, width, 24)
