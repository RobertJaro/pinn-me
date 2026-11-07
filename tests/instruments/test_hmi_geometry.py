"""Geometry regressions for HMI rasters crossing the solar limb."""

import numpy as np
import pytest
from astropy.io import fits
from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

from prom3theus.instruments.hmi.geometry import build_geometry


def _full_disk_map():
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CDELT1": 100.0,
            "CDELT2": 100.0,
            "CRPIX1": 17.0,
            "CRPIX2": 17.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "DATE-OBS": "2024-03-23T22:11:21",
            "DSUN_OBS": 1.496e11,
            "RSUN_REF": 6.957e8,
            "CRLN_OBS": 180.0,
            "CRLT_OBS": -7.0,
            "CROTA2": 17.0,
        }
    )
    return Map(np.ones((33, 33)), header)


def test_build_geometry_masks_off_disk_before_velocity_projection():
    s_map = _full_disk_map()
    on_disk = coordinate_is_on_solar_disk(all_coordinates_from_map(s_map))
    assert on_disk.any() and (~on_disk).any()

    ray, surface, basis, mu, velocity, metadata = build_geometry(
        s_map, on_disk, (1200.0, -300.0, 50.0)
    )

    assert np.isfinite(ray[on_disk]).all()
    assert np.isfinite(surface[on_disk]).all()
    assert np.isfinite(basis[on_disk]).all()
    assert np.isfinite(velocity[on_disk]).all()
    assert np.all(mu[on_disk] > 0)
    assert np.isnan(mu[~on_disk]).all()
    assert np.isnan(velocity[~on_disk]).all()
    np.testing.assert_allclose(basis[on_disk, 2], -ray[on_disk], atol=1.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(surface[on_disk], axis=-1),
        metadata["solar_radius_m"],
        rtol=0,
        atol=0.1,
    )


def test_build_geometry_reports_empty_valid_region():
    s_map = _full_disk_map()
    with pytest.raises(ValueError, match="no finite on-disk Carrington intersections"):
        build_geometry(s_map, np.zeros(s_map.data.shape, dtype=bool), (0.0, 0.0, 0.0))
