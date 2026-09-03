"""Scientific invariants of the modular Hinode/SOT-SP adapter."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch
from astropy.io import fits

from prom3theus.instruments.hinode_sp.calibration import (
    calibrate_stokes,
    observer_los_velocity_metadata,
)
from prom3theus.instruments.hinode_sp.fits_io import slice_from_config
from prom3theus.instruments.hinode_sp.geometry import carrington_rays
from prom3theus.instruments.hinode_sp.observation import (
    HinodeDataModule,
    load_raster,
)
from prom3theus.observations import ObservationPixelDataset
from prom3theus.rt.radiometry import (
    disk_center_continuum_radiance,
    load_solar_reference,
    neckel_continuum_limb_darkening,
)


def test_slit_slice_rejects_lossy_or_textual_coercion():
    assert slice_from_config([2, 7], name="slit_slice") == slice(2, 7)
    for invalid in ([2.0, 7.0], ["2", "7"], [False, True]):
        with pytest.raises(TypeError, match="entries must be integers"):
            slice_from_config(invalid, name="slit_slice")


def test_stokes_basis_tracks_positive_hpc_axes_and_rotation():
    obstime = datetime.fromisoformat("2007-01-06T00:47:20.497")

    def geometry(x: float, y: float, angle: float = 0.0):
        return carrington_rays(
            np.asarray([[x]]),
            np.asarray([[y]]),
            [obstime],
            stokes_reference_angle_deg=angle,
            valid_mask=np.ones((1, 1), dtype=bool),
        )

    x, y = 70.0, -26.0
    direction, _, basis, _ = geometry(x, y)
    epsilon = 0.01

    def transverse_tangent(delta):
        ray = direction[0, 0]
        delta = delta - np.dot(delta, ray) * ray
        return delta / np.linalg.norm(delta)

    hpc_x = transverse_tangent(
        geometry(x + epsilon, y)[0][0, 0] - geometry(x - epsilon, y)[0][0, 0]
    )
    hpc_y = transverse_tangent(
        geometry(x, y + epsilon)[0][0, 0] - geometry(x, y - epsilon)[0][0, 0]
    )
    np.testing.assert_allclose(np.dot(basis[0, 0, 0], hpc_x), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(basis[0, 0, 1], hpc_y), 1.0, atol=1e-12)
    np.testing.assert_allclose(basis[0, 0, 2], -direction[0, 0], atol=1e-15)
    np.testing.assert_allclose(
        np.cross(basis[0, 0, 0], basis[0, 0, 1]), basis[0, 0, 2], atol=1e-15
    )

    rotated = geometry(x, y, angle=30.0)[2][0, 0]
    angle = np.deg2rad(30.0)
    np.testing.assert_allclose(
        rotated[0],
        np.cos(angle) * basis[0, 0, 0] + np.sin(angle) * basis[0, 0, 1],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        rotated[1],
        -np.sin(angle) * basis[0, 0, 0] + np.cos(angle) * basis[0, 0, 1],
        atol=1e-15,
    )


def test_atlas_calibration_removes_limb_darkening_before_intensity_trim():
    reference = load_solar_reference()
    wavelength = np.linspace(6300.2, 6303.8, 9)
    continuum_mask = np.ones(wavelength.shape, dtype=bool)
    mu = np.broadcast_to(np.linspace(0.35, 1.0, 10)[:, None], (10, 10)).copy()
    disk_center = disk_center_continuum_radiance(reference, wavelength)
    expected = np.mean(
        disk_center[None, None, :]
        * neckel_continuum_limb_darkening(wavelength[None, None, :], mu[..., None]),
        axis=-1,
    )
    intrinsic_contrast = np.broadcast_to(np.linspace(0.9, 1.1, 10)[None, :], mu.shape)
    detector_to_radiance = 2.0e9
    raw_continuum = expected * intrinsic_contrast / detector_to_radiance
    raw_stokes = np.zeros((*mu.shape, 4, wavelength.size), dtype=np.float64)
    raw_stokes[..., 0, :] = raw_continuum[..., None]

    calibrated, metadata = calibrate_stokes(
        raw_stokes,
        wavelength,
        continuum_mask,
        raw_continuum,
        mu,
        np.ones(mu.shape, dtype=bool),
        solar_reference=reference,
        quiet_sun_max_fractional_polarization=0.01,
        quiet_sun_continuum_trim_quantiles=(0.1, 0.9),
        minimum_quiet_sun_pixels=50,
    )
    assert metadata["detector_to_radiance_w_m3_sr_per_raw_unit"] == pytest.approx(
        detector_to_radiance, rel=2e-4
    )
    quiet_sun = metadata["quiet_sun"]
    assert "disk-center-equivalent" in quiet_sun["selection"]
    assert quiet_sun["mean_limb_darkening_range"][0] < 0.7
    assert quiet_sun["mean_limb_darkening_range"][1] == pytest.approx(1.0)
    calibrated_continuum = calibrated[..., 0, :].mean(axis=-1)
    assert calibrated_continuum[0].mean() < calibrated_continuum[-1].mean()


def test_observer_motion_stays_removed_from_recovered_solar_gauge():
    dispersion = 0.021549

    def header(velocity: float, total_shift: float) -> fits.Header:
        value = fits.Header()
        value["DOP_RCV"] = velocity
        value["SPWLSHFT"] = 4.72823
        value["SPWLSFT0"] = total_shift
        value["DOPVUSED"] = 0
        return value

    original_velocity = 2011.0
    original_total_shift = -0.887668
    reference = observer_los_velocity_metadata(
        [header(original_velocity, original_total_shift)], dispersion
    )
    replacement_velocity = -1500.0
    orbital_delta = (
        (replacement_velocity - original_velocity)
        / 299_792_458.0
        * 6301.5091
        / dispersion
    )
    changed = observer_los_velocity_metadata(
        [header(replacement_velocity, original_total_shift + orbital_delta)],
        dispersion,
    )
    assert changed["observer_los_velocity_m_per_s"] == [replacement_velocity]
    np.testing.assert_allclose(
        changed["removed_solar_los_velocity_m_per_s"],
        reference["removed_solar_los_velocity_m_per_s"],
        rtol=0.0,
        atol=1e-8,
    )


def test_raster_preserves_detector_order_and_exact_velocity_gauge(
    synthetic_hinode_files, expected_hinode_wavelength
):
    raster = load_raster(synthetic_hinode_files)
    assert raster.stokes.shape == (3, 2, 4, 112)
    assert raster.coordinates.shape == (3, 2, 3)
    assert raster.valid_mask.shape == (3, 2)
    assert all(
        value.dtype == torch.float32
        for value in (
            raster.stokes,
            raster.wavelength_angstrom,
            raster.coordinates,
            raster.ray_direction,
            raster.stokes_basis,
        )
    )
    np.testing.assert_allclose(
        raster.wavelength_angstrom.numpy(),
        expected_hinode_wavelength,
        rtol=0.0,
        atol=3e-4,
    )
    assert torch.all(raster.wavelength_angstrom[1:] > raster.wavelength_angstrom[:-1])

    raw = np.empty((4, 112), dtype=np.float32)
    wavelength_index = np.arange(112, dtype=np.float32)
    raw[0] = 10_000 + 100 + wavelength_index
    for stokes_index in range(1, 4):
        raw[stokes_index] = stokes_index + 0.01 + wavelength_index * 1e-3
    calibration = raster.metadata["normalization"]["radiometric_calibration"]
    expected_profile = torch.from_numpy(raw) * (
        calibration["detector_to_radiance_w_m3_sr_per_raw_unit"]
        / calibration["atlas_disk_center_continuum_radiance_w_m3_sr"]
    )
    torch.testing.assert_close(raster.stokes[1, 0], expected_profile)

    velocity = raster.metadata["observer_velocity_correction"]
    assert velocity["observer_los_velocity_m_per_s"] == [2011.0, 2011.0]
    dataset = ObservationPixelDataset(raster)
    sample = dataset[0]
    assert "observer_los_velocity_m_per_s" not in sample
    assert sample["removed_solar_los_velocity_m_per_s"].ndim == 0
    assert set(sample) == {
        "coordinates",
        "ray_direction",
        "stokes_basis",
        "removed_solar_los_velocity_m_per_s",
        "stokes",
        "pixel_index",
    }


def test_nested_hinode_raster_sequences_are_rejected(synthetic_hinode_files, tmp_path):
    archive = tmp_path / "sequence"
    for raster_index, name in enumerate(("first", "second")):
        target = archive / name
        target.mkdir(parents=True)
        for scan_index, source in enumerate(synthetic_hinode_files):
            header = fits.getheader(source)
            data = fits.getdata(source).astype(np.float32)
            header["DATE_OBS"] = (
                f"2007-01-0{5 + raster_index}T23:59:0{7 + scan_index}.816"
            )
            if raster_index == 1:
                data *= 2.0
            fits.writeto(target / source.name, data, header)

    module = HinodeDataModule(directory=archive, progress=False)
    with pytest.raises(ValueError, match="nested raster sequences are unsupported"):
        module.setup()
