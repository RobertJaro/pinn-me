import numpy as np
import pytest
import torch
from datetime import datetime
from pathlib import Path
from astropy import units as u
from astropy.constants import R_sun
from astropy.coordinates import get_sun
from astropy.io import fits
from astropy.time import Time

from pme.lte.atmosphere import StratifiedAtmosphere
from pme.lte.examples import synthetic_temperature_profile
from pme.lte.geometry import chart_to_direction, direction_to_chart_mm
from pme.lte.hinode import (
    HinodeLTEDataModule,
    HinodePixelDataset,
    _hinode_carrington_rays,
    _atlas_calibrate_stokes,
    load_hinode_raster,
)
from pme.lte.synthesis import LTESynthesizer
from pme.lte.radiometry import (
    disk_center_continuum_radiance,
    load_solar_reference,
    neckel_continuum_limb_darkening,
)
from pme.train.lte_module import LTEModule


def test_hinode_stokes_basis_tracks_positive_hpc_axes_and_rotation():
    """Guard the physical signs, not just orthogonality, of the Stokes basis."""

    obstime = datetime.fromisoformat("2007-01-06T00:47:20.497")
    tx = np.asarray([[70.0]])
    ty = np.asarray([[-26.0]])

    def geometry(x, y, angle=0.0):
        return _hinode_carrington_rays(
            np.asarray([[x]]),
            np.asarray([[y]]),
            [obstime],
            stokes_reference_angle_deg=angle,
            valid_mask=np.ones((1, 1), dtype=bool),
        )

    _, direction, _, basis, _ = geometry(tx.item(), ty.item())
    epsilon_arcsec = 0.01
    direction_x_minus = geometry(tx.item() - epsilon_arcsec, ty.item())[1][0, 0]
    direction_x_plus = geometry(tx.item() + epsilon_arcsec, ty.item())[1][0, 0]
    direction_y_minus = geometry(tx.item(), ty.item() - epsilon_arcsec)[1][0, 0]
    direction_y_plus = geometry(tx.item(), ty.item() + epsilon_arcsec)[1][0, 0]

    def transverse_tangent(delta):
        ray = direction[0, 0]
        delta = delta - np.dot(delta, ray) * ray
        return delta / np.linalg.norm(delta)

    hpc_x = transverse_tangent(direction_x_plus - direction_x_minus)
    hpc_y = transverse_tangent(direction_y_plus - direction_y_minus)
    np.testing.assert_allclose(np.dot(basis[0, 0, 0], hpc_x), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(np.dot(basis[0, 0, 1], hpc_y), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(basis[0, 0, 2], -direction[0, 0], atol=1.0e-15)
    np.testing.assert_allclose(
        np.cross(basis[0, 0, 0], basis[0, 0, 1]),
        basis[0, 0, 2],
        atol=1.0e-15,
    )

    rotated = geometry(tx.item(), ty.item(), angle=30.0)[3][0, 0]
    angle = np.deg2rad(30.0)
    np.testing.assert_allclose(
        rotated[0],
        np.cos(angle) * basis[0, 0, 0] + np.sin(angle) * basis[0, 0, 1],
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        rotated[1],
        -np.sin(angle) * basis[0, 0, 0] + np.cos(angle) * basis[0, 0, 1],
        atol=1.0e-15,
    )


def test_prepared_fts_reference_has_absolute_continuum_and_neckel_clv():
    reference = load_solar_reference(
        Path(__file__).resolve().parents[2] / "pme" / "lte" / "data"
    )


def test_atlas_calibration_removes_clv_before_quiet_sun_intensity_selection():
    data_directory = Path(__file__).resolve().parents[2] / "pme" / "lte" / "data"
    reference = load_solar_reference(data_directory)
    wavelength = np.linspace(6300.2, 6303.8, 9)
    continuum_mask = np.ones(wavelength.shape, dtype=bool)
    mu = np.broadcast_to(np.linspace(0.35, 1.0, 10)[:, None], (10, 10)).copy()
    disk_center = disk_center_continuum_radiance(reference, wavelength)
    expected = np.mean(
        disk_center[None, None, :]
        * neckel_continuum_limb_darkening(wavelength[None, None, :], mu[..., None]),
        axis=-1,
    )
    intrinsic_contrast = np.broadcast_to(
        np.linspace(0.9, 1.1, 10)[None, :], mu.shape
    )
    true_detector_to_radiance = 2.0e9
    raw_continuum = expected * intrinsic_contrast / true_detector_to_radiance
    raw_stokes = np.zeros((*mu.shape, 4, wavelength.size), dtype=np.float64)
    raw_stokes[..., 0, :] = raw_continuum[..., None]

    calibrated, metadata = _atlas_calibrate_stokes(
        raw_stokes,
        wavelength,
        continuum_mask,
        raw_continuum,
        mu,
        np.ones(mu.shape, dtype=bool),
        calibration_data_directory=data_directory,
        quiet_sun_max_fractional_polarization=0.01,
        quiet_sun_continuum_trim_quantiles=(0.1, 0.9),
        minimum_quiet_sun_pixels=50,
    )

    assert metadata["detector_to_radiance_w_m3_sr_per_raw_unit"] == pytest.approx(
        true_detector_to_radiance, rel=2.0e-4
    )
    quiet_sun = metadata["quiet_sun"]
    assert "disk-center-equivalent" in quiet_sun["selection"]
    assert quiet_sun["mean_limb_darkening_range"][0] < 0.7
    assert quiet_sun["mean_limb_darkening_range"][1] == pytest.approx(1.0)
    calibrated_continuum = calibrated[..., 0, :].mean(axis=-1)
    assert calibrated_continuum[0].mean() < calibrated_continuum[-1].mean()
    continuum = disk_center_continuum_radiance(reference, np.asarray([6302.0]))
    assert continuum[0] == pytest.approx(3.06049e13, rel=2.0e-4)
    np.testing.assert_allclose(
        neckel_continuum_limb_darkening(6302.0, np.asarray([1.0, 0.5])),
        np.asarray([1.0, 0.745269]),
        rtol=2.0e-5,
    )


def test_hinode_loader_uses_one_based_wcs_and_preserves_detector_order(
    synthetic_hinode_files, expected_hinode_wavelength
):
    raster = load_hinode_raster(synthetic_hinode_files)
    for value in (
        raster.stokes,
        raster.wavelength_angstrom,
        raster.coords,
        raster.mu,
        raster.ray_origin_m,
        raster.ray_direction,
        raster.stokes_basis,
    ):
        assert value.dtype == torch.float32

    assert raster.stokes.shape == (3, 2, 4, 112)
    assert raster.wavelength_angstrom.shape == (112,)
    torch.testing.assert_close(
        raster.wavelength_angstrom,
        torch.from_numpy(expected_hinode_wavelength).to(raster.wavelength_angstrom),
        rtol=0,
        atol=2e-6,
    )
    assert torch.all(raster.wavelength_angstrom[1:] > raster.wavelength_angstrom[:-1])

    wavelength_index = np.arange(112, dtype=np.float32)
    raw = np.empty((4, 112), dtype=np.float32)
    raw[0] = 10000 + 1 * 100 + wavelength_index
    for stokes_index in range(1, 4):
        raw[stokes_index] = stokes_index + 1 * 0.01 + wavelength_index * 1.0e-3
    expected_profile = torch.from_numpy(raw.copy()).to(raster.stokes)
    calibration = raster.metadata["normalization"]["radiometric_calibration"]
    expected_profile *= (
        calibration["detector_to_radiance_w_m3_sr_per_raw_unit"]
        / calibration["atlas_disk_center_continuum_radiance_w_m3_sr"]
    )
    torch.testing.assert_close(raster.stokes[1, 0], expected_profile)

    assert raster.wavelength_angstrom[0] < 6301.5008 < raster.wavelength_angstrom[-1]
    assert raster.wavelength_angstrom[0] < 6302.4932 < raster.wavelength_angstrom[-1]
    assert raster.coords.shape == (3, 2, 3)
    assert raster.mu.shape == (3, 2, 1)
    assert torch.all((raster.mu > 0) & (raster.mu <= 1))
    assert raster.valid_mask.shape == (3, 2)
    wavelength_metadata = raster.metadata["wavelength"]
    assert wavelength_metadata["raw_cdelt1_angstrom"] < 0
    assert wavelength_metadata["effective_cdelt1_angstrom"] > 0
    assert wavelength_metadata["raw_cdelt1_sign_overridden"]
    assert wavelength_metadata["legacy_header_repair_applied"]
    assert wavelength_metadata["effective_crval1_angstrom"] == pytest.approx(6302.0801485)
    assert wavelength_metadata["detector_order_preserved"]
    assert not wavelength_metadata["jointly_reordered_with_stokes"]
    assert "thermd_sbsp" in wavelength_metadata["sign_convention_provenance"]["evidence"]
    assert raster.metadata["wavelength"]["spwlshft"] == [4.72823, 4.72823]
    assert raster.metadata["wavelength"]["shift_application"].startswith("not reapplied")
    observer_velocity = raster.metadata["observer_velocity_correction"]
    assert observer_velocity["source_keyword"] == "DOP_RCV"
    assert observer_velocity["observer_los_velocity_m_per_s"] == [2011.0, 2011.0]
    assert "positive" in observer_velocity["sign_convention"]
    assert observer_velocity["calibration_stage"].startswith("already removed")
    assert "double-correct" in observer_velocity["inversion_action"]
    assert "no absolute solar velocity zero" in observer_velocity["inferred_velocity_frame"]


def test_hinode_loader_applies_one_fixed_atlas_radiometric_scale(
    synthetic_hinode_files,
):
    data_module = HinodeLTEDataModule(
        files=synthetic_hinode_files,
        batch_size=6,
        validation_batch_size=6,
    )
    data_module.setup("fit")
    indices = data_module.normalization_metadata["indices"]
    valid = data_module.raster.valid_mask
    continua = data_module.raster.stokes[..., 0, indices].mean(dim=-1)[valid]
    calibration = data_module.normalization_metadata["radiometric_calibration"]
    assert calibration["stored_stokes_unit"] == "I_c,atlas(mu=1)"
    assert calibration["quiet_sun"]["pixel_count"] > 0
    assert continua.max() > continua.min()
    assert data_module.normalization_metadata["scope"] == "none"
    assert data_module.normalization_metadata["operation"] == (
        "no raster or per-pixel continuum normalization"
    )

    batch = next(iter(data_module.val_dataloader()))
    pixel_indices = batch["pixel_index"]
    expected = data_module.raster.stokes[
        pixel_indices[:, 0], pixel_indices[:, 1]
    ]
    torch.testing.assert_close(batch["stokes"], expected)


def test_hinode_wavelength_wcs_converts_angstrom_equivalent_units(
    synthetic_hinode_files, expected_hinode_wavelength
):
    for path in synthetic_hinode_files:
        fits.setval(path, "CUNIT1", value="nm")
        fits.setval(path, "CRVAL1", value=fits.getval(path, "CRVAL1") / 10.0)
        fits.setval(path, "CDELT1", value=fits.getval(path, "CDELT1") / 10.0)

    raster = load_hinode_raster(synthetic_hinode_files)
    np.testing.assert_allclose(
        raster.wavelength_angstrom.numpy(),
        expected_hinode_wavelength,
        rtol=0.0,
        atol=3e-4,
    )
    wavelength = raster.metadata["wavelength"]
    assert wavelength["source_cunit1"] == "nm"
    assert wavelength["cunit1_conversion_factor_to_angstrom"] == pytest.approx(10.0)


def test_level1_observer_velocity_is_provenance_not_a_model_input(
    synthetic_hinode_files,
):
    reference = load_hinode_raster(synthetic_hinode_files)
    for scan, path in enumerate(synthetic_hinode_files):
        fits.setval(path, "DOP_RCV", value=-1500.0 + 750.0 * scan)
    changed_header = load_hinode_raster(synthetic_hinode_files)

    torch.testing.assert_close(changed_header.wavelength_angstrom, reference.wavelength_angstrom)
    torch.testing.assert_close(changed_header.stokes, reference.stokes)
    torch.testing.assert_close(changed_header.coords, reference.coords)
    assert changed_header.metadata["observer_velocity_correction"][
        "observer_los_velocity_m_per_s"
    ] == [-1500.0, -750.0]

    sample = HinodePixelDataset(changed_header)[0]
    assert "observer_los_velocity_m_per_s" not in sample
    assert "observer_velocity_correction" not in sample


def test_hinode_wavelength_wcs_rejects_non_wavelength_unit(synthetic_hinode_files):
    fits.setval(synthetic_hinode_files[0], "CUNIT1", value="s")
    with pytest.raises(ValueError, match="not wavelength-equivalent"):
        load_hinode_raster(synthetic_hinode_files)


def test_hinode_mu_uses_degree_slit_rotation_and_exact_ray_impact(
    synthetic_hinode_files,
):
    for scan, path in enumerate(synthetic_hinode_files):
        fits.setval(path, "CROTA2", value=30.0)
        # Move the fixture toward the limb so the exact ray-impact expression
        # is measurably different from rho/apparent-radius.
        fits.setval(path, "XCEN", value=600.0 + scan * 0.14857)
        fits.setval(path, "YCEN", value=600.0)
    raster = load_hinode_raster(synthetic_hinode_files)

    rows = np.arange(3, dtype=np.float64)
    offset = (rows + 1.0 - 1.5) * 0.1585
    theta = np.deg2rad(30.0)
    for scan, date_obs in enumerate((
        "2007-01-05T23:59:07.816",
        "2007-01-05T23:59:08.816",
    )):
        x = 600.0 + scan * 0.14857 - offset * np.sin(theta)
        y = 600.0 + offset * np.cos(theta)
        distance = get_sun(Time(date_obs)).distance.to_value(u.m)
        tx = x * u.arcsec.to(u.rad)
        ty = y * u.arcsec.to(u.rad)
        sin_rho = np.sqrt(1.0 - np.square(np.cos(tx) * np.cos(ty)))
        expected = np.sqrt(
            1.0 - np.square(distance * sin_rho / R_sun.to_value(u.m))
        )
        np.testing.assert_allclose(
            raster.mu[:, scan, 0].numpy(), expected, rtol=0.0, atol=5e-8
        )

    surface = chart_to_direction(
        raster.coords[..., 1:],
        torch.tensor(raster.metadata["ray_geometry"]["scene_basis_rows"]),
        R_sun.to_value(u.m),
    ) * R_sun.to_value(u.m)
    torch.testing.assert_close(
        torch.linalg.vector_norm(surface, dim=-1),
        torch.full(raster.spatial_shape, R_sun.to_value(u.m), dtype=surface.dtype),
        rtol=0.0,
        atol=128.0,
    )
    derived_mu = (
        surface / torch.linalg.vector_norm(surface, dim=-1, keepdim=True)
        * (-raster.ray_direction)
    ).sum(dim=-1)
    torch.testing.assert_close(derived_mu.float(), raster.mu[..., 0], rtol=0, atol=1e-6)
    expected_chart = direction_to_chart_mm(
        surface,
        torch.tensor(raster.metadata["ray_geometry"]["scene_basis_rows"]),
        R_sun.to_value(u.m),
    )
    torch.testing.assert_close(raster.coords[..., 1:], expected_chart.float(), atol=1e-4, rtol=0)

    geometry = raster.metadata["ray_geometry"]
    assert geometry["rotation_unit"] == "degree"
    assert geometry["crota2_deg"] == [30.0, 30.0]
    assert geometry["stokes_reference_angle_deg_from_hpc_x_toward_hpc_y"] == 0.0
    reference = geometry["stokes_reference_provenance"]
    assert reference["stored_level1_frame"].endswith(
        "+Q along solar east-west"
    )
    assert reference["crota2_application"].startswith("already applied")
    assert "no second CROTA2 rotation" in geometry["stokes_reference_status"]
    assert geometry["slit_scale_arcsec_per_pixel"] == [0.1585, 0.1585]
    assert geometry["apparent_solar_radius_formula"].startswith("asin")
    assert geometry["impact_parameter_formula"] == (
        "b=D*sqrt(1-(cos(Tx)*cos(Ty))^2)"
    )
    assert geometry["transfer_model"].startswith("differentiable 3-D")


def test_hinode_coords_are_physical_solar_xy_mm_with_saved_affine(
    synthetic_hinode_files,
):
    raster = load_hinode_raster(synthetic_hinode_files)

    np.testing.assert_allclose(
        raster.coords[..., 0].numpy(),
        np.broadcast_to(np.asarray((0.0, 1.0 / 3600.0)), (3, 2)),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert torch.isfinite(raster.coords[..., 1:]).all()
    assert torch.linalg.vector_norm(
        raster.coords[..., 1:].reshape(-1, 2).mean(dim=0)
    ) < 1.0e-3
    assert torch.all(torch.linalg.vector_norm(raster.stokes_basis, dim=-1) > 0.99999)

    coordinates = raster.metadata["coordinates"]
    assert coordinates["order"] == [
        "time_hours", "carrington_chart_x_mm", "carrington_chart_y_mm"
    ]
    assert coordinates["units"] == ["hour", "Mm", "Mm"]
    assert coordinates["surface_deprojection_applied"]
    affine = coordinates["network_affine"]
    assert affine["formula"] == "normalized_xy=(solar_xy_mm-center_mm)/scale_mm"
    assert affine["scale_mm"][0] == affine["scale_mm"][1]
    assert affine["scale_mm"][0] > 0
    normalized = (
        raster.coords[..., 1:].double()
        - torch.tensor(affine["center_mm"], dtype=torch.float64)
    ) / torch.tensor(affine["scale_mm"], dtype=torch.float64)
    assert torch.isfinite(normalized).all()


def test_hinode_loader_crops_before_returning_raster(synthetic_hinode_files):
    raster = load_hinode_raster(
        synthetic_hinode_files,
        scan_slice=slice(1, 2),
        slit_slice=slice(1, 3),
    )
    assert raster.stokes.shape == (2, 1, 4, 112)
    assert raster.coords.shape == (2, 1, 3)
    assert raster.valid_mask.shape == (2, 1)


def test_one_pixel_crop_uses_native_plate_scale_for_saved_affine(
    synthetic_hinode_files,
):
    raster = load_hinode_raster(
        synthetic_hinode_files,
        scan_slice=slice(1, 2),
        slit_slice=slice(1, 2),
    )
    affine = raster.metadata["coordinates"]["network_affine"]
    np.testing.assert_allclose(
        affine["center_mm"], raster.coords[0, 0, 1:].double().numpy(), atol=1.0e-6
    )
    assert affine["native_neighbor_spacing_mm"] > 0
    assert affine["scale_mm"][0] == affine["scale_mm"][1]


def test_pixel_dataset_and_data_module_keep_existing_stokes_format(synthetic_hinode_files):
    raster = load_hinode_raster(synthetic_hinode_files)
    dataset = HinodePixelDataset(raster)
    assert len(dataset) == 6
    sample = dataset[0]
    assert set(sample) >= {
        "coords",
        "mu",
        "stokes",
        "pixel_index",
    }
    assert sample["coords"].shape == (3,)
    assert sample["mu"].shape == (1,)
    assert sample["stokes"].shape == (4, 112)

    module = HinodeLTEDataModule(
        synthetic_hinode_files,
        batch_size=2,
        validation_batch_size=3,
        validation_stride=2,
        num_workers=0,
    )
    module.setup()
    batch = next(iter(module.train_dataloader()))
    assert batch["coords"].shape == (2, 3)
    assert batch["mu"].shape == (2, 1)
    assert batch["stokes"].shape == (2, 4, 112)
    assert module.wavelength_angstrom.shape == (112,)
    validation_batch = next(iter(module.val_dataloader()))
    assert validation_batch["coords"].shape == (3, 3)
    metadata = module.checkpoint_metadata()
    assert metadata["training_sampling"] == {
        "type": "full valid raster once per epoch",
        "samples_per_epoch": 6,
        "valid_pixel_count": 6,
    }
    assert metadata["validation"] == {
        "type": "deterministic diagnostic subset of inversion pixels",
        "target_area_stride": 2,
        "spatial_lattice_stride_slit_scan": [1, 2],
        "pixel_count": 3,
        "held_out": False,
    }

def test_physics_sampling_uses_only_observed_rays_inside_the_shell(synthetic_hinode_files):
    module = HinodeLTEDataModule(synthetic_hinode_files, batch_size=2, num_workers=0)
    module.setup()
    bounds = (1.5, -0.1)
    module.configure_physics_sampling(
        torch.linspace(-5.0, 1.0, 11),
        volume_points_per_step=8,
        height_layers_per_step=2,
        optical_depth_anchor_points_per_step=3,
        shell_height_bounds_Mm=bounds,
        tangent_margin_m=1000.0,
    )
    loaders = module.train_dataloader()
    assert set(loaders) == {"stokes", "physics_volume", "optical_depth_anchor"}
    volume = next(iter(loaders["physics_volume"]))
    anchor = next(iter(loaders["optical_depth_anchor"]))
    assert set(volume) == {"coords", "ray_origin_m", "ray_direction", "geometric_height_m"}
    assert volume["coords"].shape == (8, 3)
    heights, counts = torch.unique(volume["geometric_height_m"], return_counts=True)
    assert heights.numel() == 2
    torch.testing.assert_close(counts, torch.full((2,), 4, dtype=counts.dtype))
    assert torch.all(volume["geometric_height_m"] <= bounds[0] * 1.0e6)
    torch.testing.assert_close(
        anchor["geometric_height_m"], torch.full((3,), bounds[0] * 1.0e6)
    )
    sampling = module.checkpoint_metadata()["physics_sampling"]
    assert sampling["height_layers_per_step"] == 2
    assert sampling["optical_depth_anchor_stream_enabled"]
    assert sampling["spatial_sampling"]["type"] == "observed_ray_bundle"
    assert sampling["spatial_sampling"]["coordinate_frame"] == (
        "heliocentric Carrington Cartesian"
    )
    domain = sampling["observed_domain_bounds"]
    assert domain["height_Mm"] == [bounds[1], bounds[0]]
    assert domain["radius_Rsun"][1] > domain["radius_Rsun"][0]
    assert domain["surface_latitude_deg"][1] > domain["surface_latitude_deg"][0]
    assert domain["surface_longitude_offset_deg"][1] > (
        domain["surface_longitude_offset_deg"][0]
    )
    assert "not sampled as a rectangular box" in domain["support"]
    assert "coordinate_mode" not in sampling


def test_hinode_lte_rejects_pinned_host_memory(synthetic_hinode_files):
    with pytest.raises(ValueError, match="pin_memory=false"):
        HinodeLTEDataModule(synthetic_hinode_files, pin_memory=True)


def test_integer_detector_saturation_is_excluded_from_training(
    synthetic_hinode_files, tmp_path
):
    header = fits.getheader(synthetic_hinode_files[0])
    data = np.full((4, 3, 112), 100, dtype=np.int16)
    data[0, 0, 0] = np.iinfo(np.int16).max
    path = tmp_path / "saturated.fits"
    fits.writeto(path, data, header)

    raster = load_hinode_raster(
        path,
        quiet_sun_max_fractional_polarization=2.0,
        quiet_sun_continuum_trim_quantiles=(0.0, 1.0),
    )
    assert not raster.valid_mask[0, 0]
    assert torch.all(raster.valid_mask[1:, 0])
    assert raster.metadata["quality_mask"]["rejected_pixel_count"] == 1


@pytest.mark.integration
def test_real_hinode_small_patch_loads_all_wavelengths(real_hinode_files):
    raster = load_hinode_raster(
        real_hinode_files[:2],
        slit_slice=slice(0, 8),
    )
    assert raster.stokes.shape == (8, 2, 4, 112)
    assert raster.wavelength_angstrom.shape == (112,)
    assert torch.isfinite(raster.stokes[raster.valid_mask]).all()
    assert torch.all(raster.wavelength_angstrom[1:] > raster.wavelength_angstrom[:-1])
    mean_intensity = raster.stokes[..., 0, :].mean(dim=(0, 1))
    line_minima = {}
    for laboratory_wavelength in (6301.5008, 6302.4932):
        neighborhood = (
            (raster.wavelength_angstrom > laboratory_wavelength - 0.2)
            & (raster.wavelength_angstrom < laboratory_wavelength + 0.2)
        )
        observed_minimum = raster.wavelength_angstrom[neighborhood][
            torch.argmin(mean_intensity[neighborhood])
        ]
        minimum_intensity = torch.min(mean_intensity[neighborhood])
        line_minima[laboratory_wavelength] = (observed_minimum, minimum_intensity)
        # The documented pre-1.05 sp_prep repair removes the stale ~0.08 A
        # header offset. Solar/spacecraft velocities remain physical fit terms.
        assert abs(float(observed_minimum) - laboratory_wavelength) < 0.03

    # Fe I 6301.5 is the stronger/deeper line.  This identifies which detector
    # feature belongs to which laboratory line and guards against silently
    # accepting the raw negative CDELT1 convention with a reversed spectrum.
    wavelength_6301, intensity_6301 = line_minima[6301.5008]
    wavelength_6302, intensity_6302 = line_minima[6302.4932]
    assert wavelength_6301 < wavelength_6302
    assert intensity_6301 < intensity_6302

    # Exercise the standalone forward model on the instrument's exact FITS-WCS
    # grid, rather than on an idealized or cropped wavelength axis.
    dtype = torch.float32
    log_tau500 = torch.linspace(-4.0, 1.0, 9, dtype=dtype)
    atmosphere = StratifiedAtmosphere(
        log_tau500=log_tau500,
        temperature=synthetic_temperature_profile(log_tau500).unsqueeze(0),
        velocity_field=torch.zeros(1, 9, 3, dtype=dtype),
        microturbulence=torch.full((1, 9), 1_000.0, dtype=dtype),
        magnetic_field=torch.zeros(1, 9, 3, dtype=dtype),
        gas_pressure=torch.logspace(-0.5, 5.0, 9, dtype=dtype).unsqueeze(0),
    )
    synthesized, diagnostics = LTESynthesizer(log_tau500=log_tau500)(
        atmosphere,
        raster.wavelength_angstrom,
        return_diagnostics=True,
    )
    assert synthesized.shape == (1, 4, 112)
    assert torch.isfinite(synthesized).all()
    assert set(diagnostics.propagation.line_absorption) >= {
        "FeI_6301.5008",
        "FeI_6302.4932",
    }


@pytest.mark.integration
def test_real_hinode_small_patch_runs_an_lte_optimizer_step(real_hinode_files):
    """Exercise the actual observation -> network -> LTE -> loss update path."""

    raster = load_hinode_raster(
        real_hinode_files[:2],
        slit_slice=slice(0, 2),
    )
    mask = raster.valid_mask
    batch = {
        "coords": raster.coords[mask],
        "mu": raster.mu[mask],
        "stokes": raster.stokes[mask],
    }
    module = LTEModule(
        log_tau500=torch.linspace(-4.0, 1.0, 9),
        wavelength_angstrom=raster.wavelength_angstrom,
        atmosphere_config={
            "model_config": {
                "dim": 12,
                "n_layers": 2,
                "encoding_config": {"type": "identity"},
            }
        },
        instrument_config={"oversample": 1},
        normalization_config={
            "asinh_alphas": {"Q": 1.0e-2, "U": 1.0e-2, "V": 1.0e-2}
        },
        weight_config={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0},
        wavelength_exclude_windows_angstrom=((6302.04, 6302.14),),
    ).float()
    optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-4)
    before = module.atmosphere_model.network.out_layer.bias.detach().clone()

    optimizer.zero_grad()
    # Exercise the same newly perturbed, nonuniform depth grid used by
    # ``training_step`` rather than the deterministic evaluation grid.
    prediction = module.synthesize(
        batch["coords"], batch["mu"], randomize_depth=True
    )["stokes"]
    _, loss = module._stokes_objective(prediction, batch["stokes"])
    loss.backward()
    trainable_gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert prediction.shape == batch["stokes"].shape == (4, 4, 112)
    assert torch.isfinite(loss)
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
    optimizer.step()

    after = module.atmosphere_model.network.out_layer.bias.detach()
    assert not torch.equal(before, after)
    assert torch.isfinite(module(batch["coords"], batch["mu"])).all()
