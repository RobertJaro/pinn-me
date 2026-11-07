"""Synthetic products must round-trip through the actual SDO training readers."""

from types import SimpleNamespace

import numpy as np
import torch
from astropy.io import fits

from prom3theus.muram.synthesize import synthesize_hmi, synthesize_aia
from prom3theus.muram.fits import (
    write_hmi_fits,
    write_aia_fits,
    prepare_aia_fits,
    observation_time,
)


def snapshot(shape=(3, 4, 30)):
    q = {name: np.zeros(shape) for name in ("bx", "by", "bz", "vx", "vy", "vz")}
    q.update(
        temperature=np.broadcast_to(
            np.r_[np.linspace(7000, 4500, 20), np.linspace(1e5, 2e6, 10)], shape
        ).copy(),
        pressure=np.broadcast_to(np.geomspace(3e6, 0.01, 30), shape).copy(),
        tau=np.broadcast_to(np.geomspace(100, 1e-9, 30), shape).copy(),
        ds_m=(1e5, 1e5, 1e4),
    )
    q["bx"][:] = 300
    q["by"][:] = 450
    q["bz"][:] = 900
    return SimpleNamespace(
        time_s=3600.0, iteration=10, synthesis_ready_quantities=lambda: q
    )


def test_hmi_fits_roundtrip_through_sdo_data_module(tmp_path):
    from prom3theus.instruments.hmi.timeline import HMIDataModule
    from prom3theus.instruments.hmi.observation import _describe_data
    from prom3theus.observations.registry import _hmi_source_files
    from prom3theus.artifacts.validation import (
        validate_raster_scientific_contract,
        validate_raster_velocity_contract,
    )

    snap = snapshot((4, 4, 30))
    result = synthesize_hmi(
        snap, progress=False
    )
    root = tmp_path / "hmi"
    paths = write_hmi_fits(root, result, time_s=snap.time_s, iteration=snap.iteration)
    assert len(paths) == 24
    from prom3theus.muram.cli import main
    from prom3theus.muram.fits import SyntheticHMIResponse, synthetic_response_path

    response_root = tmp_path / "responses"
    main(["prepare-responses", "--source-path", str(root), "--out", str(response_root)])
    response_path = synthetic_response_path(response_root, paths[0])
    response = SyntheticHMIResponse(response_path)
    assert torch.equal(response.quadrature, result.raw_wavelength_angstrom[1:-1])
    assert torch.equal(response.weights, result.response["spectral_weights"])
    with fits.open(paths[0]) as hdus:
        assert "BZTAU1" not in hdus
        assert hdus[0].header["CRLT_OBS"] == 0.0
        assert hdus[0].header["CRLN_OBS"] == 0.0
        assert hdus[0].header["HGLT_OBS"] == 0.0
        np.testing.assert_allclose(
            hdus[0].data / 1e13, result.stokes[..., 0, 5], rtol=2e-7
        )
    data = HMIDataModule(
        directory=root,
        transmission_profile_directory=response_root,
        num_workers=0,
        data_loading_workers=1,
        progress=False,
    )
    data.setup()
    spec = _describe_data(data)
    from prom3theus.instruments.hmi.calibration import hmi_continuum_radiance
    from prom3theus.rt.radiometry import load_solar_reference

    atlas_scale = hmi_continuum_radiance(load_solar_reference("hmi_stokes/solar_reference_617nm.json"))
    calibration = data.raster.metadata["normalization"]["radiometric_calibration"]
    assert calibration["atlas_disk_center_continuum_radiance_w_m3_sr"] == atlas_scale
    assert calibration["stored_stokes_unit"] == "I_c,atlas(mu=1)"
    torch.testing.assert_close(
        torch.as_tensor(data.raster.stokes[:]), result.stokes * (1e13 / atlas_scale), atol=1e-7, rtol=3e-7
    )
    torch.testing.assert_close(
        data.response_quadrature_wavelength_angstrom,
        result.raw_wavelength_angstrom[1:-1],
    )
    validate_raster_scientific_contract(data.raster, spec)
    validate_raster_velocity_contract(data.raster, spec.velocity_synthesis_mode)
    assert (
        len(
            _hmi_source_files(
                {
                    "directory": root,
                    "calibration": {"transmission_profile_directory": response_root},
                }
            )
        )
        == 25
    )
    from prom3theus.muram.binning import bin_spatial
    binned = HMIDataModule(
        directory=root, transmission_profile_directory=response_root,
        binning=2, num_workers=0, data_loading_workers=1, progress=False,
    )
    binned.setup()
    torch.testing.assert_close(binned.raster.stokes[:], bin_spatial(data.raster.stokes, 2))
    assert "ground_truth_bz_gauss" not in binned.raster.auxiliary
    assert binned.raster.spatial_shape == (2, 2)
    validate_raster_scientific_contract(binned.raster, spec)
    validate_raster_velocity_contract(binned.raster, spec.velocity_synthesis_mode)
    # A normalized but incorrect external profile must not silently alter training.
    import pytest
    from prom3theus.instruments.hmi.raster import load_raster

    with np.load(response_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["spectral_weights"] = arrays["spectral_weights"][::-1].copy()
    np.savez_compressed(response_path, **arrays)
    with pytest.raises(ValueError, match="does not match"):
        load_raster(paths, response_root)
    response_path.unlink()
    with pytest.raises(FileNotFoundError, match="transmission profile not found"):
        load_raster(paths, response_root)


def test_aia_fits_roundtrip_through_sdo_preparation(tmp_path):
    from prom3theus.observations import ImageObservationStore
    from prom3theus.instruments.aia_euv.observation import load_prepared_aia_observation
    from prom3theus.instruments.aia_euv.preparation import _verified_response_contract

    snap = snapshot((4, 4, 30))
    hmi = synthesize_hmi(snap, progress=False)
    aia = synthesize_aia(snap, progress=False)
    paths = write_aia_fits(
        tmp_path / "fits", aia, hmi.geometry, time_s=snap.time_s, iteration=10
    )
    prepare_aia_fits(paths, tmp_path / "prepared")
    rasters, _, _ = ImageObservationStore.load_sequence(tmp_path / "prepared")
    for i, raster in enumerate(rasters):
        torch.testing.assert_close(
            torch.as_tensor(raster.intensity[:]), aia.intensity[..., i]
        )
        assert raster.absolute_tai_seconds == observation_time(snap.time_s).tai.unix_tai
    loaded = load_prepared_aia_observation(
        {
            "type": "aia_euv",
            "binning": 2,
            "directory": tmp_path / "prepared",
            "channels_angstrom": [171, 193, 211],
            "selection": {"validation_exposure_group": "muram_000010"},
            "loader": {
                "batch_size": 2,
                "validation_batch_size": 2,
                "workers": 0,
                "pin_memory": False,
            },
        },
        expected_calibration_convention_id=_verified_response_contract()[0],
    )
    from prom3theus.muram.binning import bin_spatial
    from prom3theus.instruments.aia_euv.observation import robust_asinh_scales
    for native, binned in zip(rasters, loaded.data_module.rasters, strict=True):
        torch.testing.assert_close(binned.intensity, bin_spatial(native.intensity, 2))
        assert binned.spatial_shape == (2, 2)
    scales = robust_asinh_scales(loaded.data_module.rasters, [171, 193, 211])
    assert loaded.asinh_scales == tuple(scales[str(channel)] for channel in [171, 193, 211])
    manifest = ImageObservationStore.manifest(tmp_path / "prepared")
    assert loaded.stream.source_signature != manifest["source_signature"]
    assert ImageObservationStore.load_sequence(tmp_path / "prepared")[0][0].spatial_shape == (4, 4)


def test_sdo_cli_and_joint_training_runtime(tmp_path, monkeypatch):
    import json
    import weakref
    from prom3theus.muram.cli import main
    from prom3theus.muram.reader import _FIELD_IDS, _UNIT_SCALE
    from prom3theus.config import load_config, parse_config
    from prom3theus.application.joint_assembly import build_joint_runtime
    import prom3theus.muram.sdo as sdo

    source = tmp_path / "source"
    source.mkdir()
    quantities = snapshot().synthesis_ready_quantities()
    for iteration, time_s in [(10, 3600), (20, 7200)]:
        np.savetxt(
            source / f"Header.{iteration:06d}", [3, 4, 30, 1e7, 1e7, 1e6, time_s]
        )
        for name, value in quantities.items():
            if name != "ds_m":
                (value / _UNIT_SCALE[name]).astype("float32").ravel(order="F").tofile(
                    source / f"{_FIELD_IDS[name]}.{iteration:06d}"
                )
    previous = []
    original = sdo.synthesize_hmi

    def checked(proxy, **kwargs):
        assert all(ref() is None for ref in previous)
        result = original(proxy, **kwargs)
        previous[:] = [
            weakref.ref(proxy.synthesis_ready_quantities()["temperature"]),
            weakref.ref(result),
        ]
        return result

    monkeypatch.setattr(sdo, "synthesize_hmi", checked)
    root = tmp_path / "sdo"
    main(
        [
            "synthesize",
            "--source-path",
            str(source),
            "--out",
            str(root),
            "--iterations",
            "10",
            "20",
            "--stride",
            "2",
            "--no-progress",
            "--device",
            "cpu",
        ]
    )
    assert all(ref() is None for ref in previous)
    assert len(list((root / "hmi" / "prepared").glob("*.fits"))) == 48
    assert len(list((root / "hmi" / "responses").glob("*.npz"))) == 2
    assert len(list((root / "aia" / "fits").glob("*.fits"))) == 6
    assert not list(root.rglob("*.jpg"))
    assert (
        json.loads((root / "synthesis_status.json").read_text())["state"] == "complete"
    )
    raw = load_config("configs/muram/sdo/joint.yaml").to_dict()
    raw["solver"].update(
        output_directory=str(tmp_path / "run"), work_directory=str(tmp_path / "work")
    )
    raw["atmosphere"]["network"].update(hidden_dimension=8, hidden_layers=1)
    raw["diagnostics"]["visualization"]["enabled"] = False
    raw["physics"]["potential_boundary"]["enabled"] = False
    raw["physics"]["potential_boundary"]["photosphere"]["enabled"] = False
    for stream in raw["streams"]:
        obs = stream["observation"]
        obs["binning"] = 1
        obs["loader"].update(batch_size=2, validation_batch_size=2, workers=0)
        if obs["type"] == "hmi_stokes":
            obs["directory"] = str(root / "hmi" / "prepared")
            obs["calibration"]["transmission_profile_directory"] = str(root / "hmi" / "responses")
            obs["loader"].update(preparation_workers=1, validation_stride=1)
        else:
            obs["directory"] = str(root / "aia" / "prepared")
            obs["selection"]["validation_exposure_group"] = "muram_000010"
    runtime = build_joint_runtime(parse_config(raw, base_directory=tmp_path))
    from prom3theus.instruments.hmi.calibration import hmi_continuum_radiance
    from prom3theus.rt.radiometry import load_solar_reference
    atlas_scale = hmi_continuum_radiance(load_solar_reference("hmi_stokes/solar_reference_617nm.json"))
    np.testing.assert_allclose(
        runtime.model.terms["photospheric_stokes"].atlas_continuum_radiance_w_m3_sr.detach().cpu(),
        atlas_scale, rtol=1e-7,
    )
    for stream_id, batch in runtime.validation_batches.items():
        loss = runtime.model.terms[stream_id].evaluate_batch(batch).likelihood_loss
        assert torch.isfinite(loss)
        loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in runtime.model.parameters()
    )


def test_sdo_keeps_completed_fits_when_next_snapshot_fails(tmp_path, monkeypatch):
    import json
    import pytest
    import prom3theus.muram.sdo as sdo

    def read_snapshot(path, iteration, **kwargs):
        if iteration == 20:
            raise OSError("missing snapshot")
        return snapshot()

    monkeypatch.setattr(sdo, "MURaMSnapshot", read_snapshot)
    root = tmp_path / "sdo"
    args = SimpleNamespace(
        out=root, source_path=tmp_path, iterations=[10, 20], vertical_axis=None,
        stride=None, batch_size=256, no_progress=True,
    )
    with pytest.raises(OSError, match="missing snapshot"):
        sdo.synthesize_sdo_sequence(
            args, {},
            device=torch.device("cpu"), dtype=torch.float32,
        )
    state = json.loads((root / "synthesis_status.json").read_text())
    assert state["state"] == "failed"
    assert state["completed_iterations"] == [10]
    assert state["current_iteration"] == 20
    assert len(list((root / "hmi" / "prepared").glob("*.fits"))) == 24
    assert len(list((root / "hmi" / "responses").glob("*.npz"))) == 1
    assert len(list((root / "aia" / "fits").glob("*.fits"))) == 3
    assert not list(root.rglob("*.jpg"))


def test_equatorial_geometry_produces_finite_area_shell_maps(tmp_path, monkeypatch):
    from prom3theus.muram.geometry import build_disk_center_geometry, SOLAR_RADIUS_M
    from prom3theus.diagnostics.evaluation import AtmosphereEvaluator, AtmosphereSampling
    from prom3theus.diagnostics.atmosphere_rendering import AtmospherePlotter

    geometry = build_disk_center_geometry(33, 17, 1e6, 1e6)
    torch.testing.assert_close(
        geometry.surface_position_m[8, 16],
        torch.tensor([SOLAR_RADIUS_M, 0., 0.], dtype=torch.float64),
    )
    basis = geometry.stokes_basis[0, 0]
    assert torch.linalg.det(basis) == 1
    torch.testing.assert_close(geometry.ray_direction[0, 0], -basis[2])
    # A local vertical field points along the equatorial radius in Carrington.
    torch.testing.assert_close(basis.T @ torch.tensor([0., 0., 1.], dtype=torch.float64),
                               torch.tensor([1., 0., 0.], dtype=torch.float64))
    model = SimpleNamespace(scene_basis=basis, solar_radius_m=torch.tensor(SOLAR_RADIUS_M))
    raster = SimpleNamespace(
        surface_position_m=geometry.surface_position_m,
        stokes_basis=geometry.stokes_basis, coordinates=geometry.coordinates,
        valid_mask=torch.ones((17, 33), dtype=torch.bool),
    )
    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options(
        slice_sampling={"longitude_points": 33, "latitude_points": 17}
    ))

    def evaluate(module, position, **kwargs):
        heights = torch.linalg.vector_norm(position.double(), dim=-1) - SOLAR_RADIUS_M
        np.testing.assert_allclose(heights.numpy(), np.broadcast_to([0., 5e7], heights.shape), atol=100.)
        return {"temperature": (5000 + position[..., 1] / 1e5 + heights / 50).numpy()}

    monkeypatch.setattr(evaluator, "_evaluate_physical_positions", evaluate)
    evaluated = evaluator.evaluate_shell_layers(SimpleNamespace(atmosphere_model=model), raster, [0., 5e7])
    assert np.max(np.abs(evaluated["map_latitude_deg"])) < 1.
    assert np.max(np.abs(evaluated["map_longitude_deg"])) < 2.
    figure = AtmospherePlotter().field_panel_figure(
        evaluated, [0, 1], "equatorial MURaM geometry regression",
        field_names=("temperature",), title="Shell layers",
    )
    try:
        figure.savefig(tmp_path / "equatorial_shells.png", dpi=100)
        for axis in figure.axes[:2]:
            bounds = axis.get_window_extent()
            assert 1.5 < bounds.width / bounds.height < 3.
    finally:
        figure.clear()


def test_synthetic_and_detector_hmi_have_identical_training_units_and_loss(tmp_path):
    """Known detector gain and known SI radiance must reach the same objective."""
    from dataclasses import replace
    from prom3theus.muram.fits import SyntheticHMIResponse
    from prom3theus.instruments.hmi.calibration import calibrate_stokes
    from prom3theus.instruments.hmi.raster import load_raster
    from prom3theus.rt.radiometry import load_solar_reference, disk_center_intensity_radiance
    from prom3theus.inversion.objective import StokesObjective, weighted_stokes_loss

    snap = snapshot()
    result = synthesize_hmi(snap, progress=False)
    reference = load_solar_reference("hmi_stokes/solar_reference_617nm.json")
    atlas = disk_center_intensity_radiance(reference, result.raw_wavelength_angstrom[1:-1])
    spatial = tuple(result.stokes.shape[:2])
    physical = np.zeros((*spatial, 4, 6), dtype=np.float64)
    physical[..., 0, :] = (atlas * result.response["spectral_weights"].numpy()).sum(-1)
    # Small signed polarization keeps these eligible for the ordinary gain fit.
    physical[..., 1:, :] = physical[..., :1, :] * np.array([1e-3, -2e-3, 3e-3])[:, None]
    result = replace(result, stokes=torch.tensor(physical / 1e13, dtype=torch.float32))
    paths = write_hmi_fits(tmp_path / "fits", result, time_s=snap.time_s, iteration=10)
    synthetic = load_raster(paths, tmp_path / "fits")
    detector_gain = 2.5e8
    observed, metadata = calibrate_stokes(
        physical.copy() / detector_gain, np.ones(spatial), np.ones(spatial, dtype=bool),
        np.zeros((*spatial, 2)), SyntheticHMIResponse(paths[0]), reference, np.zeros(spatial),
        quiet_sun_max_fractional_polarization=0.01,
        quiet_sun_trim_quantiles=(0., 1.), minimum_quiet_sun_pixels=1,
        calibration_sample_limit=12,
    )
    np.testing.assert_allclose(metadata["detector_to_radiance_w_m3_sr_per_raw_unit"], detector_gain)
    target = torch.tensor(observed, dtype=torch.float32)
    torch.testing.assert_close(synthetic.stokes, target, rtol=3e-7, atol=1e-8)
    prediction = (target * 0.9).requires_grad_()
    losses, gradients = [], []
    for data in (target, synthetic.stokes):
        _, loss = weighted_stokes_loss(
            StokesObjective(type="mse"), prediction, data,
            wavelength_weights=torch.ones(6), stokes_weights=torch.tensor([1., 20., 20., 20.]),
        )
        losses.append(loss)
        gradients.append(torch.autograd.grad(loss, prediction)[0])
    torch.testing.assert_close(losses[0], losses[1], rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=3e-6, atol=1e-9)


def test_synthetic_and_degraded_aia_share_calibrated_preparation(tmp_path):
    from dataclasses import replace
    from prom3theus.muram.fits import load_aia_fits
    from prom3theus.instruments.aia_euv.preparation import prepare_aia_observation_store
    from prom3theus.observations import ImageObservationStore

    snap = snapshot()
    hmi = synthesize_hmi(snap, progress=False)
    aia = synthesize_aia(snap, progress=False)
    paths = write_aia_fits(tmp_path / "fits", aia, hmi.geometry, time_s=snap.time_s, iteration=10)
    prepare_aia_fits(paths, tmp_path / "synthetic")
    sources = [load_aia_fits(path) for path in paths]
    degraded = [replace(source, intensity=source.intensity * 0.5,
                        degradation_factor=0.5) for source in sources]
    prepare_aia_observation_store(
        tmp_path / "degraded", rasters=degraded, source_files_sha256="a" * 64,
        preparation_dependencies={"test": "known factor at the calibrated preparation boundary"},
    )
    synthetic, _, synthetic_metadata = ImageObservationStore.load_sequence(tmp_path / "synthetic")
    observed, _, observed_metadata = ImageObservationStore.load_sequence(tmp_path / "degraded")
    assert synthetic_metadata["objective_statistics"] == observed_metadata["objective_statistics"]
    for first, second in zip(synthetic, observed, strict=True):
        torch.testing.assert_close(torch.as_tensor(first.intensity[:]), torch.as_tensor(second.intensity[:]))
        torch.testing.assert_close(torch.as_tensor(first.valid_mask[:]), torch.as_tensor(second.valid_mask[:]))
