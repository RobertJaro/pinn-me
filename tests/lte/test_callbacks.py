from types import SimpleNamespace

import numpy as np
import pytest
import torch
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from pme.coordinates import (
    cartesian_to_spherical,
)
from pme.lte.atmosphere import StratifiedAtmosphereModel
from pme.lte.hinode import load_hinode_raster
from pme.lte.synthesis import LTESynthesizer
from pme.inversion_lte import _build_visualization_callback
from pme.train.lte_callbacks import LTEAtmosphereVisualizationCallback
from pme.train.lte_callbacks import (
    _FIELD_STYLES,
    _MAGNETIC_FIELDS,
    _VELOCITY_FIELDS,
)


def test_parameter_colormaps_match_field_topology_and_magnetic_palette():
    assert _MAGNETIC_FIELDS == ("b_r", "b_theta", "b_phi")
    assert _VELOCITY_FIELDS == ("v_r", "v_theta", "v_phi")
    for name in ("temperature", "density", "pressure"):
        assert _FIELD_STYLES[name]["log_norm"] is True
        assert "log" not in _FIELD_STYLES[name]["label"]
    for name in (
        "v_r", "v_theta", "v_phi", "v_q", "v_u", "v_toward",
        "v_rotation_toward",
        "b_r", "b_theta", "b_phi", "b_q", "b_u", "b_toward",
    ):
        assert _FIELD_STYLES[name]["signed"] is True
    for name in ("b_r", "b_theta", "b_phi", "b_q", "b_u", "b_toward"):
        assert _FIELD_STYLES[name]["cmap"] == "RdBu_r"
    assert _FIELD_STYLES["b_magnitude"]["cmap"] == "cividis"


class _ImageLogger:
    def __init__(self):
        self.calls = []

    def log_image(self, **kwargs):
        self.calls.append(kwargs)


def _raster_affine(raster):
    affine = raster.metadata["coordinates"]["network_affine"]
    return {
        "spatial_coordinate_center_mm": affine["center_mm"],
        "spatial_coordinate_scale_mm": affine["scale_mm"],
    }


def _raster_scene(raster):
    geometry = raster.metadata["ray_geometry"]
    return {
        "scene_geometry_config": {
            "solar_radius_m": geometry["solar_radius_m"],
            "scene_basis": geometry["scene_basis_rows"],
        }
    }


def test_physical_shell_validation_uses_traced_coordinates_and_height_labels(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 7),
        **_raster_affine(raster),
        **_raster_scene(raster),
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    module = SimpleNamespace(
        atmosphere_model=model,
        synthesizer=LTESynthesizer(line_ids=["FeI_6301.5008", "FeI_6302.4932"]),
    )
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path / "shell_visualizations",
        ray_sampling={"batch_size": 2, "max_pixels": 4},
        slice_sampling={
            "batch_size": 8,
            "layer_count": 4,
            "longitude_points": 4,
            "latitude_points": 4,
            "radial_points": 7,
        },
        dpi=70,
        meridional_slice={"enabled": True, "longitude_deg": 25.0},
    )
    rows = np.asarray((0, 2))
    columns = np.asarray((0, 1))
    evaluated = callback._evaluate_ray_optical_depth(
        module, raster, rows=rows, columns=columns
    )

    assert evaluated["map_x_mm"].shape == (2, 2, 7)
    assert evaluated["map_y_mm"].shape == (2, 2, 7)
    assert evaluated["map_longitude_deg"].shape == (2, 2, 7)
    assert evaluated["map_latitude_deg"].shape == (2, 2, 7)
    np.testing.assert_allclose(
        evaluated["shell_height_levels_m"],
        model.depth_to_height(model.log_tau500).detach().numpy(),
        atol=100.0,
    )
    valid = raster.valid_mask[rows][:, columns]
    coords = raster.coords[rows][:, columns][valid]
    with torch.no_grad():
        _, trace = model.trace_rays(
            coords,
            raster.ray_origin_m[rows][:, columns][valid],
            raster.ray_direction[rows][:, columns][valid],
            model.log_tau500,
        )
    np.testing.assert_allclose(
        evaluated["map_x_mm"][valid.numpy()],
        trace.chart_xy_mm[..., 0].cpu().numpy(),
        atol=1.0e-5,
    )
    assert evaluated["sampling"] == (
        "subsampled observed rays for continuum optical depth only"
    )
    assert set(evaluated["profile_fields"]) == {
        "tau500_ray",
        "geometric_height",
    }
    tau500_ray = evaluated["profile_fields"]["tau500_ray"]
    np.testing.assert_allclose(tau500_ray[:, 0], 0.0)
    assert np.all(np.diff(tau500_ray, axis=-1) >= 0.0)
    tau_figure = callback._tau_figure(evaluated, "test")
    assert len(tau_figure.axes) == 3  # profile, map, and map colorbar
    assert "derived" in tau_figure.axes[0].get_title()
    shell_evaluated = callback._evaluate_shell_layers(
        module,
        raster,
        model.depth_to_height(model.log_tau500[[0, -1]]).detach().cpu().numpy(),
    )
    figure = callback._field_panel_figure(
        shell_evaluated,
        [0, 1],
        "test",
        field_names=("temperature",),
        title="parameters",
    )
    annotations = [text.get_text() for axis in figure.axes[:2] for text in axis.texts]
    expected = 1.0 + (
        model.depth_to_height(model.log_tau500[[0, -1]]) / model.solar_radius_m
    )
    assert annotations == [
        rf"$r={value:.6f}\,R_\odot$" for value in expected.tolist()
    ]
    assert not any("tau" in text.lower() for text in annotations)

    logger = _ImageLogger()
    paths = callback.render(
        SimpleNamespace(logger=logger, global_step=0),
        module,
        raster,
        label="initial",
    )
    assert len(paths) == 7
    assert [path.name for path in paths if "tau" in path.name] == ["initial_tau500.png"]
    assert [call["key"] for call in logger.calls[-3:]] == [
        "Parameters meridional slice",
        "Magnetic field meridional slice",
        "Velocity meridional slice",
    ]


def test_atmosphere_callback_writes_depth_maps(synthetic_hinode_files, tmp_path):
    raster = load_hinode_raster(synthetic_hinode_files)
    model = StratifiedAtmosphereModel(
        torch.linspace(-4.0, 1.0, 7),
        **_raster_affine(raster),
        **_raster_scene(raster),
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    model.train()
    module = SimpleNamespace(
        atmosphere_model=model,
        synthesizer=LTESynthesizer(line_ids=["FeI_6301.5008", "FeI_6302.4932"]),
    )
    logger = _ImageLogger()
    trainer = SimpleNamespace(logger=logger, global_step=17)
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path / "visualizations",
        ray_sampling={"batch_size": 2, "max_pixels": 4},
        slice_sampling={
            "batch_size": 8,
            "layer_count": 4,
            "longitude_points": 4,
            "latitude_points": 4,
            "radial_points": 7,
        },
        dpi=70,
    )

    paths = callback.render(trainer, module, raster, label="epoch_0001")

    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 1_000 for path in paths)
    assert [path.name for path in paths] == [
        "epoch_0001_parameters.png",
        "epoch_0001_magnetic_field.png",
        "epoch_0001_velocity.png",
        "epoch_0001_tau500.png",
    ]
    assert [call["key"] for call in logger.calls] == [
        "Parameters",
        "Magnetic field",
        "Velocity",
        "Optical depth",
    ]
    assert all(call["step"] == 17 for call in logger.calls)
    assert all(call["images"][0].endswith(".png") for call in logger.calls)
    assert model.training

    assert callback.slice_layer_count == 4



def test_atmosphere_callback_writes_parameter_b_and_v_meridional_slices(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    longitude_deg = 25.0
    model = StratifiedAtmosphereModel(
        torch.linspace(-4.0, 1.0, 7),
        **_raster_affine(raster),
        **_raster_scene(raster),
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    module = SimpleNamespace(
        atmosphere_model=model,
        synthesizer=LTESynthesizer(line_ids=["FeI_6301.5008", "FeI_6302.4932"]),
    )
    logger = _ImageLogger()
    trainer = SimpleNamespace(logger=logger, global_step=3)
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path / "visualizations",
        ray_sampling={"batch_size": 2, "max_pixels": 4},
        slice_sampling={
            "batch_size": 8,
            "layer_count": 4,
            "longitude_points": 4,
            "latitude_points": 4,
            "radial_points": 7,
        },
        dpi=70,
        meridional_slice={"enabled": True, "longitude_deg": longitude_deg},
    )

    paths = callback.render(trainer, module, raster, label="epoch_0001")

    assert len(paths) == 7
    assert [path.name for path in paths[-3:]] == [
        "epoch_0001_meridional_parameters.png",
        "epoch_0001_meridional_magnetic_field.png",
        "epoch_0001_meridional_velocity.png",
    ]
    assert [call["key"] for call in logger.calls[-3:]] == [
        "Parameters meridional slice",
        "Magnetic field meridional slice",
        "Velocity meridional slice",
    ]
    assert all(path.is_file() and path.stat().st_size > 1_000 for path in paths[-3:])

    evaluated = callback._evaluate_meridional_slice(module, raster)
    assert evaluated["requested_longitude_deg"] == longitude_deg
    assert evaluated["slice_longitude_deg"] == pytest.approx(longitude_deg)
    assert evaluated["map_fields"]["geometric_height"].shape == (
        4,
        1,
        7,
    )
    assert evaluated["map_fields"]["b_theta"].shape == (4, 1, 7)
    assert evaluated["map_fields"]["b_phi"].shape == (4, 1, 7)
    assert evaluated["map_fields"]["v_theta"].shape == (4, 1, 7)
    assert evaluated["map_fields"]["v_phi"].shape == (4, 1, 7)
    assert evaluated["sampling"] == (
        "explicit constant-Carrington-longitude radial plane"
    )
    assert "tau500_ray" not in evaluated["map_fields"]
    assert np.isfinite(evaluated["map_fields"]["temperature"]).all()
    xy_evaluated = callback._evaluate_shell_layers(
        module,
        raster,
        model.depth_to_height(model.log_tau500).detach().cpu().numpy(),
    )
    assert xy_evaluated["map_fields"]["b_r"].shape == (4, 4, 7)
    assert xy_evaluated["map_fields"]["b_theta"].shape == (4, 4, 7)
    assert xy_evaluated["map_fields"]["b_phi"].shape == (4, 4, 7)
    assert xy_evaluated["map_fields"]["v_theta"].shape == (4, 4, 7)
    assert xy_evaluated["map_fields"]["v_phi"].shape == (4, 4, 7)
    assert "tau500_ray" not in xy_evaluated["map_fields"]
    figure = callback._meridional_field_panel_figure(
        evaluated,
        "test",
        field_names=("v_r",),
        title="Velocity field",
    )
    assert len(figure.axes) == 2  # one panel and one colorbar
    assert figure._supxlabel.get_text() == "Carrington latitude [deg]"
    assert figure._supylabel.get_text() == r"radius $r/R_\odot$"
    assert not figure.axes[0].get_xlabel()
    assert not figure.axes[0].get_ylabel()
    assert figure.axes[0].get_aspect() == "auto"
    assert figure.axes[1].get_xlabel() == _FIELD_STYLES["v_r"]["label"]
    assert not figure.axes[1].get_title()



def test_callback_renders_integrated_maps_ensemble_profiles_and_scatter(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path,
        slice_sampling={
            "longitude_points": 8,
            "latitude_points": 6,
            "radial_points": 7,
        },
        dpi=70,
    )
    pixel_index = torch.nonzero(raster.valid_mask, as_tuple=False)
    reference = raster.stokes[raster.valid_mask]
    callback.on_validation_epoch_start(None, None)
    callback.on_validation_batch_end(
        None,
        SimpleNamespace(
            wavelength_angstrom=raster.wavelength_angstrom,
            wavelength_weights=torch.ones_like(raster.wavelength_angstrom),
        ),
        {
            "stokes_pred": reference * 0.9,
            "stokes_reference": reference,
            "pixel_index": pixel_index,
        },
        None,
        0,
    )
    outputs = callback._local_validation_payload()
    assert outputs is not None
    figure = callback._stokes_validation_figure(
        outputs,
        raster,
        raster.wavelength_angstrom,
        "epoch_0001",
    )
    assert len(figure.axes) == 20  # 16 panels plus four column-shared map colorbars
    panel_titles = [axis.get_title() for axis in figure.axes]
    assert any("ensemble" in title for title in panel_titles)
    assert any("all validation pixels" in title for title in panel_titles)
    assert not any("strongest-polarization" in title for title in panel_titles)
    assert "I_{c,\\,atlas}" in figure.subfigs[1]._supylabel.get_text()
    assert figure.subfigs[0]._supxlabel.get_text() == "Carrington chart X [Mm]"
    assert figure.subfigs[0]._supylabel.get_text() == "Carrington chart Y [Mm]"
    colorbar_axes = figure.axes[16:]
    assert len(colorbar_axes) == 4
    assert all(axis.get_xlabel() for axis in colorbar_axes)
    assert all(not axis.get_title() for axis in colorbar_axes)
    assert "no plot-time normalization" in figure._suptitle.get_text()
    assert "fixed disk-center atlas-$I_c$ units" in figure._suptitle.get_text()
    stokes_i_scatter = figure.axes[12]
    assert stokes_i_scatter.get_xlim()[0] != 0.0
    assert stokes_i_scatter.get_xlim() == stokes_i_scatter.get_ylim()
    for component in range(1, 4):
        assert figure.axes[12 + component].get_xlim()[0] == 0.0
    trainer = SimpleNamespace(logger=_ImageLogger(), global_step=2)
    path = callback._save_figure(
        trainer,
        figure,
        "epoch_0001_stokes_validation.png",
        "Validation/Stokes comparison",
    )
    assert path.is_file() and path.stat().st_size > 1_000


def test_validation_callback_streams_integrals_and_bounds_profile_memory(tmp_path):
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path,
        ray_sampling={"max_profile_samples": 5},
        dpi=70,
    )
    module = SimpleNamespace(
        wavelength_angstrom=torch.linspace(6301.0, 6303.0, 9),
        wavelength_weights=torch.ones(9),
    )
    callback.on_validation_epoch_start(None, None)
    for batch_index in range(3):
        reference = torch.arange(7 * 4 * 9, dtype=torch.float32).reshape(7, 4, 9)
        reference = reference / reference.amax()
        pixel_index = torch.stack(
            (
                torch.arange(7),
                torch.full((7,), batch_index, dtype=torch.long),
            ),
            dim=-1,
        )
        callback.on_validation_batch_end(
            None,
            module,
            {
                "stokes_pred": 0.9 * reference,
                "stokes_reference": reference,
                "pixel_index": pixel_index,
            },
            None,
            batch_index,
        )

    payload = callback._local_validation_payload()
    assert payload is not None
    assert payload["integrated_prediction"].shape == (21, 4)
    assert payload["integrated_reference"].shape == (21, 4)
    assert payload["pixel_index"].shape == (21, 2)
    assert payload["stokes_pred"].shape == (5, 4, 9)
    assert payload["stokes_reference"].shape == (5, 4, 9)


def test_validation_integrals_follow_the_fitted_wavelength_mask(tmp_path):
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path, ray_sampling={"max_profile_samples": 2}
    )
    module = SimpleNamespace(
        wavelength_angstrom=torch.tensor((6300.0, 6301.0, 6302.0, 6303.0)),
        wavelength_weights=torch.tensor((1.0, 1.0, 0.0, 1.0)),
    )
    profiles = torch.ones(1, 4, 4)
    callback.on_validation_epoch_start(None, None)
    callback.on_validation_batch_end(
        None,
        module,
        {
            "stokes_pred": profiles,
            "stokes_reference": profiles,
            "pixel_index": torch.tensor(((0, 0),)),
        },
        None,
        0,
    )
    payload = callback._local_validation_payload()
    # Only the first one-Angstrom interval has both endpoints in the fit mask.
    torch.testing.assert_close(payload["integrated_prediction"], torch.ones(1, 4))
    torch.testing.assert_close(payload["integrated_reference"], torch.ones(1, 4))


def test_validation_callback_skips_collection_between_render_epochs(tmp_path):
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path,
        every_n_epochs=3,
        ray_sampling={"max_profile_samples": 5},
    )
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=False)
    module = SimpleNamespace(
        wavelength_angstrom=torch.linspace(6301.0, 6303.0, 9),
        wavelength_weights=torch.ones(9),
    )
    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(
        trainer,
        module,
        {
            "stokes_pred": torch.zeros(2, 4, 9),
            "stokes_reference": torch.zeros(2, 4, 9),
            "pixel_index": torch.tensor(((0, 0), (0, 1))),
        },
        None,
        0,
    )
    assert callback._local_validation_payload() is None


def test_parameter_maps_use_physical_shell_while_stokes_maps_use_detector_rays(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    model = StratifiedAtmosphereModel(
        torch.linspace(-4.0, 1.0, 7),
        **_raster_affine(raster),
        **_raster_scene(raster),
        model_config={
            "dim": 8,
            "n_layers": 2,
            "encoding_config": {"type": "identity"},
        },
    )
    module = SimpleNamespace(
        atmosphere_model=model,
        synthesizer=LTESynthesizer(line_ids=["FeI_6301.5008", "FeI_6302.4932"]),
    )
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path,
        slice_sampling={
            "longitude_points": 8,
            "latitude_points": 6,
            "radial_points": 7,
        },
        dpi=70,
    )
    rows = np.asarray((0, 2))
    columns = np.asarray((0, 1))
    heights = model.depth_to_height(model.log_tau500[[0]]).detach().cpu().numpy()
    evaluated = callback._evaluate_shell_layers(
        module, raster, heights
    )
    parameter_figure = callback._field_panel_figure(
        evaluated,
        [0],
        "test",
        field_names=("temperature",),
        title="parameters",
    )
    pixel_index = torch.nonzero(raster.valid_mask, as_tuple=False)
    reference = raster.stokes[raster.valid_mask]
    callback.on_validation_epoch_start(None, None)
    callback.on_validation_batch_end(
        None,
        SimpleNamespace(
            wavelength_angstrom=raster.wavelength_angstrom,
            wavelength_weights=torch.ones_like(raster.wavelength_angstrom),
        ),
        {
            "stokes_pred": reference,
            "stokes_reference": reference,
            "pixel_index": pixel_index,
        },
        None,
        0,
    )
    outputs = callback._local_validation_payload()
    assert outputs is not None
    stokes_figure = callback._stokes_validation_figure(
        outputs,
        raster,
        raster.wavelength_angstrom,
        "test",
        rows=rows,
        columns=columns,
    )
    parameter_mesh = parameter_figure.axes[0].collections[0]
    stokes_mesh = stokes_figure.axes[0].collections[0]
    assert parameter_mesh.get_array().shape == (6, 8)
    assert stokes_mesh.get_array().shape == (2, 2)
    assert parameter_figure._supxlabel.get_text() == "Carrington longitude [deg]"
    assert parameter_figure._supylabel.get_text() == "Carrington latitude [deg]"
    assert stokes_figure.subfigs[0]._supxlabel.get_text() == "Carrington chart X [Mm]"
    assert stokes_figure.subfigs[0]._supylabel.get_text() == "Carrington chart Y [Mm]"


def test_tau_panels_share_limits_across_the_displayed_stratification(tmp_path):
    callback = LTEAtmosphereVisualizationCallback(tmp_path, dpi=70)
    x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(2, dtype=float))
    # The two displayed surfaces differ by 100 units, and an undisplayed middle
    # surface is much larger. Limits must cover the complete evaluated
    # stratification, not merely the selected display slices.
    field = np.stack((x, 1_000.0 + x, 100.0 + x), axis=-1)
    evaluated = {
        "log_tau500": np.asarray((-4.0, -2.0, 0.0)),
        "solar_radius_m": 695_700_000.0,
        "shell_height_levels_m": np.asarray((600_000.0, 200_000.0, -50_000.0)),
        "map_x_mm": x,
        "map_y_mm": y,
        "map_longitude_deg": x,
        "map_latitude_deg": y,
        "map_fields": {"temperature": field},
    }

    figure = callback._field_panel_figure(
        evaluated,
        [0, 2],
        "test",
        field_names=("temperature",),
        title="parameters",
    )
    panel_axes = figure.axes[:2]
    first_norm = panel_axes[0].collections[0].norm
    second_norm = panel_axes[1].collections[0].norm
    assert isinstance(first_norm, LogNorm)
    assert first_norm.vmin == second_norm.vmin
    assert first_norm.vmax == second_norm.vmax
    assert first_norm.vmax - first_norm.vmin > 900.0
    assert len(figure.axes) == 3  # two depth panels and one shared colorbar
    assert figure.axes[2].get_xlabel() == _FIELD_STYLES["temperature"]["label"]
    assert not figure.axes[2].get_title()


def test_row_shared_colorbar_remains_vertical_on_the_right(tmp_path):
    callback = LTEAtmosphereVisualizationCallback(tmp_path, dpi=70)
    figure = Figure()
    axes = figure.subplots(1, 2)
    image = axes[0].imshow(np.arange(4).reshape(2, 2))

    colorbar = callback._add_shared_colorbar(
        figure,
        image,
        axes,
        "shared row scale",
        shared_by="row",
    )

    assert colorbar.orientation == "vertical"
    assert colorbar.ax.get_ylabel() == "shared row scale"
    assert not colorbar.ax.get_xlabel()
    assert not colorbar.ax.get_title()


def test_atmosphere_callback_cadence_and_final_snapshot(monkeypatch, tmp_path):
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path,
        every_n_epochs=2,
        include_initial=False,
    )
    calls = []
    monkeypatch.setattr(
        callback,
        "render",
        lambda trainer, module, raster, *, label, **kwargs: calls.append(label),
    )
    monkeypatch.setattr(callback, "_gather_validation_outputs", lambda: {})
    monkeypatch.setattr(callback, "_stokes_validation_figure", lambda *args, **kwargs: object())
    monkeypatch.setattr(callback, "_save_figure", lambda *args, **kwargs: None)
    trainer = SimpleNamespace(
        current_epoch=0,
        global_step=1,
        logger=False,
        datamodule=SimpleNamespace(raster=SimpleNamespace(spatial_shape=(2, 2))),
        is_global_zero=True,
        sanity_checking=False,
    )
    module = SimpleNamespace(
        wavelength_angstrom=torch.arange(2),
        wavelength_exclude_windows_angstrom=(),
    )

    callback.on_validation_end(trainer, module)
    assert calls == []
    trainer.current_epoch = 1
    callback.on_validation_end(trainer, module)
    assert calls == ["epoch_0002"]
    callback.on_fit_end(trainer, module)
    assert calls == ["epoch_0002"]

    trainer.current_epoch = 2
    trainer.global_step = 2
    callback.on_fit_end(trainer, module)
    assert calls == ["epoch_0002", "final_epoch_0002"]


def test_visualization_config_can_be_toggled_and_resolves_relative_output(tmp_path):
    assert (
        _build_visualization_callback(
            {"enabled": False, "every_n_epochs": 10}, tmp_path
        )
        is None
    )
    callback = _build_visualization_callback(
        {"enabled": True, "output_directory": "figures", "every_n_epochs": 3},
        tmp_path,
    )
    assert callback.output_directory == (tmp_path / "figures").resolve()
    assert callback.every_n_epochs == 3
    meridional_callback = _build_visualization_callback(
        {
            "enabled": True,
            "meridional_slice": {"enabled": True, "longitude_deg": 25.0},
        },
        tmp_path,
    )
    assert meridional_callback.meridional_slice_enabled is True
    assert meridional_callback.meridional_slice_longitude_deg == 25.0
