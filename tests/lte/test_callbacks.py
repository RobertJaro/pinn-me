from types import SimpleNamespace

import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from pme.lte.atmosphere import StratifiedAtmosphereModel
from pme.lte.hinode import load_hinode_raster
from pme.lte.synthesis import LTESynthesizer
from pme.inversion_lte import _build_visualization_callback
from pme.train.lte_callbacks import LTEAtmosphereVisualizationCallback
from pme.train.lte_callbacks import _FIELD_STYLES


def test_parameter_colormaps_match_field_topology_and_magnetic_palette():
    for name in ("temperature", "density", "pressure"):
        assert _FIELD_STYLES[name]["log_norm"] is True
        assert "log" not in _FIELD_STYLES[name]["label"]
    for name in ("v_x", "v_y", "v_z", "b_x", "b_y", "b_los"):
        assert _FIELD_STYLES[name]["signed"] is True
    for name in ("b_x", "b_y", "b_los"):
        assert _FIELD_STYLES[name]["cmap"] == "RdBu_r"
    assert _FIELD_STYLES["b_magnitude"]["cmap"] == "cividis"
    assert _FIELD_STYLES["b_azimuth"]["cmap"] == "twilight"
    assert _FIELD_STYLES["b_azimuth"]["limits"] == (-180.0, 180.0)
    assert _FIELD_STYLES["b_inclination"]["cmap"] == "PiYG"
    assert _FIELD_STYLES["b_inclination"]["limits"] == (0.0, 180.0)


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


def test_atmosphere_callback_writes_depth_maps(synthetic_hinode_files, tmp_path):
    raster = load_hinode_raster(synthetic_hinode_files)
    model = StratifiedAtmosphereModel(
        torch.linspace(-4.0, 1.0, 7),
        **_raster_affine(raster),
        height_mapping_config={
            "gauge_reference_coords": raster.coords[raster.valid_mask],
        },
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
        depth_layer_count=4,
        evaluation_batch_size=2,
        max_map_pixels=4,
        dpi=70,
    )

    paths = callback.render(trainer, module, raster, label="epoch_0001")

    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 1_000 for path in paths)
    assert [path.name for path in paths] == [
        "epoch_0001_parameters.png",
        "epoch_0001_magnetic_field.png",
        "epoch_0001_velocity.png",
        "epoch_0001_tau_mapping.png",
    ]
    assert [call["key"] for call in logger.calls] == [
        "Parameters",
        "Magnetic field",
        "Velocity",
        "Tau mapping",
    ]
    assert all(call["step"] == 17 for call in logger.calls)
    assert all(call["images"][0].endswith(".png") for call in logger.calls)
    assert model.training

    depth_grid = model.log_tau500.detach().numpy()
    selected_q = depth_grid[callback._depth_indices(depth_grid)]
    np.testing.assert_allclose(
        selected_q,
        (-4.0, -7.0 / 3.0, -2.0 / 3.0, 1.0),
        atol=1.0e-6,
    )

    evaluated = callback._evaluate(module, raster)
    mapping_figure = callback._tau_mapping_figure(evaluated, [0, 5], "test")
    assert len(mapping_figure.axes) == 2
    assert mapping_figure.axes[0].get_yscale() == "linear"
    assert mapping_figure.axes[1].get_yscale() == "symlog"
    assert "dimensionless" in mapping_figure.axes[0].get_xlabel()
    assert mapping_figure.axes[0].get_xlim()[0] > mapping_figure.axes[0].get_xlim()[1]
    assert mapping_figure.axes[1].get_xlim()[0] > mapping_figure.axes[1].get_xlim()[1]
    assert np.isfinite(evaluated["profile_fields"]["height_metric"]).all()
    gauge_height = model.height_mapping(
        raster.coords[raster.valid_mask], torch.tensor([0.0])
    )
    assert gauge_height.mean().abs() < 1.0e-5 * gauge_height.abs().max().clamp_min(1.0)


def test_direct_log_tau_callback_omits_mapping_and_plots_y_depth(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    scan_index = int(raster.metadata["scan_indices"][0])
    model = StratifiedAtmosphereModel(
        torch.linspace(-4.0, 1.0, 7),
        coordinate_mode="log_tau",
        **_raster_affine(raster),
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
    callback = LTEAtmosphereVisualizationCallback(
        tmp_path / "tau_visualizations",
        depth_layer_count=4,
        evaluation_batch_size=2,
        max_map_pixels=4,
        dpi=70,
        yz_slice={"enabled": True, "scan_index": scan_index},
    )
    paths = callback.render(
        SimpleNamespace(logger=logger, global_step=0),
        module,
        raster,
        label="initial",
    )
    assert len(paths) == 6
    assert not any("tau_mapping" in path.name for path in paths)
    assert [call["key"] for call in logger.calls[-3:]] == [
        "Parameters Y-tau", "Magnetic field Y-tau", "Velocity Y-tau"
    ]
    evaluated = callback._evaluate_yz_slice(module, raster)
    figure = callback._yz_field_panel_figure(
        evaluated,
        "test",
        field_names=("temperature", "pressure"),
        title="Parameters",
        vertical_coordinate="log_tau500",
    )
    panel_axes = figure.axes[:2]
    assert all(axis.get_ylim()[0] > axis.get_ylim()[1] for axis in panel_axes)
    assert "tau" in figure._supylabel.get_text()

def test_atmosphere_callback_writes_parameter_b_and_v_yz_slices(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    scan_index = int(raster.metadata["scan_indices"][0])
    model = StratifiedAtmosphereModel(
        torch.linspace(-4.0, 1.0, 7),
        **_raster_affine(raster),
        height_mapping_config={
            "gauge_reference_coords": raster.coords[raster.valid_mask],
        },
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
        depth_layer_count=4,
        evaluation_batch_size=2,
        max_map_pixels=4,
        dpi=70,
        yz_slice={"enabled": True, "scan_index": scan_index},
    )

    paths = callback.render(trainer, module, raster, label="epoch_0001")

    assert len(paths) == 10
    assert [path.name for path in paths[4:7]] == [
        "epoch_0001_y_tau_parameters.png",
        "epoch_0001_y_tau_magnetic_field.png",
        "epoch_0001_y_tau_velocity.png",
    ]
    assert [path.name for path in paths[-3:]] == [
        "epoch_0001_yz_parameters.png",
        "epoch_0001_yz_magnetic_field.png",
        "epoch_0001_yz_velocity.png",
    ]
    assert [call["key"] for call in logger.calls[-3:]] == [
        "Parameters YZ",
        "Magnetic field YZ",
        "Velocity YZ",
    ]
    assert all(path.is_file() and path.stat().st_size > 1_000 for path in paths[-3:])

    evaluated = callback._evaluate_yz_slice(module, raster)
    assert evaluated["slice_scan_index"] == scan_index
    assert evaluated["map_fields"]["geometric_height"].shape == (
        raster.spatial_shape[0],
        1,
        model.log_tau500.numel(),
    )
    figure = callback._yz_field_panel_figure(
        evaluated,
        "test",
        field_names=("v_x", "v_y", "v_z", "v_magnitude"),
        title="Velocity field",
    )
    assert len(figure.axes) == 8  # four panels and four colorbars
    assert figure._supxlabel.get_text() == "Solar-Y [Mm]"
    assert "geometric height" in figure._supylabel.get_text()
    assert all(not axis.get_xlabel() for axis in figure.axes[:4])
    assert all(not axis.get_ylabel() for axis in figure.axes[:4])
    assert all(axis.get_aspect() == "auto" for axis in figure.axes[:4])
    assert [axis.get_xlabel() for axis in figure.axes[4:]] == [
        _FIELD_STYLES[name]["label"]
        for name in ("v_x", "v_y", "v_z", "v_magnitude")
    ]
    assert all(not axis.get_title() for axis in figure.axes[4:])

    map_evaluated = callback._evaluate(module, raster)
    mapping_figure = callback._tau_mapping_figure(
        map_evaluated,
        callback._depth_indices(map_evaluated["log_tau500"]),
        "test",
        yz_evaluated=evaluated,
    )
    assert len(mapping_figure.axes) == 6  # four panels and two YZ colorbars
    assert mapping_figure.axes[2].get_xlabel() == "Solar-Y [Mm]"
    assert mapping_figure.axes[3].get_xlabel() == "Solar-Y [Mm]"
    assert "geometric height" in mapping_figure.axes[2].get_ylabel()
    assert mapping_figure.axes[2].get_aspect() == "auto"
    assert mapping_figure.axes[3].get_aspect() == "auto"
    colorbar_labels = [axis.get_xlabel() for axis in mapping_figure.axes[4:]]
    assert any("tau" in value for value in colorbar_labels)
    assert any("km" in value and "dex" in value for value in colorbar_labels)


def test_callback_renders_integrated_maps_ensemble_profiles_and_scatter(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    callback = LTEAtmosphereVisualizationCallback(tmp_path, dpi=70)
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
    assert figure.subfigs[0]._supxlabel.get_text() == "Solar-X [Mm]"
    assert figure.subfigs[0]._supylabel.get_text() == "Solar-Y [Mm]"
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
        max_profile_samples=5,
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
    callback = LTEAtmosphereVisualizationCallback(tmp_path, max_profile_samples=2)
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
        max_profile_samples=5,
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


def test_parameter_and_stokes_maps_share_physical_solar_xy_grid(
    synthetic_hinode_files, tmp_path
):
    raster = load_hinode_raster(synthetic_hinode_files)
    model = StratifiedAtmosphereModel(
        torch.linspace(-4.0, 1.0, 7),
        **_raster_affine(raster),
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
    callback = LTEAtmosphereVisualizationCallback(tmp_path, dpi=70)
    rows = np.asarray((0, 2))
    columns = np.asarray((0, 1))
    evaluated = callback._evaluate(module, raster, rows=rows, columns=columns)
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
    assert parameter_mesh.get_array().shape == stokes_mesh.get_array().shape == (2, 2)
    np.testing.assert_allclose(
        parameter_mesh.get_coordinates(), stokes_mesh.get_coordinates()
    )
    assert parameter_figure._supxlabel.get_text() == "Solar-X [Mm]"
    assert parameter_figure._supylabel.get_text() == "Solar-Y [Mm]"
    assert stokes_figure.subfigs[0]._supxlabel.get_text() == "Solar-X [Mm]"
    assert stokes_figure.subfigs[0]._supylabel.get_text() == "Solar-Y [Mm]"


def test_tau_panels_share_limits_across_the_displayed_stratification(tmp_path):
    callback = LTEAtmosphereVisualizationCallback(tmp_path, dpi=70)
    x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(2, dtype=float))
    # The two displayed surfaces differ by 100 units, and an undisplayed middle
    # surface is much larger. Limits must cover the complete evaluated
    # stratification, not merely the selected display slices.
    field = np.stack((x, 1_000.0 + x, 100.0 + x), axis=-1)
    evaluated = {
        "log_tau500": np.asarray((-4.0, -2.0, 0.0)),
        "map_x_mm": x,
        "map_y_mm": y,
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
    yz_callback = _build_visualization_callback(
        {"enabled": True, "yz_slice": {"enabled": True, "scan_index": 150}},
        tmp_path,
    )
    assert yz_callback.yz_slice_enabled is True
    assert yz_callback.yz_slice_scan_index == 150
