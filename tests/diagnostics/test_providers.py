"""The lifecycle logs explicit provider artifacts, never report-shaped guesses."""
from types import SimpleNamespace
from pathlib import Path

from prom3theus.diagnostics import providers


def test_only_declared_artifacts_are_logged_and_names_are_stream_qualified(monkeypatch):
    output = providers.DiagnosticOutput(
        {"input_filename": "not_an_output.png", "summary": "complete"},
        (Path("plot.png"), Path("plot.png")),
    )
    monkeypatch.setattr(providers, "_PROVIDERS", {"probe": lambda *args: output})
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            streams=[
                SimpleNamespace(id=name, data_term=SimpleNamespace(type="probe"))
                for name in ("first", "second")
            ]
        )
    )
    report = providers.render_diagnostics(runtime, None, None, Path("work"))
    assert report["artifacts"] == [
        {"stream_id": "first", "path": "plot.png"},
        {"stream_id": "second", "path": "plot.png"},
    ]
    assert set(report["streams"]) == {"first", "second"}


def test_validation_steps_and_filenames_never_create_new_panels():
    from unittest.mock import Mock
    from prom3theus.application.joint_training import _log_validation

    logger = Mock()
    for step in (100, 200):
        artifacts = [
            {
                "stream_id": stream,
                "path": f"/work/step_{step}/{stream}_{plot}_{step}.png",
                **({"media_key": f"AIA {plot}"} if stream != "hmi" else {}),
            }
            for stream in ("hmi", "aia_first", "aia_second")
            for plot in ("observation comparison", "side view")
        ]
        _log_validation(
            logger, {"metrics": {}, "rendering": {"artifacts": artifacts}}, step
        )
    calls = [call.kwargs for call in logger.log_image.call_args_list]
    assert [call["key"] for call in calls] == [
        "Observation comparison",
        "AIA observation comparison",
        "AIA side view",
    ] * 2
    assert [len(call["images"]) for call in calls] == [2, 2, 2, 2, 2, 2]
    assert all(len(call["images"]) == len(call["caption"]) for call in calls)
    assert [call["step"] for call in calls] == [100, 100, 100, 200, 200, 200]


def test_aia_provider_declares_two_panels_and_keeps_contribution_plots_local(
    monkeypatch,
):
    from unittest.mock import Mock
    from prom3theus.application import joint_rendering
    from prom3theus.application.joint_training import _log_validation

    monkeypatch.setattr(
        joint_rendering,
        "_render_aia_diagnostics",
        lambda *args, **kwargs: {
            "enabled": True,
            "side_views": ["side.png"],
            "report": {
                "streams": {
                    "aia": {
                        "diagnostics": {
                            "paths": ["comparison.png"],
                            "contribution_paths": ["contribution.png"],
                        }
                    }
                }
            },
        },
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            streams=[
                SimpleNamespace(
                    id="aia", data_term=SimpleNamespace(type="aia_optically_thin")
                )
            ]
        )
    )
    report = providers.render_diagnostics(runtime, None, None, Path("work"))
    assert len(report["artifacts"]) == 3
    logger = Mock()
    _log_validation(logger, {"metrics": {}, "rendering": report}, 100)
    assert {
        call.kwargs["key"]: call.kwargs["images"]
        for call in logger.log_image.call_args_list
    } == {
        "AIA observation comparison": ["comparison.png"],
        "AIA side view": ["side.png"],
    }


def test_validation_images_use_real_lightning_wandb_image_conversion(tmp_path):
    """Exercise the SDK boundary; permissive logger mocks miss invalid kwargs."""
    from unittest.mock import Mock

    from PIL import Image
    from pytorch_lightning.loggers import WandbLogger
    import wandb

    from prom3theus.application.joint_training import _log_validation

    experiment = Mock()
    logger = WandbLogger(experiment=experiment, save_dir=str(tmp_path))
    keys = ("AIA observation comparison", "AIA side view")
    for step in (0, 10):
        artifacts = []
        for panel, key in enumerate(keys):
            for stream in ("aia_first", "aia_second"):
                path = tmp_path / f"{stream}_{panel}_{step}.png"
                Image.new("RGB", (8, 8), color="red").save(path)
                artifacts.append(
                    {"stream_id": stream, "path": str(path), "media_key": key}
                )
        _log_validation(
            logger,
            {"metrics": {}, "rendering": {"artifacts": artifacts}},
            step,
        )

    payloads = [call.args[0] for call in experiment.log.call_args_list]
    image_payloads = [payload for payload in payloads if any(key in payload for key in keys)]
    assert len(image_payloads) == 4
    for payload, (step, panel) in zip(image_payloads, ((0, 0), (0, 1), (10, 0), (10, 1))):
        images = payload[keys[panel]]
        assert payload["trainer/global_step"] == step
        assert len(images) == 2
        assert all(isinstance(image, wandb.Image) for image in images)
        assert [image._caption for image in images] == [
            f"{stream}_{panel}_{step}" for stream in ("aia_first", "aia_second")
        ]


def test_stokes_renderer_titles_reach_lifecycle_without_stream_or_step_prefixes(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from matplotlib.figure import Figure
    from prom3theus.application import joint_rendering
    from prom3theus.application.joint_training import _log_validation
    from prom3theus.diagnostics.rendering import AtmosphereRenderer

    renderer = AtmosphereRenderer(tmp_path, dpi=72, evaluator=None)
    trainer = SimpleNamespace(logger=None, global_step=0)
    for name in ("first_0000", "second_0000"):
        renderer._save_figure(trainer, Figure(), f"{name}.png", "Magnetic field")
    paths = tuple(renderer.media_groups["Magnetic field"])
    monkeypatch.setattr(
        joint_rendering, "_render_stokes_diagnostics",
        lambda *args, **kwargs: providers.DiagnosticOutput(
            list(paths), paths, media_groups=renderer.media_groups
        ),
    )
    runtime = SimpleNamespace(config=SimpleNamespace(streams=[
        SimpleNamespace(id="hmi_123", data_term=SimpleNamespace(type="lte_stokes"))
    ]))
    rendering = providers.render_diagnostics(runtime, trainer, None, tmp_path)
    logger = Mock()
    _log_validation(logger, {"metrics": {}, "rendering": rendering}, 0)
    logger.log_image.assert_called_once_with(
        key="Magnetic field", images=list(paths),
        caption=[Path(path).stem for path in paths], step=0,
    )


def test_shared_physics_artifacts_are_logged_once_with_stable_media_keys(monkeypatch, tmp_path):
    from prom3theus.diagnostics import physics_equations
    runtime = SimpleNamespace(model=object(), config=SimpleNamespace(streams=[], diagnostics=object()))
    monkeypatch.setattr(physics_equations, 'render_physics_equations', lambda *args: providers.DiagnosticOutput(
        {'MHS': {'weight': 1e-5}}, ('mhs_step100.png',),
        media_groups={'Physics equation MHS': ('mhs_step100.png',)}))
    result = providers.render_diagnostics(runtime, None, None, tmp_path)
    assert result['artifacts'] == [{'stream_id': 'physics', 'path': 'mhs_step100.png', 'media_key': 'Physics equation MHS'}]
