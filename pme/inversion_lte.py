"""Command-line entry point for an isolated Hinode depth-stratified LTE inversion."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    OnExceptionCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from pme.lte.hinode import HinodeLTEDataModule
from pme.lte.resources import validate_resource_bundle
from pme.lte.synthesis import HINODE_SP_FE_LINE_IDS
from pme.train.lte_callbacks import LTEAtmosphereVisualizationCallback
from pme.train.lte_module import LTEModule
from pme.train.util import load_yaml_config


def _resolve_log_tau500(value) -> list[float]:
    """Resolve an explicit sequence or ``{min, max, count}`` grid config."""

    if not isinstance(value, Mapping):
        return [float(item) for item in value]
    unknown = set(value) - {"min", "max", "count"}
    if unknown:
        raise KeyError(f"Unknown atmosphere.log_tau500 options: {sorted(unknown)}")
    minimum = float(value["min"])
    maximum = float(value["max"])
    count = int(value["count"])
    if count < 2 or not maximum > minimum:
        raise ValueError("log_tau500 requires count >= 2 and max > min")
    return torch.linspace(minimum, maximum, count, dtype=torch.float64).tolist()


def _inject_spatial_coordinate_affine(
    atmosphere_config: dict,
    raster_metadata: dict,
) -> dict:
    """Bind both atmosphere networks to the raster's saved physical affine."""

    try:
        affine = raster_metadata["coordinates"]["network_affine"]
        expected = {
            "spatial_coordinate_center_mm": [float(value) for value in affine["center_mm"]],
            "spatial_coordinate_scale_mm": [float(value) for value in affine["scale_mm"]],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Hinode raster metadata lacks a valid physical network-coordinate affine."
        ) from error
    for key, values in expected.items():
        if len(values) != 2 or not torch.isfinite(torch.tensor(values)).all():
            raise ValueError(f"Raster coordinate affine {key!r} must be a finite pair.")
        if key.endswith("scale_mm") and any(value <= 0 for value in values):
            raise ValueError("Raster spatial coordinate scales must be positive.")
        configured = atmosphere_config.get(key)
        if configured is not None:
            configured_tensor = torch.as_tensor(configured, dtype=torch.float64)
            expected_tensor = torch.as_tensor(values, dtype=torch.float64)
            if configured_tensor.shape != (2,) or not torch.allclose(
                configured_tensor,
                expected_tensor,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    f"atmosphere.{key} conflicts with the affine derived from the "
                    "selected Hinode raster. Remove it from YAML or use the exact "
                    f"saved value {values}."
                )
        atmosphere_config[key] = values
    return atmosphere_config


def _inject_stic_thermodynamic_bounds(
    atmosphere_config: dict,
    resource_metadata: dict,
) -> dict:
    """Bind T/P decoder bounds to the verified lookup axes exactly."""

    try:
        lookup_bounds = resource_metadata["stic_lookup_bounds"]
        expected = {
            "temperature_log10_bounds": [
                float(value) for value in lookup_bounds["temperature_log10_k"]
            ],
            "gas_pressure_log10_bounds": [
                float(value) for value in lookup_bounds["gas_pressure_log10_pa"]
            ],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Prepared LTE resources do not expose valid STiC thermodynamic bounds."
        ) from error
    for key, values in expected.items():
        if len(values) != 2 or not values[0] < values[1]:
            raise ValueError(f"Invalid STiC lookup bounds for atmosphere.{key}.")
        configured = atmosphere_config.get(key)
        if configured is not None and list(map(float, configured)) != values:
            raise ValueError(
                f"atmosphere.{key} must match the verified STiC table exactly: "
                f"expected {values}, got {configured}."
            )
        atmosphere_config[key] = values
    return atmosphere_config


def _inject_height_gauge_quadrature(
    atmosphere_config: dict,
    raster,
) -> dict:
    """Bind the mean-height gauge to a fixed, deterministic valid-FOV lattice."""

    height_config = atmosphere_config.setdefault("height_mapping_config", {})
    if not isinstance(height_config, dict):
        raise TypeError("atmosphere.height_mapping_config must be a mapping.")
    if "gauge_reference_coords" in height_config:
        raise ValueError(
            "height_mapping_config.gauge_reference_coords is derived from the selected "
            "Hinode raster and must not be specified manually."
        )
    target_count = int(height_config.pop("gauge_quadrature_points", 1024))
    if target_count < 1:
        raise ValueError("height_mapping_config.gauge_quadrature_points must be positive.")

    valid = raster.valid_mask.detach().cpu()
    height, width = valid.shape
    row_count = min(
        height,
        max(1, round((target_count * height / max(width, 1)) ** 0.5)),
    )
    column_count = min(width, max(1, target_count // row_count))
    rows = torch.linspace(0, height - 1, row_count).round().long().unique()
    columns = torch.linspace(0, width - 1, column_count).round().long().unique()
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    selected = torch.stack((row_grid.reshape(-1), column_grid.reshape(-1)), dim=-1)
    selected = selected[valid[selected[:, 0], selected[:, 1]]]

    if selected.shape[0] < min(target_count, int(valid.sum())):
        chosen_flat = selected[:, 0] * width + selected[:, 1]
        all_valid = torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)
        remaining = all_valid[~torch.isin(all_valid, chosen_flat)]
        required = min(target_count - selected.shape[0], remaining.shape[0])
        if required > 0:
            positions = torch.linspace(0, remaining.shape[0] - 1, required).round().long()
            extra_flat = remaining[positions]
            extra = torch.stack(
                (torch.div(extra_flat, width, rounding_mode="floor"), extra_flat % width),
                dim=-1,
            )
            selected = torch.cat((selected, extra), dim=0)
    if selected.shape[0] > target_count:
        positions = torch.linspace(0, selected.shape[0] - 1, target_count).round().long()
        selected = selected[positions]
    if selected.numel() == 0:
        raise ValueError("The selected Hinode raster has no valid height-gauge pixels.")

    gauge_coords = raster.coords.detach().cpu()[selected[:, 0], selected[:, 1]]
    height_config["gauge_reference_coords"] = gauge_coords.tolist()
    return atmosphere_config


def _resolve_physics_reference(
    physics_config: dict,
    log_tau500,
    resource_metadata: dict,
) -> dict:
    """Inject the verified STiC/FALC gravity and pressure when required."""

    equations = physics_config.get("equations", {})
    requires_gravity = any(
        bool(equations.get(name, {}).get("enabled", False))
        for name in ("hse", "mhs", "momentum")
    )
    requires_pressure = bool(
        equations.get("pressure_boundary", {}).get("enabled", False)
    )
    if not (requires_gravity or requires_pressure):
        return physics_config
    try:
        boundary = resource_metadata["falc_top_boundary"]
        target_log_tau = float(boundary["target_log10_tau500"])
        pressure_pa = float(boundary["gas_pressure_pa"])
        gravity_m_per_s2 = float(boundary["gravity_cm_s2"]) / 100.0
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Configured geometric physics requires a verified "
            "resources.falc_top_boundary optical-depth, "
            "pressure, and gravity tuple."
        ) from error
    if not torch.isfinite(
        torch.tensor((target_log_tau, pressure_pa, gravity_m_per_s2))
    ).all() or pressure_pa <= 0 or gravity_m_per_s2 <= 0:
        raise ValueError(
            "The verified FALC/STiC top boundary and gravity are not finite and positive."
        )
    configured_top = float(log_tau500[0])
    if requires_pressure and configured_top != target_log_tau:
        raise ValueError(
            "The configured atmosphere top does not match the verified FALC/STiC "
            f"pressure boundary: log_tau500={configured_top} != {target_log_tau}."
        )
    configured_pressure = physics_config.get("top_pressure_pa")
    if configured_pressure is not None and not torch.isclose(
        torch.tensor(float(configured_pressure), dtype=torch.float64),
        torch.tensor(pressure_pa, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "training.physics_config.top_pressure_pa conflicts with the checksum-verified "
            f"FALC/STiC boundary ({pressure_pa} Pa). Remove the YAML override."
        )
    configured_gravity = physics_config.get("gravity_m_per_s2")
    if configured_gravity is not None and not torch.isclose(
        torch.tensor(float(configured_gravity), dtype=torch.float64),
        torch.tensor(gravity_m_per_s2, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "training.physics_config.gravity_m_per_s2 conflicts with the checksum-"
            "verified FALC gravity "
            f"({gravity_m_per_s2} m s^-2). Remove the YAML override."
        )
    if requires_pressure:
        physics_config["top_pressure_pa"] = pressure_pa
    if requires_gravity:
        physics_config["gravity_m_per_s2"] = gravity_m_per_s2
    return physics_config


def _resolve_physics_activation(
    physics_config: dict,
    coordinate_mode: str = "geometric_height",
) -> tuple[bool, bool]:
    """Validate physics equations against the chosen depth coordinate."""

    equations = physics_config.get("equations", {})
    tau_mapping_enabled = bool(
        equations.get("tau_mapping", {}).get("enabled", False)
    )
    if coordinate_mode == "geometric_height" and not tau_mapping_enabled:
        raise ValueError(
            "Hinode LTE inversion requires "
            "training.physics_config.equations.tau_mapping.enabled=true: F is "
            "evaluated through "
            "the learned Z mapping, so an unconstrained height metric is not "
            "physically identifiable."
        )
    if coordinate_mode == "log_tau":
        unsupported = {
            name
            for name in (
                "tau_mapping", "divergence_b", "mhs", "continuity",
                "induction", "momentum",
            )
            if bool(equations.get(name, {}).get("enabled", False))
        }
        if unsupported:
            raise ValueError(
                "Direct log_tau inversion cannot enable geometric spatial physics: "
                f"{sorted(unsupported)}."
            )
    volume_enabled = any(
        bool(equations.get(name, {}).get("enabled", False))
        for name in (
            "hse", "tau_mapping", "divergence_b", "mhs", "continuity",
            "induction", "momentum",
        )
    )
    boundary_enabled = bool(
        equations.get("pressure_boundary", {}).get("enabled", False)
    )
    return volume_enabled, boundary_enabled


def _build_logger(logging_config: dict, work_directory: Path):
    config = deepcopy(logging_config)
    logger_type = str(config.pop("type", "csv")).lower()
    if logger_type in ("none", "false", "disabled"):
        return False
    if logger_type == "csv":
        name = config.pop("name", "lte")
        return CSVLogger(save_dir=str(work_directory), name=name, **config)
    if logger_type == "wandb":
        return WandbLogger(save_dir=str(work_directory), **config)
    raise ValueError("logging.type must be 'csv', 'wandb', or 'none'.")


def _build_visualization_callback(visualization_config: dict, work_directory: Path):
    config = deepcopy(visualization_config)
    if not bool(config.pop("enabled", True)):
        return None
    output_directory = Path(
        config.pop("output_directory", work_directory / "atmosphere_visualizations")
    ).expanduser()
    if not output_directory.is_absolute():
        output_directory = work_directory / output_directory
    return LTEAtmosphereVisualizationCallback(output_directory, **config)


def run(config: dict):
    """Build and fit an LTE inversion from an already parsed configuration."""

    config = deepcopy(config)
    allowed_top_level = {
        "base_path",
        "work_directory",
        "seed",
        "data",
        "resources",
        "atmosphere",
        "synthesis",
        "instrument",
        "training",
        "normalization",
        "stokes_loss",
        "weight",
        "check_val_every_n_epoch",
        "visualization",
        "trainer",
        "logging",
    }
    unknown_top_level = set(config) - allowed_top_level
    if unknown_top_level:
        raise KeyError(f"Unknown LTE configuration sections: {sorted(unknown_top_level)}")
    # A seed is optional.  Omit it for a genuinely fresh random network
    # initialization; specify it only when an exactly reproducible realization
    # is required for a controlled comparison or regression run.
    if config.get("seed") is not None:
        seed_everything(int(config["seed"]), workers=True)
    base_path = Path(config["base_path"]).expanduser().resolve()
    work_directory = Path(config.get("work_directory", base_path)).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)

    resources_config = deepcopy(config.get("resources", {}))
    unknown_resources = set(resources_config) - {"directory", "instrument"}
    if unknown_resources:
        raise KeyError(f"Unknown LTE resource options: {sorted(unknown_resources)}")
    if "directory" not in resources_config:
        raise KeyError(
            "Configuration must define resources.directory for a prepared offline bundle."
        )
    resource_metadata = validate_resource_bundle(resources_config["directory"])
    expected_instrument = resources_config.get("instrument")
    if (
        expected_instrument is not None
        and resource_metadata["instrument"] != expected_instrument
    ):
        raise ValueError(
            "Prepared LTE bundle instrument does not match resources.instrument: "
            f"{resource_metadata['instrument']!r} != {expected_instrument!r}."
        )
    resource_directory = resource_metadata["directory"]

    data_config = deepcopy(config["data"])
    data_type = data_config.pop("type", "hinode_lte")
    if data_type not in ("hinode_lte", "hinode-lte"):
        raise ValueError(f"LTE entry point only accepts Hinode LTE data, got {data_type!r}.")
    configured_calibration_directory = data_config.get("calibration_data_directory")
    if configured_calibration_directory is not None and Path(
        configured_calibration_directory
    ).expanduser().resolve() != Path(resource_directory):
        raise ValueError(
            "data.calibration_data_directory conflicts with resources.directory."
        )
    data_config["calibration_data_directory"] = resource_directory
    data_module = HinodeLTEDataModule(**data_config)
    # Loading here makes the exact observed wavelength and data provenance
    # available when the synthesis module and checkpoint metadata are created.
    data_module.setup("fit")

    atmosphere_config = deepcopy(config.get("atmosphere", {}))
    try:
        log_tau500 = _resolve_log_tau500(atmosphere_config.pop("log_tau500"))
    except KeyError as error:
        raise KeyError("Configuration must define atmosphere.log_tau500.") from error
    _inject_spatial_coordinate_affine(atmosphere_config, data_module.raster.metadata)
    _inject_stic_thermodynamic_bounds(atmosphere_config, resource_metadata)
    coordinate_mode = str(atmosphere_config.get("coordinate_mode", "log_tau")).lower()
    atmosphere_config["coordinate_mode"] = coordinate_mode
    if coordinate_mode == "geometric_height":
        _inject_height_gauge_quadrature(atmosphere_config, data_module.raster)

    training_config = deepcopy(config.get("training", {}))
    physics_config = deepcopy(training_config.get("physics_config", {}))
    _resolve_physics_reference(physics_config, log_tau500, resource_metadata)
    volume_enabled, boundary_enabled = _resolve_physics_activation(
        physics_config, coordinate_mode
    )
    if volume_enabled or boundary_enabled:
        boundary_points_per_step = (
            int(physics_config.get("boundary_points_per_step", 64))
            if boundary_enabled
            else 0
        )
        physics_config["boundary_points_per_step"] = boundary_points_per_step
        data_module.configure_physics_sampling(
            log_tau500,
            volume_points_per_step=int(physics_config.get("volume_points_per_step", 256)),
            points_per_tau=int(physics_config.get("points_per_tau", 16)),
            boundary_points_per_step=boundary_points_per_step,
            top_pressure_pa=(
                float(physics_config["top_pressure_pa"])
                if boundary_enabled
                else None
            ),
        )
    if "physics_config" in training_config:
        module_physics_config = deepcopy(physics_config)
        training_config["physics_config"] = module_physics_config
    synthesizer_config = deepcopy(config.get("synthesis", {}))
    configured_line_ids = tuple(
        synthesizer_config.setdefault("line_ids", list(HINODE_SP_FE_LINE_IDS))
    )
    if len(set(configured_line_ids)) != len(configured_line_ids):
        raise ValueError("synthesis.line_ids must not contain duplicate transitions.")
    missing_line_ids = set(HINODE_SP_FE_LINE_IDS) - set(configured_line_ids)
    if missing_line_ids:
        raise ValueError(
            "Hinode/SOT-SP LTE inversion must synthesize both Fe I lines; "
            f"missing {sorted(missing_line_ids)}."
        )
    instrument_config = deepcopy(config.get("instrument", {}))
    for target, key in (
        (synthesizer_config, "atomic_data_directory"),
        (instrument_config, "data_directory"),
    ):
        configured = target.get(key)
        if configured is not None and Path(configured).expanduser().resolve() != Path(
            resource_directory
        ):
            raise ValueError(
                f"{key} conflicts with the shared resources.directory. Configure it only once."
            )
        target[key] = resource_directory
    checkpoint_metadata = data_module.checkpoint_metadata()
    checkpoint_metadata["lte_resources"] = resource_metadata
    module = LTEModule(
        log_tau500=log_tau500,
        wavelength_angstrom=data_module.wavelength_angstrom,
        atmosphere_config=atmosphere_config,
        synthesizer_config=synthesizer_config,
        instrument_config=instrument_config,
        normalization_config=deepcopy(config.get("normalization", {})),
        stokes_loss_config=deepcopy(config.get("stokes_loss", {})),
        weight_config=deepcopy(config.get("weight", {})),
        continuum_indices=data_module.normalization_metadata["indices"],
        atlas_continuum_radiance_w_m3_sr=data_module.normalization_metadata[
            "radiometric_calibration"
        ]["atlas_disk_center_continuum_radiance_w_m3_sr"],
        checkpoint_metadata=checkpoint_metadata,
        **training_config,
    )
    selected_lines = {line.id: line for line in module.synthesizer.lines}
    observed_min = float(data_module.wavelength_angstrom.min())
    observed_max = float(data_module.wavelength_angstrom.max())
    for line_id in HINODE_SP_FE_LINE_IDS:
        line = selected_lines.get(line_id)
        if line is None:
            raise RuntimeError(f"Required Hinode/SP line {line_id!r} was not constructed.")
        if not observed_min < line.wavelength_air_angstrom < observed_max:
            raise ValueError(
                f"Observed wavelength grid [{observed_min}, {observed_max}] Angstrom "
                f"does not contain required line {line_id} at "
                f"{line.wavelength_air_angstrom} Angstrom."
            )
    validation_cadence = int(config.get("check_val_every_n_epoch", 1))
    if validation_cadence < 1:
        raise ValueError("check_val_every_n_epoch must be positive.")
    checkpoint = ModelCheckpoint(
        dirpath=str(base_path),
        filename="lte-{epoch:04d}-{step}",
        monitor="valid.loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=validation_cadence,
    )
    callbacks = [
        checkpoint,
        OnExceptionCheckpoint(dirpath=str(base_path), filename="on_exception"),
    ]
    visualization_callback = _build_visualization_callback(
        config.get("visualization", {}), work_directory
    )
    if visualization_callback is not None:
        callbacks.append(visualization_callback)
    trainer_config = deepcopy(config.get("trainer", {}))
    resume_checkpoint = trainer_config.pop("resume_from_checkpoint", None)
    # Hardware selection belongs to Lightning: use the best available
    # accelerator and every device it discovers. LTE synthesis always runs in
    # true float32 so atomic and instrument buffers share the compute dtype.
    trainer_config.pop("accelerator", None)
    trainer_config.pop("devices", None)
    trainer_config.pop("precision", None)
    module.float()
    inference_mode = bool(trainer_config.pop("inference_mode", False))
    derivative_equations = {
        "hse", "divergence_b", "mhs", "continuity", "induction", "momentum"
    }
    if inference_mode and any(
        bool(physics_config.get("equations", {}).get(name, {}).get("enabled", False))
        for name in derivative_equations
    ):
        raise ValueError(
            "trainer.inference_mode must be false when validation evaluates "
            "derivative-based physics equations."
        )
    logger = _build_logger(config.get("logging", {}), work_directory)
    if logger is not False:
        logger.log_hyperparams(config)
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=int(trainer_config.pop("max_epochs", 50)),
        accelerator="auto",
        devices="auto",
        precision="32-true",
        gradient_clip_val=float(trainer_config.pop("gradient_clip_val", 0.5)),
        num_sanity_val_steps=int(trainer_config.pop("num_sanity_val_steps", 0)),
        log_every_n_steps=int(trainer_config.pop("log_every_n_steps", 1)),
        check_val_every_n_epoch=validation_cadence,
        deterministic=trainer_config.pop("deterministic", False),
        inference_mode=inference_mode,
        **trainer_config,
    )
    if resume_checkpoint is None:
        # Omit ckpt_path entirely. Passing an explicit None allows Lightning's
        # checkpoint connector to reuse a previously resolved checkpoint in
        # some repeated-fit contexts, which can silently revive an
        # on_exception checkpoint from the failed optimizer step.
        trainer.fit(module, datamodule=data_module)
    else:
        trainer.fit(module, datamodule=data_module, ckpt_path=resume_checkpoint)
    trainer.save_checkpoint(str(base_path / "inversion_lte.ckpt"))
    return module, data_module, trainer


def main(argv=None):
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(
        description="Invert Hinode/SOT-SP Stokes spectra with a stratified LTE atmosphere."
    )
    parser.add_argument("--config", required=True, help="LTE inversion YAML configuration")
    args, overwrite_args = parser.parse_known_args(argv)
    config = load_yaml_config(args.config, overwrite_args)
    run(config)


if __name__ == "__main__":
    main()
