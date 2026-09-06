"""Application runner for strict, versioned LTE inversion configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from pytorch_lightning import Trainer

from prom3theus import __version__
from prom3theus.artifacts.model import (
    OBSERVATION_DIRECTORY_NAME,
    ArtifactManifest,
    save_artifact,
)
from prom3theus.config import InversionConfig
from prom3theus.instruments import resolve_instrument_config
from prom3theus.inversion.constraints import BOUNDARY_EQUATIONS
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.observations import (
    OBSERVATION_STORE_FORMAT,
    OBSERVATION_STORE_VERSION,
    ObservationStore,
    load_or_prepare_observation,
)
from prom3theus.resources import validate_resource_bundle
from prom3theus.training.lightning import LTEInversionModule


@dataclass(frozen=True, slots=True)
class InversionRun:
    """Objects and durable paths produced by one completed training run."""

    module: LTEInversionModule
    data_module: Any
    trainer: Trainer
    artifact_directory: Path


def _depth_grid(config: InversionConfig) -> list[float]:
    depth = config.atmosphere.depth_grid
    return torch.linspace(
        depth.minimum,
        depth.maximum,
        depth.count,
        dtype=torch.float32,
    ).tolist()


def _atmosphere_config(config: InversionConfig) -> dict[str, Any]:
    atmosphere = config.atmosphere
    geometry = atmosphere.geometry
    parameters = atmosphere.parameters
    network = atmosphere.network
    return {
        "height_input_scale_m": geometry.height_input_scale_m,
        "time_dependent": geometry.time_dependent,
        "shell_height_bounds_Mm": [
            geometry.outer_height_megameter,
            geometry.inner_height_megameter,
        ],
        "line_formation_height_bounds_Mm": [
            geometry.line_formation_outer_height_megameter,
            geometry.inner_height_megameter,
        ],
        "tangent_margin_m": geometry.tangent_margin_m,
        "reference_atmosphere_config": atmosphere.reference_atmosphere,
        "upper_atmosphere_config": (
            None
            if atmosphere.upper_atmosphere is None
            else atmosphere.upper_atmosphere.to_dict()
        ),
        "temperature_log_scale": parameters.temperature.log_scale,
        "velocity_scale_m_per_s": parameters.velocity.scale_m_per_s,
        "velocity_max_m_per_s": parameters.velocity.maximum_m_per_s,
        "magnetic_scale_gauss": parameters.magnetic_field.scale_gauss,
        "microturbulence_log_scale": parameters.microturbulence.log_scale,
        "gas_pressure_log_scale": parameters.gas_pressure.log_scale,
        "model_config": {
            "type": network.type,
            "dim": network.hidden_dimension,
            "n_layers": network.hidden_layers,
            "activation": network.activation,
            "encoding_config": network.encoding.to_dict(),
        },
    }


def _physics_config(config: InversionConfig) -> dict[str, Any]:
    physics = config.training.physics
    return {
        **physics.collocation.to_dict(),
        "vector_basis_matches_spatial_coordinates": (
            physics.vector_basis_matches_spatial_coordinates
        ),
        "upper_boundary_current_free_ramp_steps": (
            physics.upper_boundary_current_free_ramp_steps
        ),
        "adiabatic_index": physics.adiabatic_index,
        "normalization": physics.normalization.to_dict(),
        "equations": physics.equations.to_dict(),
        "gravity_m_per_s2": None,
        "sampling_domain": None,
        "upper_sampling_domain": None,
    }


def _inject_coordinate_contract(
    atmosphere: dict[str, Any], raster_metadata: Mapping[str, Any]
) -> None:
    """Bind neural coordinates to the canonical observation geometry."""

    try:
        coordinates = raster_metadata["coordinates"]
        spatial = coordinates["network_affine"]
        ray_geometry = raster_metadata["ray_geometry"]
        center = [float(value) for value in spatial["center_mm"]]
        scale = [float(value) for value in spatial["scale_mm"]]
        scene = {
            "solar_radius_m": float(ray_geometry["solar_radius_m"]),
            "scene_basis": [
                [float(component) for component in row]
                for row in ray_geometry["scene_basis_rows"]
            ],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Observation metadata does not contain a valid coordinate/scene contract."
        ) from error
    if len(center) != 2 or len(scale) != 2 or any(value <= 0 for value in scale):
        raise ValueError("Observation network affine must contain two positive scales.")
    atmosphere["spatial_coordinate_center_mm"] = center
    atmosphere["spatial_coordinate_scale_mm"] = scale
    atmosphere["scene_geometry_config"] = scene

    if atmosphere["time_dependent"]:
        try:
            temporal = coordinates["time_affine"]
            atmosphere["time_coordinate_center_hours"] = float(temporal["center_hours"])
            atmosphere["time_coordinate_scale_hours"] = float(temporal["scale_hours"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "A time-dependent atmosphere requires the observation time affine."
            ) from error


def _inject_resource_contract(
    atmosphere: dict[str, Any],
    physics: dict[str, Any],
    resources: Mapping[str, Any],
) -> None:
    """Bind resource-derived physical constants to the runtime configuration."""

    del atmosphere

    gravity_equations = (
        "hydrostatic_equilibrium",
        "magnetohydrostatic_equilibrium",
        "momentum",
    )
    if any(physics["equations"][name]["enabled"] for name in gravity_equations):
        try:
            physics["gravity_m_per_s2"] = (
                float(resources["falc_top_boundary"]["gravity_cm_s2"]) / 100.0
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Active force balance requires verified FALC gravity metadata."
            ) from error


def _physics_sampling(
    physics: dict[str, Any],
    data_module: Any,
    atmosphere: Mapping[str, Any],
) -> dict[str, Any] | None:
    equations = physics["equations"]
    volume_names = (
        "hydrostatic_equilibrium",
        "magnetohydrostatic_equilibrium",
        "momentum",
        "magnetic_divergence",
        "radial_magnetic_energy_gradient",
        "induction",
        "continuity",
        "adiabatic_pressure",
    )
    upper_volume_names = (
        "upper_domain_microturbulence_prior",
        "upper_domain_temperature_prior",
    )
    volume_active = any(equations[name]["enabled"] for name in volume_names)
    upper_volume_active = any(equations[name]["enabled"] for name in upper_volume_names)
    boundary_active = any(equations[name]["enabled"] for name in BOUNDARY_EQUATIONS)
    if not volume_active and not upper_volume_active and not boundary_active:
        physics["sampling_domain"] = None
        physics["upper_sampling_domain"] = None
        return None
    bounds = getattr(data_module, "observation_sampling_bounds", None)
    if bounds is None:
        raise TypeError(
            "Active physical constraints require observation sampling bounds."
        )
    domain = SphericalShellDomain.from_observation_bounds(
        bounds,
        atmosphere["shell_height_bounds_Mm"],
    )
    physics["sampling_domain"] = domain.configuration()
    line_outer, _ = atmosphere["line_formation_height_bounds_Mm"]
    full_outer, _ = atmosphere["shell_height_bounds_Mm"]
    upper_domain = None
    if upper_volume_active:
        if not full_outer > line_outer:
            raise ValueError(
                "Active upper-domain physics requires a non-empty height extension."
            )
        upper_domain = domain.with_height_bounds((line_outer, full_outer))
        physics["upper_sampling_domain"] = upper_domain.configuration()
    metadata = {"full_domain": domain.metadata()}
    if upper_domain is not None:
        metadata["upper_domain"] = upper_domain.metadata()
    metadata["line_formation_height_bounds_Mm"] = list(
        atmosphere["line_formation_height_bounds_Mm"]
    )
    return metadata


def _logger(config: InversionConfig):
    logging = config.logging
    if logging.type == "disabled":
        return False
    try:
        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(
            save_dir=str(config.solver.work_directory),
            project=logging.project,
            name=logging.run_name,
            tags=list(logging.tags),
        )
    except ImportError as error:
        raise ImportError(
            "W&B logging requires the 'visualization' optional dependencies."
        ) from error


def _visualization_callback(config: InversionConfig):
    visualization = config.diagnostics.visualization
    if not visualization.enabled:
        return None
    try:
        from prom3theus.training.callbacks import AtmosphereVisualizationCallback
    except ImportError as error:
        raise ImportError(
            "Visualization requires the 'visualization' optional dependencies."
        ) from error
    options = visualization.to_dict()
    options.pop("enabled")
    return AtmosphereVisualizationCallback(
        config.solver.output_directory / "diagnostics",
        **options,
    )


def _instrument_config(
    config: InversionConfig,
    observation: Any,
) -> tuple[dict[str, Any], float, bool]:
    instrument = config.instrument.to_dict()
    line_of_sight_velocity = float(
        instrument.pop("line_of_sight_velocity_correction_m_per_s")
    )
    optimize_line_of_sight_velocity = bool(
        instrument.pop("optimize_line_of_sight_velocity_correction")
    )
    for name, value in observation.instrument_options.items():
        if name in instrument and instrument[name] != value:
            raise ValueError(
                f"Configured instrument option {name!r} conflicts with the observation."
            )
        instrument[name] = value
    return (
        resolve_instrument_config(instrument),
        line_of_sight_velocity,
        optimize_line_of_sight_velocity,
    )


def run_inversion(
    config: InversionConfig,
    *,
    rebuild_observations: bool = False,
) -> InversionRun:
    """Build and fit one LTE inversion from an already validated config."""

    if not isinstance(config, InversionConfig):
        raise TypeError("run_inversion requires a validated InversionConfig.")
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")

    output_directory = config.solver.output_directory
    work_directory = config.solver.work_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory = output_directory / "artifact"
    if artifact_directory.exists():
        raise FileExistsError(
            f"Artifact directory already exists: {artifact_directory}. "
            "Choose a fresh solver.output_directory."
        )

    resources = validate_resource_bundle()
    resource_contract = deepcopy(resources)
    data_module, observation, adapter = load_or_prepare_observation(
        config.observation.to_dict(),
        resources,
        work_directory / "observation-cache",
        rebuild=rebuild_observations,
    )
    raster = data_module.raster
    if raster is None:
        raise RuntimeError("Observation adapter returned no prepared raster.")

    log_tau500 = _depth_grid(config)
    atmosphere = _atmosphere_config(config)
    physics = _physics_config(config)
    _inject_coordinate_contract(atmosphere, raster.metadata)
    _inject_resource_contract(atmosphere, physics, resources)
    physics_sampling = _physics_sampling(physics, data_module, atmosphere)

    line_ids = list(config.synthesis.line_ids)
    missing = set(observation.required_line_ids) - set(line_ids)
    if missing:
        raise ValueError(
            f"Observation {observation.observation_id!r} requires missing lines: "
            f"{sorted(missing)}."
        )
    instrument, line_of_sight_velocity, optimize_line_of_sight_velocity = (
        _instrument_config(
            config,
            observation,
        )
    )
    run_metadata = data_module.run_metadata()
    run_metadata.update(
        {
            "physics_sampling": deepcopy(physics_sampling),
            "resources": deepcopy(resource_contract),
            "observation": observation.metadata(),
            "adapter": adapter.name,
        }
    )

    weights = config.loss.stokes_weights
    module = LTEInversionModule(
        log_tau500=log_tau500,
        wavelength_angstrom=observation.wavelength_angstrom,
        atmosphere_config=atmosphere,
        synthesizer_config={
            "line_ids": line_ids,
        },
        instrument_config=instrument,
        stokes_loss_config={
            "type": config.loss.type,
            "stokes_sigmas": {
                "I": config.loss.stokes_sigmas.i,
                "Q": config.loss.stokes_sigmas.q,
                "U": config.loss.stokes_sigmas.u,
                "V": config.loss.stokes_sigmas.v,
            },
            "huber_delta": config.loss.huber_delta,
        },
        weight_config={
            "I": weights.i,
            "Q": weights.q,
            "U": weights.u,
            "V": weights.v,
        },
        wavelength_weights=None,
        wavelength_exclude_windows_angstrom=(
            config.synthesis.excluded_wavelength_windows_angstrom
        ),
        continuum_indices=observation.continuum_indices,
        atlas_continuum_radiance_w_m3_sr=observation.radiance_scale_w_m3_sr,
        depth_sampling_config=config.training.depth_sampling.to_dict(),
        physics_config=physics,
        learning_rate=config.training.learning_rate.to_dict(),
        run_metadata=run_metadata,
        observation_id=observation.observation_id,
        velocity_synthesis_mode=str(observation.velocity_synthesis_mode),
        instrument_line_of_sight_velocity_correction_m_per_s=line_of_sight_velocity,
        optimize_instrument_line_of_sight_velocity_correction=(
            optimize_line_of_sight_velocity
        ),
        vector_regularization_config=config.training.vector_regularization.to_dict(),
    ).float()

    selected_lines = {line.id: line for line in module.synthesizer.lines}
    for line_id in observation.required_line_ids:
        line = selected_lines[line_id]
        support = observation.line_support_angstrom
        if support is not None and not (
            support[0] < line.wavelength_air_angstrom < support[1]
        ):
            raise ValueError(
                f"Observation support {support} does not contain {line_id} at "
                f"{line.wavelength_air_angstrom} Angstrom."
            )

    callbacks: list[Any] = []
    visualization = _visualization_callback(config)
    if visualization is not None:
        callbacks.append(visualization)

    logger = _logger(config)
    if logger is not False:
        logger.log_hyperparams(config.to_dict())
    runtime = config.runtime
    trainer_options = dict(
        logger=logger,
        callbacks=callbacks,
        max_epochs=runtime.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="32-true",
        gradient_clip_val=runtime.gradient_clip_norm,
        num_sanity_val_steps=0,
        log_every_n_steps=runtime.log_every_n_steps,
        check_val_every_n_epoch=config.diagnostics.validation_every_n_epochs,
        inference_mode=False,
        enable_checkpointing=False,
    )
    if runtime.validation_check_interval_steps is not None:
        trainer_options["val_check_interval"] = runtime.validation_check_interval_steps
    trainer = Trainer(**trainer_options)
    trainer.fit(module, datamodule=data_module)

    observation_store = Path(data_module.observation_store_path).resolve()
    store_manifest = ObservationStore.manifest(observation_store)
    manifest = ArtifactManifest.create(
        package_version=__version__,
        resolved_config=config.to_dict(),
        resources=resource_contract,
        observation={
            "store": {
                "path": OBSERVATION_DIRECTORY_NAME,
                "format": OBSERVATION_STORE_FORMAT,
                "version": OBSERVATION_STORE_VERSION,
                "source_signature": store_manifest["source_signature"],
            },
            "spec": observation.metadata(),
        },
        model=dict(module.hparams),
        provenance={"training_runtime": "pytorch_lightning"},
    )
    save_artifact(
        artifact_directory,
        model=module,
        manifest=manifest,
        observation_store=observation_store,
    )
    return InversionRun(
        module=module,
        data_module=data_module,
        trainer=trainer,
        artifact_directory=artifact_directory,
    )


__all__ = ["InversionRun", "run_inversion"]
