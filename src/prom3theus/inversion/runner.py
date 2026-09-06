"""Application runner for strict, versioned LTE inversion configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from prom3theus import __version__
from prom3theus.artifacts.checkpoint import p3s_context
from prom3theus.config import InversionConfig
from prom3theus.instruments import resolve_instrument_config
from prom3theus.inversion.constraints import BOUNDARY_EQUATIONS
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.observations import ObservationStore, load_or_prepare_observation
from prom3theus.resources import validate_resource_bundle
from prom3theus.training.lightning import LTEInversionModule, P3S_CONTEXT_KEY


class P3SSaveStateCallback(Callback):
    """Write one compact, atomic post-validation PROM3THEUS save state."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._last_saved_step: int | None = None

    def _save(self, trainer: Trainer, module: LTEInversionModule) -> None:
        if trainer.is_global_zero:
            context = module.save_state_context
            if context is None:
                raise RuntimeError("P3S saving requires a configured context.")
            parameters = {
                name: parameter.detach()
                for name, parameter in module.named_parameters()
            }
            nonfinite = [
                name
                for name, value in parameters.items()
                if (value.is_floating_point() or value.is_complex())
                and not torch.isfinite(value).all()
            ]
            if nonfinite:
                raise RuntimeError(
                    "Refusing to save non-finite P3S parameters: "
                    + ", ".join(nonfinite[:12])
                )
            payload = {
                P3S_CONTEXT_KEY: deepcopy(context),
                "parameters": parameters,
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
            }
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            try:
                torch.save(payload, temporary)
                os.replace(temporary, self.path)
                self._last_saved_step = int(trainer.global_step)
            finally:
                temporary.unlink(missing_ok=True)
        trainer.strategy.barrier()

    def on_validation_end(
        self, trainer: Trainer, pl_module: LTEInversionModule
    ) -> None:
        if not trainer.sanity_checking:
            self._save(trainer, pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LTEInversionModule) -> None:
        if self._last_saved_step != int(trainer.global_step):
            self._save(trainer, pl_module)


@dataclass(frozen=True, slots=True)
class InversionRun:
    """Objects and durable paths produced by one completed training run."""

    module: LTEInversionModule
    data_module: Any
    trainer: Trainer
    save_state_path: Path
    lightning_checkpoint_path: Path


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


def _observation_times(data_module: Any) -> list[dict[str, Any]]:
    """Extract only the time-selection metadata required by P3SLoader."""

    records = []
    for raster in data_module.rasters:
        times = raster.metadata.get("times")
        coordinates = raster.metadata.get("coordinates")
        if (
            not isinstance(times, list)
            or not times
            or any(not isinstance(value, str) or not value for value in times)
            or not isinstance(coordinates, Mapping)
        ):
            raise ValueError("Every observation raster must declare explicit times.")
        # Hinode DATE_OBS values are UTC and predate the explicit time_scale
        # metadata added for HMI TAI observations.
        scale = str(coordinates.get("time_scale", "utc")).lower()
        if scale not in {"tai", "utc"}:
            raise ValueError("Every observation raster time_scale must be TAI or UTC.")
        records.append({"values": list(times), "scale": scale})
    return records


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

    observation_store = Path(data_module.observation_store_path).resolve()
    store_manifest = ObservationStore.manifest(observation_store)
    save_state_path = (output_directory / "state.p3s").resolve()
    lightning_checkpoint_path = (output_directory / "last.ckpt").resolve()
    module.set_save_state_context(
        p3s_context(
            package_version=__version__,
            resolved_config=config.to_dict(),
            resources=resource_contract,
            observation_spec=observation.metadata(),
            source_signature=store_manifest["source_signature"],
            raster_names=data_module.raster_names,
            validation_raster_index=data_module.validation_raster_index,
            times=_observation_times(data_module),
            bounds=data_module.observation_sampling_bounds,
            model=dict(module.hparams),
        )
    )

    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=output_directory,
            filename="last",
            monitor=None,
            save_top_k=1,
            save_last=False,
            save_on_exception=True,
            save_weights_only=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        ),
        P3SSaveStateCallback(save_state_path),
    ]
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
        enable_checkpointing=True,
    )
    if runtime.validation_check_interval_steps is not None:
        trainer_options["val_check_interval"] = runtime.validation_check_interval_steps
    trainer = Trainer(**trainer_options)
    fit_options = {"datamodule": data_module}
    if lightning_checkpoint_path.is_file():
        fit_options["ckpt_path"] = lightning_checkpoint_path
    trainer.fit(module, **fit_options)

    return InversionRun(
        module=module,
        data_module=data_module,
        trainer=trainer,
        save_state_path=save_state_path,
        lightning_checkpoint_path=lightning_checkpoint_path,
    )


__all__ = ["InversionRun", "run_inversion"]
