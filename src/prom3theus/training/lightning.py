"""PyTorch-Lightning integration for depth-stratified LTE inversion."""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import Mapping

import torch
from pytorch_lightning import LightningModule

from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.instruments import resolve_instrument_config
from prom3theus.observations import resolve_velocity_synthesis_mode
from prom3theus.inversion.forward import (
    ForwardRuntime,
    LTEForwardComposition,
)
from prom3theus.core import (
    CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
    NormalizationModule,
    SPEED_OF_LIGHT,
)
from prom3theus.inversion.constraints.magnetofluid import (
    EQUATION_NAMES,
    PhysicsResult,
    VOLUME_EQUATIONS,
)
from prom3theus.inversion.objective import StokesObjective

from .assembly import build_forward_assembly, build_physics_assembly
from .batch import validate_batch_tensors
from .configuration import (
    resolve_coordinate_grids,
    resolve_depth_sampling,
    resolve_learning_rate,
    resolve_objective_weighting,
    resolve_vector_regularization,
)


class LTEInversionModule(LightningModule):
    """Compose a shared atmosphere, LTE synthesis, and one observation operator.

    The observation boundary is deliberately instrument-neutral. Data modules
    may attach a mapping of batch-local tensors under ``instrument_response``;
    the configured observation operator receives that mapping as keyword
    arguments. This is the single-observation form of the contract that a
    future mixed-data module can invoke once per observation stream.
    """

    stokes_names = ("I", "Q", "U", "V")

    def __init__(
        self,
        *,
        log_tau500,
        wavelength_angstrom,
        atmosphere_config: Mapping,
        synthesizer_config: Mapping,
        instrument_config: Mapping,
        normalization_config: Mapping,
        stokes_loss_config: Mapping,
        weight_config: Mapping,
        wavelength_weights,
        wavelength_exclude_windows_angstrom,
        continuum_indices,
        atlas_continuum_radiance_w_m3_sr: float,
        depth_sampling_config: Mapping,
        physics_config: Mapping,
        learning_rate: Mapping,
        run_metadata: Mapping,
        observation_id: str,
        velocity_synthesis_mode: str,
        instrument_radial_velocity_correction_m_per_s: float,
        vector_regularization_config: Mapping,
    ):
        super().__init__()
        mappings = {
            "atmosphere_config": atmosphere_config,
            "synthesizer_config": synthesizer_config,
            "instrument_config": instrument_config,
            "normalization_config": normalization_config,
            "stokes_loss_config": stokes_loss_config,
            "weight_config": weight_config,
            "depth_sampling_config": depth_sampling_config,
            "physics_config": physics_config,
            "learning_rate": learning_rate,
            "run_metadata": run_metadata,
            "vector_regularization_config": vector_regularization_config,
        }
        invalid_mappings = [
            name for name, value in mappings.items() if not isinstance(value, Mapping)
        ]
        if invalid_mappings:
            raise TypeError(f"LTE runtime mappings are invalid: {invalid_mappings}.")
        grids = resolve_coordinate_grids(log_tau500, wavelength_angstrom)
        log_tau500 = grids.log_tau500
        wavelength_angstrom = grids.wavelength_angstrom
        atmosphere_config = deepcopy(dict(atmosphere_config))
        synthesizer_config = deepcopy(dict(synthesizer_config))
        instrument_config = resolve_instrument_config(
            deepcopy(dict(instrument_config)),
        )
        self.instrument_type = instrument_config["type"]
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be non-empty.")
        self.observation_id = observation_id
        self.velocity_synthesis_mode = resolve_velocity_synthesis_mode(
            velocity_synthesis_mode
        )
        self.atmosphere_model = StratifiedAtmosphereModel(
            log_tau500=log_tau500,
            **atmosphere_config,
        )
        if not isinstance(
            instrument_radial_velocity_correction_m_per_s, Real
        ) or isinstance(instrument_radial_velocity_correction_m_per_s, bool):
            raise TypeError(
                "instrument_radial_velocity_correction_m_per_s must be numeric."
            )
        initial_radial_velocity_correction = float(
            instrument_radial_velocity_correction_m_per_s
        )
        if (
            not math.isfinite(initial_radial_velocity_correction)
            or abs(initial_radial_velocity_correction) >= SPEED_OF_LIGHT
        ):
            raise ValueError(
                "instrument_radial_velocity_correction_m_per_s must be finite "
                "and subluminal."
            )
        # Optimize the instrument zero point in the atmosphere decoder's natural
        # velocity units.  This gives the scalar a gradient scale comparable to
        # the learned velocity field while exposing and applying it in m/s.
        self.instrument_radial_velocity_correction_normalized = torch.nn.Parameter(
            torch.tensor(
                initial_radial_velocity_correction
                / self.atmosphere_model.velocity_scale_m_per_s,
                dtype=torch.float32,
            )
        )
        vector_regularization = resolve_vector_regularization(
            vector_regularization_config
        )
        self.vector_regularization_enabled = vector_regularization.enabled
        self.vector_regularization_magnetic_weight = (
            vector_regularization.magnetic_weight
        )
        self.vector_regularization_velocity_weight = (
            vector_regularization.velocity_weight
        )
        self.vector_regularization_decay_steps = vector_regularization.decay_steps
        self.vector_regularization_configuration = vector_regularization.metadata(
            magnetic_scale_gauss=self.atmosphere_model.magnetic_scale_gauss,
            velocity_scale_m_per_s=(self.atmosphere_model.velocity_scale_m_per_s),
        )
        depth_sampling = resolve_depth_sampling(
            depth_sampling_config,
        )
        self.coarse_depth_sample_count = depth_sampling.sample_count
        self.coarse_to_fine_enabled = depth_sampling.coarse_to_fine_enabled
        self.coarse_to_fine_sample_count = depth_sampling.fine_sample_count
        self.coarse_to_fine_uniform_weight_floor = depth_sampling.uniform_weight_floor
        self.register_buffer(
            "_coarse_depth_grid",
            torch.linspace(
                log_tau500[0],
                log_tau500[-1],
                self.coarse_depth_sample_count,
                dtype=log_tau500.dtype,
                device=log_tau500.device,
            ),
            persistent=False,
        )
        forward = build_forward_assembly(
            atmosphere_model=self.atmosphere_model,
            synthesizer_config=synthesizer_config,
            instrument_config=instrument_config,
            velocity_synthesis_mode=self.velocity_synthesis_mode,
            depth_sampling=depth_sampling,
            wavelength_angstrom=wavelength_angstrom,
        )
        self.synthesizer = forward.synthesizer
        continuum = forward.continuum_opacity
        self.instrument = forward.instrument
        self.register_buffer("wavelength_angstrom", wavelength_angstrom)
        self._forward_composition = forward.composition
        self.register_buffer(
            "_synthesis_wavelength_base",
            forward.synthesis_wavelength_angstrom.detach().clone(),
        )
        self.register_buffer(
            "carrington_angular_velocity_rad_per_s",
            wavelength_angstrom.new_tensor(CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S),
        )
        objective_weighting = resolve_objective_weighting(
            wavelength_angstrom,
            weight_config=weight_config,
            wavelength_weights=wavelength_weights,
            wavelength_exclude_windows_angstrom=(wavelength_exclude_windows_angstrom),
            continuum_indices=continuum_indices,
            atlas_continuum_radiance_w_m3_sr=(atlas_continuum_radiance_w_m3_sr),
        )
        self.register_buffer("continuum_indices", objective_weighting.continuum_indices)
        self.register_buffer(
            "atlas_continuum_radiance_w_m3_sr",
            wavelength_angstrom.new_tensor(
                objective_weighting.atlas_continuum_radiance_w_m3_sr
            ),
        )
        normalization_config = deepcopy(dict(normalization_config))
        stokes_loss_config = deepcopy(dict(stokes_loss_config))
        self.normalization = NormalizationModule(**normalization_config)
        self.stokes_loss = StokesObjective(**stokes_loss_config)
        self.stokes_weight_config = dict(objective_weighting.stokes_weight_config)
        self.register_buffer("stokes_weights", objective_weighting.stokes_weights)
        self.register_buffer(
            "wavelength_weights", objective_weighting.wavelength_weights
        )
        self.wavelength_exclude_windows_angstrom = (
            objective_weighting.wavelength_exclude_windows_angstrom
        )

        learning_rate_settings = resolve_learning_rate(learning_rate)
        self.learning_rate = learning_rate_settings.start
        self.learning_rate_schedule = learning_rate_settings.schedule
        self.learning_rate_configuration = learning_rate_settings.model_configuration
        self.resolved_learning_rate_iterations: int | None = None

        self.run_metadata = deepcopy(dict(run_metadata))

        reference_gravity = getattr(continuum, "reference_gravity_m_per_s2", None)
        physics = build_physics_assembly(
            physics_config,
            reference_gravity_m_per_s2=reference_gravity,
        )
        self.physics = physics.constraints
        self.gravity_m_per_s2 = physics.gravity_m_per_s2
        self.physics_volume_points_per_step = physics.volume_points_per_step
        self.physics_height_layers_per_step = physics.height_layers_per_step
        self.upper_boundary_points_per_step = physics.upper_boundary_points_per_step
        self.physics_validation_height_layers = physics.validation_height_layers
        self.physics_validation_points_per_height = physics.validation_points_per_height
        self.physics_sampling_domain = physics.sampling_domain
        self.register_buffer(
            "physics_validation_position_m",
            physics.validation_position_m,
        )
        self.register_buffer(
            "physics_validation_time_hours",
            physics.validation_time_hours,
        )
        self.register_buffer(
            "physics_validation_boundary_position_m",
            physics.validation_boundary_position_m,
            persistent=False,
        )
        self.register_buffer(
            "physics_validation_boundary_time_hours",
            physics.validation_boundary_time_hours,
            persistent=False,
        )
        # Record the exact constructor inputs used by the tensor-artifact
        # manifest to reconstruct this module.
        self.save_hyperparameters(
            {
                "log_tau500": log_tau500.tolist(),
                "wavelength_angstrom": wavelength_angstrom.tolist(),
                "atmosphere_config": atmosphere_config,
                "synthesizer_config": synthesizer_config,
                "instrument_config": instrument_config,
                "normalization_config": normalization_config,
                "stokes_loss_config": self.stokes_loss.configuration(),
                "weight_config": deepcopy(self.stokes_weight_config),
                "wavelength_weights": (
                    None
                    if wavelength_weights is None
                    else objective_weighting.wavelength_weights.tolist()
                ),
                "wavelength_exclude_windows_angstrom": (
                    objective_weighting.wavelength_exclude_windows_angstrom
                ),
                "continuum_indices": objective_weighting.continuum_indices.tolist(),
                "atlas_continuum_radiance_w_m3_sr": (
                    objective_weighting.atlas_continuum_radiance_w_m3_sr
                ),
                "depth_sampling_config": depth_sampling.model_configuration(),
                "physics_config": physics.model_configuration(),
                "learning_rate": deepcopy(self.learning_rate_configuration),
                "run_metadata": self.run_metadata,
                "observation_id": self.observation_id,
                "velocity_synthesis_mode": self.velocity_synthesis_mode,
                "instrument_radial_velocity_correction_m_per_s": (
                    initial_radial_velocity_correction
                ),
                "vector_regularization_config": deepcopy(vector_regularization_config),
            }
        )
        # Lookup tables are generated and regression-tested at high precision,
        # but the inversion graph has one explicit runtime dtype.
        self.float()

    @property
    def synthesis_wavelength_base(self) -> torch.Tensor:
        """Prepared synthesis grid in the module's current dtype and device."""

        return self._synthesis_wavelength_base

    @property
    def instrument_radial_velocity_correction_m_per_s(self) -> torch.Tensor:
        """Trainable instrument wavelength zero point expressed as radial m/s."""

        return (
            self.instrument_radial_velocity_correction_normalized
            * self.atmosphere_model.velocity_scale_m_per_s
        )

    def _vector_regularization_factor(self) -> float:
        if not self.vector_regularization_enabled:
            return 0.0
        trainer = getattr(self, "_trainer", None)
        step = int(getattr(trainer, "global_step", 0))
        return max(
            0.0,
            1.0 - step / float(self.vector_regularization_decay_steps),
        )

    def _vector_regularization(self, atmosphere) -> tuple[torch.Tensor, float]:
        """Isotropically shrink synthesis-constrained vectors during warm-up."""

        factor = self._vector_regularization_factor()
        loss = atmosphere.magnetic_field.new_zeros(())
        if factor == 0.0:
            return loss, factor
        if self.vector_regularization_magnetic_weight > 0:
            loss = (
                loss
                + self.vector_regularization_magnetic_weight
                * (
                    atmosphere.magnetic_field
                    / self.atmosphere_model.magnetic_scale_gauss
                )
                .square()
                .mean()
            )
        if self.vector_regularization_velocity_weight > 0:
            loss = (
                loss
                + self.vector_regularization_velocity_weight
                * (
                    atmosphere.velocity_field
                    / self.atmosphere_model.velocity_scale_m_per_s
                )
                .square()
                .mean()
            )
        return loss * factor, factor

    def sample_depth_grid(
        self,
        randomize: bool = True,
    ) -> torch.Tensor:
        """Return the configured coarse grid, optionally jittered for training.

        ``sample_count`` controls resolution for every synthesis path,
        including validation and export. Randomization changes only the
        interior locations: each point receives an independent uniform shift
        bounded by 40% of the base-grid spacing, leaving a finite gap between
        neighboring strata, while the domain endpoints stay fixed. Thus
        disabling jitter never falls back to the higher-resolution atmosphere
        representation grid.
        """

        return self._forward_composition.sample_depth_grid(
            self._coarse_depth_grid,
            randomize=randomize,
        )

    @property
    def forward_composition(self) -> LTEForwardComposition:
        """Numerical atmosphere-to-observation composition."""

        return self._forward_composition

    def synthesize(
        self,
        coords: torch.Tensor,
        *,
        ray_direction: torch.Tensor | None = None,
        stokes_basis: torch.Tensor | None = None,
        observer_los_velocity_m_per_s: torch.Tensor | None = None,
        removed_solar_los_velocity_m_per_s: torch.Tensor | None = None,
        randomize_depth: bool = False,
        depth_grid: torch.Tensor | None = None,
        return_details: bool = False,
        instrument_response: Mapping[str, torch.Tensor] | None = None,
    ) -> dict:
        return self._forward_composition.synthesize(
            coords,
            runtime=ForwardRuntime(
                coarse_depth_grid=self._coarse_depth_grid,
                observed_wavelength_angstrom=self.wavelength_angstrom,
                synthesis_wavelength_angstrom=self.synthesis_wavelength_base,
                radiance_scale=self.atlas_continuum_radiance_w_m3_sr,
                carrington_angular_velocity_rad_per_s=(
                    self.carrington_angular_velocity_rad_per_s
                ),
                instrument_radial_velocity_correction_m_per_s=(
                    self.instrument_radial_velocity_correction_m_per_s
                ),
            ),
            ray_direction=ray_direction,
            stokes_basis=stokes_basis,
            observer_los_velocity_m_per_s=observer_los_velocity_m_per_s,
            removed_solar_los_velocity_m_per_s=(removed_solar_los_velocity_m_per_s),
            randomize_depth=randomize_depth,
            depth_grid=depth_grid,
            return_details=return_details,
            instrument_response=instrument_response,
        )

    def forward(
        self,
        coords: torch.Tensor,
        **ray_geometry,
    ) -> torch.Tensor:
        return self.synthesize(coords, **ray_geometry)["stokes"]

    def _stokes_objective(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Predicted and observed Stokes shapes differ: {prediction.shape} != {target.shape}."
            )
        if prediction.ndim < 3 or prediction.shape[-2] != 4:
            raise ValueError("Stokes profiles must have shape [..., 4, wavelength].")
        if not torch.isfinite(prediction).all():
            count = int((~torch.isfinite(prediction)).sum().detach().cpu())
            raise FloatingPointError(
                f"LTE synthesis produced {count} non-finite Stokes values."
            )
        if not torch.isfinite(target).all():
            count = int((~torch.isfinite(target)).sum().detach().cpu())
            raise FloatingPointError(
                f"Observed batch contains {count} non-finite Stokes values."
            )
        spectral_weights = self.wavelength_weights.to(prediction)
        transformed_squared_error = self.stokes_loss(
            prediction,
            target,
            self.normalization,
        )
        weighted_squared_error = transformed_squared_error * spectral_weights
        # Average over every leading sample and normalize by the sum of active
        # wavelength weights.  With unit weights this is exactly the original
        # elementwise mean; excluded samples contribute neither numerator nor
        # denominator.
        leading_dimensions = tuple(range(prediction.ndim - 2))
        component_mse = weighted_squared_error.sum(
            dim=(*leading_dimensions, prediction.ndim - 1)
        )
        leading_count = prediction.numel() // (
            prediction.shape[-2] * prediction.shape[-1]
        )
        component_mse = component_mse / (leading_count * spectral_weights.sum())
        total = torch.dot(component_mse, self.stokes_weights.to(component_mse))
        return component_mse, total

    def _physics_objective(
        self,
        volume_batch,
        upper_boundary_batch,
        *,
        create_graph: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Evaluate every nonzero-weight equation from one collocation state."""

        zero = next(self.atmosphere_model.parameters()).new_zeros(())
        if self.physics.volume_active:
            if volume_batch is None:
                raise KeyError("Active volume physics requires a 'volume' sample.")
            grouped_position = volume_batch["position_m"]
            radial_group_shape = tuple(map(int, grouped_position.shape[:2]))
            physics_result = self.physics.volume(
                self.atmosphere_model,
                self.synthesizer.continuum_opacity,
                grouped_position.reshape(-1, 3),
                volume_batch["time_hours"].reshape(-1, 1),
                create_graph=create_graph,
                return_state=False,
                radial_group_shape=radial_group_shape,
            )
        else:
            physics_result = PhysicsResult(
                losses={name: zero for name in VOLUME_EQUATIONS},
                weights=self.physics.loss_weights,
            )

        boundary_losses = {}
        pressure_prior_name = "upper_boundary_gas_pressure_prior"
        if self.physics.is_active(pressure_prior_name):
            if upper_boundary_batch is None:
                raise KeyError(
                    "Active upper-boundary physics requires an 'upper_boundary' sample."
                )
            with torch.set_grad_enabled(create_graph):
                pressure_prior = self.physics.upper_boundary_gas_pressure_prior(
                    self.atmosphere_model,
                    upper_boundary_batch["position_m"].reshape(-1, 3),
                    upper_boundary_batch["time_hours"].reshape(-1, 1),
                )
            boundary_losses[pressure_prior_name] = pressure_prior.losses[
                pressure_prior_name
            ]

        available = {**physics_result.losses, **boundary_losses}
        losses = {
            name: available[name]
            for name in EQUATION_NAMES
            if self.physics.is_active(name)
        }
        weighted = sum(
            (losses[name] * physics_result.weights[name] for name in losses),
            start=zero,
        )
        return losses, weighted

    def _sample_training_physics(self):
        """Draw one grouped shell state directly on the model device."""

        if not self.physics.any_active:
            return None, None
        if self.physics_sampling_domain is None:
            raise RuntimeError(
                "Active physics requires an initialized spherical sampling domain."
            )
        parameter = next(self.atmosphere_model.parameters())
        volume = None
        if self.physics.volume_active:
            volume = self.physics_sampling_domain.random_grouped(
                self.physics_height_layers_per_step,
                self.physics_volume_points_per_step
                // self.physics_height_layers_per_step,
                device=parameter.device,
                dtype=parameter.dtype,
            )
        upper_boundary = None
        if self.physics.boundary_active:
            grouped = self.physics_sampling_domain.random_top(
                self.upper_boundary_points_per_step,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            upper_boundary = {name: value.squeeze(0) for name, value in grouped.items()}
        return volume, upper_boundary

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        *,
        return_callback_payload: bool = False,
    ):
        validate_batch_tensors(
            batch,
            reference=next(self.atmosphere_model.parameters()),
        )
        required_fields = {
            "coordinates",
            "ray_direction",
            "stokes_basis",
            "stokes",
        }
        if self.velocity_synthesis_mode.value == "carrington_registered_relative":
            required_fields.add("removed_solar_los_velocity_m_per_s")
            forbidden_velocity_field = "observer_los_velocity_m_per_s"
        else:
            required_fields.add("observer_los_velocity_m_per_s")
            forbidden_velocity_field = "removed_solar_los_velocity_m_per_s"
        if self.instrument_type == "hmi_filter_profiles":
            required_fields.add("instrument_response")
        missing_fields = sorted(required_fields - set(batch))
        if missing_fields:
            raise KeyError(f"LTE batch is missing required fields: {missing_fields}.")
        if forbidden_velocity_field in batch:
            raise ValueError(
                "LTE batch contains a velocity field from the wrong synthesis frame: "
                f"{forbidden_velocity_field}."
            )
        physics_volume_batch, upper_boundary_batch = (
            self._sample_training_physics()
            if stage == "train" and self.physics.any_active
            else (None, None)
        )
        batch_observation = batch.get("observation_id")
        if batch_observation is not None:
            identifiers = (
                {batch_observation}
                if isinstance(batch_observation, str)
                else {str(value) for value in batch_observation}
            )
            if identifiers != {self.observation_id}:
                raise ValueError(
                    "LTE batch observation_id does not match the configured operator: "
                    f"{sorted(identifiers)} != {self.observation_id!r}."
                )
        result = self.synthesize(
            batch["coordinates"],
            ray_direction=batch["ray_direction"],
            stokes_basis=batch["stokes_basis"],
            observer_los_velocity_m_per_s=(
                batch["observer_los_velocity_m_per_s"]
                if self.velocity_synthesis_mode.value == "carrington_observer_relative"
                else None
            ),
            removed_solar_los_velocity_m_per_s=(
                batch["removed_solar_los_velocity_m_per_s"]
                if self.velocity_synthesis_mode.value
                == "carrington_registered_relative"
                else None
            ),
            randomize_depth=stage == "train",
            return_details=(stage == "train" and self.vector_regularization_enabled),
            instrument_response=(
                batch["instrument_response"]
                if self.instrument_type == "hmi_filter_profiles"
                else None
            ),
        )
        component_mse, stokes_loss = self._stokes_objective(
            result["stokes"], batch["stokes"]
        )
        if stage == "train":
            if self.physics.any_active:
                physics_losses, physics_loss = self._physics_objective(
                    physics_volume_batch,
                    upper_boundary_batch,
                    create_graph=True,
                )
                loss = stokes_loss + physics_loss
            else:
                physics_losses = {}
                loss = stokes_loss
            vector_regularization, _ = (
                self._vector_regularization(result["atmosphere"])
                if self.vector_regularization_enabled
                else (stokes_loss.new_zeros(()), 0.0)
            )
            loss = loss + vector_regularization
        else:
            physics_losses = {}
            loss = stokes_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"The {stage} LTE objective became non-finite.")
        batch_size = int(batch["coordinates"].shape[0])
        if stage == "train":
            # Match the established spherical inversion dashboard: expose the
            # transformed objective for every Stokes component plus all
            # physical objectives on every optimizer step.
            step_metrics = {
                f"train.{name}": component_mse[index]
                for index, name in enumerate(self.stokes_names)
            }
            step_metrics["train.stokes_loss"] = stokes_loss
            step_metrics["train.loss"] = loss
            step_metrics["train.instrument_radial_velocity_correction_km_s"] = (
                self.instrument_radial_velocity_correction_m_per_s / 1_000.0
            )
            if self.vector_regularization_enabled:
                step_metrics["train.vector_regularization"] = vector_regularization
            for name, value in physics_losses.items():
                step_metrics[f"train.{name}"] = value
            self.log_dict(
                step_metrics,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )
        else:
            validation_losses = {
                f"valid.{name}": component_mse[index]
                for index, name in enumerate(self.stokes_names)
            }
            validation_losses.update(
                {
                    "valid.loss": loss,
                    "valid.stokes_loss": stokes_loss,
                }
            )
            self.log_dict(
                validation_losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
        if return_callback_payload:
            return {
                "loss": loss.detach(),
                "stokes_pred": result["stokes"].detach(),
                "stokes_reference": batch["stokes"].detach(),
                "pixel_index": batch["pixel_index"].detach(),
            }
        return loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "train")

    def on_train_batch_start(self, batch, batch_idx) -> None:
        """Stop before using model parameters that are already non-finite."""

        del batch, batch_idx
        invalid = [
            name
            for name, parameter in self.named_parameters()
            if not torch.isfinite(parameter).all()
        ]
        if invalid:
            raise FloatingPointError(
                "Non-finite LTE parameters at batch start: "
                + ", ".join(invalid[:12])
                + (" ..." if len(invalid) > 12 else "")
            )

    def on_after_backward(self) -> None:
        """Stop before an optimizer can incorporate non-finite gradients."""

        invalid = [
            name
            for name, parameter in self.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if invalid:
            self.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Non-finite LTE gradients before the optimizer step: "
                + ", ".join(invalid[:12])
                + (" ..." if len(invalid) > 12 else "")
            )

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            result = self._shared_step(batch, "valid", return_callback_payload=True)
        trainer = getattr(self, "_trainer", None)
        run_physics_validation = (
            batch_idx == 0
            and self.physics.any_active
            and (trainer is None or trainer.is_global_zero)
        )
        if run_physics_validation:
            volume = None
            if self.physics.volume_active:
                if self.physics_validation_position_m.numel() == 0:
                    raise RuntimeError(
                        "Volume physics validation requires an initialized shell domain."
                    )
                volume = {
                    "position_m": self.physics_validation_position_m,
                    "time_hours": self.physics_validation_time_hours,
                }
            upper_boundary = None
            if self.physics.boundary_active:
                if self.physics_validation_boundary_position_m.numel() == 0:
                    raise RuntimeError(
                        "Boundary physics validation requires an initialized shell domain."
                    )
                upper_boundary = {
                    "position_m": self.physics_validation_boundary_position_m,
                    "time_hours": self.physics_validation_boundary_time_hours,
                }
            with torch.inference_mode(False), torch.enable_grad():
                physics_losses, _ = self._physics_objective(
                    volume,
                    upper_boundary,
                    create_graph=False,
                )
            self.log_dict(
                {
                    f"valid.{name}": value.detach()
                    for name, value in physics_losses.items()
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
                sync_dist=False,
                rank_zero_only=True,
            )
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.learning_rate_schedule is None:
            return optimizer
        configured_iterations = self.learning_rate_schedule["iterations"]
        iterations = (
            int(self.trainer.estimated_stepping_batches)
            if configured_iterations == "auto"
            else int(configured_iterations)
        )
        if iterations < 1:
            raise ValueError("Resolved learning-rate iterations must be positive.")
        self.resolved_learning_rate_iterations = iterations
        gamma = (
            self.learning_rate_schedule["end"] / self.learning_rate_schedule["start"]
        ) ** (1.0 / iterations)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
