"""PyTorch-Lightning integration for depth-stratified LTE inversion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from pme.lte.atmosphere import StratifiedAtmosphereModel
from pme.lte.instrument import HinodeSpectralPSF
from pme.lte.synthesis import LTESynthesizer
from pme.model import NormalizationModule
from pme.train.lte_physics import LTEPhysicsModule, LTEPhysicsResult, VOLUME_EQUATIONS
from pme.train.physics import PhysicsWeightSchedule
from pme.train.stokes_loss import StokesLossModule


class LTEModule(LightningModule):
    """Compose the atmosphere network, LTE synthesis, and Hinode response."""

    stokes_names = ("I", "Q", "U", "V")

    def __init__(
        self,
        log_tau500,
        wavelength_angstrom,
        atmosphere_config=None,
        synthesizer_config=None,
        instrument_config=None,
        normalization_config=None,
        stokes_loss_config=None,
        weight_config=None,
        wavelength_weights=None,
        wavelength_exclude_windows_angstrom=None,
        continuum_indices=None,
        atlas_continuum_radiance_w_m3_sr: float = 3.06e13,
        depth_sampling_config=None,
        physics_config=None,
        lr_params=None,
        checkpoint_metadata=None,
    ):
        super().__init__()
        log_tau500 = torch.as_tensor(log_tau500)
        if not log_tau500.is_floating_point():
            log_tau500 = log_tau500.to(dtype=torch.float32)
        wavelength_angstrom = torch.as_tensor(wavelength_angstrom)
        if not wavelength_angstrom.is_floating_point():
            wavelength_angstrom = wavelength_angstrom.to(dtype=torch.float32)
        if log_tau500.ndim != 1 or log_tau500.numel() < 2:
            raise ValueError(
                "log_tau500 must be one-dimensional with at least two points."
            )
        if not torch.all(log_tau500[1:] > log_tau500[:-1]):
            raise ValueError("log_tau500 must increase from top to bottom.")
        if wavelength_angstrom.ndim != 1 or wavelength_angstrom.numel() < 2:
            raise ValueError(
                "wavelength_angstrom must be a one-dimensional observed grid."
            )
        if not torch.all(wavelength_angstrom[1:] > wavelength_angstrom[:-1]):
            raise ValueError("wavelength_angstrom must be strictly increasing.")
        atmosphere_config = deepcopy(atmosphere_config or {})
        synthesizer_config = deepcopy(synthesizer_config or {})
        instrument_config = deepcopy(instrument_config or {})
        self.atmosphere_model = StratifiedAtmosphereModel(
            log_tau500=log_tau500,
            **atmosphere_config,
        )
        depth_sampling = deepcopy(depth_sampling_config or {})
        self.training_depth_sample_count = int(
            depth_sampling.pop("sample_count", log_tau500.numel())
        )
        if self.training_depth_sample_count < 2:
            raise ValueError("Depth sample_count must be at least two.")
        if depth_sampling:
            raise TypeError(f"Unknown depth-sampling options: {sorted(depth_sampling)}")
        self.synthesizer = LTESynthesizer(
            # The atmosphere is a continuous coordinate model.  Training uses
            # a newly jittered stratified quadrature grid for every batch, so
            # the forward solver must accept the grid carried by the atmosphere.
            log_tau500=None,
            **synthesizer_config,
        )
        continuum = getattr(self.synthesizer, "continuum_opacity", None)
        self.instrument = HinodeSpectralPSF(**instrument_config)
        self.register_buffer("wavelength_angstrom", wavelength_angstrom)
        continuum_index_tensor = (
            torch.arange(wavelength_angstrom.numel(), dtype=torch.long)
            if continuum_indices is None
            else torch.as_tensor(continuum_indices, dtype=torch.long)
        )
        if continuum_index_tensor.ndim != 1 or continuum_index_tensor.numel() < 1:
            raise ValueError("continuum_indices must be a non-empty one-dimensional sequence.")
        if torch.any(continuum_index_tensor < 0) or torch.any(
            continuum_index_tensor >= wavelength_angstrom.numel()
        ):
            raise ValueError("continuum_indices contains an out-of-range wavelength index.")
        if torch.unique(continuum_index_tensor).numel() != continuum_index_tensor.numel():
            raise ValueError("continuum_indices must not contain duplicates.")
        self.register_buffer("continuum_indices", continuum_index_tensor)
        # One immutable physical radiance unit keeps Planck synthesis near
        # unity without fitting or estimating a continuum scale during training.
        atlas_continuum_radiance_w_m3_sr = float(
            atlas_continuum_radiance_w_m3_sr
        )
        if (
            not math.isfinite(atlas_continuum_radiance_w_m3_sr)
            or atlas_continuum_radiance_w_m3_sr <= 0.0
        ):
            raise ValueError(
                "atlas_continuum_radiance_w_m3_sr must be finite and positive."
            )
        self.register_buffer(
            "atlas_continuum_radiance_w_m3_sr",
            wavelength_angstrom.new_tensor(atlas_continuum_radiance_w_m3_sr),
        )
        normalization_config = deepcopy(normalization_config or {})
        stokes_loss_config = deepcopy(stokes_loss_config or {})
        self.normalization = NormalizationModule(**normalization_config)
        self.stokes_loss = StokesLossModule(**stokes_loss_config)
        components = ("I", "Q", "U", "V")
        weight_config = deepcopy(weight_config or {})
        unknown_weights = set(weight_config) - set(components)
        if unknown_weights:
            raise KeyError(
                f"Unknown Stokes weight components: {sorted(unknown_weights)}"
            )
        self.stokes_weight_schedules = {
            component: PhysicsWeightSchedule.from_config(
                weight_config.get(component, 1.0)
            )
            for component in components
        }
        self.stokes_weight_config = {
            component: schedule.configuration()
            for component, schedule in self.stokes_weight_schedules.items()
        }
        resolved_weights = {
            component: schedule.value_at(0)
            for component, schedule in self.stokes_weight_schedules.items()
        }
        weights = torch.tensor(
            [resolved_weights[component] for component in components],
            dtype=torch.float32,
        )
        if (
            weights.shape != (4,)
            or not torch.isfinite(weights).all()
            or torch.any(weights < 0)
        ):
            raise ValueError("Stokes weights must be finite and non-negative.")
        if not any(
            schedule.start > 0 or schedule.end > 0
            for schedule in self.stokes_weight_schedules.values()
        ):
            raise ValueError("At least one Stokes weight schedule must be nonzero.")
        self.register_buffer("stokes_weights", weights)

        if wavelength_weights is None:
            spectral_weights = torch.ones_like(wavelength_angstrom)
        else:
            spectral_weights = torch.as_tensor(
                wavelength_weights,
                dtype=wavelength_angstrom.dtype,
                device=wavelength_angstrom.device,
            )
            if spectral_weights.shape != wavelength_angstrom.shape:
                raise ValueError(
                    "wavelength_weights must have the same shape as wavelength_angstrom."
                )
        if not torch.isfinite(spectral_weights).all() or torch.any(
            spectral_weights < 0
        ):
            raise ValueError("wavelength_weights must be finite and non-negative.")

        exclusion_windows = []
        for window in wavelength_exclude_windows_angstrom or ():
            if len(window) != 2:
                raise ValueError(
                    "Each wavelength exclusion window must contain [minimum, maximum]."
                )
            minimum, maximum = map(float, window)
            if (
                not torch.isfinite(torch.tensor((minimum, maximum))).all()
                or minimum > maximum
            ):
                raise ValueError(
                    "Wavelength exclusion windows must contain finite, ordered bounds."
                )
            exclusion_windows.append([minimum, maximum])
            excluded = (wavelength_angstrom >= minimum) & (
                wavelength_angstrom <= maximum
            )
            spectral_weights = spectral_weights.masked_fill(excluded, 0.0)
        if not torch.any(spectral_weights > 0):
            raise ValueError(
                "At least one wavelength must have positive objective weight."
            )
        self.register_buffer("wavelength_weights", spectral_weights)
        self.wavelength_exclude_windows_angstrom = exclusion_windows

        lr_params = deepcopy(
            lr_params
            or {
                "start": 3e-4,
                "end": 3e-5,
                "iterations": 10_000,
            }
        )
        self.lr_start = float(lr_params["start"])
        self.lr_end = float(lr_params.get("end", self.lr_start))
        raw_lr_iterations = lr_params.get("iterations", 1)
        self.lr_iterations = (
            None
            if raw_lr_iterations is None or str(raw_lr_iterations).lower() == "auto"
            else int(raw_lr_iterations)
        )
        if (
            self.lr_start <= 0
            or self.lr_end <= 0
            or (self.lr_iterations is not None and self.lr_iterations < 1)
        ):
            raise ValueError(
                "Learning-rate start/end must be positive and iterations must be "
                "positive or 'auto'."
            )
        self.resolved_lr_iterations: int | None = None

        self.checkpoint_metadata = deepcopy(checkpoint_metadata or {})

        physics = deepcopy(physics_config or {})
        reference_top_pressure = getattr(continuum, "reference_top_pressure_pa", None)
        reference_gravity = getattr(continuum, "reference_gravity_m_per_s2", None)
        equations = deepcopy(physics.pop("equations", {}))
        configured_top_pressure = physics.pop("top_pressure_pa", reference_top_pressure)
        configured_gravity = physics.pop("gravity_m_per_s2", reference_gravity)
        self.top_pressure_pa = (
            None if configured_top_pressure is None else float(configured_top_pressure)
        )
        self.gravity_m_per_s2 = (
            None if configured_gravity is None else float(configured_gravity)
        )
        self.physics_volume_points_per_step = int(
            physics.pop("volume_points_per_step", 256)
        )
        self.physics_points_per_tau = int(physics.pop("points_per_tau", 16))
        self.pressure_boundary_points_per_step = int(
            physics.pop("boundary_points_per_step", 64)
        )
        self.physics_validation_depth_points = int(
            physics.pop("validation_depth_points", 65)
        )
        vector_basis_matches = bool(
            physics.pop("vector_basis_matches_spatial_coordinates", False)
        )
        physics_normalization = physics.pop("normalization", None)
        if physics:
            raise TypeError(f"Unknown physics options: {sorted(physics)}")
        self.physics = LTEPhysicsModule(
            equations,
            gravity_m_per_s2=self.gravity_m_per_s2,
            top_pressure_pa=self.top_pressure_pa,
            vector_basis_matches_spatial_coordinates=vector_basis_matches,
            normalization=physics_normalization,
        )
        self.pressure_boundary_enabled = self.physics.enabled["pressure_boundary"]
        if self.top_pressure_pa is not None and self.top_pressure_pa <= 0:
            raise ValueError("HSE top pressure must be positive.")
        if self.gravity_m_per_s2 is not None and self.gravity_m_per_s2 <= 0:
            raise ValueError("HSE gravity must be positive.")
        if self.physics_volume_points_per_step < 1:
            raise ValueError("Physics volume sample count must be positive.")
        if self.physics_points_per_tau < 1:
            raise ValueError("Physics points_per_tau must be positive.")
        if self.physics_volume_points_per_step % self.physics_points_per_tau:
            raise ValueError(
                "Physics volume_points_per_step must be divisible by points_per_tau."
            )
        if self.pressure_boundary_points_per_step < 0 or (
            self.pressure_boundary_enabled
            and self.pressure_boundary_points_per_step < 1
        ):
            raise ValueError(
                "Boundary sample count must be non-negative and positive when the "
                "pressure-boundary equation is enabled."
            )
        if self.physics_validation_depth_points < 2:
            raise ValueError(
                "Physics validation requires at least two points per component."
            )
        physical_height_validator = getattr(
            continuum, "validate_physical_height_contract", None
        )
        if self.pressure_boundary_enabled and callable(physical_height_validator):
            physical_height_validator(
                float(self.atmosphere_model.log_tau500[0]),
                self.top_pressure_pa,
            )


        # Make Lightning checkpoints reconstructible without relying on a
        # separately copied YAML file.  Module state still contains the exact
        # tensors and is authoritative.
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
                    None if wavelength_weights is None else spectral_weights.tolist()
                ),
                "wavelength_exclude_windows_angstrom": exclusion_windows,
                "continuum_indices": continuum_index_tensor.tolist(),
                "atlas_continuum_radiance_w_m3_sr": float(
                    self.atlas_continuum_radiance_w_m3_sr.detach().cpu()
                ),
                "depth_sampling_config": {
                    "sample_count": self.training_depth_sample_count,
                },
                "physics_config": {
                    **self.physics.configuration(),
                    "top_pressure_pa": self.top_pressure_pa,
                    "gravity_m_per_s2": self.gravity_m_per_s2,
                    "volume_points_per_step": self.physics_volume_points_per_step,
                    "points_per_tau": self.physics_points_per_tau,
                    "boundary_points_per_step": self.pressure_boundary_points_per_step,
                    "validation_depth_points": self.physics_validation_depth_points,
                },
                "lr_params": {
                    "start": self.lr_start,
                    "end": self.lr_end,
                    "iterations": "auto"
                    if self.lr_iterations is None
                    else self.lr_iterations,
                },
                "checkpoint_metadata": self.checkpoint_metadata,
            }
        )

    @property
    def synthesis_wavelength_base(self) -> torch.Tensor:
        """Uniform synthesis grid in the module's current dtype and device.

        Regenerating this inexpensive grid avoids retaining float32 wavelength
        quantization when Lightning converts the module to float64 for the LTE
        hydrostatic-equilibrium calculation.
        """

        return self.instrument.synthesis_grid(self.wavelength_angstrom)

    def sample_depth_grid(
        self,
        randomize: bool = True,
    ) -> torch.Tensor:
        """Draw one ordered depth sample per stratum of the reference grid.

        The domain endpoints remain fixed so every formal solution covers the
        complete atmosphere. Interior points start on a linearly spaced
        log-tau grid. Each interior point receives an independent uniform shift
        bounded by half the base-grid spacing, so it cannot cross the midpoint
        to either neighbor. The resulting grid is passed unchanged to the
        formal solver.
        """

        reference = self.atmosphere_model.log_tau500
        if not randomize:
            return reference
        count = self.training_depth_sample_count
        base = torch.linspace(
            reference[0],
            reference[-1],
            count,
            dtype=reference.dtype,
            device=reference.device,
        )
        if count <= 2:
            return base
        interior = base[1:-1]
        maximum_shift = 0.5 * (base[1] - base[0])
        random_shift = (2.0 * torch.rand_like(interior) - 1.0) * maximum_shift
        sampled = interior + random_shift
        return torch.cat((base[:1], sampled, base[-1:]))

    def physics_validation_grid(self) -> torch.Tensor:
        """Deterministic linear log-tau grid independent of RT quadrature nodes."""

        reference = self.atmosphere_model.log_tau500
        return torch.linspace(
            reference[0],
            reference[-1],
            self.physics_validation_depth_points,
            dtype=reference.dtype,
            device=reference.device,
        )

    def synthesize(
        self,
        coords: torch.Tensor,
        mu: torch.Tensor,
        *,
        randomize_depth: bool = False,
        depth_grid: torch.Tensor | None = None,
        return_physics_diagnostics: bool = False,
    ) -> dict:
        if depth_grid is None:
            depth_grid = self.sample_depth_grid(randomize=randomize_depth)
        else:
            depth_grid = torch.as_tensor(
                depth_grid,
                dtype=self.atmosphere_model.log_tau500.dtype,
                device=self.atmosphere_model.log_tau500.device,
            )
            if (
                depth_grid.ndim != 1
                or depth_grid.numel() < 2
                or not torch.all(depth_grid[1:] > depth_grid[:-1])
            ):
                raise ValueError("An explicit synthesis depth_grid must be ordered and one-dimensional.")
        sampled_atmosphere = self.atmosphere_model(coords, log_tau500=depth_grid)
        synthesis_wavelength = self.synthesis_wavelength_base
        synthesis_kwargs = {}
        if return_physics_diagnostics:
            synthesis_kwargs["return_diagnostics"] = True
        synthesis_result = self.synthesizer(
            sampled_atmosphere,
            synthesis_wavelength,
            mu,
            radiance_scale=self.atlas_continuum_radiance_w_m3_sr,
            **synthesis_kwargs,
        )
        if return_physics_diagnostics:
            high_resolution_stokes, physics_diagnostics = synthesis_result
        else:
            high_resolution_stokes = synthesis_result
            physics_diagnostics = None
        sampled_stokes = self.instrument(
            high_resolution_stokes,
            synthesis_wavelength,
            self.wavelength_angstrom,
        )
        atmosphere = sampled_atmosphere
        predicted_continuum = sampled_stokes[..., 0, :].index_select(
            -1, self.continuum_indices.to(sampled_stokes.device)
        ).mean()
        if not torch.isfinite(predicted_continuum) or predicted_continuum <= 0:
            raise FloatingPointError(
                "The batch-mean synthesized Stokes-I continuum must be finite "
                "and positive."
            )
        return {
            "atmosphere": atmosphere,
            "stokes": sampled_stokes,
            "predicted_continuum": predicted_continuum,
            "physics_diagnostics": physics_diagnostics,
        }

    def forward(self, coords: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return self.synthesize(coords, mu)["stokes"]

    def _stokes_objective(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        component_weights: torch.Tensor | None = None,
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
        weights = (
            self.stokes_weights
            if component_weights is None
            else component_weights.to(component_mse)
        )
        total = torch.dot(component_mse, weights)
        return component_mse, total

    def _current_stokes_weights(self, *, final: bool) -> torch.Tensor:
        step = int(getattr(self, "global_step", 0))
        values = [
            schedule.end if final else schedule.value_at(step)
            for schedule in self.stokes_weight_schedules.values()
        ]
        return self.stokes_weights.new_tensor(values)

    def _physics_grid_samples(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.physics_validation_grid().to(coords)
        sample_count = coords.shape[0]
        return (
            coords[:, None, :].expand(sample_count, q.numel(), 3).reshape(-1, 3),
            q[None, :].expand(sample_count, q.numel()).reshape(-1),
        )

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        *,
        return_callback_payload: bool = False,
        depth_grid: torch.Tensor | None = None,
        log_metrics: bool = True,
    ):
        physics_volume_batch = None
        physics_boundary_batch = None
        if stage == "train" and "physics_volume" in batch:
            physics_volume_batch = batch["physics_volume"]
            physics_boundary_batch = batch.get("pressure_boundary")
            batch = batch["stokes"]
        result = self.synthesize(
            batch["coords"],
            batch["mu"],
            randomize_depth=stage == "train",
            depth_grid=depth_grid,
            return_physics_diagnostics=False,
        )
        # Validation reports the objective at the current optimizer step, not
        # a future end-of-schedule objective.
        final_weights = False
        stokes_weights = self._current_stokes_weights(final=final_weights)
        if stage == "train":
            self.stokes_weights.copy_(stokes_weights)
        component_mse, stokes_loss = self._stokes_objective(
            result["stokes"], batch["stokes"], stokes_weights
        )
        step = int(getattr(self, "global_step", 0))
        if self.physics.volume_enabled:
            if physics_volume_batch is None:
                physics_coords, physics_q = self._physics_grid_samples(batch["coords"])
            else:
                physics_coords = physics_volume_batch["coords"]
                physics_q = physics_volume_batch["log_tau500"]
            physics_result = self.physics.volume(
                self.atmosphere_model,
                self.synthesizer.continuum_opacity,
                physics_coords,
                physics_q,
                global_step=step,
                final_weights=final_weights,
                create_graph=stage == "train",
                return_state=False,
            )
        else:
            zero = stokes_loss.new_zeros(())
            physics_result = LTEPhysicsResult(
                losses={name: zero for name in VOLUME_EQUATIONS},
                residual_norms={},
                weights=self.physics.weights(step, final=final_weights),
            )
        if self.pressure_boundary_enabled:
            if stage == "train" and physics_boundary_batch is None:
                raise KeyError(
                    "pressure_boundary training requires an independent "
                    "'pressure_boundary' batch."
                )
            boundary_result = self.physics.pressure_boundary(
                self.atmosphere_model,
                (
                    physics_boundary_batch["coords"]
                    if physics_boundary_batch is not None
                    else batch["coords"]
                ),
                (
                    physics_boundary_batch["log_tau500"]
                    if physics_boundary_batch is not None
                    else self.atmosphere_model.log_tau500[0].expand(
                        batch["coords"].shape[0]
                    )
                ),
                (
                    physics_boundary_batch.get("gas_pressure_pa")
                    if physics_boundary_batch is not None
                    else None
                ),
                global_step=step,
                final_weights=final_weights,
            )
            boundary_loss = boundary_result.losses["pressure_boundary"]
        else:
            boundary_loss = stokes_loss.new_zeros(())
        available_physics_losses = {
            **physics_result.losses,
            "pressure_boundary": boundary_loss,
        }
        physics_losses = {
            name: available_physics_losses[name]
            for name, enabled in self.physics.enabled.items()
            if enabled
        }
        physics_weights = physics_result.weights
        loss = stokes_loss + sum(
            physics_losses[name] * physics_weights[name] for name in physics_losses
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {stage} Stokes objective: {stokes_loss}."
            )
        batch_size = int(batch["coords"].shape[0])
        if stage == "train" and log_metrics:
            # Match the established spherical inversion dashboard: expose the
            # transformed objective for every Stokes component plus all
            # physical objectives on every optimizer step.
            step_metrics = {
                f"train.{name}": component_mse[index]
                for index, name in enumerate(self.stokes_names)
            }
            step_metrics["train.stokes_loss"] = stokes_loss
            step_metrics["train.loss"] = loss
            for name, value in physics_losses.items():
                step_metrics[f"train.{name}"] = value
            self.log_dict(
                step_metrics,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )
        elif stage != "train" and log_metrics:
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
            validation_losses.update(
                {f"valid.{name}": value for name, value in physics_losses.items()}
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

    def on_after_backward(self) -> None:
        """Stop before Adam can turn a non-finite gradient into model state."""

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

    def on_train_batch_start(self, batch, batch_idx) -> None:
        """Reject parameters corrupted by an earlier or resumed optimizer step."""

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

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "valid", return_callback_payload=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_start)
        if self.lr_start == self.lr_end:
            return optimizer
        if self.lr_iterations is None:
            iterations = int(self.trainer.estimated_stepping_batches)
        else:
            iterations = self.lr_iterations
        if iterations < 1:
            raise RuntimeError("Trainer estimated no LTE optimizer steps.")
        self.resolved_lr_iterations = iterations
        gamma = (self.lr_end / self.lr_start) ** (1.0 / iterations)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Drop obsolete fitted scales and restore the configured atlas unit."""

        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            state_dict.pop("log_radiometric_gain", None)
            state_dict.pop("synthesized_radiance_reference", None)
            state_dict["atlas_continuum_radiance_w_m3_sr"] = (
                self.atlas_continuum_radiance_w_m3_sr.detach().cpu().clone()
            )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Keep checkpointing usable with injected/lightweight synthesizers in
        # tests and downstream experiments.  Atomic provenance is optional;
        # the unit/schema metadata remains useful when a synthesizer does not
        # expose an ``atomic`` database.
        atomic = getattr(self.synthesizer, "atomic", None)
        synthesizer_metadata = getattr(self.synthesizer, "metadata", None)
        atomic_lines = [asdict(line) for line in getattr(atomic, "lines", ())]
        continuum = getattr(self.synthesizer, "continuum_opacity", None)
        continuum_metadata = (
            continuum.metadata()
            if callable(getattr(continuum, "metadata", None))
            else None
        )
        production_continuum_metadata = deepcopy(continuum_metadata)
        if isinstance(production_continuum_metadata, dict):
            production_continuum_metadata.pop("hminus_coefficient_sha256", None)
            production_continuum_metadata.pop("hminus_diagnostic_provenance", None)
        physics_sampling = deepcopy(self.checkpoint_metadata.get("physics_sampling"))
        checkpoint["lte_metadata"] = {
            "schema_version": 28,
            "compute_dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
            "units": {
                "wavelength": "standard-air angstrom at data/instrument boundary",
                "wavelength_internal": "vacuum angstrom for frequency, Planck, and opacity",
                "log_tau500": "log10 of dimensionless vertical continuum optical depth at 5000 vacuum angstrom",
                "mu": "dimensionless ray cosine",
                "stokes": (
                    "observations and synthesis are expressed in fixed disk-center "
                    "atlas-continuum units I_c,atlas(mu=1); one quiet-Sun-derived "
                    "detector-to-radiance scalar is applied uniformly to I,Q,U,V"
                ),
                "temperature": "K",
                "velocity_field": (
                    "m/s, observer Stokes frame [vx, vy, vz]; +vz points toward "
                    "observer and radiative-transfer redshift velocity is -vz"
                ),
                "magnetic_field": (
                    "gauss, observer Stokes frame [Bx(+Q reference), "
                    "By(increasing azimuth), B_los(positive toward observer)]"
                ),
                "microturbulence": "m/s",
                "gas_pressure": "Pa",
                "geometric_height": (
                    "m, increasing upward; the fixed valid-FOV quadrature has "
                    "mean z=0 at log_tau500=0 while individual columns remain corrugated"
                ),
                "spatial_coordinates": (
                    "[time_hours, helioprojective Solar-X Mm, helioprojective "
                    "Solar-Y Mm]; the static model ignores time"
                ),
                "mass_density": "kg/m^3 from the pinned STiC/Wittmann lookup",
                "number_density": "m^-3",
                "continuum_extinction": "m^-1",
                "continuum_mass_extinction": "m^2/kg",
            },
            "atmosphere_parameterization": self.atmosphere_model.reference_metadata(),
            "instrument": self.instrument.metadata(),
            "synthesizer": (
                synthesizer_metadata()
                if callable(synthesizer_metadata)
                else {
                    "class": type(self.synthesizer).__name__,
                }
            ),
            "atomic_data": {
                "lines": atomic_lines,
                "line_sources": deepcopy(getattr(atomic, "line_sources", {})),
                "line_field_conventions": deepcopy(
                    getattr(atomic, "line_conventions", {})
                ),
                "element_metadata_sources": deepcopy(
                    getattr(atomic, "abundance_sources", {})
                ),
                "element_metadata_role": (
                    "atomic masses supply production Doppler widths; the stored "
                    "abundances and ionization energies support reference-only "
                    "EOS diagnostics and do not set production Fe populations"
                ),
                "verified_production_sha256": {
                    key: value
                    for key, value in deepcopy(
                        getattr(atomic, "verified_resource_sha256", {})
                    ).items()
                    if key in ("lines", "abundances")
                },
                "continuum_model": production_continuum_metadata,
            },
            "optional_reference_diagnostics": {
                "partition_source": str(
                    getattr(
                        getattr(atomic, "_partition_table", None),
                        "source",
                        "not loaded",
                    )
                ),
                "hminus_coefficient_sha256": getattr(
                    continuum, "coefficient_sha256", None
                ),
                "hminus_provenance": deepcopy(getattr(continuum, "provenance", None)),
            },
            "continuum_normalization": {
                "type": "absolute atlas radiometry in one fixed numerical unit",
                "indices": self.continuum_indices.detach().cpu().tolist(),
                "operation": (
                    "the loader applies one quiet-Sun detector-to-radiance calibration "
                    "uniformly to observed I,Q,U,V; observed and synthesized profiles "
                    "are expressed in the immutable disk-center atlas-continuum unit"
                ),
                "atlas_continuum_radiance_w_m3_sr": float(
                    self.atlas_continuum_radiance_w_m3_sr.detach().cpu()
                ),
                "learned_gain": None,
                "raster_continuum_normalization": False,
            },
            "stokes_objective": {
                "loss": self.stokes_loss.configuration(),
                "normalization": {
                    "asinh_alphas": (
                        None
                        if self.normalization.asinh_alphas is None
                        else self.normalization.asinh_alphas.detach()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    ),
                },
                "weight": deepcopy(self.stokes_weight_config),
            },
            "depth_sampling": {
                "training_grid": (
                    "linear log10(tau_500) strata with fixed endpoint and adjacent-"
                    "midpoint bounds, independently uniform-sampled within each "
                    "interior stratum, then evaluated by the continuous atmosphere"
                ),
                "evaluation_grid": "configured deterministic reference grid",
                "sample_count": int(self.atmosphere_model.log_tau500.numel()),
                "training_sample_count": self.training_depth_sample_count,
                "domain": self.atmosphere_model.log_tau500[[0, -1]]
                .detach()
                .cpu()
                .tolist(),
                "strata": (
                    "linear equally spaced log_tau500 centers with fixed adjacent-"
                    "midpoint bounds; independent uniform interior draws and fixed endpoints"
                ),
                "integration": (
                    "actual geometric-height line elements from the learned "
                    "Z(x,y,log_tau500) mapping at every realized nonuniform "
                    "optical-depth sample"
                    if self.atmosphere_model.coordinate_mode == "geometric_height"
                    else "actual line elements from successive realized nonuniform "
                    "delta(tau_500) intervals"
                ),
            },
            "physics": {
                **self.physics.configuration(),
                "equation_definitions": {
                    "hse": (
                        "[dP/dlog10(tau500) - ln(10)*tau500*rho*g/alpha500] / mean_xy(P at fixed tau500) = 0"
                        if self.atmosphere_model.coordinate_mode == "log_tau"
                        else "[dP/dlog10(tau500) - rho*g*(-dz/dlog10(tau500))] / mean_xy(P) = 0"
                    ),
                    "tau_mapping": ("alpha500*(-dz/dlog10(tau500)) = ln(10)*tau500"),
                    "pressure_boundary": "log10(Pgas/Ptop) = 0 at q_top",
                    "divergence_b": "div(B) = 0",
                    "mhs": ("grad(P) - rho*g - curl(B)xB/(4*pi) = 0 (Gaussian cgs)"),
                    "continuity": "div(rho*v) = 0 (stationary)",
                    "induction": "curl(v cross B) = 0 (stationary ideal MHD)",
                    "momentum": (
                        "rho*(v dot grad)v + grad(P) - rho*g - "
                        "curl(B)xB/(4*pi) = 0 (stationary ideal MHD, Gaussian cgs)"
                    ),
                },
                "coordinate": (
                    "[Solar-X Mm, Solar-Y Mm, log10(tau500)]"
                    if self.atmosphere_model.coordinate_mode == "log_tau"
                    else "geometric position [x_m,y_m,z_m]"
                ),
                "unit_contract": (
                    "direct-tau HSE compares separately asinh-scaled dimensionless "
                    "log-pressure derivatives assembled from SI P, rho, g, and alpha500"
                    if self.atmosphere_model.coordinate_mode == "log_tau"
                    else "geometric HSE differentiates P directly, transforms it "
                    "with the learned tau-height metric, and normalizes only by "
                    "sampled tau-surface mean pressure; other "
                    "differential physics uses SI except B in gauss and is "
                    "nondimensionalized from configured L0, t0, and B0"
                ),
                "density_source": "pinned STiC/Wittmann differentiable lookup",
                "iterative_forward_solve": False,
                "derivative_strategy": (
                    "one log-pressure/log-tau derivative for direct-tau HSE"
                    if self.atmosphere_model.coordinate_mode == "log_tau"
                    else "one selectively populated primitive Jacobian, including "
                    "direct pressure rather than log-pressure derivatives, shared "
                    "by all active equations; algebraic product rules reuse it"
                ),
                "volume_sampling": (
                    "independent uniform Solar-X, Solar-Y, and log_tau500 samples "
                    "over the configured coordinate bounds"
                ),
                "boundary_sampling": (
                    "independent valid-raster spatial samples on the top face"
                    if self.pressure_boundary_enabled
                    else "none"
                ),
                "sampling_contract": physics_sampling,
                "top_pressure_pa": self.top_pressure_pa,
                "gravity_m_per_s2": self.gravity_m_per_s2,
                "volume_points_per_step": self.physics_volume_points_per_step,
                "points_per_tau": self.physics_points_per_tau,
                "tau_surfaces_per_step": (
                    self.physics_volume_points_per_step // self.physics_points_per_tau
                ),
                "boundary_points_per_step": self.pressure_boundary_points_per_step,
                "validation_grid": "deterministic linear log_tau grid",
                "validation_depth_points": self.physics_validation_depth_points,
                "optical_depth_mapping": {
                    "equation": "alpha500*dz/dlog10(tau500) + ln(10)*tau500 = 0",
                    "training_residual": "squared log10 ratio at random collocation points",
                    "validation_residual": "same pointwise differential equation",
                    "height_model": (
                        self.atmosphere_model.height_mapping.metadata()
                        if self.atmosphere_model.height_mapping is not None
                        else None
                    ),
                },
            },
            "wavelength_objective": {
                "weights": self.wavelength_weights.detach().cpu().tolist(),
                "exclude_windows_angstrom": deepcopy(
                    self.wavelength_exclude_windows_angstrom
                ),
            },
            "optimization": {
                "optimizer": "Adam",
                "lr_start": self.lr_start,
                "lr_end": self.lr_end,
                "lr_iterations": (
                    "auto" if self.lr_iterations is None else self.lr_iterations
                ),
            },
            "velocity_zero_point": (
                "relative solar velocity in the fixed Hinode Level-1 gauge: "
                "sp_prep removed DOP_RCV and registered the slit-averaged Fe I "
                "6301.5 centre"
            ),
            "data": deepcopy(self.checkpoint_metadata),
        }
        if self.atmosphere_model.coordinate_mode == "log_tau":
            checkpoint["lte_metadata"]["units"].pop("geometric_height", None)
            checkpoint["lte_metadata"]["physics"]["optical_depth_mapping"] = None
