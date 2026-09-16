"""Schema-v3 wrapper for the established LTE Stokes forward objective."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from prom3theus.core import CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S, SPEED_OF_LIGHT
from prom3theus.inversion.assembly import build_forward_assembly
from prom3theus.inversion.configuration import (
    resolve_depth_sampling,
    resolve_objective_weighting,
)
from prom3theus.inversion.forward import ForwardRuntime
from prom3theus.inversion.objective import (
    STOKES_COMPONENTS,
    StokesObjective,
    weighted_stokes_loss,
)
from prom3theus.inversion.data_terms.disambiguation import ObserverPhaseRotation
from prom3theus.observations import resolve_velocity_synthesis_mode

from .base import DataTermBatchResult, ObservationDataTerm


class StokesObservationTerm(ObservationDataTerm):
    """One LTE spectropolarimetric likelihood sharing the joint atmosphere."""

    observation_kind = "stokes"

    def __init__(
        self,
        *,
        atmosphere_model,
        observation_id: str,
        wavelength_angstrom,
        continuum_indices,
        atlas_continuum_radiance_w_m3_sr: float,
        velocity_synthesis_mode: str,
        synthesizer_config: Mapping[str, Any],
        instrument_config: Mapping[str, Any],
        objective_config: Mapping[str, Any],
        weight_config: Mapping[str, float],
        wavelength_exclude_windows_angstrom,
        depth_sampling_config: Mapping[str, Any],
        instrument_line_of_sight_velocity_correction_m_per_s: float = 0.0,
        optimize_instrument_line_of_sight_velocity_correction: bool = False,
        disambiguation_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be non-empty.")
        wavelength = torch.as_tensor(wavelength_angstrom, dtype=torch.float32)
        if wavelength.ndim != 1 or wavelength.numel() < 2:
            raise ValueError("wavelength_angstrom must be a non-empty spectral grid.")
        self.observation_id = observation_id
        self.velocity_synthesis_mode = resolve_velocity_synthesis_mode(
            velocity_synthesis_mode
        )
        depth_sampling = resolve_depth_sampling(depth_sampling_config)
        assembly = build_forward_assembly(
            atmosphere_model=atmosphere_model,
            synthesizer_config=synthesizer_config,
            instrument_config=instrument_config,
            velocity_synthesis_mode=self.velocity_synthesis_mode,
            depth_sampling=depth_sampling,
            wavelength_angstrom=wavelength,
        )
        self.synthesizer = assembly.synthesizer
        self.instrument = assembly.instrument
        self._composition = assembly.composition
        self.instrument_type = str(instrument_config["type"])
        self.register_buffer("wavelength_angstrom", wavelength)
        self.register_buffer(
            "synthesis_wavelength_angstrom",
            assembly.synthesis_wavelength_angstrom.detach().clone(),
        )
        self.register_buffer(
            "coarse_depth_grid",
            torch.linspace(
                depth_sampling.reference_log_tau500_bounds[0],
                depth_sampling.reference_log_tau500_bounds[1],
                depth_sampling.sample_count,
                dtype=wavelength.dtype,
            ),
            persistent=False,
        )
        weighting = resolve_objective_weighting(
            wavelength,
            weight_config=weight_config,
            wavelength_weights=None,
            wavelength_exclude_windows_angstrom=(wavelength_exclude_windows_angstrom),
            continuum_indices=continuum_indices,
            atlas_continuum_radiance_w_m3_sr=(atlas_continuum_radiance_w_m3_sr),
        )
        self.register_buffer("stokes_weights", weighting.stokes_weights)
        self.register_buffer("wavelength_weights", weighting.wavelength_weights)
        self.register_buffer(
            "atlas_continuum_radiance_w_m3_sr",
            wavelength.new_tensor(weighting.atlas_continuum_radiance_w_m3_sr),
        )
        self.register_buffer(
            "carrington_angular_velocity_rad_per_s",
            wavelength.new_tensor(CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S),
        )
        self.disambiguation_options = self._resolve_disambiguation_options(
            disambiguation_config
        )
        self.phase_rotation = None
        if self.disambiguation_options["enabled"]:
            self.phase_rotation = ObserverPhaseRotation(
                spatial_coordinate_center_mm=atmosphere_model.spatial_coordinate_center_mm,
                spatial_coordinate_scale_mm=atmosphere_model.spatial_coordinate_scale_mm,
                time_dependent=atmosphere_model.time_dependent,
                time_coordinate_center_hours=atmosphere_model.time_coordinate_center_hours,
                time_coordinate_scale_hours=atmosphere_model.time_coordinate_scale_hours,
                hidden_dimension=self.disambiguation_options["hidden_dimension"],
                hidden_layers=self.disambiguation_options["hidden_layers"],
                first_omega_0=self.disambiguation_options["first_omega_0"],
                hidden_omega_0=self.disambiguation_options["hidden_omega_0"],
            )
        objective_options = dict(objective_config)
        self.qu_warmup_steps = objective_options.pop("qu_warmup_steps", 0)
        if type(self.qu_warmup_steps) is not int or self.qu_warmup_steps < 0:
            raise ValueError("qu_warmup_steps must be a non-negative integer")
        self.step = 0
        self.objective = StokesObjective(**objective_options)
        correction = float(instrument_line_of_sight_velocity_correction_m_per_s)
        if not math.isfinite(correction) or abs(correction) >= SPEED_OF_LIGHT:
            raise ValueError("Instrument LOS correction must be finite and subluminal.")
        if type(optimize_instrument_line_of_sight_velocity_correction) is not bool:
            raise TypeError("LOS correction optimization flag must be boolean.")
        self.velocity_scale_m_per_s = float(atmosphere_model.velocity_scale_m_per_s)
        self.instrument_line_of_sight_velocity_correction_normalized = (
            torch.nn.Parameter(
                wavelength.new_tensor(correction / self.velocity_scale_m_per_s),
                requires_grad=optimize_instrument_line_of_sight_velocity_correction,
            )
        )

    @staticmethod
    def _resolve_disambiguation_options(
        config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        defaults = {
            "enabled": False,
            "cold_steps": 2_000,
            "warmup_steps": 5_000,
            "handoff_step": 15_000,
            "binary_weight": 0.01,
            "hidden_dimension": 64,
            "hidden_layers": 3,
            "first_omega_0": 1.0,
            "hidden_omega_0": 1.0,
        }
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("disambiguation_config must be a mapping or null")
        options = {**defaults, **({} if config is None else dict(config))}
        expected = set(defaults)
        unknown = set(options) - expected
        if unknown:
            raise TypeError(
                f"Unknown disambiguation options: {sorted(unknown)}"
            )
        if type(options["enabled"]) is not bool:
            raise TypeError("disambiguation.enabled must be boolean")
        for name in (
            "cold_steps",
            "warmup_steps",
            "handoff_step",
            "hidden_dimension",
            "hidden_layers",
        ):
            if type(options[name]) is not int or options[name] < 0:
                raise ValueError(f"disambiguation.{name} must be a non-negative integer")
        if options["hidden_dimension"] < 1 or options["hidden_layers"] < 1:
            raise ValueError(
                "disambiguation hidden_dimension and hidden_layers must be positive"
            )
        for name in ("first_omega_0", "hidden_omega_0"):
            value = options[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(
                    f"disambiguation.{name} must be finite and positive"
                )
        if (
            isinstance(options["binary_weight"], bool)
            or not isinstance(options["binary_weight"], (int, float))
            or not math.isfinite(float(options["binary_weight"]))
            or options["binary_weight"] < 0.0
        ):
            raise ValueError("disambiguation.binary_weight must be finite and non-negative")
        if options["enabled"]:
            if options["warmup_steps"] < 1:
                raise ValueError(
                    "Enabled disambiguation requires warmup_steps > 0"
                )
            if options["binary_weight"] <= 0.0:
                raise ValueError(
                    "Enabled disambiguation requires binary_weight > 0"
                )
            if options["handoff_step"] <= (
                options["cold_steps"] + options["warmup_steps"]
            ):
                raise ValueError(
                    "disambiguation.handoff_step must be greater than cold_steps "
                    "plus warmup_steps"
                )
        return options

    @property
    def instrument_line_of_sight_velocity_correction_m_per_s(self) -> torch.Tensor:
        return (
            self.instrument_line_of_sight_velocity_correction_normalized
            * self.velocity_scale_m_per_s
        )

    def _runtime(self) -> ForwardRuntime:
        return ForwardRuntime(
            coarse_depth_grid=self.coarse_depth_grid,
            observed_wavelength_angstrom=self.wavelength_angstrom,
            synthesis_wavelength_angstrom=self.synthesis_wavelength_angstrom,
            radiance_scale=self.atlas_continuum_radiance_w_m3_sr,
            carrington_angular_velocity_rad_per_s=(
                self.carrington_angular_velocity_rad_per_s
            ),
            instrument_line_of_sight_velocity_correction_m_per_s=(
                self.instrument_line_of_sight_velocity_correction_m_per_s
            ),
        )

    def set_step(self, step):
        if type(step) is not int or step < 0:
            raise ValueError("Stokes objective step must be a non-negative integer")
        self.step = step

    def _phase_is_active(self) -> bool:
        options = getattr(self, "disambiguation_options", None)
        return bool(
            self.training
            and options is not None
            and options["enabled"]
            and self.step < options["handoff_step"]
            and getattr(self, "phase_rotation", None) is not None
        )

    def _phase_binary_weight(self) -> float:
        if not self.training or not self._phase_is_active():
            return 0.0
        options = self.disambiguation_options
        elapsed_steps = self.step - options["cold_steps"]
        warmup_steps = options["warmup_steps"]
        fraction = min(1.0, max(0.0, elapsed_steps / warmup_steps))
        return float(options["binary_weight"]) * fraction

    def _phase_for_coordinates(self, coordinates: torch.Tensor):
        if not self._phase_is_active():
            return None
        return self.phase_rotation(coordinates)

    def phase_for_diagnostics(self, coordinates: torch.Tensor) -> torch.Tensor | None:
        """Return the scheduled observer-frame phase for validation plots.

        Validation runs put the observation term in evaluation mode, which
        intentionally bypasses the temporary phase in Stokes synthesis.  The
        diagnostic plot still needs to show the phase network itself, so this
        method exposes the scheduled phase without changing the forward
        prediction or adding it to the validation loss.
        """

        options = self.disambiguation_options
        phase_rotation = self.phase_rotation
        if not options["enabled"] or phase_rotation is None:
            return None
        if self.step >= options["handoff_step"]:
            return coordinates[..., 0].new_zeros(coordinates.shape[:-1])
        return phase_rotation(coordinates)

    def _synthesize_batch(
        self,
        batch: Mapping[str, Any],
        *,
        randomize_depth: bool,
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        phase = self._phase_for_coordinates(batch["coordinates"])
        synthesis_options = {
            "runtime": self._runtime(),
            "ray_direction": batch["ray_direction"],
            "stokes_basis": batch["stokes_basis"],
            "observer_los_velocity_m_per_s": (
                batch["observer_los_velocity_m_per_s"]
                if self.velocity_synthesis_mode.value == "carrington_observer_relative"
                else None
            ),
            "removed_solar_los_velocity_m_per_s": (
                batch["removed_solar_los_velocity_m_per_s"]
                if self.velocity_synthesis_mode.value
                == "carrington_registered_relative"
                else None
            ),
            "randomize_depth": randomize_depth,
            "instrument_response": (
                batch["instrument_response"]
                if self.instrument_type == "hmi_filter_profiles"
                else None
            ),
        }
        if phase is not None:
            synthesis_options["magnetic_azimuth_phase_rad"] = phase
        result = self._composition.synthesize(
            batch["coordinates"],
            **synthesis_options,
        )
        return result, phase

    def _qu_weight_factor(self):
        return 0.0 if self.training and self.step < self.qu_warmup_steps else 1.0

    def _loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.stokes_weights
        if self._qu_weight_factor() == 0:
            # Preserve I/V coefficients and unweighted component diagnostics.
            weights = weights * weights.new_tensor([1.0, 0.0, 0.0, 1.0])
        return weighted_stokes_loss(
            self.objective,
            prediction,
            target,
            wavelength_weights=self.wavelength_weights,
            stokes_weights=weights,
        )

    def predict(self, batch):
        return self._synthesize_batch(batch, randomize_depth=False)[0]["stokes"]

    def evaluate_batch(self, batch: Mapping[str, Any]) -> DataTermBatchResult:
        required = {"coordinates", "ray_direction", "stokes_basis", "stokes"}
        if self.velocity_synthesis_mode.value == "carrington_observer_relative":
            required.add("observer_los_velocity_m_per_s")
        else:
            required.add("removed_solar_los_velocity_m_per_s")
        if self.instrument_type == "hmi_filter_profiles":
            required.add("instrument_response")
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"LTE Stokes batch is missing fields: {missing}.")
        result, phase = self._synthesize_batch(
            batch,
            randomize_depth=self.training,
        )
        prediction = result["stokes"]
        component, total = self._loss(prediction, batch["stokes"])
        phase_mean_sin2 = (
            prediction.new_zeros(())
            if phase is None
            else torch.sin(phase).square().mean()
        )
        phase_binary_weight = self._phase_binary_weight()
        phase_binary_loss = phase_mean_sin2 * phase_binary_weight
        total = total + phase_binary_loss

        component_losses = {
            name: component[index] for index, name in enumerate(STOKES_COMPONENTS)
        }
        component_losses["phase_binary"] = phase_binary_loss
        residual = prediction - batch["stokes"]
        metrics = {
            "line_of_sight_velocity_correction_m_per_s": (
                self.instrument_line_of_sight_velocity_correction_m_per_s.detach()
            ),
            "qu_weight_factor": prediction.new_tensor(self._qu_weight_factor()),
            "phase_rotation_active": prediction.new_tensor(
                float(phase is not None)
            ),
            "phase_binary_weight": prediction.new_tensor(phase_binary_weight),
            "phase_binary_mean_sin2": phase_mean_sin2.detach(),
            "phase_binary_loss": phase_binary_loss.detach(),
            "residual_bias": residual.mean(),
            "residual_rms": residual.square().mean().sqrt(),
        }
        diagnostics = {
            "prediction": prediction,
            "target": batch["stokes"],
            "pixel_index": batch.get("pixel_index"),
        }
        return DataTermBatchResult(
            likelihood_loss=total,
            component_losses=component_losses,
            metrics=metrics,
            sample_count=int(prediction.shape[0]),
            diagnostics=diagnostics,
        )

    def metadata(self) -> dict[str, Any]:
        result = {
            "observation_id": self.observation_id,
            "observation_kind": self.observation_kind,
            "objective": dict(self.objective.configuration()),
            "disambiguation": dict(self.disambiguation_options),
        }
        if self.phase_rotation is not None:
            result["disambiguation"]["phase_network"] = dict(
                self.phase_rotation.metadata()
            )
        return result


__all__ = ["StokesObservationTerm"]
