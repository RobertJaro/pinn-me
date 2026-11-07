"""Observed Stokes likelihood for differentiable formal LTE integration."""
from __future__ import annotations

from collections.abc import Mapping
import math
import torch

from prom3theus.core import SPEED_OF_LIGHT
from prom3theus.core.validation import deferred_validation, require_tensor
from prom3theus.instruments import build_instrument
from prom3theus.inversion.configuration import resolve_objective_weighting
from prom3theus.inversion.objective import STOKES_COMPONENTS, StokesObjective, reduce_stokes_loss
from prom3theus.observations import resolve_velocity_synthesis_mode
from prom3theus.rt.wavelength import CenteredWavelength, air_to_vacuum_angstrom
from .base import DataTermBatchResult, ObservationDataTerm


class StokesObservationTerm(ObservationDataTerm):
    """Instrument response and likelihood for scene-owned LTE synthesis."""
    observation_kind = "stokes"

    def __init__(self, *, atmosphere_model, synthesis_window, synthesis_window_id,
                 observation_id, wavelength_angstrom, continuum_indices,
                 atlas_continuum_radiance_w_m3_sr, velocity_synthesis_mode,
                 instrument_config, objective_config, weight_config,
                 wavelength_exclude_windows_angstrom=(),
                 instrument_line_of_sight_velocity_correction_m_per_s=0.,
                 optimize_instrument_line_of_sight_velocity_correction=False,
                 maximum_line_of_sight_velocity_correction_m_per_s=10000.):
        super().__init__()
        object.__setattr__(self, "synthesis_window", synthesis_window)
        object.__setattr__(self, "atmosphere_model", atmosphere_model)
        self.synthesis_window_id = synthesis_window_id
        self.observation_id = observation_id
        self.velocity_synthesis_mode = resolve_velocity_synthesis_mode(velocity_synthesis_mode)
        self.instrument = build_instrument(instrument_config)
        self.instrument_type = instrument_config["type"]
        wavelength = torch.as_tensor(wavelength_angstrom, dtype=torch.float64)
        synthesis = self.instrument.synthesis_grid(wavelength)
        prepare = getattr(self.instrument, "prepare", None)
        if prepare is not None:
            prepare(synthesis, wavelength)
        self.register_buffer("wavelength_angstrom", wavelength)
        self.register_buffer("synthesis_wavelength_angstrom", synthesis.to(torch.float64))
        self.register_buffer("synthesis_wavelength_vacuum_angstrom", air_to_vacuum_angstrom(synthesis.to(torch.float64)))
        # Center fixed detector coordinates once on CPU before model-dtype
        # arithmetic. Keep this metadata exact across float()/double() calls.
        self.spectral_origin_angstrom = float(int(self.synthesis_wavelength_vacuum_angstrom[0]))
        self.register_buffer("synthesis_wavelength_vacuum_offset_angstrom",
            self.synthesis_wavelength_vacuum_angstrom - self.spectral_origin_angstrom,
            persistent=False)
        weighting = resolve_objective_weighting(
            wavelength, weight_config=weight_config, wavelength_weights=None,
            wavelength_exclude_windows_angstrom=wavelength_exclude_windows_angstrom,
            continuum_indices=continuum_indices,
            atlas_continuum_radiance_w_m3_sr=atlas_continuum_radiance_w_m3_sr)
        reference = atmosphere_model.solar_radius_m
        self.register_buffer("stokes_weights", weighting.stokes_weights.to(reference))
        self.register_buffer("wavelength_weights", weighting.wavelength_weights.to(reference))
        self.register_buffer("atlas_continuum_radiance_w_m3_sr", reference.new_tensor(weighting.atlas_continuum_radiance_w_m3_sr))
        options = dict(objective_config)
        self.qu_warmup_steps = options.pop("qu_warmup_steps", 0)
        self.objective = StokesObjective(**options, continuum_indices=weighting.continuum_indices)
        self.step = 0
        correction = float(instrument_line_of_sight_velocity_correction_m_per_s)
        if not math.isfinite(correction) or abs(correction) >= SPEED_OF_LIGHT:
            raise ValueError("Instrument wavelength correction must be finite and subluminal")
        maximum = float(maximum_line_of_sight_velocity_correction_m_per_s)
        if not math.isfinite(maximum) or not 0 < maximum < SPEED_OF_LIGHT or abs(correction) >= maximum:
            raise ValueError("Initial correction must lie strictly inside its positive subluminal spectral bound")
        self.register_buffer("maximum_velocity_correction", reference.new_tensor(maximum))
        self.instrument_line_of_sight_velocity_correction_normalized = torch.nn.Parameter(
            reference.new_tensor(math.atanh(correction / maximum)),
            requires_grad=optimize_instrument_line_of_sight_velocity_correction)
        self.instrument.to(reference)

    def _apply(self, fn, recurse=True):
        # Preserve sub-milliangstrom detector offsets through model.float().
        names = ("wavelength_angstrom", "synthesis_wavelength_angstrom",
                 "synthesis_wavelength_vacuum_angstrom", "synthesis_wavelength_vacuum_offset_angstrom")
        exact = {name: getattr(self, name) for name in names}
        super()._apply(fn, recurse=recurse)
        for name, value in exact.items():
            setattr(self, name, value.to(device=getattr(self, name).device, dtype=torch.float64))
        return self

    @property
    def instrument_line_of_sight_velocity_correction_m_per_s(self):
        return torch.tanh(self.instrument_line_of_sight_velocity_correction_normalized) * self.maximum_velocity_correction

    def set_step(self, step):
        self.step = int(step)

    def _qu_weight_factor(self):
        return 0. if self.training and self.step < self.qu_warmup_steps else 1.

    def _score(self, prediction, target):
        comparison = self.objective.compare(prediction, target)
        weights = self.stokes_weights
        if self._qu_weight_factor() == 0:
            weights = weights * weights.new_tensor([1., 0., 0., 1.])
        component, total = reduce_stokes_loss(
            (comparison.prediction - comparison.target).square(),
            wavelength_weights=self.wavelength_weights, stokes_weights=weights)
        return component, total

    def _loss(self, prediction, target):
        return self._score(prediction, target)

    def spectral_queries(self, batch):
        coordinates = batch["coordinates"]
        mode = self.velocity_synthesis_mode.value
        name = ("observer_los_velocity_m_per_s" if mode == "carrington_observer_relative"
                else "removed_solar_los_velocity_m_per_s")
        if name not in batch:
            raise KeyError(f"Stokes batch requires {name}")
        correction = self.instrument_line_of_sight_velocity_correction_m_per_s
        shift = batch[name].to(correction).reshape(coordinates.shape[:-1])

        def inverse_doppler_delta(velocity):
            beta = velocity / SPEED_OF_LIGHT
            require_tensor((torch.isfinite(beta) & (beta.abs() < 1)).all(),
                           "Spectral shifts must be finite and subluminal")
            inverse = torch.sqrt((1 - beta) / (1 + beta))
            # Rationalize inverse-1: direct subtraction loses velocities of
            # a few m/s in float32, even before forming absolute wavelengths.
            return -2 * beta / ((1 + beta) * (inverse + 1))

        observer_delta = inverse_doppler_delta(shift)
        correction_delta = inverse_doppler_delta(correction)
        delta = observer_delta + correction_delta + observer_delta * correction_delta
        offset = self.synthesis_wavelength_vacuum_offset_angstrom.to(correction)
        shifted = offset + (self.spectral_origin_angstrom + offset) * delta[..., None]
        return CenteredWavelength(self.spectral_origin_angstrom, shifted)

    def _prediction_and_stokes(self, batch):
        with deferred_validation():
            geometry = self.synthesis_window.boundary_queries(batch["coordinates"], batch["ray_direction"])
            direction = geometry["direction"]
            physical = self.synthesis_window(geometry, self.spectral_queries(batch))
            prediction = self.observe_stokes(
                batch, physical, direction=direction,
                canonical_basis=geometry.get("canonical_stokes_basis"))
            return prediction, physical

    def observe_stokes(self, batch, physical, *, direction=None, canonical_basis=None):
        """Project canonical Stokes on spectral_queries(batch) into detector units.

        Shared by predictions and independent formal-RT diagnostics so basis,
        radiance calibration and the spectral response cannot drift apart.
        """
        if direction is None:
            geometry = self.synthesis_window.boundary_queries(batch["coordinates"], batch["ray_direction"])
            direction = geometry["direction"]
            if canonical_basis is None:
                canonical_basis = geometry.get("canonical_stokes_basis")
        canonical = (self.synthesis_window.basis(direction)
                     if canonical_basis is None else canonical_basis)
        detector = batch["stokes_basis"].to(canonical)
        require_tensor(torch.isclose(detector[..., 2, :], direction, atol=2e-5, rtol=0).all(),
                       "Detector Stokes axis must point along photon propagation")
        cosine = (detector[..., 0, :] * canonical[..., 0, :]).sum(-1)
        sine = (detector[..., 0, :] * canonical[..., 1, :]).sum(-1)
        angle = torch.atan2(sine, cosine)
        # The old instrument convention actively rotated transverse B by this
        # angle; the equivalent output operation is a negative basis rotation.
        angle = angle - math.radians(self.instrument.polarization_convention.offset_deg)
        c = torch.cos(2 * angle)[..., None].to(physical)
        s = torch.sin(2 * angle)[..., None].to(physical)
        i, q, u, v = physical.unbind(-2)
        stokes = torch.stack((i, c*q+s*u, -s*q+c*u, v), dim=-2)
        stokes = stokes * (torch.as_tensor(self.synthesis_window.radiance_scale).to(stokes)
                           / self.atlas_continuum_radiance_w_m3_sr)
        response = batch.get("instrument_response", {})
        prediction = self.instrument(stokes, self.synthesis_wavelength_angstrom.to(stokes),
                                     self.wavelength_angstrom.to(stokes), **response)
        return prediction

    def predict(self, batch):
        return self._prediction_and_stokes(batch)[0]

    def evaluate_batch(self, batch):
        """Score and diagnose a batch; public evaluations retain full diagnostics."""
        return self._evaluate_batch(batch, diagnostics=True)

    def evaluate_training_batch(self, batch):
        """Score an optimizer batch without computing unused display metrics."""
        return self._evaluate_batch(batch, diagnostics=False)

    def _evaluate_batch(self, batch, *, diagnostics):
        prediction, physical = self._prediction_and_stokes(batch)
        component, total = self._score(prediction, batch["stokes"])
        metrics = (self._fit_metrics(prediction, batch["stokes"], physical)
                   if diagnostics else {})
        return DataTermBatchResult(
            likelihood_loss=total,
            component_losses=dict(zip(STOKES_COMPONENTS, component.unbind(), strict=True)),
            metrics=metrics,
            sample_count=prediction.numel() // (4 * prediction.shape[-1]),
            diagnostics={"prediction": prediction, "target": batch["stokes"],
                         "pixel_index": batch.get("pixel_index")},
        )

    @torch.no_grad()
    def _fit_metrics(self, prediction, target, physical):
        """Summarize fit and physical admissibility without extending autograd."""
        intensity = physical[..., 0, :]
        polarization = torch.linalg.vector_norm(physical[..., 1:, :], dim=-2)
        violation = torch.maximum(torch.relu(-intensity), torch.relu(polarization - intensity))
        residual = prediction - target
        metrics = {
            "stokes_outgoing_admissibility_max": violation.max(),
        }
        if self.instrument_line_of_sight_velocity_correction_normalized.requires_grad:
            metrics["line_of_sight_velocity_correction_m_per_s"] = (
                self.instrument_line_of_sight_velocity_correction_m_per_s.detach()
            )
        if self.qu_warmup_steps > 0:
            metrics["qu_weight_factor"] = prediction.new_tensor(self._qu_weight_factor())
        # Component MSE already reports the objective-space fit. Retain only
        # the complementary raw per-component errors here.
        dimensions = (*range(prediction.ndim - 2), prediction.ndim - 1)
        rms = residual.square().mean(dim=dimensions).sqrt()
        for name, value in zip(STOKES_COMPONENTS, rms, strict=True):
            metrics[f"stokes_{name}_raw_residual_rms"] = value
        return metrics

    def metadata(self):
        return {"observation_id": self.observation_id, "observation_kind": self.observation_kind,
                "synthesis_window": self.synthesis_window_id,
                "objective": dict(self.objective.configuration())}


__all__ = ["StokesObservationTerm"]
