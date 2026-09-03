"""Numerical forward composition for atmospheric inversions.

This module is deliberately independent of PyTorch Lightning.  It connects an
atmosphere representation, a spectral-synthesis backend, ray geometry, velocity
gauges, and an observation operator.  LTE is the sole backend implemented by
the package today; :class:`ForwardSynthesisBackend` defines the narrow boundary
at which a future NLTE solver can enter without changing training orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import torch

from prom3theus.core import (
    SPEED_OF_LIGHT,
    carrington_rotation_velocity_cartesian,
    cartesian_to_spherical,
    project_cartesian_to_spherical,
)
from prom3theus.inversion.depth_sampling import (
    importance_fine_distances,
    jitter_depth_grid,
    merge_depth_samples,
)
from prom3theus.observations.contracts import (
    CARRINGTON_OBSERVER_RELATIVE_VELOCITY,
    CARRINGTON_REGISTERED_RELATIVE_VELOCITY,
    resolve_velocity_synthesis_mode,
)
from prom3theus.rt import (
    RayDistancePath,
    RayTraceResult,
    StratifiedAtmosphere,
    TransferPath,
    project_vectors_to_stokes,
)


@runtime_checkable
class ForwardSynthesisBackend(Protocol):
    """Solver-neutral spectral backend required by forward composition.

    A future NLTE backend may maintain very different internal state, but it
    must accept a sampled atmosphere and an explicit transfer path and return
    emergent, high-resolution Stokes profiles.  The reference-extinction method
    supplies only an opacity-guided sampling proposal; its result is detached
    before selecting fine ray points.
    """

    @property
    def uses_prepared_wavelength(self) -> bool:
        """Whether synthesis uses backend-prepared wavelength state."""

    def prepare_wavelength_grid(self, wavelength_angstrom: torch.Tensor) -> None:
        """Prepare immutable state for a synthesis wavelength grid."""

    def reference_extinction(
        self,
        atmosphere: StratifiedAtmosphere,
    ) -> torch.Tensor:
        """Return a positive reference extinction for sampling refinement."""

    def synthesize(
        self,
        atmosphere: StratifiedAtmosphere,
        wavelength_angstrom: torch.Tensor,
        *,
        radiance_scale: torch.Tensor | float,
        path: TransferPath,
    ) -> torch.Tensor:
        """Return high-resolution emergent Stokes profiles."""


class LTESynthesisBackend:
    """Adapt the package LTE synthesizer to :class:`ForwardSynthesisBackend`."""

    def __init__(self, synthesizer: Any):
        self.synthesizer = synthesizer
        self._uses_prepared_wavelength = False

    @property
    def uses_prepared_wavelength(self) -> bool:
        return self._uses_prepared_wavelength

    def prepare_wavelength_grid(self, wavelength_angstrom: torch.Tensor) -> None:
        prepare = getattr(self.synthesizer, "prepare_wavelength_grid", None)
        self._uses_prepared_wavelength = callable(prepare)
        if self._uses_prepared_wavelength:
            prepare(wavelength_angstrom)

    def reference_extinction(
        self,
        atmosphere: StratifiedAtmosphere,
    ) -> torch.Tensor:
        if atmosphere.gas_pressure is None:
            raise ValueError("LTE refinement requires atmospheric gas pressure.")
        return self.synthesizer.continuum_opacity.volume_extinction_at_5000(
            atmosphere.temperature,
            atmosphere.gas_pressure,
        )

    def synthesize(
        self,
        atmosphere: StratifiedAtmosphere,
        wavelength_angstrom: torch.Tensor,
        *,
        radiance_scale: torch.Tensor | float,
        path: TransferPath,
    ) -> torch.Tensor:
        return self.synthesizer(
            atmosphere,
            None if self.uses_prepared_wavelength else wavelength_angstrom,
            radiance_scale=radiance_scale,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class DepthRefinement:
    """Opacity-guided fine-ray sampling configuration."""

    enabled: bool
    sample_count: int
    uniform_weight_floor: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("DepthRefinement.enabled must be boolean.")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("DepthRefinement.sample_count must be a positive integer.")
        if not 0.0 <= self.uniform_weight_floor <= 1.0:
            raise ValueError("DepthRefinement.uniform_weight_floor must lie in [0, 1].")


@dataclass(frozen=True, slots=True)
class ForwardRuntime:
    """Current registered tensors supplied by the owning inversion module."""

    coarse_depth_grid: torch.Tensor
    observed_wavelength_angstrom: torch.Tensor
    synthesis_wavelength_angstrom: torch.Tensor
    radiance_scale: torch.Tensor | float
    carrington_angular_velocity_rad_per_s: torch.Tensor | float
    instrument_radial_velocity_correction_m_per_s: torch.Tensor | float


class LTEForwardComposition:
    """Compose atmosphere sampling, LTE synthesis, and observation response.

    This is intentionally not an ``nn.Module``.  The owning inversion module
    registers the atmosphere, LTE synthesizer, instrument, parameters, and
    buffers under their established names.  Holding references here therefore
    adds no state-dict prefixes and preserves existing artifact weights exactly.
    """

    def __init__(
        self,
        *,
        atmosphere_model: Any,
        backend: ForwardSynthesisBackend,
        instrument: Any,
        velocity_synthesis_mode: str,
        depth_refinement: DepthRefinement,
    ):
        if not isinstance(backend, ForwardSynthesisBackend):
            raise TypeError("backend must implement ForwardSynthesisBackend.")
        self.atmosphere_model = atmosphere_model
        self.backend = backend
        self.instrument = instrument
        self.velocity_synthesis_mode = resolve_velocity_synthesis_mode(
            velocity_synthesis_mode
        )
        self.depth_refinement = depth_refinement

    def prepare_wavelength_grid(
        self,
        observed_wavelength_angstrom: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare backend/operator state and return their synthesis grid."""

        synthesis_wavelength = self.instrument.synthesis_grid(
            observed_wavelength_angstrom
        )
        self.backend.prepare_wavelength_grid(synthesis_wavelength)
        prepare_instrument = getattr(self.instrument, "prepare", None)
        if callable(prepare_instrument):
            prepare_instrument(
                synthesis_wavelength,
                observed_wavelength_angstrom,
            )
        return synthesis_wavelength

    @staticmethod
    def sample_depth_grid(
        coarse_depth_grid: torch.Tensor,
        *,
        randomize: bool,
    ) -> torch.Tensor:
        return jitter_depth_grid(coarse_depth_grid, randomize=randomize)

    def _refine_ray_sampling(
        self,
        coordinates: torch.Tensor,
        ray_direction: torch.Tensor,
        coarse_atmosphere: StratifiedAtmosphere,
        coarse_trace: RayTraceResult,
    ) -> tuple[StratifiedAtmosphere, RayTraceResult]:
        """Add fine points while preserving both differentiable field paths."""

        refinement = self.depth_refinement
        with torch.no_grad():
            alpha500 = self.backend.reference_extinction(coarse_atmosphere)
            fine_distance_m = importance_fine_distances(
                alpha500,
                coarse_trace.distance_m,
                refinement.sample_count,
                refinement.uniform_weight_floor,
            )
            direction = ray_direction.to(fine_distance_m)
            direction = direction / torch.linalg.vector_norm(
                direction,
                dim=-1,
                keepdim=True,
            )
            outer_position_rsun = (
                coarse_trace.position_m[..., 0, :]
                / self.atmosphere_model.solar_radius_m
            )
            fine_position_rsun = (
                outer_position_rsun[..., None, :]
                + (fine_distance_m / self.atmosphere_model.solar_radius_m)[..., None]
                * direction[..., None, :]
            )
            fine_chart_xy_mm, fine_geometric_height_m = (
                self.atmosphere_model._position_chart_height(fine_position_rsun)
            )
            fine_position_m = fine_position_rsun * self.atmosphere_model.solar_radius_m
            combined_distance_m = torch.cat(
                (coarse_trace.distance_m, fine_distance_m),
                dim=-1,
            )
            order = torch.argsort(combined_distance_m, dim=-1)
            distance_m = torch.gather(combined_distance_m, -1, order)
            position_m = merge_depth_samples(
                coarse_trace.position_m,
                fine_position_m,
                order,
            )
            chart_xy_mm = merge_depth_samples(
                coarse_trace.chart_xy_mm,
                fine_chart_xy_mm,
                order,
            )
            geometric_height_m = merge_depth_samples(
                coarse_trace.geometric_height_m,
                fine_geometric_height_m,
                order,
            )
            depth_grid = torch.linspace(
                coarse_atmosphere.log_tau500[0],
                coarse_atmosphere.log_tau500[-1],
                distance_m.shape[-1],
                dtype=distance_m.dtype,
                device=distance_m.device,
            )
            ray_time = (
                coordinates.to(fine_position_rsun)[..., 2]
                .unsqueeze(-1)
                .expand(fine_position_rsun.shape[:-1])
            )
            fine_evaluation_coordinates = torch.cat(
                (fine_chart_xy_mm, ray_time[..., None]),
                dim=-1,
            )
        fine_fields = self.atmosphere_model.evaluate_chart_height_points(
            fine_evaluation_coordinates,
            fine_geometric_height_m,
        )
        fields = {
            name: merge_depth_samples(
                getattr(coarse_atmosphere, name),
                fine_fields[name],
                order,
            )
            for name in (
                "temperature",
                "velocity_field",
                "microturbulence",
                "magnetic_field",
                "gas_pressure",
            )
        }
        atmosphere = replace(
            coarse_atmosphere,
            log_tau500=depth_grid,
            geometric_height_m=geometric_height_m,
            **fields,
        )
        trace = RayTraceResult(
            position_m=position_m,
            distance_m=distance_m,
            chart_xy_mm=chart_xy_mm,
            geometric_height_m=geometric_height_m,
        )
        return atmosphere, trace

    def trace_atmosphere(
        self,
        coordinates: torch.Tensor,
        ray_direction: torch.Tensor,
        depth_grid: torch.Tensor,
    ) -> tuple[StratifiedAtmosphere, RayTraceResult]:
        """Trace and optionally refine one batch of atmosphere rays."""

        atmosphere, trace = self.atmosphere_model.trace_rays(
            coordinates,
            ray_direction,
            depth_grid,
        )
        if self.depth_refinement.enabled:
            atmosphere, trace = self._refine_ray_sampling(
                coordinates,
                ray_direction,
                atmosphere,
                trace,
            )
        return atmosphere, trace

    def apply_instrument(
        self,
        high_resolution_stokes: torch.Tensor,
        synthesis_wavelength_angstrom: torch.Tensor,
        observed_wavelength_angstrom: torch.Tensor,
        instrument_response: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply the configured observation operator to synthesized Stokes."""

        if instrument_response is not None and not isinstance(
            instrument_response,
            Mapping,
        ):
            raise TypeError("instrument_response must be a mapping of keyword tensors.")
        return self.instrument(
            high_resolution_stokes,
            synthesis_wavelength_angstrom,
            observed_wavelength_angstrom,
            **({} if instrument_response is None else instrument_response),
        )

    @staticmethod
    def _line_of_sight_offset(
        velocity_field_observer: torch.Tensor,
        value: torch.Tensor | float,
        coordinates: torch.Tensor,
        *,
        sign: float,
        name: str,
    ) -> torch.Tensor:
        offset = torch.as_tensor(
            value,
            dtype=velocity_field_observer.dtype,
            device=velocity_field_observer.device,
        )
        expected_shape = coordinates.shape[:-1]
        if offset.shape == (*expected_shape, 1):
            offset = offset.squeeze(-1)
        if offset.shape != expected_shape:
            raise ValueError(
                f"{name} must match the leading coordinate shape {expected_shape}."
            )
        if not torch.isfinite(offset).all() or torch.any(
            offset.abs() >= SPEED_OF_LIGHT
        ):
            raise ValueError(f"{name} must be finite and subluminal.")
        return torch.cat(
            (
                velocity_field_observer[..., :2],
                velocity_field_observer[..., 2:] + sign * offset[..., None, None],
            ),
            dim=-1,
        )

    def _validate_velocity_inputs(
        self,
        observer_los_velocity_m_per_s: torch.Tensor | None,
        removed_solar_los_velocity_m_per_s: torch.Tensor | None,
    ) -> tuple[bool, bool]:
        uses_observer_gauge = (
            self.velocity_synthesis_mode.value == CARRINGTON_OBSERVER_RELATIVE_VELOCITY
        )
        uses_registered_gauge = (
            self.velocity_synthesis_mode.value
            == CARRINGTON_REGISTERED_RELATIVE_VELOCITY
        )
        if uses_observer_gauge and observer_los_velocity_m_per_s is None:
            raise ValueError(
                "carrington_observer_relative synthesis requires "
                "observer_los_velocity_m_per_s from the observation."
            )
        if not uses_observer_gauge and observer_los_velocity_m_per_s is not None:
            raise ValueError(
                "registered_relative synthesis must not apply a second "
                "observer LOS velocity correction."
            )
        if uses_registered_gauge and removed_solar_los_velocity_m_per_s is None:
            raise ValueError(
                "carrington_registered_relative synthesis requires "
                "removed_solar_los_velocity_m_per_s from the observation."
            )
        if not uses_registered_gauge and removed_solar_los_velocity_m_per_s is not None:
            raise ValueError(
                "Only carrington_registered_relative synthesis may apply the "
                "Hinode spectral-registration velocity."
            )
        return uses_observer_gauge, uses_registered_gauge

    def synthesize(
        self,
        coordinates: torch.Tensor,
        *,
        runtime: ForwardRuntime,
        ray_direction: torch.Tensor | None = None,
        stokes_basis: torch.Tensor | None = None,
        observer_los_velocity_m_per_s: torch.Tensor | None = None,
        removed_solar_los_velocity_m_per_s: torch.Tensor | None = None,
        randomize_depth: bool = False,
        depth_grid: torch.Tensor | None = None,
        return_details: bool = False,
        instrument_response: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the complete atmosphere-to-observed-Stokes graph."""

        uses_observer_gauge, uses_registered_gauge = self._validate_velocity_inputs(
            observer_los_velocity_m_per_s,
            removed_solar_los_velocity_m_per_s,
        )
        if ray_direction is None or stokes_basis is None:
            raise ValueError(
                "The spherical LTE pipeline requires ray_direction and "
                "stokes_basis for every synthesis."
            )
        if depth_grid is None:
            depth_grid = self.sample_depth_grid(
                runtime.coarse_depth_grid,
                randomize=randomize_depth,
            )
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
                raise ValueError(
                    "An explicit synthesis depth_grid must be ordered and one-dimensional."
                )
        sampled_atmosphere, ray_trace = self.trace_atmosphere(
            coordinates,
            ray_direction,
            depth_grid,
        )
        uses_carrington_frame = uses_observer_gauge or uses_registered_gauge
        rotation_velocity_cartesian = (
            carrington_rotation_velocity_cartesian(
                ray_trace.position_m,
                torch,
                runtime.carrington_angular_velocity_rad_per_s,
            )
            if uses_carrington_frame
            else torch.zeros_like(sampled_atmosphere.velocity_field)
        )
        velocity_field_inertial_cartesian = (
            sampled_atmosphere.velocity_field + rotation_velocity_cartesian
        )
        radial_unit = ray_trace.position_m / torch.linalg.vector_norm(
            ray_trace.position_m,
            dim=-1,
            keepdim=True,
        )
        radial_velocity_correction = torch.as_tensor(
            runtime.instrument_radial_velocity_correction_m_per_s
        ).to(sampled_atmosphere.velocity_field)
        if radial_velocity_correction.numel() != 1:
            raise ValueError(
                "instrument_radial_velocity_correction_m_per_s must be scalar."
            )
        if not torch.isfinite(radial_velocity_correction).all() or torch.any(
            radial_velocity_correction.abs() >= SPEED_OF_LIGHT
        ):
            raise ValueError(
                "instrument_radial_velocity_correction_m_per_s must be finite "
                "and subluminal."
            )
        radial_velocity_correction_cartesian = radial_unit * radial_velocity_correction
        velocity_field_synthesis_cartesian = (
            velocity_field_inertial_cartesian + radial_velocity_correction_cartesian
        )
        magnetic_field_observer = project_vectors_to_stokes(
            sampled_atmosphere.magnetic_field,
            stokes_basis,
        )
        velocity_field_synthesis_observer = project_vectors_to_stokes(
            velocity_field_synthesis_cartesian,
            stokes_basis,
        )
        velocity_field_observer = velocity_field_synthesis_observer
        if observer_los_velocity_m_per_s is not None:
            velocity_field_observer = self._line_of_sight_offset(
                velocity_field_observer,
                observer_los_velocity_m_per_s,
                coordinates,
                sign=-1.0,
                name="observer_los_velocity_m_per_s",
            )
        if removed_solar_los_velocity_m_per_s is not None:
            velocity_field_observer = self._line_of_sight_offset(
                velocity_field_observer,
                removed_solar_los_velocity_m_per_s,
                coordinates,
                # Hinode Level-1 registration removed this solar LOS shift.
                # Reproduce that registered wavelength gauge by subtracting it
                # from the synthesized toward-observer velocity component.
                sign=-1.0,
                name="removed_solar_los_velocity_m_per_s",
            )
        if not torch.isfinite(velocity_field_observer).all() or torch.any(
            velocity_field_observer[..., 2].abs() >= SPEED_OF_LIGHT
        ):
            raise FloatingPointError(
                "The synthesized LOS velocity must remain finite and subluminal."
            )
        synthesis_atmosphere = replace(
            sampled_atmosphere,
            velocity_field=velocity_field_observer,
            magnetic_field=magnetic_field_observer,
        )
        transfer_path = RayDistancePath(ray_trace.distance_m)
        high_resolution_stokes = self.backend.synthesize(
            synthesis_atmosphere,
            runtime.synthesis_wavelength_angstrom,
            radiance_scale=runtime.radiance_scale,
            path=transfer_path,
        )
        sampled_stokes = self.apply_instrument(
            high_resolution_stokes,
            runtime.synthesis_wavelength_angstrom,
            runtime.observed_wavelength_angstrom,
            instrument_response,
        )
        result: dict[str, Any] = {"stokes": sampled_stokes}
        if not return_details:
            return result
        velocity_field_inertial_observer = project_vectors_to_stokes(
            velocity_field_inertial_cartesian,
            stokes_basis,
        )
        spherical_coordinates = cartesian_to_spherical(
            ray_trace.position_m,
            torch,
        )
        magnetic_field_spherical = project_cartesian_to_spherical(
            sampled_atmosphere.magnetic_field,
            spherical_coordinates,
            torch,
        )
        velocity_field_inertial_spherical = project_cartesian_to_spherical(
            velocity_field_inertial_cartesian,
            spherical_coordinates,
            torch,
        )
        velocity_field_synthesis_spherical = project_cartesian_to_spherical(
            velocity_field_synthesis_cartesian,
            spherical_coordinates,
            torch,
        )
        result.update(
            {
                "atmosphere": sampled_atmosphere,
                "ray_trace": ray_trace,
                "spherical_coordinates": spherical_coordinates,
                "magnetic_field_spherical": magnetic_field_spherical,
                "rotation_velocity_cartesian": rotation_velocity_cartesian,
                "velocity_field_inertial_cartesian": (
                    velocity_field_inertial_cartesian
                ),
                "velocity_field_inertial_spherical": (
                    velocity_field_inertial_spherical
                ),
                "velocity_field_synthesis_spherical": (
                    velocity_field_synthesis_spherical
                ),
                "instrument_radial_velocity_correction_m_per_s": (
                    runtime.instrument_radial_velocity_correction_m_per_s
                ),
                "instrument_radial_velocity_correction_cartesian": (
                    radial_velocity_correction_cartesian
                ),
                "velocity_field_synthesis_cartesian": (
                    velocity_field_synthesis_cartesian
                ),
                "magnetic_field_observer": magnetic_field_observer,
                "velocity_field_inertial_observer": (velocity_field_inertial_observer),
                "velocity_field_synthesis_observer": (
                    velocity_field_synthesis_observer
                ),
                "velocity_field_observer": velocity_field_observer,
            }
        )
        return result


__all__ = [
    "DepthRefinement",
    "ForwardRuntime",
    "ForwardSynthesisBackend",
    "LTEForwardComposition",
    "LTESynthesisBackend",
]
