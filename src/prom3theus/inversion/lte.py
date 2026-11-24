"""Atmosphere-to-emergent-Stokes synthesis, without a learned radiation field."""

import copy
import math
from collections.abc import Mapping
from types import SimpleNamespace
from dataclasses import dataclass

import torch
from torch import nn

from prom3theus.core import carrington_rotation_velocity_cartesian
from prom3theus.core.validation import deferred_validation, require_tensor
from prom3theus.inversion.depth_sampling import (
    DepthRefinement, contribution_fine_distances, merge_depth_samples, stokes_interval_proposal,
)
from prom3theus.rt.formal_transfer import formal_stokes
from prom3theus.rt.polarized_exponential import independent_coefficients
from prom3theus.rt.geometry import position_to_chart_height
from prom3theus.rt.coefficient_mapper import CoefficientMapper
from prom3theus.rt.local_coefficients import LTELocalCoefficients
from prom3theus.rt.opacity import planck_lambda
from prom3theus.rt.ray_geometry import canonical_basis, sample_shell_characteristic, shell_characteristic, unit_direction
from prom3theus.rt.wavelength import CenteredWavelength


@dataclass(frozen=True)
class RaySamples:
    """Observer-to-boundary sampling and one canonical polarization frame."""

    distance_m: torch.Tensor
    intervals_m: torch.Tensor
    position_m: torch.Tensor
    chart_megameter: torch.Tensor
    height_m: torch.Tensor
    time_hours: torch.Tensor
    direction: torch.Tensor
    stokes_basis: torch.Tensor


class LTEWindow(nn.Module):
    """Shared local atomic physics and numerical integration for a spectral window.

    The atmosphere is owned by the joint model. This module has no trainable
    radiation parameters. Its boundary is the local bottom Planck source unless
    an explicit, fixed incoming temperature is configured.
    """

    def __init__(self, atmosphere_model, *, configuration, mapper_spec=None):
        super().__init__()
        # Keep numerical composition independent of the application's schema.
        # Public YAML is validated by config; direct constructors and restored
        # artifacts still enforce the physical/numerical requirements here.
        self.configuration = dict(depth_count=257, jitter_fraction=0., coarse_to_fine=None,
            transfer_precision="float32", shell_height_bounds_megameter=None,
            lateral_halo_megameter=0., incoming_temperature_k=None)
        allowed = set(self.configuration) | {"coefficient_backend", "mapper_checkpoint", "id", "line_ids", "radiance_scale", "wavelength_bounds_vacuum_angstrom"}
        unknown = set(configuration) - allowed
        if unknown:
            raise ValueError(f"Unknown LTE synthesis options: {sorted(unknown)}")
        self.configuration.update(copy.deepcopy(dict(configuration)))
        options = SimpleNamespace(**self.configuration)
        if type(options.depth_count) is not int or options.depth_count < 2:
            raise ValueError("depth_count must be an integer >= 2")
        if options.transfer_precision not in ("float32", "float64"):
            raise ValueError("transfer_precision must be float32 or float64")
        if (isinstance(options.jitter_fraction, bool)
                or not math.isfinite(options.jitter_fraction)
                or not 0 <= options.jitter_fraction <= 1):
            raise ValueError("jitter_fraction must be finite and between 0 and 1")
        if not math.isfinite(options.radiance_scale) or options.radiance_scale <= 0:
            raise ValueError("radiance_scale must be finite and positive")
        if not math.isfinite(options.lateral_halo_megameter) or options.lateral_halo_megameter < 0:
            raise ValueError("lateral_halo_megameter must be finite and nonnegative")
        if options.incoming_temperature_k is not None and (
            not math.isfinite(options.incoming_temperature_k) or options.incoming_temperature_k <= 0
        ):
            raise ValueError("incoming_temperature_k must be finite and positive")
        bounds = options.wavelength_bounds_vacuum_angstrom
        if len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("Wavelength bounds must be finite, positive and increasing")
        object.__setattr__(self, "atmosphere_model", atmosphere_model)
        refinement = options.coarse_to_fine
        if refinement is None:
            refinement = {"enabled": False}
        if not isinstance(refinement, Mapping) or set(refinement) - {"enabled", "fine_sample_count", "uniform_weight_floor"}:
            raise ValueError("coarse_to_fine requires enabled, fine_sample_count and uniform_weight_floor")
        # The typed schema serializes omitted fine-sampling options as None.
        refinement = {key: value for key, value in refinement.items() if key == "enabled" or value is not None}
        if refinement.get("enabled") and not {"fine_sample_count", "uniform_weight_floor"} <= set(refinement):
            raise ValueError("Enabled coarse_to_fine requires fine_sample_count and uniform_weight_floor")
        self.depth_refinement = DepthRefinement(refinement.get("enabled", False),
            refinement.get("fine_sample_count", 1), refinement.get("uniform_weight_floor", 0.))
        self.configuration["coarse_to_fine"] = dict(enabled=self.depth_refinement.enabled,
            fine_sample_count=self.depth_refinement.sample_count,
            uniform_weight_floor=self.depth_refinement.uniform_weight_floor)
        self.depth_count = options.depth_count
        self.jitter_fraction = options.jitter_fraction
        self.transfer_precision = options.transfer_precision
        reference = atmosphere_model.solar_radius_m
        self.register_buffer("radiance_scale", reference.new_tensor(options.radiance_scale))
        self.register_buffer("scene_basis", atmosphere_model.scene_basis.detach().clone())
        self.register_buffer("solar_radius_m", reference.detach().clone())
        center = atmosphere_model.spatial_coordinate_center_megameter.detach()
        scale = atmosphere_model.spatial_coordinate_scale_megameter.detach()
        halo = options.lateral_halo_megameter
        self.register_buffer("chart_bounds_megameter", torch.stack((center-scale-halo, center+scale+halo)))
        outer, inner = getattr(atmosphere_model, "line_formation_height_bounds_megameter",
                               atmosphere_model.shell_height_bounds_megameter)
        if options.shell_height_bounds_megameter is not None:
            inner, outer = options.shell_height_bounds_megameter
        atmosphere_outer, atmosphere_inner = atmosphere_model.shell_height_bounds_megameter
        if not atmosphere_inner <= inner < outer <= atmosphere_outer:
            raise ValueError("LTE synthesis shell must lie inside the atmosphere")
        self.configuration["shell_height_bounds_megameter"] = (inner, outer)
        self.register_buffer("height_bounds_m", reference.new_tensor((inner*1e6, outer*1e6)))
        self.register_buffer("wavelength_bounds", reference.new_tensor(options.wavelength_bounds_vacuum_angstrom))
        self.coefficients = LTELocalCoefficients(line_ids=options.line_ids).to(reference)

        self.mapper = None
        backend = getattr(options, "coefficient_backend", "exact_lte")
        checkpoint = getattr(options, "mapper_checkpoint", None)
        if backend == "nn_surrogate":
            if mapper_spec is None:
                if checkpoint is None:
                    raise ValueError("nn_surrogate requires a mapper_checkpoint")
                self.mapper = CoefficientMapper.load(checkpoint)
            else:
                # Portable inversion artifacts supply the learned tensors in their state dict.
                self.mapper = CoefficientMapper.from_saved_spec(mapper_spec)
            spec = self.mapper.spec
            if list(options.line_ids) != spec["line_ids"]:
                raise ValueError("Mapper line list differs from the synthesis window")
            if bounds[0] < spec["bounds"][0] or bounds[1] > spec["bounds"][1]:
                raise ValueError("Mapper wavelength support must cover the configured LTE window")
            self.mapper.to(reference).requires_grad_(False)
        elif backend != "exact_lte" or checkpoint is not None or mapper_spec is not None:
            raise ValueError("Invalid coefficient backend or misplaced mapper checkpoint")

    @property
    def construction(self):
        options = copy.deepcopy(self.configuration)
        if self.mapper is not None:
            options.pop("mapper_checkpoint", None)
            options["mapper_spec"] = copy.deepcopy(self.mapper.spec)
        return options

    def basis(self, direction):
        # Each ray is independent: choose a nonsingular axis for that ray.
        # The same basis is used at every depth and rotated into the detector
        # basis afterwards, so axis selection cannot change physical Stokes.
        n = unit_direction(direction)
        axes = self.scene_basis.to(n)
        axis = axes[torch.argmin(torch.abs(n @ axes.T), dim=-1)]
        return canonical_basis(n, axis)

    def boundary_queries(self, coordinates, ray_direction):
        geometry = shell_characteristic(coordinates.to(self.solar_radius_m), ray_direction,
            scene_basis=self.scene_basis, solar_radius_m=self.solar_radius_m,
            height_bounds_m=self.height_bounds_m,
            chart_bounds_megameter=self.chart_bounds_megameter)
        geometry["canonical_stokes_basis"] = self.basis(geometry["direction"])
        return geometry

    def evaluate_atmosphere(self, position_m, time_hours, *, chart=None, height=None):
        if chart is None:
            chart, height = position_to_chart_height(position_m, self.scene_basis, self.solar_radius_m)
        time = torch.broadcast_to(time_hours.to(chart), chart.shape[:-1])
        state = dict(self.atmosphere_model.evaluate_chart_height_points(
            torch.cat((chart, time[..., None]), dim=-1), height))
        state["velocity_field"] = state["velocity_field"] + carrington_rotation_velocity_cartesian(position_m).to(state["velocity_field"])
        if self.mapper is not None and self.mapper.thermodynamic_input == "mass_density":
            state["mass_density"] = self.coefficients.plasma_provider.mass_density(
                state["temperature"], state["gas_pressure"]
            )
        return state

    def sample_rays(self, geometry):
        """Stratify physical ray distance, preserving both shell boundaries.

        During training each interior node moves independently within its
        midpoint stratum. Evaluation uses the nominal uniform grid. The same
        sampled atmosphere is shared by all wavelengths.
        """
        direction = geometry["direction"]
        index = torch.arange(self.depth_count, device=direction.device, dtype=direction.dtype)
        offset = torch.zeros((*direction.shape[:-1], self.depth_count), device=direction.device, dtype=direction.dtype)
        if self.training and self.jitter_fraction > 0 and self.depth_count > 2:
            # Midpoint bounds leave a small gap, scaled to the precision of the
            # largest node index. No sorting or per-point checks are needed.
            eps = torch.finfo(direction.dtype).eps * self.depth_count
            if self.depth_refinement.enabled:
                # Leave room for all fine points even in the narrowest stratum.
                eps *= 4 * (self.depth_refinement.sample_count + 1)
            draws = torch.rand((*direction.shape[:-1], self.depth_count - 2),
                               device=direction.device, dtype=direction.dtype)
            offset[..., 1:-1] = (draws - .5) * ((1 - 2 * eps) * self.jitter_fraction)
        fraction = (index + offset) / (self.depth_count - 1)
        distance = geometry["path_length_m"][..., None] * fraction
        # Compute actual line elements from local offsets, before large ray
        # distances can erase small separations in float32.
        intervals = geometry["path_length_m"][..., None] * ((1 + offset.diff(dim=-1)) / (self.depth_count - 1))
        return self.samples_at_distances(geometry, distance, intervals)

    def samples_at_distances(self, geometry, distance, intervals=None):
        """Map physical ray distances to the stable local shell coordinates."""
        direction = geometry["direction"]
        if intervals is None:
            intervals = distance.diff(dim=-1)
        chart, height, position = sample_shell_characteristic(geometry, distance,
            solar_radius_m=self.solar_radius_m)
        basis = geometry.get("canonical_stokes_basis")
        if basis is None:
            basis = self.basis(direction)
        return RaySamples(distance, intervals, position, chart, height,
            geometry["time_hours"][..., None].expand_as(distance),
            direction[..., None, :].expand(*distance.shape, 3),
            basis[..., None, :, :].expand(*distance.shape, 3, 3))

    def refine_rays(self, geometry, coarse, proposal):
        """Draw only new points; return their sorted merge and physical widths."""
        with torch.no_grad():
            fine_distance = contribution_fine_distances(
                proposal, coarse.distance_m, self.depth_refinement.sample_count,
                self.depth_refinement.uniform_weight_floor,
                randomize=self.training and self.jitter_fraction > 0,
                minimum_separation=4 * torch.finfo(coarse.distance_m.dtype).eps
                    * geometry["path_length_m"][..., None],
            )
            distance, order = torch.sort(torch.cat((coarse.distance_m, fine_distance), dim=-1), dim=-1)
        fine = self.samples_at_distances(geometry, fine_distance)
        return fine, distance, distance.diff(dim=-1), order

    def incoming_stokes(self, wave, batch_shape, dtype):
        temperature = self.configuration["incoming_temperature_k"]
        if temperature is None:
            return None
        temperature = torch.full((*batch_shape, wave.shape[-1]), temperature,
                                 dtype=dtype, device=wave.device)
        intensity = planck_lambda(temperature, wave.expand_as(temperature),
            radiance_scale=self.radiance_scale, paired_wavelengths=True)
        zero = torch.zeros_like(intensity)
        return torch.stack((intensity, zero, zero, zero), dim=-1)

    def _transfer_coefficients(self, prepared, spectrum):
        # The exact teacher also serves standalone callers that consume the full matrix.
        if self.mapper is None:
            local = self.coefficients.evaluate_spectrum(prepared, spectrum, radiance_scale=self.radiance_scale)
            return independent_coefficients(local.propagation_m1), local.source
        local = self.mapper.evaluate_spectrum(prepared, spectrum, radiance_scale=self.radiance_scale, compact=True)
        return local.propagation_m1, local.source

    def forward(self, geometry, wavelength_vacuum_angstrom):
        """Sample -> atmosphere -> local physics -> spectra -> formal integration.

        Optional refinement reuses coarse coefficients and evaluates only new
        fine points. Proposals are detached; both atmosphere evaluations retain
        gradients. No intermediate radiation is fed back into the atmosphere.
        """
        with deferred_validation():
            query = wavelength_vacuum_angstrom
            wave = (query.vacuum() if isinstance(query, CenteredWavelength)
                    else torch.as_tensor(query, device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype))
            if wave.ndim < 1 or not wave.numel():
                raise ValueError("Synthesis wavelengths must be finite and nonempty")
            require_tensor(lambda: torch.isfinite(wave).all(), "Synthesis wavelengths must be finite")
            require_tensor(lambda: ((wave >= self.wavelength_bounds[0]) & (wave <= self.wavelength_bounds[1])).all(),
                           "Detector spectral queries exceed the configured LTE window")
            batch_shape = geometry["direction"].shape[:-1]
            if wave.ndim != 1 and wave.shape[:-1] != batch_shape:
                raise ValueError("Per-ray wavelength axes must match the ray batch shape")
            samples = self.sample_rays(geometry)
            atmosphere = self.evaluate_atmosphere(samples.position_m, samples.time_hours,
                chart=samples.chart_megameter, height=samples.height_m)
            backend = self.coefficients if self.mapper is None else self.mapper
            prepared = backend.prepare_atmosphere(atmosphere,
                direction=samples.direction, stokes_basis=samples.stokes_basis)
            # A singleton depth axis broadcasts per-ray queries without copying
            # any atmospheric point or its continuum interpolation table.
            if isinstance(query, CenteredWavelength):
                spectral_query = (query if wave.ndim == 1 else
                    CenteredWavelength(query.origin_angstrom, query.offset_angstrom[..., None, :]))
            else:
                spectral_query = wave if wave.ndim == 1 else wave[..., None, :]
            spectrum = backend.prepare_spectrum(spectral_query)
            incoming = self.incoming_stokes(wave, batch_shape, atmosphere["temperature"].dtype)
            propagation, source = self._transfer_coefficients(prepared, spectrum)
            distance, intervals = samples.distance_m, samples.intervals_m
            if self.depth_refinement.enabled:
                with torch.no_grad():
                    proposal = stokes_interval_proposal(propagation, intervals)
                fine, distance, intervals, order = self.refine_rays(geometry, samples, proposal)
                fine_atmosphere = self.evaluate_atmosphere(fine.position_m, fine.time_hours,
                    chart=fine.chart_megameter, height=fine.height_m)
                fine_prepared = backend.prepare_atmosphere(fine_atmosphere,
                    direction=fine.direction, stokes_basis=fine.stokes_basis)
                fine_propagation, fine_source = self._transfer_coefficients(fine_prepared, spectrum)
                propagation = merge_depth_samples(propagation, fine_propagation, order)
                source = merge_depth_samples(source, fine_source, order)

            stokes = formal_stokes(propagation, source, distance, bottom_boundary=incoming,
                precision=self.transfer_precision, polarized=True, intervals_m=intervals)
            return stokes.transpose(-1, -2)


__all__ = ["LTEWindow"]
