"""PyTorch-Lightning integration for depth-stratified LTE inversion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import math
from numbers import Real
from typing import Mapping

import torch
from pytorch_lightning import LightningModule

from pme.lte.atmosphere import StratifiedAtmosphereModel
from pme.lte.geometry import (
    RayTraceResult,
    chart_to_direction,
    direction_to_chart_mm,
    intersect_sphere_near_side_from_local_point,
)
from pme.lte.instrument import HinodeSpectralPSF
from pme.coordinates import (
    cartesian_to_spherical,
    project_cartesian_to_spherical,
    project_spherical_to_observer,
)
from pme.lte.synthesis import LTESynthesizer
from pme.model import NormalizationModule
from pme.solar_velocity import (
    CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
    CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS,
    carrington_rotation_velocity_cartesian,
)
from pme.train.lte_physics import (
    BOUNDARY_EQUATIONS,
    LTEPhysicsModule,
    LTEPhysicsResult,
    VOLUME_EQUATIONS,
)
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
        learning_rate: float | Mapping = 1.0e-5,
        checkpoint_metadata=None,
    ):
        super().__init__()
        log_tau500 = torch.as_tensor(log_tau500, dtype=torch.float32)
        wavelength_angstrom = torch.as_tensor(
            wavelength_angstrom, dtype=torch.float32
        )
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
        refinement = dict(depth_sampling.pop("coarse_to_fine", {}))
        self.coarse_to_fine_enabled = bool(refinement.pop("enabled", False))
        self.coarse_to_fine_sample_count = int(
            refinement.pop("fine_sample_count", 32)
        )
        self.coarse_to_fine_uniform_weight_floor = float(
            refinement.pop("uniform_weight_floor", 0.05)
        )
        if refinement:
            raise TypeError(
                f"Unknown coarse-to-fine sampling options: {sorted(refinement)}"
            )
        if self.training_depth_sample_count < 2:
            raise ValueError("Depth sample_count must be at least two.")
        if self.coarse_to_fine_sample_count < 1:
            raise ValueError("fine_sample_count must be positive.")
        if not 0.0 <= self.coarse_to_fine_uniform_weight_floor <= 1.0:
            raise ValueError("uniform_weight_floor must lie between zero and one.")
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
        self.register_buffer(
            "carrington_angular_velocity_rad_per_s",
            wavelength_angstrom.new_tensor(CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S),
        )
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
        resolved_weights = {}
        for component in components:
            value = weight_config.get(component, 1.0)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"Stokes weight {component} must be a fixed number; schedules are unsupported."
                )
            resolved_weights[component] = float(value)
        self.stokes_weight_config = dict(resolved_weights)
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
        if not torch.any(weights > 0):
            raise ValueError("At least one Stokes weight must be nonzero.")
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

        if isinstance(learning_rate, Mapping):
            learning_rate_config = dict(learning_rate)
            unknown = set(learning_rate_config) - {"start", "end", "iterations"}
            if unknown:
                raise TypeError(f"Unknown learning-rate options: {sorted(unknown)}")
            try:
                start = float(learning_rate_config["start"])
                end = float(learning_rate_config["end"])
                iterations = learning_rate_config["iterations"]
            except KeyError as error:
                raise KeyError(
                    "Scheduled learning_rate requires start, end, and iterations."
                ) from error
            if isinstance(iterations, str):
                if iterations.lower() != "auto":
                    raise ValueError("learning_rate.iterations must be positive or 'auto'.")
                iterations = "auto"
            elif (
                not isinstance(iterations, Real)
                or isinstance(iterations, bool)
                or int(iterations) != iterations
                or iterations <= 0
            ):
                raise ValueError("learning_rate.iterations must be positive or 'auto'.")
            else:
                iterations = int(iterations)
            self.learning_rate_schedule = {
                "start": start,
                "end": end,
                "iterations": iterations,
            }
            self.learning_rate_configuration = deepcopy(self.learning_rate_schedule)
            self.learning_rate = start
        elif isinstance(learning_rate, Real) and not isinstance(learning_rate, bool):
            self.learning_rate = float(learning_rate)
            self.learning_rate_schedule = None
            self.learning_rate_configuration = self.learning_rate
        else:
            raise TypeError("learning_rate must be a positive number or schedule mapping.")
        rate_values = (
            (self.learning_rate,)
            if self.learning_rate_schedule is None
            else (self.learning_rate_schedule["start"], self.learning_rate_schedule["end"])
        )
        if any(not math.isfinite(value) or value <= 0 for value in rate_values):
            raise ValueError("Learning rates must be finite and positive.")
        self.resolved_learning_rate_iterations: int | None = None

        self.checkpoint_metadata = deepcopy(checkpoint_metadata or {})

        physics = deepcopy(physics_config or {})
        reference_gravity = getattr(continuum, "reference_gravity_m_per_s2", None)
        equations = deepcopy(physics.pop("equations", {}))
        configured_gravity = physics.pop("gravity_m_per_s2", reference_gravity)
        self.gravity_m_per_s2 = (
            None if configured_gravity is None else float(configured_gravity)
        )
        self.physics_volume_points_per_step = int(
            physics.pop("volume_points_per_step", 256)
        )
        self.physics_height_layers_per_step = int(
            physics.pop("height_layers_per_step", 8)
        )
        self.optical_depth_anchor_points_per_step = int(
            physics.pop("optical_depth_anchor_points_per_step", 64)
        )
        self.optical_depth_anchor_depth_points = int(
            physics.pop("optical_depth_anchor_depth_points", 65)
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
            vector_basis_matches_spatial_coordinates=vector_basis_matches,
            normalization=physics_normalization,
        )
        self.optical_depth_anchor_enabled = self.physics.enabled[
            "mean_radial_optical_depth_anchor"
        ]
        self.upper_boundary_pressure_prior_enabled = self.physics.enabled[
            "upper_boundary_gas_pressure_prior"
        ]
        self.upper_boundary_sampling_enabled = any(
            self.physics.enabled[name] for name in BOUNDARY_EQUATIONS
        )
        if self.gravity_m_per_s2 is not None and self.gravity_m_per_s2 <= 0:
            raise ValueError("Magnetohydrostatic-equilibrium gravity must be positive.")
        if self.physics_volume_points_per_step < 1:
            raise ValueError("Physics volume sample count must be positive.")
        if (
            self.physics_height_layers_per_step < 1
            or self.physics_height_layers_per_step > self.physics_volume_points_per_step
            or self.physics_volume_points_per_step % self.physics_height_layers_per_step != 0
        ):
            raise ValueError(
                "Physics volume sample count must be exactly divisible by the positive "
                "height layer count."
            )
        if self.optical_depth_anchor_points_per_step < 0 or (
            self.upper_boundary_sampling_enabled
            and self.optical_depth_anchor_points_per_step < 1
        ):
            raise ValueError(
                "Optical-depth anchor sample count must be non-negative and positive "
                "when the anchor is enabled."
            )
        if self.optical_depth_anchor_depth_points < 2:
            raise ValueError("The optical-depth anchor requires at least two radial points.")
        if self.physics_validation_depth_points < 2:
            raise ValueError(
                "Physics validation requires at least two points per component."
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
                    "coarse_to_fine": {
                        "enabled": self.coarse_to_fine_enabled,
                        "fine_sample_count": self.coarse_to_fine_sample_count,
                        "uniform_weight_floor": self.coarse_to_fine_uniform_weight_floor,
                    },
                },
                "physics_config": {
                    **self.physics.configuration(),
                    "gravity_m_per_s2": self.gravity_m_per_s2,
                    "volume_points_per_step": self.physics_volume_points_per_step,
                    "height_layers_per_step": self.physics_height_layers_per_step,
                    "optical_depth_anchor_points_per_step": (
                        self.optical_depth_anchor_points_per_step
                    ),
                    "optical_depth_anchor_depth_points": (
                        self.optical_depth_anchor_depth_points
                    ),
                    "validation_depth_points": self.physics_validation_depth_points,
                },
                "learning_rate": deepcopy(self.learning_rate_configuration),
                "checkpoint_metadata": self.checkpoint_metadata,
            }
        )
        # Lookup tables are generated and regression-tested at high precision,
        # but the inversion graph has one explicit runtime dtype.
        self.float()

    @property
    def synthesis_wavelength_base(self) -> torch.Tensor:
        """Uniform synthesis grid in the module's current dtype and device.

        Regenerating this inexpensive grid keeps it aligned with the module's
        active compute dtype and device.
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

    @staticmethod
    @torch.no_grad()
    def _importance_refined_distances(
        alpha500: torch.Tensor,
        distance_m: torch.Tensor,
        fine_sample_count: int,
        uniform_weight_floor: float,
    ) -> torch.Tensor:
        """Insert deterministic per-ray samples using detached tau contribution weights."""

        if alpha500.shape != distance_m.shape or alpha500.ndim < 2:
            raise ValueError("alpha500 and distance_m must have matching [..., depth] shapes.")
        if fine_sample_count < 1 or alpha500.shape[-1] < 2:
            raise ValueError("Coarse-to-fine sampling requires positive samples and two coarse points.")
        if torch.any(distance_m[..., 1:] <= distance_m[..., :-1]):
            raise ValueError("Coarse ray distances must increase strictly.")
        interval_m = distance_m[..., 1:] - distance_m[..., :-1]
        delta_tau = 0.5 * (alpha500[..., 1:] + alpha500[..., :-1]) * interval_m
        tau_edge = torch.cat(
            (torch.zeros_like(delta_tau[..., :1]), torch.cumsum(delta_tau, dim=-1)),
            dim=-1,
        )
        tau_mid = 0.5 * (tau_edge[..., 1:] + tau_edge[..., :-1])
        contribution = delta_tau * torch.exp(-tau_mid.clamp_max(80.0))
        contribution = contribution.detach()
        interval_count = contribution.shape[-1]
        total_contribution = contribution.sum(dim=-1, keepdim=True)
        normalized = torch.where(
            total_contribution > torch.finfo(contribution.dtype).tiny,
            contribution / total_contribution.clamp_min(
                torch.finfo(contribution.dtype).tiny
            ),
            torch.full_like(contribution, 1.0 / interval_count),
        )
        probability = (
            (1.0 - uniform_weight_floor) * normalized
            + uniform_weight_floor / interval_count
        )
        cdf = torch.cumsum(probability, dim=-1)
        quantiles = (
            torch.arange(
                fine_sample_count, dtype=distance_m.dtype, device=distance_m.device
            )
            + 0.5
        ) / fine_sample_count
        quantiles = quantiles.expand(*distance_m.shape[:-1], fine_sample_count)
        interval = torch.searchsorted(
            cdf.contiguous(), quantiles.contiguous(), right=False
        ).clamp_max(interval_count - 1)
        cdf_before = torch.cat((torch.zeros_like(cdf[..., :1]), cdf[..., :-1]), dim=-1)
        lower_cdf = torch.gather(cdf_before, -1, interval)
        upper_cdf = torch.gather(cdf, -1, interval)
        fraction = (quantiles - lower_cdf) / (upper_cdf - lower_cdf).clamp_min(
            torch.finfo(distance_m.dtype).eps
        )
        lower_distance = torch.gather(distance_m[..., :-1], -1, interval)
        upper_distance = torch.gather(distance_m[..., 1:], -1, interval)
        fine_distance = lower_distance + fraction * (upper_distance - lower_distance)
        return torch.sort(torch.cat((distance_m, fine_distance), dim=-1), dim=-1).values

    def _refine_ray_sampling(
        self,
        coords: torch.Tensor,
        ray_direction: torch.Tensor,
        coarse_atmosphere,
        coarse_trace: RayTraceResult,
    ):
        """Build and evaluate the final per-ray grid from a coarse continuum trace."""

        with torch.no_grad():
            alpha500 = self.synthesizer.continuum_opacity.volume_extinction_at_5000(
                coarse_atmosphere.temperature, coarse_atmosphere.gas_pressure
            )
            distance_m = self._importance_refined_distances(
                alpha500,
                coarse_trace.distance_m,
                self.coarse_to_fine_sample_count,
                self.coarse_to_fine_uniform_weight_floor,
            )
            direction = ray_direction.to(distance_m)
            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            )
            geometry_basis = self.atmosphere_model.scene_basis.to(distance_m)
            surface_reference_rsun = chart_to_direction(
                coords.to(distance_m)[..., 1:],
                geometry_basis,
                self.atmosphere_model.solar_radius_m,
            )
            outer_radius = 1.0 + (
                self.atmosphere_model.shell_height_bounds_Mm[0] * 1.0e6
                / self.atmosphere_model.solar_radius_m
            )
            outer_offset = intersect_sphere_near_side_from_local_point(
                surface_reference_rsun,
                direction,
                outer_radius.expand_as(distance_m[..., 0]),
            )
            outer_position_rsun = (
                surface_reference_rsun + outer_offset[..., None] * direction
            )
            position_rsun = (
                outer_position_rsun[..., None, :]
                + (distance_m / self.atmosphere_model.solar_radius_m)[..., None]
                * direction[..., None, :]
            )
            chart_xy_mm = direction_to_chart_mm(
                position_rsun,
                geometry_basis,
                self.atmosphere_model.solar_radius_m,
            )
            geometric_height_m = (
                torch.linalg.vector_norm(position_rsun, dim=-1) - 1.0
            ) * self.atmosphere_model.solar_radius_m
            position_m = position_rsun * self.atmosphere_model.solar_radius_m
            depth_grid = torch.linspace(
                coarse_atmosphere.log_tau500[0],
                coarse_atmosphere.log_tau500[-1],
                distance_m.shape[-1],
                dtype=distance_m.dtype,
                device=distance_m.device,
            )
        # This is the only network evaluation in the refinement pass that
        # participates in autograd.
        fields = self.atmosphere_model.evaluate_position_rsun(position_rsun)
        refined_atmosphere = replace(
            coarse_atmosphere,
            log_tau500=depth_grid,
            geometric_height_m=geometric_height_m,
            **fields,
        )
        refined_trace = RayTraceResult(
            position_m=position_m,
            distance_m=distance_m,
            chart_xy_mm=chart_xy_mm,
            geometric_height_m=geometric_height_m,
            maximum_surface_residual_m=coarse_trace.maximum_surface_residual_m,
        )
        return refined_atmosphere, refined_trace

    def synthesize(
        self,
        coords: torch.Tensor,
        mu: torch.Tensor,
        *,
        ray_origin_m: torch.Tensor | None = None,
        ray_direction: torch.Tensor | None = None,
        stokes_basis: torch.Tensor | None = None,
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
        if any(value is None for value in (ray_origin_m, ray_direction, stokes_basis)):
            raise ValueError(
                "The spherical LTE pipeline requires ray_origin_m, ray_direction, "
                "and stokes_basis for every synthesis."
            )
        if self.coarse_to_fine_enabled:
            # The proposal pass selects coordinates only. Building an autograd
            # graph here would retain a second atmosphere graph without adding
            # a valid gradient through the discrete importance selection.
            with torch.no_grad():
                sampled_atmosphere, ray_trace = self.atmosphere_model.trace_rays(
                    coords,
                    ray_origin_m,
                    ray_direction,
                    depth_grid,
                )
            sampled_atmosphere, ray_trace = self._refine_ray_sampling(
                coords,
                ray_direction,
                sampled_atmosphere,
                ray_trace,
            )
        else:
            sampled_atmosphere, ray_trace = self.atmosphere_model.trace_rays(
                coords,
                ray_origin_m,
                ray_direction,
                depth_grid,
            )
        spherical_coordinates = cartesian_to_spherical(ray_trace.position_m, torch)
        magnetic_field_spherical = project_cartesian_to_spherical(
            sampled_atmosphere.magnetic_field, spherical_coordinates, torch
        )
        velocity_field_spherical = project_cartesian_to_spherical(
            sampled_atmosphere.velocity_field, spherical_coordinates, torch
        )
        rotation_velocity_cartesian = carrington_rotation_velocity_cartesian(
            ray_trace.position_m,
            torch,
            self.carrington_angular_velocity_rad_per_s,
        )
        velocity_field_inertial_cartesian = (
            sampled_atmosphere.velocity_field + rotation_velocity_cartesian
        )
        velocity_field_inertial_spherical = project_cartesian_to_spherical(
            velocity_field_inertial_cartesian, spherical_coordinates, torch
        )
        magnetic_field_observer = project_spherical_to_observer(
            magnetic_field_spherical, spherical_coordinates, stokes_basis, torch
        )
        velocity_field_observer = project_spherical_to_observer(
            velocity_field_inertial_spherical,
            spherical_coordinates,
            stokes_basis,
            torch,
        )
        synthesis_atmosphere = replace(
            sampled_atmosphere,
            velocity_field=velocity_field_observer,
            magnetic_field=magnetic_field_observer,
        )
        synthesis_wavelength = self.synthesis_wavelength_base
        synthesis_kwargs = {}
        if return_physics_diagnostics:
            synthesis_kwargs["return_diagnostics"] = True
        synthesis_kwargs["ray_distance_m"] = ray_trace.distance_m
        synthesis_result = self.synthesizer(
            synthesis_atmosphere,
            synthesis_wavelength,
            1.0,
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
            "ray_trace": ray_trace,
            "spherical_coordinates": spherical_coordinates,
            "magnetic_field_spherical": magnetic_field_spherical,
            "velocity_field_spherical": velocity_field_spherical,
            "rotation_velocity_cartesian": rotation_velocity_cartesian,
            "velocity_field_inertial_cartesian": velocity_field_inertial_cartesian,
            "velocity_field_inertial_spherical": velocity_field_inertial_spherical,
            "magnetic_field_observer": magnetic_field_observer,
            "velocity_field_observer": velocity_field_observer,
        }

    def forward(
        self,
        coords: torch.Tensor,
        mu: torch.Tensor,
        **ray_geometry,
    ) -> torch.Tensor:
        return self.synthesize(coords, mu, **ray_geometry)["stokes"]

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

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        *,
        return_callback_payload: bool = False,
        depth_grid: torch.Tensor | None = None,
        log_metrics: bool = True,
    ):
        def assert_float32_tree(value, path="batch"):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() and value.dtype != torch.float32:
                    raise TypeError(
                        f"{path} must use float32 during LTE inversion; got {value.dtype}."
                    )
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    assert_float32_tree(item, f"{path}.{key}")

        assert_float32_tree(batch)
        physics_volume_batch = None
        optical_depth_anchor_batch = None
        if stage == "train" and "physics_volume" in batch:
            physics_volume_batch = batch["physics_volume"]
            optical_depth_anchor_batch = batch.get("optical_depth_anchor")
            batch = batch["stokes"]
        result = self.synthesize(
            batch["coords"],
            batch["mu"],
            ray_origin_m=batch.get("ray_origin_m"),
            ray_direction=batch.get("ray_direction"),
            stokes_basis=batch.get("stokes_basis"),
            randomize_depth=stage == "train",
            depth_grid=depth_grid,
            return_physics_diagnostics=False,
        )
        final_weights = False
        stokes_weights = self.stokes_weights
        component_mse, stokes_loss = self._stokes_objective(
            result["stokes"], batch["stokes"], stokes_weights
        )
        step = int(getattr(self, "global_step", 0))
        if self.physics.volume_enabled:
            if physics_volume_batch is None:
                sample_count = batch["coords"].shape[0]
                height = self.atmosphere_model.depth_to_height(
                    self.physics_validation_grid(), (sample_count,)
                )
                physics_height_m = height.reshape(-1)
                expanded_coords = batch["coords"].unsqueeze(-2).expand(
                    *height.shape, 3
                )
                physics_coords = expanded_coords.reshape(-1, 3)
                expanded_origin = batch["ray_origin_m"].unsqueeze(-2).expand(
                    *height.shape, 3
                )
                expanded_direction = batch["ray_direction"].unsqueeze(-2).expand_as(
                    expanded_origin
                )
                physics_position_m = self.atmosphere_model.ray_shell_positions(
                    expanded_origin,
                    expanded_direction,
                    height,
                    expanded_coords,
                ).reshape(-1, 3)
            else:
                physics_coords = physics_volume_batch["coords"]
                height = physics_volume_batch["geometric_height_m"].to(
                    self.atmosphere_model.log_tau500
                )
                physics_height_m = height
                physics_position_m = self.atmosphere_model.ray_shell_positions(
                    physics_volume_batch["ray_origin_m"],
                    physics_volume_batch["ray_direction"],
                    height,
                    physics_volume_batch["coords"],
                )
            physics_result = self.physics.volume(
                self.atmosphere_model,
                self.synthesizer.continuum_opacity,
                physics_coords,
                physics_height_m,
                global_step=step,
                final_weights=final_weights,
                create_graph=stage == "train",
                return_state=False,
                position_m=physics_position_m,
            )
        else:
            zero = stokes_loss.new_zeros(())
            physics_result = LTEPhysicsResult(
                losses={name: zero for name in VOLUME_EQUATIONS},
                residual_norms={},
                weights=self.physics.weights(step, final=final_weights),
            )
        if self.upper_boundary_sampling_enabled:
            if stage == "train" and optical_depth_anchor_batch is None:
                raise KeyError(
                    "Upper-boundary physics requires an independent "
                    "'optical_depth_anchor' batch."
                )
            upper_boundary_coords = (
                optical_depth_anchor_batch["coords"]
                if optical_depth_anchor_batch is not None
                else batch["coords"]
            )
        else:
            upper_boundary_coords = None
        boundary_losses = {}
        anchor_name = "mean_radial_optical_depth_anchor"
        if self.optical_depth_anchor_enabled:
            anchor_result = self.physics.mean_radial_optical_depth_anchor(
                self.atmosphere_model,
                self.synthesizer.continuum_opacity,
                upper_boundary_coords,
                global_step=step,
                final_weights=final_weights,
                depth_points=self.optical_depth_anchor_depth_points,
            )
            boundary_losses[anchor_name] = anchor_result.losses[anchor_name]
        else:
            boundary_losses[anchor_name] = stokes_loss.new_zeros(())
        pressure_prior_name = "upper_boundary_gas_pressure_prior"
        if self.upper_boundary_pressure_prior_enabled:
            pressure_prior_result = self.physics.upper_boundary_gas_pressure_prior(
                self.atmosphere_model,
                upper_boundary_coords,
                global_step=step,
                final_weights=final_weights,
            )
            boundary_losses[pressure_prior_name] = pressure_prior_result.losses[
                pressure_prior_name
            ]
        else:
            boundary_losses[pressure_prior_name] = stokes_loss.new_zeros(())
        available_physics_losses = {
            **physics_result.losses,
            **boundary_losses,
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
            self.learning_rate_schedule["end"]
            / self.learning_rate_schedule["start"]
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
            "schema_version": 32,
            "compute_dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
            "units": {
                "wavelength": "standard-air angstrom at data/instrument boundary",
                "wavelength_internal": "vacuum angstrom for frequency, Planck, and opacity",
                "log_tau500": (
                    "derived log10 dimensionless continuum optical depth at 5000 "
                    "vacuum angstrom; stored depth grid is only a shell quadrature label"
                ),
                "mu": "dimensionless ray cosine",
                "stokes": (
                    "observations and synthesis are expressed in fixed disk-center "
                    "atlas-continuum units I_c,atlas(mu=1); one quiet-Sun-derived "
                    "detector-to-radiance scalar is applied uniformly to I,Q,U,V"
                ),
                "temperature": "K",
                "velocity_field": (
                    "learned co-rotating residual in m/s expressed in the "
                    "Heliographic Carrington Cartesian basis [Xc,Yc,Zc]; rigid "
                    "Carrington rotation is excluded from this field"
                ),
                "magnetic_field": (
                    "gauss, Heliographic Carrington Cartesian [Xc,Yc,Zc], with +Zc "
                    "toward solar north"
                ),
                "vector_projection": (
                    "model Cartesian [Xc,Yc,Zc] -> local spherical "
                    "[r,theta(colatitude),phi(longitude)]; rigid Carrington rotation "
                    "Omega cross r is added to velocity before projection into the supplied per-ray "
                    "observer basis [+Q,+U,toward_observer]; physics losses use "
                    "the original Cartesian fields"
                ),
                "carrington_rotation": {
                    "sidereal_period_days": CARRINGTON_SIDEREAL_ROTATION_PERIOD_DAYS,
                    "angular_velocity_rad_per_s": CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S,
                    "axis": "+Zc",
                    "formula": "v_rotation = Omega_Carrington cross r",
                },
                "microturbulence": "m/s",
                "gas_pressure": "Pa",
                "geometric_height": "m, radius minus R_sun in the fixed spherical shell",
                "ray_geometry": (
                    "heliocentric Carrington Cartesian in solar-radius units during "
                    "intersection; exported positions and local path offsets are metres"
                ),
                "spatial_coordinates": (
                    "[time_hours, Carrington gnomonic chart-X Mm, chart-Y Mm]; "
                    "the static model ignores time"
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
                    "reference-log-tau strata mapped through the pinned radial "
                    "stratification; tau500 is derived from absolute opacity"
                ),
                "evaluation_grid": "configured deterministic outer-to-inner shell grid",
                "sample_count": int(self.atmosphere_model.log_tau500.numel()),
                "training_sample_count": self.training_depth_sample_count,
                "coarse_to_fine": {
                    "enabled": self.coarse_to_fine_enabled,
                    "coarse_sample_count": self.training_depth_sample_count,
                    "fine_sample_count": self.coarse_to_fine_sample_count,
                    "final_sample_count": (
                        self.training_depth_sample_count
                        + self.coarse_to_fine_sample_count
                        if self.coarse_to_fine_enabled
                        else self.training_depth_sample_count
                    ),
                    "importance": "detached alpha500*exp(-tau500)*delta_s contribution",
                    "selection": "deterministic per-ray probability quantiles",
                    "uniform_weight_floor": self.coarse_to_fine_uniform_weight_floor,
                    "coarse_points_retained": True,
                },
                "domain": self.atmosphere_model.log_tau500[[0, -1]]
                .detach()
                .cpu()
                .tolist(),
                "strata": (
                    "linear equally spaced log_tau500 centers with fixed adjacent-"
                    "midpoint bounds; independent uniform interior draws and fixed endpoints"
                ),
                "integration": (
                    "exact observer-ray delta-s between ordered physical points; "
                    "unreachable inner radii are compressed to a pre-tangent endpoint; "
                    "optional detached per-ray importance refinement preserves gradients "
                    "through the final atmosphere and formal-solver evaluations"
                ),
            },
            "physics": {
                **self.physics.configuration(),
                "equation_definitions": {
                    "magnetohydrostatic_equilibrium": (
                        "[grad(P) + rho*g*radial_unit - (curl(B) cross B)/mu0] "
                        "*L0/mean_same_height_layer(P + rho*g*L0 + |B|^2/mu0) = 0"
                    ),
                    "mean_radial_optical_depth_anchor": (
                        "mean_observed_domain(log10(integral_from_shell_top_to_Rsun "
                        "alpha500 dr)) = 0"
                    ),
                    "upper_boundary_gas_pressure_prior": (
                        "log(P_gas/P_FALC) at the upper shell boundary = 0"
                    ),
                    "magnetic_divergence": (
                        "div(B)*L0/mean_same_height_layer(|B|) = 0"
                    ),
                },
                "coordinate": "Carrington Cartesian position [x_m,y_m,z_m]",
                "unit_contract": (
                    "Magnetohydrostatic equilibrium differentiates SI pressure and tesla magnetic field "
                    "in Carrington Cartesian space; its force residual is normalized by the layer mean of "
                    "P + rho*g*L0 + |B|^2/mu0. div(B) uses gauss and is normalized by "
                    "the same sampled-height layer mean |B| with configured physical length L0; "
                    "neither normalization uses an optical-depth coordinate"
                ),
                "density_source": "pinned STiC/Wittmann differentiable lookup",
                "iterative_forward_solve": False,
                "derivative_strategy": (
                    "one selectively populated primitive Jacobian shared by magnetohydrostatic equilibrium and div(B)"
                ),
                "volume_sampling": (
                    "uniform random geometric-height layers with Carrington-Cartesian "
                    "points on observed rays that reach each layer"
                ),
                "upper_boundary_sampling": (
                    "independent valid observed-domain chart coordinates at the "
                    "upper shell boundary; the optical-depth anchor additionally "
                    "integrates radial columns to the exact solar radius"
                    if self.upper_boundary_sampling_enabled
                    else "none"
                ),
                "sampling_contract": physics_sampling,
                "gravity_m_per_s2": self.gravity_m_per_s2,
                "volume_points_per_step": self.physics_volume_points_per_step,
                "height_layers_per_step": self.physics_height_layers_per_step,
                "optical_depth_anchor_points_per_step": self.optical_depth_anchor_points_per_step,
                "optical_depth_anchor_depth_points": self.optical_depth_anchor_depth_points,
                "validation_grid": "deterministic linear log_tau grid",
                "validation_depth_points": self.physics_validation_depth_points,
                "optical_depth_mapping": {
                    "equation": "tau500(s)=integral_from_outer_boundary^s alpha500 ds",
                    "training_residual": (
                        "mean radial log10(tau500) at the solar radius"
                        if self.optical_depth_anchor_enabled
                        else None
                    ),
                    "validation_residual": "cumulative optical depth along each realized ray",
                    "height_model": "fixed FALC radial shell labels plus learned atmospheric perturbations",
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
                "learning_rate": deepcopy(self.learning_rate_configuration),
                "schedule": (
                    None
                    if self.learning_rate_schedule is None
                    else "per-step exponential decay"
                ),
                "resolved_iterations": self.resolved_learning_rate_iterations,
            },
            "velocity_zero_point": (
                "relative solar velocity in the fixed Hinode Level-1 gauge: "
                "sp_prep removed DOP_RCV and registered the slit-averaged Fe I "
                "6301.5 centre; rigid Carrington rotation is supplied explicitly "
                "and is not absorbed by the learned velocity"
            ),
            "data": deepcopy(self.checkpoint_metadata),
        }
