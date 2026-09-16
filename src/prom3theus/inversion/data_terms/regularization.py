"""Shared atmosphere and magnetofluid objectives for schema-v3 runs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from prom3theus.inversion.assembly import PhysicsAssembly
from prom3theus.inversion.constraints.magnetofluid import EQUATION_NAMES, PhysicsResult
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.rt.atmosphere import _batched_vector_jacobian

from .base import SharedObjectiveTerm, SharedTermResult


class AtmosphereRegularizationTerm(SharedObjectiveTerm):
    """Evaluate one explicit prior on deterministic physical shell samples."""

    def __init__(
        self,
        *,
        kind: str,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        component_weights: Mapping[str, float],
        smoothness_step_m: float | None = None,
    ) -> None:
        super().__init__()
        if kind not in {
            "stic_table_support",
            "vector_magnitude",
            "vector_potential",
        }:
            raise ValueError(f"Unknown atmosphere regularization kind {kind!r}.")
        if smoothness_step_m is not None:
            smoothness_step_m = float(smoothness_step_m)
            if not torch.isfinite(torch.as_tensor(smoothness_step_m)) or smoothness_step_m <= 0.0:
                raise ValueError("smoothness_step_m must be finite and positive.")
        if kind == "vector_potential" and smoothness_step_m is None:
            smoothness_step_m = 0.25e6
        position = torch.as_tensor(position_m, dtype=torch.float32)
        time = torch.as_tensor(time_hours, dtype=torch.float32)
        if position.ndim < 2 or position.shape[-1] != 3 or not position.numel():
            raise ValueError("Regularization positions must end in three values.")
        if time.shape not in {position.shape[:-1], (*position.shape[:-1], 1)}:
            raise ValueError("Regularization times must match sampled positions.")
        if not torch.isfinite(position).all() or not torch.isfinite(time).all():
            raise ValueError("Regularization samples must be finite.")
        weights = {name: float(value) for name, value in component_weights.items()}
        if not weights or any(value < 0 for value in weights.values()):
            raise ValueError(
                "Regularization weights must be non-negative and non-empty."
            )
        self.kind = kind
        self.component_weights = weights
        self.smoothness_step_m = smoothness_step_m
        self.register_buffer("position_m", position, persistent=False)
        self.register_buffer("time_hours", time, persistent=False)

    def evaluate(self, atmosphere_model) -> SharedTermResult:
        if self.kind == "vector_potential":
            if getattr(atmosphere_model, "magnetic_representation", "direct") != "vector_potential":
                raise ValueError(
                    "vector_potential regularization requires the vector_potential "
                    "magnetic representation."
                )
            parameter = next(atmosphere_model.parameters())
            position = self.position_m.reshape(-1, 3).to(parameter)
            time = self.time_hours.reshape(-1, 1).to(parameter)
            position_rsun = (
                position / atmosphere_model.solar_radius_m.to(parameter)
            ).detach().requires_grad_(True)
            normalized_vector_potential = getattr(
                atmosphere_model,
                "_evaluate_vector_potential_only_normalized",
                None,
            )
            if normalized_vector_potential is None:
                vector_potential = atmosphere_model._evaluate_vector_potential_only(
                    position_rsun,
                    time,
                ) / atmosphere_model.vector_potential_scale_gauss_m
            else:
                vector_potential = normalized_vector_potential(position_rsun, time)
            raw = {
                "gauge": parameter.new_zeros(()),
                "smoothness": parameter.new_zeros(()),
            }
            if self.component_weights.get("gauge", 0.0) > 0.0:
                jacobian = _batched_vector_jacobian(
                    vector_potential,
                    position_rsun,
                    create_graph=True,
                )
                divergence = torch.diagonal(jacobian, dim1=-2, dim2=-1).sum(dim=-1)
                # ``position_rsun`` is dimensionless and A_hat is already
                # normalized by B0*L0.  Convert d/d(position/R_sun) to
                # d/d(x/L0) using the dimensionless solar-radius ratio only.
                solar_radius_model = getattr(atmosphere_model, "solar_radius_model", None)
                if solar_radius_model is None:
                    solar_radius_model = (
                        atmosphere_model.solar_radius_m
                        / atmosphere_model.height_input_scale_m
                    )
                raw["gauge"] = (divergence / solar_radius_model.to(divergence)).square().mean()
            if self.component_weights.get("smoothness", 0.0) > 0.0:
                step_m = float(self.smoothness_step_m)
                step_model = step_m / float(atmosphere_model.height_input_scale_m)
                solar_radius_model_value = getattr(
                    atmosphere_model,
                    "solar_radius_model",
                    atmosphere_model.solar_radius_m / atmosphere_model.height_input_scale_m,
                )
                step_rsun = step_model / float(solar_radius_model_value)
                eye = torch.eye(
                    3,
                    dtype=position_rsun.dtype,
                    device=position_rsun.device,
                )
                plus = (
                    position_rsun[:, None, :] + step_rsun * eye[None, :, :]
                ).reshape(-1, 3)
                minus = (
                    position_rsun[:, None, :] - step_rsun * eye[None, :, :]
                ).reshape(-1, 3)
                repeated_time = time.repeat_interleave(3, dim=0)
                if normalized_vector_potential is None:
                    plus_potential = (
                        atmosphere_model._evaluate_vector_potential_only(
                            plus,
                            repeated_time,
                        ) / atmosphere_model.vector_potential_scale_gauss_m
                    ).reshape(-1, 3, 3)
                    minus_potential = (
                        atmosphere_model._evaluate_vector_potential_only(
                            minus,
                            repeated_time,
                        ) / atmosphere_model.vector_potential_scale_gauss_m
                    ).reshape(-1, 3, 3)
                else:
                    plus_potential = normalized_vector_potential(
                        plus,
                        repeated_time,
                    ).reshape(-1, 3, 3)
                    minus_potential = normalized_vector_potential(
                        minus,
                        repeated_time,
                    ).reshape(-1, 3, 3)
                second_derivative = (
                    plus_potential
                    - 2.0 * vector_potential[:, None, :]
                    + minus_potential
                ) / (step_model * step_model)
                raw["smoothness"] = second_derivative.square().mean()
        else:
            position = self.position_m.reshape(-1, 3)
            time = self.time_hours.reshape(-1, 1)
            fields_are_normalized = False
            if self.kind == "stic_table_support":
                # This term deliberately crosses the physical table adapter.
                fields = atmosphere_model.evaluate_position_points(
                    position,
                    time_hours=time,
                )
            else:
                normalized_evaluator = getattr(
                    atmosphere_model,
                    "evaluate_position_rsun_normalized",
                    None,
                )
                if normalized_evaluator is None:
                    fields = atmosphere_model.evaluate_position_points(
                        position,
                        time_hours=time,
                    )
                    normalizer = getattr(
                        atmosphere_model, "normalize_atmosphere_fields", None
                    )
                    if normalizer is not None:
                        fields = normalizer(fields)
                        fields_are_normalized = True
                else:
                    fields = normalized_evaluator(
                        position / atmosphere_model.solar_radius_m.to(position),
                        time_hours=time,
                    )
                    fields_are_normalized = True
            if self.kind == "stic_table_support":
                (
                    temperature,
                    pressure,
                ) = atmosphere_model.thermodynamic_eos.stic_support_residuals(
                    fields["temperature"], fields["gas_pressure"]
                )
                raw = {
                    "temperature": temperature.square().mean(),
                    "gas_pressure": pressure.square().mean(),
                }
            else:
                magnetic = fields["magnetic_field"]
                velocity = fields["velocity_field"]
                if not fields_are_normalized:
                    magnetic = magnetic / atmosphere_model.magnetic_scale_gauss
                    velocity = velocity / atmosphere_model.velocity_scale_m_per_s
                raw = {
                    "magnetic": magnetic.square().mean(),
                    "velocity": velocity.square().mean(),
                }
        unknown = set(self.component_weights) - set(raw)
        if unknown:
            raise KeyError(
                f"Regularization weights do not match {self.kind}: {sorted(unknown)}."
            )
        weighted = {
            name: raw[name] * weight
            for name, weight in self.component_weights.items()
            if weight > 0
        }
        total = sum(weighted.values(), start=next(iter(raw.values())).new_zeros(()))
        return SharedTermResult(
            loss=total,
            component_losses=weighted,
            active=bool(weighted),
            metrics={f"raw_{name}": value for name, value in raw.items()},
        )


class PhysicsConstraintTerm(SharedObjectiveTerm):
    """Evaluate equations on random fitting or fixed validation samples."""

    _VOLUME_CURRENT_FREE = "magnetic_current_free"

    def __init__(self, assembly: PhysicsAssembly, *, create_graph: bool = True) -> None:
        super().__init__()
        if type(create_graph) is not bool:
            raise TypeError("create_graph must be boolean.")
        self.constraints = assembly.constraints
        self.create_graph = create_graph
        self.magnetic_current_free_steps = int(assembly.magnetic_current_free_steps)
        self.magnetic_current_free_final_factor = float(assembly.magnetic_current_free_final_factor)
        self.sampling_domain = assembly.sampling_domain
        self.upper_sampling_domain = assembly.upper_sampling_domain
        self.volume_points_per_step = int(assembly.volume_points_per_step)
        self.height_layers_per_step = int(assembly.height_layers_per_step)
        self.upper_volume_points_per_step = int(assembly.upper_volume_points_per_step)
        self.upper_height_layers_per_step = int(assembly.upper_height_layers_per_step)
        self.upper_boundary_points_per_step = int(
            assembly.upper_boundary_points_per_step
        )
        self.side_boundary_points_per_step = int(assembly.side_boundary_points_per_step)
        self.loss_start_step = int(getattr(assembly, "loss_start_step", 0))
        self.loss_ramp_steps = int(getattr(assembly, "loss_ramp_steps", 0))
        self.height_sampling_power = float(
            getattr(assembly, "height_sampling_power", 2.0)
        )
        self.volume_sampling = str(
            getattr(assembly, "volume_sampling", "height_grouped")
        )
        self.residual_adaptive_enabled = bool(
            getattr(assembly, "residual_adaptive_enabled", False)
        )
        self.residual_adaptive_candidate_multiplier = int(
            getattr(assembly, "residual_adaptive_candidate_multiplier", 2)
        )
        self.residual_adaptive_fraction = float(
            getattr(assembly, "residual_adaptive_fraction", 0.5)
        )
        self.residual_adaptive_start_step = int(
            getattr(assembly, "residual_adaptive_start_step", 500)
        )
        self.residual_adaptive_update_every_n_steps = int(
            getattr(assembly, "residual_adaptive_update_every_n_steps", 250)
        )
        # Residual-adaptive locations are refreshed periodically and reused
        # between refreshes.  Keep these as non-persistent runtime state: the
        # next refresh is deterministic under the checkpointed torch RNG and
        # does not make collocation geometry part of the scientific contract.
        self._adaptive_position_m: torch.Tensor | None = None
        self._adaptive_time_hours: torch.Tensor | None = None
        self._adaptive_shape: tuple[int, int] | None = None
        self.hard_example_enabled = bool(
            getattr(assembly, "hard_example_enabled", False)
        )
        self.hard_example_count = int(getattr(assembly, "hard_example_count", 0))
        self.hard_example_start_step = int(
            getattr(assembly, "hard_example_start_step", 0)
        )
        self.hard_example_jitter_length_m = float(
            getattr(assembly, "hard_example_jitter_length_m", 50_000.0)
        )
        self.hard_example_jitter_time_hours = float(
            getattr(assembly, "hard_example_jitter_time_hours", 0.0)
        )
        # Carried across steps exactly like the residual-adaptive cache above:
        # non-persistent runtime state, not part of the scientific contract.
        self._hard_position_m: torch.Tensor | None = None
        self._hard_time_hours: torch.Tensor | None = None
        self.register_buffer(
            "current_step", torch.zeros((), dtype=torch.long), persistent=False
        )
        for name, value in (
            ("volume_position_m", assembly.validation_position_m),
            ("volume_time_hours", assembly.validation_time_hours),
            ("upper_position_m", assembly.validation_upper_position_m),
            ("upper_time_hours", assembly.validation_upper_time_hours),
            ("boundary_position_m", assembly.validation_boundary_position_m),
            ("boundary_time_hours", assembly.validation_boundary_time_hours),
            ("side_position_m", assembly.validation_side_position_m),
            ("side_time_hours", assembly.validation_side_time_hours),
            ("side_normal", assembly.validation_side_normal),
        ):
            self.register_buffer(name, value, persistent=False)

    def set_step(self, step: int) -> None:
        """Set the future fitting step without coupling this term to Lightning."""

        if type(step) is not int or step < 0:
            raise ValueError("Physics step must be a non-negative integer.")
        self.current_step.fill_(step)

    def _magnetic_current_free_factor(self) -> float:
        if self.magnetic_current_free_steps == 0:
            return 0.0
        if int(self.current_step) < self.magnetic_current_free_steps:
            return 1.0
        return self.magnetic_current_free_final_factor

    def _curriculum_factor(self) -> float:
        """Return the staged physics multiplier at the current fitting step."""

        step = int(self.current_step)
        if step < self.loss_start_step:
            return 0.0
        if self.loss_ramp_steps <= 0:
            return 1.0
        return min(1.0, (step - self.loss_start_step) / self.loss_ramp_steps)

    def _nontrivial_volume_active(self, atmosphere_model=None) -> bool:
        """Whether residual-adaptive sampling has a non-identically-zero target."""

        active = {
            name
            for name in (
                "hydrostatic_equilibrium",
                "magnetohydrostatic_equilibrium",
                "momentum",
                "magnetic_divergence",
                "magnetic_force_free",
                "magnetic_current_free",
                "radial_magnetic_energy_gradient",
                "radial_magnetic_field",
                "induction",
                "continuity",
                "adiabatic_pressure",
            )
            if self.constraints.is_active(name)
            and (
                name != self._VOLUME_CURRENT_FREE
                or self._magnetic_current_free_factor() > 0.0
            )
        }
        if not active:
            return False
        return not (
            active == {"magnetic_divergence"}
            and atmosphere_model is not None
            and getattr(atmosphere_model, "magnetic_representation", "direct")
            == "vector_potential"
        )

    def _adaptive_sampling_active(self, atmosphere_model=None) -> bool:
        if not self.residual_adaptive_enabled or not self.constraints.volume_active:
            return False
        if not self._nontrivial_volume_active(atmosphere_model):
            return False
        if self.residual_adaptive_fraction <= 0.0:
            return False
        step = int(self.current_step)
        return step >= self.residual_adaptive_start_step

    def _adaptive_refresh_due(self, shape: tuple[int, int]) -> bool:
        """Whether the cached high-residual locations need replacement."""

        if self._adaptive_position_m is None or self._adaptive_time_hours is None:
            return True
        if self._adaptive_shape != shape:
            return True
        step = int(self.current_step)
        if step < self.residual_adaptive_start_step:
            return False
        return (
            (step - self.residual_adaptive_start_step)
            % self.residual_adaptive_update_every_n_steps
            == 0
        )

    def _hard_example_active(self, atmosphere_model=None) -> bool:
        """Whether the persistent highest-residual carry-forward is in effect.

        Independent of ``residual_adaptive_*``: that mechanism only re-ranks
        candidate points within one already-selected discrete height layer,
        so it cannot compensate for an entire height band being sampled too
        rarely. This mechanism instead keeps the globally highest-residual
        points from the previous step (jittered) as part of every subsequent
        step's batch, regardless of how collocation heights are drawn.
        """

        if not self.hard_example_enabled or not self.constraints.volume_active:
            return False
        if not self._nontrivial_volume_active(atmosphere_model):
            return False
        if self.hard_example_count <= 0:
            return False
        return int(self.current_step) >= self.hard_example_start_step

    def _jitter_hard_examples(
        self, position_m: torch.Tensor, time_hours: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perturb carried-forward points and keep them inside the domain."""

        domain = self.sampling_domain
        if self.hard_example_jitter_length_m > 0.0:
            position_m = position_m + torch.randn_like(position_m) * (
                self.hard_example_jitter_length_m
            )
        if self.hard_example_jitter_time_hours > 0.0:
            time_hours = time_hours + torch.randn_like(time_hours) * (
                self.hard_example_jitter_time_hours
            )
        radius_m = torch.linalg.vector_norm(position_m, dim=-1, keepdim=True)
        minimum_radius_m = (
            domain.solar_radius_m + domain.height_bounds_Mm[0] * 1.0e6
        )
        maximum_radius_m = (
            domain.solar_radius_m + domain.height_bounds_Mm[1] * 1.0e6
        )
        clamped_radius_m = radius_m.clamp(min=minimum_radius_m, max=maximum_radius_m)
        safe_radius_m = radius_m.clamp_min(torch.finfo(radius_m.dtype).tiny)
        position_m = position_m * (clamped_radius_m / safe_radius_m)
        time_hours = time_hours.clamp(
            domain.time_bounds_hours[0], domain.time_bounds_hours[1]
        )
        return position_m, time_hours

    def _update_hard_examples(
        self,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        equation_diagnostics: Mapping[str, Mapping[str, torch.Tensor]] | None,
        shape: tuple[int, int],
    ) -> None:
        """Score this step's points and cache the hardest ones, jittered."""

        height_count, points_per_height = shape
        score = position_m.new_zeros((height_count, points_per_height))
        for name, diagnostic in (equation_diagnostics or {}).items():
            if (
                name == self._VOLUME_CURRENT_FREE
                and self._magnetic_current_free_factor() <= 0.0
            ):
                continue
            residual = diagnostic.get("normalized_residual")
            if residual is None:
                continue
            residual = residual.to(score).reshape(height_count, points_per_height, -1)
            score = score + torch.nan_to_num(
                residual.square().mean(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            )
        flat_score = score.reshape(-1).detach()
        flat_position = position_m.reshape(-1, 3).detach()
        flat_time = time_hours.reshape(-1, 1).detach()
        keep = min(self.hard_example_count, flat_score.numel())
        if keep <= 0:
            self._hard_position_m = None
            self._hard_time_hours = None
            return
        top_indices = flat_score.topk(keep).indices
        hard_position, hard_time = self._jitter_hard_examples(
            flat_position[top_indices], flat_time[top_indices]
        )
        self._hard_position_m = hard_position.reshape(keep, 1, 3)
        self._hard_time_hours = hard_time.reshape(keep, 1, 1)

    @staticmethod
    def _call_grouped_sampler(
        domain,
        height_count: int,
        points_per_height: int,
        *,
        height_power: float,
        device,
    ) -> dict[str, torch.Tensor]:
        try:
            return domain.random_grouped(
                height_count,
                points_per_height,
                height_power=height_power,
                device=device,
            )
        except TypeError as error:
            # Small test/d downstream domain adapters may still implement the
            # pre-height-power sampler contract.
            if "height_power" not in str(error):
                raise
            return domain.random_grouped(
                height_count,
                points_per_height,
                device=device,
            )

    def _adaptive_volume_samples(
        self,
        atmosphere_model,
        domain: SphericalShellDomain,
        height_count: int,
        points_per_height: int,
        *,
        device,
    ) -> dict[str, torch.Tensor]:
        """Keep a uniform floor while concentrating half of each height layer."""

        candidate_points = max(
            points_per_height,
            points_per_height * self.residual_adaptive_candidate_multiplier,
        )
        candidate = self._call_grouped_sampler(
            domain,
            height_count,
            candidate_points,
            height_power=self.height_sampling_power,
            device=device,
        )
        candidate_position = candidate["position_m"]
        candidate_time = candidate["time_hours"]
        shape = (height_count, candidate_points)
        with torch.enable_grad():
            scored = self.constraints.volume(
                atmosphere_model,
                candidate_position.reshape(-1, 3),
                candidate_time.reshape(-1, 1),
                height_group_shape=shape,
                create_graph=False,
                return_state=False,
                return_diagnostics=True,
            )
        score = candidate_position.new_zeros(shape)
        for name, diagnostic in (scored.equation_diagnostics or {}).items():
            if (
                name == self._VOLUME_CURRENT_FREE
                and self._magnetic_current_free_factor() <= 0.0
            ):
                continue
            residual = diagnostic.get("normalized_residual")
            if residual is None:
                continue
            residual = residual.to(score).reshape(height_count, candidate_points, -1)
            score = score + torch.nan_to_num(
                residual.square().mean(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            )
        adaptive_count = min(
            candidate_points,
            max(0, round(points_per_height * self.residual_adaptive_fraction)),
        )
        uniform_count = points_per_height - adaptive_count
        if adaptive_count:
            if float(score.max()) <= 0.0:
                ranking_score = torch.rand_like(score)
            else:
                ranking_score = score + (
                    torch.finfo(score.dtype).eps
                    * score.detach().abs().amax().clamp_min(1.0)
                    * torch.rand_like(score)
                )
            adaptive_indices = ranking_score.topk(adaptive_count, dim=1).indices
        else:
            adaptive_indices = score.new_empty((height_count, 0), dtype=torch.long)
        if uniform_count:
            uniform_rank = torch.rand(
                shape,
                dtype=score.dtype,
                device=score.device,
            )
            if adaptive_count:
                selected = torch.zeros(shape, dtype=torch.bool, device=score.device)
                selected.scatter_(1, adaptive_indices, True)
                uniform_rank = uniform_rank.masked_fill(selected, float("inf"))
            uniform_indices = uniform_rank.argsort(dim=1)[:, :uniform_count]
        else:
            uniform_indices = score.new_empty((height_count, 0), dtype=torch.long)
        indices = torch.cat((adaptive_indices, uniform_indices), dim=1)

        def select(values: torch.Tensor) -> torch.Tensor:
            expanded = indices[(...,) + (None,) * (values.ndim - 2)].expand(
                *indices.shape, *values.shape[2:]
            )
            return torch.gather(values, 1, expanded)

        return {
            "position_m": select(candidate_position),
            "time_hours": select(candidate_time),
        }

    def _samples(
        self,
        reference: torch.Tensor,
        atmosphere_model=None,
    ) -> dict[str, torch.Tensor]:
        if not self.training:
            return {
                "volume_position_m": self.volume_position_m,
                "volume_time_hours": self.volume_time_hours,
                "upper_position_m": self.upper_position_m,
                "upper_time_hours": self.upper_time_hours,
                "boundary_position_m": self.boundary_position_m,
                "boundary_time_hours": self.boundary_time_hours,
                "side_position_m": self.side_position_m,
                "side_time_hours": self.side_time_hours,
                "side_normal": self.side_normal,
            }
        if self.sampling_domain is None:
            raise RuntimeError(
                "Training-active physics requires an initialized sampling domain."
            )
        device = reference.device
        samples: dict[str, torch.Tensor] = {}
        if self.constraints.volume_active:
            height_count = self.height_layers_per_step
            points_per_height = self.volume_points_per_step // height_count
            if atmosphere_model is not None and self._hard_example_active(
                atmosphere_model
            ):
                fresh_count = self.volume_points_per_step - (
                    0
                    if self._hard_position_m is None
                    else self._hard_position_m.shape[0]
                )
                if fresh_count > 0:
                    fresh = self._call_grouped_sampler(
                        self.sampling_domain,
                        fresh_count,
                        1,
                        height_power=self.height_sampling_power,
                        device=device,
                    )
                if self._hard_position_m is not None:
                    hard_position = self._hard_position_m.to(device)
                    hard_time = self._hard_time_hours.to(device)
                    if fresh_count > 0:
                        position = torch.cat((hard_position, fresh["position_m"]), dim=0)
                        time = torch.cat((hard_time, fresh["time_hours"]), dim=0)
                    else:
                        position = hard_position
                        time = hard_time
                else:
                    # Bootstrap: no carried-forward hard set yet.
                    position = fresh["position_m"]
                    time = fresh["time_hours"]
                volume = {"position_m": position, "time_hours": time}
            elif atmosphere_model is not None and self._adaptive_sampling_active(
                atmosphere_model
            ):
                shape = (height_count, points_per_height)
                adaptive_count = min(
                    points_per_height,
                    max(0, round(points_per_height * self.residual_adaptive_fraction)),
                )
                if self._adaptive_refresh_due(shape):
                    refreshed = self._adaptive_volume_samples(
                        atmosphere_model,
                        self.sampling_domain,
                        height_count,
                        points_per_height,
                        device=device,
                    )
                    self._adaptive_position_m = refreshed["position_m"][
                        :, :adaptive_count
                    ].detach()
                    self._adaptive_time_hours = refreshed["time_hours"][
                        :, :adaptive_count
                    ].detach()
                    self._adaptive_shape = shape
                adaptive_position = self._adaptive_position_m.to(device)
                adaptive_time = self._adaptive_time_hours.to(device)
                uniform_count = points_per_height - adaptive_count
                if uniform_count:
                    uniform = self._call_grouped_sampler(
                        self.sampling_domain,
                        height_count,
                        uniform_count,
                        height_power=self.height_sampling_power,
                        device=device,
                    )
                    position = torch.cat(
                        (adaptive_position, uniform["position_m"]), dim=1
                    )
                    time = torch.cat(
                        (adaptive_time, uniform["time_hours"]), dim=1
                    )
                else:
                    position = adaptive_position
                    time = adaptive_time
                volume = {"position_m": position, "time_hours": time}
            elif self.volume_sampling == "fully_random":
                volume = self._call_grouped_sampler(
                    self.sampling_domain,
                    self.volume_points_per_step,
                    1,
                    height_power=self.height_sampling_power,
                    device=device,
                )
            else:
                volume = self._call_grouped_sampler(
                    self.sampling_domain,
                    height_count,
                    points_per_height,
                    height_power=self.height_sampling_power,
                    device=device,
                )
            samples.update(
                volume_position_m=volume["position_m"],
                volume_time_hours=volume["time_hours"],
            )
        if self.constraints.upper_volume_active:
            if self.upper_sampling_domain is None:
                raise RuntimeError(
                    "Training-active upper physics requires its sampling domain."
                )
            upper = self._call_grouped_sampler(
                self.upper_sampling_domain,
                self.upper_height_layers_per_step,
                self.upper_volume_points_per_step // self.upper_height_layers_per_step,
                height_power=self.height_sampling_power,
                device=device,
            )
            samples.update(
                upper_position_m=upper["position_m"],
                upper_time_hours=upper["time_hours"],
            )
        if self.constraints.upper_boundary_active:
            boundary = self.sampling_domain.random_top(
                self.upper_boundary_points_per_step,
                device=device,
            )
            samples.update(
                boundary_position_m=boundary["position_m"].squeeze(0),
                boundary_time_hours=boundary["time_hours"].squeeze(0),
            )
        if self.constraints.side_boundary_active:
            try:
                side = self.sampling_domain.random_sides(
                    self.side_boundary_points_per_step,
                    height_power=self.height_sampling_power,
                    device=device,
                )
            except TypeError as error:
                if "height_power" not in str(error):
                    raise
                side = self.sampling_domain.random_sides(
                    self.side_boundary_points_per_step,
                    device=device,
                )
            samples.update(
                side_position_m=side["position_m"],
                side_time_hours=side["time_hours"],
                side_normal=side["normal"],
            )
        return samples

    def evaluate(self, atmosphere_model) -> SharedTermResult:
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        curriculum_factor = self._curriculum_factor()
        if curriculum_factor <= 0.0:
            return SharedTermResult(
                loss=zero,
                component_losses={},
                active=False,
                metrics={
                    "magnetic_current_free_factor": zero,
                    "physics_curriculum_factor": zero,
                    "physics_robust_loss_delta": zero
                    + getattr(self.constraints, "robust_loss_delta", 0.0),
                },
            )
        samples = self._samples(parameter, atmosphere_model)
        available: dict[str, torch.Tensor] = {}
        hard_example_active = self._hard_example_active(atmosphere_model)
        with torch.enable_grad():
            if self.constraints.volume_active:
                position = samples["volume_position_m"]
                time = samples["volume_time_hours"]
                shape = tuple(map(int, position.shape[:2]))
                volume_kwargs = dict(
                    create_graph=self.create_graph,
                    return_state=False,
                    height_group_shape=shape,
                )
                if hard_example_active:
                    volume_kwargs["return_diagnostics"] = True
                result = self.constraints.volume(
                    atmosphere_model,
                    position.reshape(-1, 3),
                    time.reshape(-1, 1),
                    **volume_kwargs,
                )
                if hard_example_active:
                    self._update_hard_examples(
                        position, time, result.equation_diagnostics, shape
                    )
            else:
                result = PhysicsResult(losses={}, weights=self.constraints.loss_weights)
            available.update(result.losses)
            if self.constraints.upper_volume_active:
                position = samples["upper_position_m"]
                time = samples["upper_time_hours"]
                upper_shape = tuple(map(int, position.shape[:2]))
                available.update(
                    self.constraints.upper_domain(
                        atmosphere_model,
                        position.reshape(-1, 3),
                        time.reshape(-1, 1),
                        height_group_shape=upper_shape,
                        create_graph=self.create_graph,
                    ).losses
                )
            if self.constraints.upper_boundary_active:
                position = samples["boundary_position_m"]
                time = samples["boundary_time_hours"]
                available.update(
                    self.constraints.upper_boundary(
                        atmosphere_model,
                        position.reshape(-1, 3),
                        time.reshape(-1, 1),
                        create_graph=self.create_graph,
                    ).losses
                )
            if self.constraints.side_boundary_active:
                position = samples["side_position_m"]
                time = samples["side_time_hours"]
                normal = samples["side_normal"]
                available.update(
                    self.constraints.side_boundary(
                        atmosphere_model,
                        position.reshape(-1, 3),
                        time.reshape(-1, 1),
                        normal.reshape(-1, 3),
                        create_graph=self.create_graph,
                    ).losses
                )
        active = {
            name: available[name]
            for name in EQUATION_NAMES
            if self.constraints.is_active(name)
        }
        volume_current_free_factor = self._magnetic_current_free_factor()
        weighted = {
            name: value
            * self.constraints.loss_weights[name]
            * (volume_current_free_factor if name == self._VOLUME_CURRENT_FREE else 1.0)
            * curriculum_factor
            for name, value in active.items()
            if name != self._VOLUME_CURRENT_FREE or volume_current_free_factor > 0
        }
        total = sum(weighted.values(), start=zero)
        return SharedTermResult(
            loss=total,
            component_losses=weighted,
            active=bool(weighted),
            metrics={
                "magnetic_current_free_factor": zero + volume_current_free_factor,
                "physics_curriculum_factor": zero + curriculum_factor,
                "physics_robust_loss_delta": zero
                + getattr(self.constraints, "robust_loss_delta", 0.0),
                **{f"raw_{name}": value for name, value in active.items()},
            },
        )


__all__ = ["AtmosphereRegularizationTerm", "PhysicsConstraintTerm"]
