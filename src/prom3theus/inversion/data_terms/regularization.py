"""Shared atmosphere and magnetofluid objectives for schema-v5 runs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from prom3theus.inversion.assembly import PhysicsAssembly
from prom3theus.inversion.constraints.magnetofluid import EQUATION_NAMES, PhysicsResult

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
    ) -> None:
        super().__init__()
        if kind not in {"stic_table_support", "vector_magnitude"}:
            raise ValueError(f"Unknown atmosphere regularization kind {kind!r}.")
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
        self.register_buffer("position_m", position, persistent=False)
        self.register_buffer("time_hours", time, persistent=False)

    def evaluate(self, atmosphere_model) -> SharedTermResult:
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
        if not self.constraints.is_active(self._VOLUME_CURRENT_FREE):
            return 0.0
        if self.magnetic_current_free_steps == 0 or int(self.current_step) < self.magnetic_current_free_steps:
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
                "vector_potential_gauge",
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

    def _hard_example_active(self, atmosphere_model=None) -> bool:
        """Whether the persistent highest-residual carry-forward is in effect.

        Each step keeps the highest-residual points of the *previous step*
        -- jittered -- in every subsequent step's batch, reusing diagnostics
        already computed for the loss instead of a separate scored
        candidate pool. Only meaningful under ``volume_sampling:
        fully_random``, where there are no discrete height groups to carry
        a point out of.
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
            domain.solar_radius_m + domain.height_bounds_megameter[0] * 1.0e6
        )
        maximum_radius_m = (
            domain.solar_radius_m + domain.height_bounds_megameter[1] * 1.0e6
        )
        clamped_radius_m = radius_m.clamp(min=minimum_radius_m, max=maximum_radius_m)
        safe_radius_m = radius_m.clamp_min(torch.finfo(radius_m.dtype).tiny)
        position_m = position_m * (clamped_radius_m / safe_radius_m)
        time_hours = time_hours.clamp(
            domain.time_bounds_hours[0], domain.time_bounds_hours[1]
        )
        return position_m, time_hours

    def _residual_score(
        self,
        equation_diagnostics: "Mapping[str, Mapping[str, torch.Tensor]] | None",
        shape: tuple[int, ...],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Sum the mean squared normalized residual of every scoring equation."""

        score = reference.new_zeros(shape)
        for name, diagnostic in (equation_diagnostics or {}).items():
            if (
                name == self._VOLUME_CURRENT_FREE
                and self._magnetic_current_free_factor() <= 0.0
            ):
                continue
            residual = diagnostic.get("normalized_residual")
            if residual is None:
                continue
            residual = residual.to(score).reshape(*shape, -1)
            score = score + torch.nan_to_num(
                residual.square().mean(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            )
        return score

    def _update_hard_examples(
        self,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        equation_diagnostics: Mapping[str, Mapping[str, torch.Tensor]] | None,
        shape: tuple[int, int],
    ) -> None:
        """Score this step's points and cache the hardest ones, jittered."""

        score = self._residual_score(equation_diagnostics, tuple(shape), position_m)
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

    def _random_volume_samples(
        self,
        *,
        device,
        hard_example_active: bool,
    ) -> dict[str, torch.Tensor]:
        """Assemble one fully random step: hard carry-forward, then fresh.

        ``hard_example_*`` carries the previous step's hardest points
        forward jittered; whatever budget is left is drawn uniformly, so a
        uniform floor over the domain always remains.
        """

        positions: list[torch.Tensor] = []
        times: list[torch.Tensor] = []
        hard_count = 0
        if hard_example_active and self._hard_position_m is not None:
            positions.append(self._hard_position_m.to(device))
            times.append(self._hard_time_hours.to(device))
            hard_count = int(self._hard_position_m.shape[0])
        fresh_count = max(0, self.volume_points_per_step - hard_count)
        if fresh_count:
            uniform = self._call_grouped_sampler(
                self.sampling_domain,
                fresh_count,
                1,
                height_power=self.height_sampling_power,
                device=device,
            )
            positions.append(uniform["position_m"])
            times.append(uniform["time_hours"])
        return {
            "position_m": torch.cat(positions, dim=0),
            "time_hours": torch.cat(times, dim=0),
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
            if self.volume_sampling == "uniform_volume":
                volume = self.sampling_domain.random_volume(
                    self.volume_points_per_step, device=device
                )
            elif self.volume_sampling == "fully_random":
                # Every point is its own height group, so the carry-forward
                # set ranks globally over the flat pool.
                volume = self._random_volume_samples(
                    device=device,
                    hard_example_active=(
                        atmosphere_model is not None and self._hard_example_active(atmosphere_model)
                    ),
                )
            else:
                volume = self._call_grouped_sampler(
                    self.sampling_domain,
                    self.height_layers_per_step,
                    self.volume_points_per_step // self.height_layers_per_step,
                    height_power=self.height_sampling_power,
                    device=device,
                )
            samples.update(
                volume_position_m=volume["position_m"],
                volume_time_hours=volume["time_hours"],
            )
        if self.constraints.upper_volume_active:
            if self.constraints.volume_active:
                # Share the batch already drawn for the ordinary volume
                # equations (momentum, induction, continuity, ...) instead of
                # a second, independent sampling pass over largely the same
                # region. Height-restricted upper-domain equations (e.g.
                # coronal_energy) mask which points count at reduction time
                # rather than needing their own dedicated draw.
                samples.update(
                    upper_position_m=samples["volume_position_m"],
                    upper_time_hours=samples["volume_time_hours"],
                )
            else:
                if self.upper_sampling_domain is None:
                    raise RuntimeError(
                        "Training-active upper physics requires its sampling domain."
                    )
                if self.volume_sampling == "uniform_volume":
                    upper = self.upper_sampling_domain.random_volume(
                        self.upper_volume_points_per_step, device=device
                    )
                else:
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
                },
            )
        samples = self._samples(parameter, atmosphere_model)
        available: dict[str, torch.Tensor] = {}
        hard_example_active = self._hard_example_active(atmosphere_model)
        with torch.enable_grad():
            if self.constraints.volume_active:
                position = samples["volume_position_m"]
                time = samples["volume_time_hours"]
                shape = (position.shape[0], 1) if position.ndim == 2 else tuple(position.shape[:2])
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
                upper_shape = (position.shape[0], 1) if position.ndim == 2 else tuple(position.shape[:2])
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
                **{f"raw_{name}": value for name, value in active.items()},
            },
        )


__all__ = ["AtmosphereRegularizationTerm", "PhysicsConstraintTerm"]
