"""Fixed differential constraints for the spherical LTE inversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

import torch
from torch import nn

from prom3theus.core import CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S


VOLUME_EQUATIONS = (
    "hydrostatic_equilibrium",
    "magnetohydrostatic_equilibrium",
    "momentum",
    "magnetic_divergence",
    "induction",
    "continuity",
)
BOUNDARY_EQUATIONS = ("upper_boundary_gas_pressure_prior",)
EQUATION_NAMES = (*VOLUME_EQUATIONS, *BOUNDARY_EQUATIONS)

GAUSS_TO_TESLA = 1.0e-4
VACUUM_PERMEABILITY_H_PER_M = 4.0 * math.pi * 1.0e-7


@dataclass(frozen=True)
class PhysicsNormalization:
    """Independent scales used to nondimensionalize physical residuals."""

    length_m: float = 1.0e6
    time_s: float = 3_600.0

    def __post_init__(self):
        for name, value in (("length_m", self.length_m), ("time_s", self.time_s)):
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(f"Physics normalization {name} must be numeric.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Physics normalization {name} must be finite and positive."
                )

    @classmethod
    def from_config(cls, config: Mapping | None) -> "PhysicsNormalization":
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("Physics normalization must be a mapping.")
        values = dict(config or {})
        unknown = set(values) - {"length_m", "time_s"}
        if unknown:
            raise TypeError(f"Unknown physics normalization options: {sorted(unknown)}")
        return cls(**values)

    def configuration(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "time_s": self.time_s,
        }


@dataclass(frozen=True)
class PhysicsState:
    """Atmospheric values and one shared derivative state at shell points."""

    position_m: torch.Tensor
    gas_pressure: torch.Tensor
    mass_density: torch.Tensor | None
    velocity_field_m_per_s: torch.Tensor
    magnetic_field_gauss: torch.Tensor
    primitive_derivatives: "PrimitiveDerivatives"
    radial_unit: torch.Tensor
    radial_group_shape: tuple[int, int]

    def derivative(self, name: str) -> torch.Tensor:
        if name == "velocity_cross_magnetic":
            velocity_jacobian = self.derivative("velocity").transpose(1, 2)
            magnetic_jacobian = self.derivative("magnetic").transpose(1, 2)
            result = torch.linalg.cross(
                velocity_jacobian,
                self.magnetic_field_gauss[:, None, :],
                dim=-1,
            ) + torch.linalg.cross(
                self.velocity_field_m_per_s[:, None, :],
                magnetic_jacobian,
                dim=-1,
            )
            return result.transpose(1, 2)
        if name == "mass_flux":
            density_jacobian = self.derivative("density")[:, 0, :]
            return (
                self.mass_density[:, None, None] * self.derivative("velocity")
                + self.velocity_field_m_per_s[:, :, None] * density_jacobian[:, None, :]
            )
        return self.primitive_derivatives.spatial(name)

    def time_derivative(self, name: str) -> torch.Tensor:
        return self.primitive_derivatives.temporal(name)

    def grouped(self, values: torch.Tensor) -> torch.Tensor:
        """View flat collocation values as ``[height, point, ...]``."""

        height_count, points_per_height = self.radial_group_shape
        if values.shape[0] != height_count * points_per_height:
            raise ValueError("Physics values do not match the radial sample groups.")
        return values.reshape(
            height_count,
            points_per_height,
            *values.shape[1:],
        )

    def radial_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Return one mean per explicitly sampled radius without replication."""

        return self.grouped(values).mean(dim=1)


@dataclass(frozen=True)
class PrimitiveDerivatives:
    """Physical derivatives of base atmospheric outputs from one Jacobian."""

    jacobian: torch.Tensor
    slices: dict[str, slice]

    def _select(self, name: str) -> torch.Tensor:
        try:
            return self.jacobian[:, self.slices[name], :]
        except KeyError as error:
            raise KeyError(
                f"Primitive derivative {name!r} was not requested."
            ) from error

    def spatial(self, name: str) -> torch.Tensor:
        return self._select(name)[..., :3]

    def temporal(self, name: str) -> torch.Tensor:
        return self._select(name)[..., 3]


@dataclass(frozen=True)
class PhysicsResult:
    losses: dict[str, torch.Tensor]
    weights: dict[str, float]
    state: PhysicsState | None = None


def _pointwise_output_jacobian(
    primitives: Mapping[str, torch.Tensor],
    coordinates: torch.Tensor,
    *,
    create_graph: bool,
) -> PrimitiveDerivatives:
    """Differentiate all base outputs together in one batched autograd call."""

    if not primitives:
        return PrimitiveDerivatives(coordinates.new_empty((0, 0, 4)), {})
    flattened = []
    slices = {}
    offset = 0
    for name, value in primitives.items():
        value = value.reshape(coordinates.shape[0], -1)
        flattened.append(value)
        slices[name] = slice(offset, offset + value.shape[-1])
        offset += value.shape[-1]
    joined = torch.cat(flattened, dim=-1)
    output_count = joined.shape[-1]
    basis = torch.eye(output_count, dtype=joined.dtype, device=joined.device)
    batched_outputs = basis[:, None, :].expand(
        output_count, joined.shape[0], output_count
    )
    jacobian = torch.autograd.grad(
        joined,
        coordinates,
        grad_outputs=batched_outputs,
        create_graph=create_graph,
        retain_graph=create_graph,
        is_grads_batched=True,
    )[0].movedim(0, 1)
    return PrimitiveDerivatives(jacobian, slices)


def _divergence(jacobian: torch.Tensor) -> torch.Tensor:
    return jacobian[:, 0, 0] + jacobian[:, 1, 1] + jacobian[:, 2, 2]


def _curl(jacobian: torch.Tensor) -> torch.Tensor:
    """Return curl for a [sample, vector component, coordinate] Jacobian."""

    return torch.stack(
        (
            jacobian[:, 2, 1] - jacobian[:, 1, 2],
            jacobian[:, 0, 2] - jacobian[:, 2, 0],
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
        ),
        dim=-1,
    )


class MagnetofluidConstraints(nn.Module):
    """Evaluate LTE magnetofluid constraints on grouped spherical-shell points."""

    def __init__(
        self,
        equations: Mapping | None = None,
        *,
        gravity_m_per_s2: float | None = None,
        vector_basis_matches_spatial_coordinates: bool = False,
        normalization: Mapping | None = None,
    ):
        super().__init__()
        if equations is not None and not isinstance(equations, Mapping):
            raise TypeError("Physics equations must be a mapping.")
        equations = dict(equations or {})
        unknown = set(equations) - set(EQUATION_NAMES)
        if unknown:
            raise KeyError(
                f"Unknown magnetofluid equations {sorted(unknown)}; expected {EQUATION_NAMES}."
            )
        self.enabled: dict[str, bool] = {}
        self.fixed_weights: dict[str, float] = {}
        self.equation_config: dict[str, dict] = {}
        for name in EQUATION_NAMES:
            raw = equations.get(name, {})
            if not isinstance(raw, Mapping):
                raise TypeError(f"Physics equation {name!r} must be a mapping.")
            config = dict(raw)
            enabled = config.pop("enabled", False)
            if type(enabled) is not bool:
                raise TypeError(f"Physics equation {name!r} enabled must be boolean.")
            weight = config.pop("weight", 1.0 if enabled else 0.0)
            if config:
                raise TypeError(
                    f"Unknown options for physics equation {name!r}: {sorted(config)}"
                )
            if not isinstance(weight, Real) or isinstance(weight, bool):
                raise TypeError(
                    f"Physics equation {name!r} requires a fixed numeric weight; schedules are unsupported."
                )
            weight = float(weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"Physics equation {name!r} weight must be finite and non-negative."
                )
            self.enabled[name] = enabled
            self.fixed_weights[name] = weight if enabled else 0.0
            self.equation_config[name] = {"enabled": enabled, "weight": weight}

        if gravity_m_per_s2 is not None and (
            not isinstance(gravity_m_per_s2, Real) or isinstance(gravity_m_per_s2, bool)
        ):
            raise TypeError("gravity_m_per_s2 must be numeric or null.")
        self.gravity_m_per_s2 = (
            None if gravity_m_per_s2 is None else float(gravity_m_per_s2)
        )
        if self.gravity_m_per_s2 is not None and (
            not math.isfinite(self.gravity_m_per_s2) or self.gravity_m_per_s2 <= 0
        ):
            raise ValueError("gravity_m_per_s2 must be finite and positive.")
        force_balance_equations = {
            name
            for name in (
                "hydrostatic_equilibrium",
                "magnetohydrostatic_equilibrium",
                "momentum",
            )
            if self.is_active(name)
        }
        if len(force_balance_equations) > 1:
            raise ValueError(
                "hydrostatic_equilibrium, magnetohydrostatic_equilibrium, and "
                "momentum are mutually exclusive."
            )
        if (force_balance_equations) and (self.gravity_m_per_s2 is None):
            raise ValueError(
                "Hydrostatic, magnetohydrostatic, and momentum balance require "
                "positive gravity_m_per_s2."
            )
        if type(vector_basis_matches_spatial_coordinates) is not bool:
            raise TypeError("vector_basis_matches_spatial_coordinates must be boolean.")
        self.vector_basis_matches_spatial_coordinates = (
            vector_basis_matches_spatial_coordinates
        )
        vector_derivatives_active = any(
            self.is_active(name)
            for name in (
                "magnetohydrostatic_equilibrium",
                "momentum",
                "magnetic_divergence",
                "induction",
                "continuity",
            )
        )
        if (
            vector_derivatives_active
            and not self.vector_basis_matches_spatial_coordinates
        ):
            raise ValueError(
                "Vector differential equations require "
                "vector_basis_matches_spatial_coordinates=true."
            )
        self.normalization = PhysicsNormalization.from_config(normalization)

    def is_active(self, name: str) -> bool:
        """Return whether an equation can contribute to the objective."""

        if name not in self.enabled:
            raise KeyError(f"Unknown LTE physics equation {name!r}.")
        return self.enabled[name] and self.fixed_weights[name] != 0.0

    @property
    def any_active(self) -> bool:
        return any(self.is_active(name) for name in EQUATION_NAMES)

    @property
    def volume_active(self) -> bool:
        return any(self.is_active(name) for name in VOLUME_EQUATIONS)

    @property
    def boundary_active(self) -> bool:
        return any(self.is_active(name) for name in BOUNDARY_EQUATIONS)

    @property
    def loss_weights(self) -> dict[str, float]:
        """Fixed equation weights used by every training and validation call."""

        return dict(self.fixed_weights)

    def build_state(
        self,
        atmosphere_model,
        continuum_opacity,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        active_equations: set[str],
        *,
        create_graph: bool,
        radial_group_shape: tuple[int, int],
    ) -> PhysicsState:
        parameter = next(atmosphere_model.parameters())
        position_m = position_m.reshape(-1, 3)
        time_hours = time_hours.to(parameter).reshape(-1, 1)
        sample_count = position_m.shape[0]
        if time_hours.shape[0] != sample_count:
            raise ValueError(
                "Physics position and time tensors must contain the same samples."
            )
        if math.prod(radial_group_shape) != sample_count:
            raise ValueError(
                "Physics radial group shape does not match the sample count."
            )

        # Normalize before casting to the network dtype so a float64 shell
        # sample does not lose its radial offset at the solar-radius scale.
        position_rsun = (position_m / atmosphere_model.solar_radius_m).to(parameter)
        coordinates = (
            torch.cat((position_rsun, time_hours), dim=-1).detach().requires_grad_(True)
        )
        position_rsun = coordinates[:, :3]
        time_hours = coordinates[:, 3:4]
        fields = atmosphere_model.evaluate_position_rsun(
            position_rsun, time_hours=time_hours
        )
        temperature = fields["temperature"]
        pressure = fields["gas_pressure"]
        magnetic = fields["magnetic_field"]
        velocity = fields["velocity_field"]
        radial_unit = position_rsun / torch.linalg.vector_norm(
            position_rsun, dim=-1, keepdim=True
        )
        density = (
            continuum_opacity.reference_mass_density(temperature, pressure)
            if {
                "hydrostatic_equilibrium",
                "magnetohydrostatic_equilibrium",
                "momentum",
                "continuity",
            }
            & active_equations
            else None
        )
        primitives = {}
        if {
            "hydrostatic_equilibrium",
            "magnetohydrostatic_equilibrium",
            "momentum",
        } & active_equations:
            primitives["pressure"] = pressure
        if {
            "magnetohydrostatic_equilibrium",
            "momentum",
            "magnetic_divergence",
            "induction",
        } & active_equations:
            primitives["magnetic"] = magnetic
        if {"momentum", "induction", "continuity"} & active_equations:
            primitives["velocity"] = velocity
        if density is not None:
            primitives["density"] = density
        raw_derivatives = _pointwise_output_jacobian(
            primitives, coordinates, create_graph=create_graph
        )
        physical_jacobian = torch.cat(
            (
                raw_derivatives.jacobian[..., :3] / atmosphere_model.solar_radius_m,
                raw_derivatives.jacobian[..., 3:] / 3_600.0,
            ),
            dim=-1,
        )
        return PhysicsState(
            position_m=position_rsun * atmosphere_model.solar_radius_m,
            gas_pressure=pressure,
            mass_density=density,
            velocity_field_m_per_s=velocity,
            magnetic_field_gauss=magnetic,
            primitive_derivatives=PrimitiveDerivatives(
                physical_jacobian,
                raw_derivatives.slices,
            ),
            radial_unit=radial_unit,
            radial_group_shape=radial_group_shape,
        )

    def volume(
        self,
        atmosphere_model,
        continuum_opacity,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        *,
        create_graph: bool = True,
        return_state: bool = False,
        radial_group_shape: tuple[int, int] | None = None,
    ) -> PhysicsResult:
        weights = self.loss_weights
        active = {
            name
            for name in VOLUME_EQUATIONS
            if self.enabled[name] and weights[name] != 0.0
        }
        zero = next(atmosphere_model.parameters()).new_zeros(())
        losses = {name: zero for name in VOLUME_EQUATIONS}
        if not active:
            return PhysicsResult(losses, weights)
        temporal = {"momentum", "induction", "continuity"} & active
        if temporal and not getattr(atmosphere_model, "time_dependent", False):
            raise ValueError(
                f"Temporal LTE physics {sorted(temporal)} require atmosphere.time_dependent=true."
            )
        state = self.build_state(
            atmosphere_model,
            continuum_opacity,
            position_m,
            time_hours,
            active,
            create_graph=create_graph,
            radial_group_shape=(
                (1, int(position_m.numel() // 3))
                if radial_group_shape is None
                else radial_group_shape
            ),
        )
        if "hydrostatic_equilibrium" in active:
            grad_pressure = state.derivative("pressure")[:, 0]
            radial_pressure_gradient = (grad_pressure * state.radial_unit).sum(dim=-1)
            pressure_equivalent = (
                state.gas_pressure
                + state.mass_density
                * self.gravity_m_per_s2
                * self.normalization.length_m
            )
            force_scale = (
                state.radial_mean(pressure_equivalent)
                .clamp_min(torch.finfo(state.gas_pressure.dtype).tiny)
                .detach()
            )
            residual = (
                state.grouped(
                    radial_pressure_gradient
                    + state.mass_density * self.gravity_m_per_s2
                )
                * (self.normalization.length_m / force_scale)[:, None]
            )
            losses["hydrostatic_equilibrium"] = residual.square().mean()
        if "magnetohydrostatic_equilibrium" in active:
            grad_pressure = state.derivative("pressure")[:, 0]
            magnetic_tesla = state.magnetic_field_gauss * GAUSS_TO_TESLA
            curl_magnetic_tesla = _curl(state.derivative("magnetic") * GAUSS_TO_TESLA)
            lorentz_force = (
                torch.linalg.cross(curl_magnetic_tesla, magnetic_tesla, dim=-1)
                / VACUUM_PERMEABILITY_H_PER_M
            )
            pressure_equivalent = (
                state.gas_pressure
                + state.mass_density
                * self.gravity_m_per_s2
                * self.normalization.length_m
                + magnetic_tesla.square().sum(dim=-1) / VACUUM_PERMEABILITY_H_PER_M
            )
            force_scale = (
                state.radial_mean(pressure_equivalent)
                .clamp_min(torch.finfo(state.gas_pressure.dtype).tiny)
                .detach()
            )
            residual = (
                state.grouped(
                    grad_pressure
                    + state.mass_density[:, None]
                    * self.gravity_m_per_s2
                    * state.radial_unit
                    - lorentz_force
                )
                * (self.normalization.length_m / force_scale)[:, None, None]
            )
            losses["magnetohydrostatic_equilibrium"] = (
                residual.square().sum(dim=-1).mean()
            )
        if "momentum" in active:
            grad_pressure = state.derivative("pressure")[:, 0]
            magnetic_tesla = state.magnetic_field_gauss * GAUSS_TO_TESLA
            curl_magnetic_tesla = _curl(state.derivative("magnetic") * GAUSS_TO_TESLA)
            lorentz_force = (
                torch.linalg.cross(curl_magnetic_tesla, magnetic_tesla, dim=-1)
                / VACUUM_PERMEABILITY_H_PER_M
            )

            velocity = state.velocity_field_m_per_s
            material_acceleration = state.time_derivative("velocity") + torch.einsum(
                "nij,nj->ni", state.derivative("velocity"), velocity
            )
            omega = velocity.new_tensor(
                [0.0, 0.0, CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S]
            ).expand_as(velocity)
            coriolis_acceleration = 2.0 * torch.linalg.cross(omega, velocity, dim=-1)
            centrifugal_acceleration = torch.linalg.cross(
                omega,
                torch.linalg.cross(omega, state.position_m, dim=-1),
                dim=-1,
            )
            rotating_frame_acceleration = (
                material_acceleration + coriolis_acceleration + centrifugal_acceleration
            )
            pressure_equivalent = (
                state.gas_pressure
                + state.mass_density
                * (
                    self.gravity_m_per_s2
                    + torch.linalg.vector_norm(rotating_frame_acceleration, dim=-1)
                )
                * self.normalization.length_m
                + magnetic_tesla.square().sum(dim=-1) / VACUUM_PERMEABILITY_H_PER_M
            )
            force_scale = (
                state.radial_mean(pressure_equivalent)
                .clamp_min(torch.finfo(state.gas_pressure.dtype).tiny)
                .detach()
            )
            residual = (
                state.grouped(
                    state.mass_density[:, None] * rotating_frame_acceleration
                    + grad_pressure
                    + state.mass_density[:, None]
                    * self.gravity_m_per_s2
                    * state.radial_unit
                    - lorentz_force
                )
                * (self.normalization.length_m / force_scale)[:, None, None]
            )
            losses["momentum"] = residual.square().sum(dim=-1).mean()
        field_scale = None
        if {"magnetic_divergence", "induction"} & active:
            field_scale = (
                state.radial_mean(
                    torch.linalg.vector_norm(state.magnetic_field_gauss, dim=-1)
                )
                .clamp_min(torch.finfo(state.magnetic_field_gauss.dtype).tiny)
                .detach()
            )
        if "magnetic_divergence" in active:
            residual = (
                state.grouped(_divergence(state.derivative("magnetic")))
                * (self.normalization.length_m / field_scale)[:, None]
            )
            losses["magnetic_divergence"] = residual.square().mean()
        if "induction" in active:
            residual = (
                state.grouped(
                    state.time_derivative("magnetic")
                    - _curl(state.derivative("velocity_cross_magnetic"))
                )
                * (self.normalization.time_s / field_scale)[:, None, None]
            )
            losses["induction"] = residual.square().sum(dim=-1).mean()
        if "continuity" in active:
            density_scale = (
                state.radial_mean(state.mass_density)
                .clamp_min(torch.finfo(state.mass_density.dtype).tiny)
                .detach()
            )
            residual = (
                state.grouped(
                    state.time_derivative("density")[:, 0]
                    + _divergence(state.derivative("mass_flux"))
                )
                * (self.normalization.time_s / density_scale)[:, None]
            )
            losses["continuity"] = residual.square().mean()
        return PhysicsResult(losses, weights, state if return_state else None)

    def upper_boundary_gas_pressure_prior(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
    ) -> PhysicsResult:
        """Weakly retain the radial-reference pressure at the shell top."""

        weights = self.loss_weights
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        name = "upper_boundary_gas_pressure_prior"
        if not self.enabled[name] or weights[name] == 0.0:
            return PhysicsResult({name: zero}, weights)
        position_m = position_m.reshape(-1, 3)
        time_hours = time_hours.to(parameter).reshape(-1, 1)
        if position_m.shape[0] < 1 or time_hours.shape[0] != position_m.shape[0]:
            raise ValueError("The upper-boundary pressure prior requires samples.")
        position_rsun = (position_m / atmosphere_model.solar_radius_m).to(parameter)
        pressure = atmosphere_model.evaluate_position_rsun(
            position_rsun,
            time_hours=time_hours,
        )["gas_pressure"]
        upper_height_m = (
            torch.linalg.vector_norm(position_rsun, dim=-1) - 1.0
        ) * atmosphere_model.solar_radius_m
        reference_log_pressure = atmosphere_model.reference_atmosphere.logs_at_height(
            upper_height_m
        )[1]
        residual = torch.log(pressure) - reference_log_pressure
        return PhysicsResult(
            {name: residual.square().mean()},
            weights,
        )

    def configuration(self) -> dict:
        return {
            "equations": self.equation_config,
            "gravity_m_per_s2": self.gravity_m_per_s2,
            "vector_basis_matches_spatial_coordinates": self.vector_basis_matches_spatial_coordinates,
            "normalization": self.normalization.configuration(),
        }


__all__ = [
    "BOUNDARY_EQUATIONS",
    "EQUATION_NAMES",
    "MagnetofluidConstraints",
    "PhysicsNormalization",
    "PhysicsResult",
    "PhysicsState",
    "VOLUME_EQUATIONS",
]
