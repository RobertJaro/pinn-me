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
    "radial_magnetic_energy_gradient",
    "induction",
    "continuity",
    "adiabatic_pressure",
)
UPPER_VOLUME_EQUATIONS = (
    "upper_domain_microturbulence_prior",
    "upper_domain_temperature_prior",
)
UPPER_BOUNDARY_EQUATIONS = (
    "upper_boundary_open_velocity",
    "upper_boundary_current_free",
    "upper_boundary_gas_pressure_prior",
)
SIDE_BOUNDARY_EQUATIONS = (
    "side_boundary_open_velocity",
    "side_boundary_current_free",
)
BOUNDARY_EQUATIONS = (*UPPER_BOUNDARY_EQUATIONS, *SIDE_BOUNDARY_EQUATIONS)
EQUATION_NAMES = (*VOLUME_EQUATIONS, *UPPER_VOLUME_EQUATIONS, *BOUNDARY_EQUATIONS)

GAUSS_TO_TESLA = 1.0e-4
VACUUM_PERMEABILITY_H_PER_M = 4.0 * math.pi * 1.0e-7


@dataclass(frozen=True)
class PhysicsNormalization:
    """Independent scales used to nondimensionalize physical residuals.

    Characteristic field magnitudes are evaluated independently at each height.
    The transported magnetic field uses the common rate ``1 / T + V0 / L``.
    Continuity and adiabatic pressure are written as logarithmic specific-rate
    equations and therefore use that rate directly. Every height-local scale is
    detached. The normalization therefore conditions each residual without
    providing an optimizer path through the current field magnitude or
    thermodynamic state.
    """

    length_m: float = 1.0e6
    time_s: float = 3_600.0
    magnetic_field_floor_gauss: float = 1.0
    velocity_scale_m_per_s: float = 1_000.0

    def __post_init__(self):
        for name, value in (
            ("length_m", self.length_m),
            ("time_s", self.time_s),
            ("magnetic_field_floor_gauss", self.magnetic_field_floor_gauss),
            ("velocity_scale_m_per_s", self.velocity_scale_m_per_s),
        ):
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
        unknown = set(values) - {
            "length_m",
            "time_s",
            "magnetic_field_floor_gauss",
            "velocity_scale_m_per_s",
        }
        if unknown:
            raise TypeError(f"Unknown physics normalization options: {sorted(unknown)}")
        return cls(**values)

    def configuration(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "time_s": self.time_s,
            "magnetic_field_floor_gauss": self.magnetic_field_floor_gauss,
            "velocity_scale_m_per_s": self.velocity_scale_m_per_s,
        }

    @property
    def transport_rate_per_s(self) -> float:
        """Combined fixed temporal and advective characteristic rate."""

        return 1.0 / self.time_s + self.velocity_scale_m_per_s / self.length_m

    @property
    def transport_time_s(self) -> float:
        return 1.0 / self.transport_rate_per_s


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
    height_group_shape: tuple[int, int]

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

        height_count, points_per_height = self.height_group_shape
        if values.shape[0] != height_count * points_per_height:
            raise ValueError("Physics values do not match the height sample groups.")
        return values.reshape(
            height_count,
            points_per_height,
            *values.shape[1:],
        )

    def height_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Average angular/time samples independently at every sampled height."""

        return self.grouped(values).mean(dim=1)

    def normalize_height_groups(
        self,
        residual: torch.Tensor,
        normalization_quantity: torch.Tensor,
        dimensional_factor: float,
        *,
        scale_floor: float = 0.0,
    ) -> torch.Tensor:
        """Normalize a dimensional residual by its height-mean scale.

        ``normalization_quantity`` is the positive characteristic field with the
        residual's units after multiplication by ``dimensional_factor``.  Keeping
        one scale per sampled height prevents dense low-atmosphere values from
        suppressing the upper-atmosphere equations. Every scale is detached so
        learned fields cannot alter a physical residual through its denominator.
        ``scale_floor`` supplies a smooth physical floor before detachment.
        """

        grouped_residual = self.grouped(residual)
        scale = self.height_mean(normalization_quantity)
        if scale_floor > 0.0:
            floor = scale.new_tensor(float(scale_floor))
            scale = torch.sqrt(scale.square() + floor.square())
        scale = scale.clamp_min(torch.finfo(normalization_quantity.dtype).tiny)
        scale = scale.detach()
        broadcast_shape = (scale.shape[0],) + (1,) * (grouped_residual.ndim - 1)
        return grouped_residual * (
            float(dimensional_factor) / scale.reshape(broadcast_shape)
        )

    def height_group_robust_loss(
        self,
        grouped_residual: torch.Tensor,
        *,
        delta: float = 1.0,
    ) -> torch.Tensor:
        """Average a stable pseudo-Huber penalty equally over heights/components."""

        if tuple(grouped_residual.shape[:2]) != self.height_group_shape:
            raise ValueError("Normalized residual does not retain its height groups.")
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("Pseudo-Huber delta must be finite and positive.")
        square = grouped_residual.square()
        # Algebraically equal to 2*d^2*(sqrt(1 + (r/d)^2) - 1), but stable
        # for the small residuals that determine final constraint accuracy.
        penalty = 2.0 * square / (
            torch.sqrt(1.0 + square / float(delta) ** 2) + 1.0
        )
        return penalty.mean()


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
        adiabatic_index: float = 5.0 / 3.0,
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
        if not isinstance(adiabatic_index, Real) or isinstance(adiabatic_index, bool):
            raise TypeError("adiabatic_index must be numeric.")
        self.adiabatic_index = float(adiabatic_index)
        if not math.isfinite(self.adiabatic_index) or self.adiabatic_index <= 1.0:
            raise ValueError("adiabatic_index must be finite and greater than one.")
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
                "radial_magnetic_energy_gradient",
                "induction",
                "continuity",
                "adiabatic_pressure",
                "upper_boundary_open_velocity",
                "upper_boundary_current_free",
                "side_boundary_open_velocity",
                "side_boundary_current_free",
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
    def upper_volume_active(self) -> bool:
        return any(self.is_active(name) for name in UPPER_VOLUME_EQUATIONS)

    @property
    def boundary_active(self) -> bool:
        return any(self.is_active(name) for name in BOUNDARY_EQUATIONS)

    @property
    def upper_boundary_active(self) -> bool:
        return any(self.is_active(name) for name in UPPER_BOUNDARY_EQUATIONS)

    @property
    def side_boundary_active(self) -> bool:
        return any(self.is_active(name) for name in SIDE_BOUNDARY_EQUATIONS)

    @property
    def loss_weights(self) -> dict[str, float]:
        """Fixed equation weights used by every training and validation call."""

        return dict(self.fixed_weights)

    def build_state(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        active_equations: set[str],
        *,
        create_graph: bool,
        height_group_shape: tuple[int, int],
    ) -> PhysicsState:
        parameter = next(atmosphere_model.parameters())
        position_m = position_m.reshape(-1, 3)
        time_hours = time_hours.to(parameter).reshape(-1, 1)
        sample_count = position_m.shape[0]
        if time_hours.shape[0] != sample_count:
            raise ValueError(
                "Physics position and time tensors must contain the same samples."
            )
        if (
            len(height_group_shape) != 2
            or any(type(value) is not int or value < 1 for value in height_group_shape)
            or math.prod(height_group_shape) != sample_count
        ):
            raise ValueError(
                "Physics height_group_shape must contain two positive integers "
                "whose product matches the sample count."
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
            atmosphere_model.thermodynamic_eos.mass_density(temperature, pressure)
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
        if "adiabatic_pressure" in active_equations:
            primitives["log_pressure"] = torch.log(pressure)
        if {
            "magnetohydrostatic_equilibrium",
            "momentum",
            "magnetic_divergence",
            "radial_magnetic_energy_gradient",
            "induction",
            "upper_boundary_current_free",
            "side_boundary_current_free",
        } & active_equations:
            primitives["magnetic"] = magnetic
        if {
            "momentum",
            "induction",
            "continuity",
            "adiabatic_pressure",
            "upper_boundary_open_velocity",
            "side_boundary_open_velocity",
        } & active_equations:
            primitives["velocity"] = velocity
        if density is not None and "continuity" in active_equations:
            primitives["log_density"] = torch.log(density)
        if (
            density is not None
            and {
                "hydrostatic_equilibrium",
                "magnetohydrostatic_equilibrium",
                "momentum",
            }
            & active_equations
        ):
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
            height_group_shape=height_group_shape,
        )

    def volume(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        *,
        height_group_shape: tuple[int, int],
        create_graph: bool = True,
        return_state: bool = False,
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
        temporal = {
            "momentum",
            "induction",
            "continuity",
            "adiabatic_pressure",
        } & active
        if temporal and not getattr(atmosphere_model, "time_dependent", False):
            raise ValueError(
                f"Temporal LTE physics {sorted(temporal)} require atmosphere.time_dependent=true."
            )
        state = self.build_state(
            atmosphere_model,
            position_m,
            time_hours,
            active,
            create_graph=create_graph,
            height_group_shape=height_group_shape,
        )
        gravity_m_per_s2 = None
        if {
            "hydrostatic_equilibrium",
            "magnetohydrostatic_equilibrium",
            "momentum",
        } & active:
            radius_m = torch.linalg.vector_norm(state.position_m, dim=-1)
            surface_radius_m = torch.as_tensor(
                atmosphere_model.solar_radius_m,
                dtype=radius_m.dtype,
                device=radius_m.device,
            )
            gravity_m_per_s2 = (
                self.gravity_m_per_s2 * (surface_radius_m / radius_m).square()
            )
        if "hydrostatic_equilibrium" in active:
            grad_pressure = state.derivative("pressure")[:, 0]
            radial_pressure_gradient = (grad_pressure * state.radial_unit).sum(dim=-1)
            pressure_equivalent = (
                state.gas_pressure
                + state.mass_density * gravity_m_per_s2 * self.normalization.length_m
            )
            residual = state.normalize_height_groups(
                radial_pressure_gradient + state.mass_density * gravity_m_per_s2,
                pressure_equivalent,
                self.normalization.length_m,
            )
            losses["hydrostatic_equilibrium"] = state.height_group_robust_loss(residual)
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
                + state.mass_density * gravity_m_per_s2 * self.normalization.length_m
                + magnetic_tesla.square().sum(dim=-1) / VACUUM_PERMEABILITY_H_PER_M
            )
            residual = state.normalize_height_groups(
                grad_pressure
                + state.mass_density[:, None]
                * gravity_m_per_s2[:, None]
                * state.radial_unit
                - lorentz_force,
                pressure_equivalent,
                self.normalization.length_m,
            )
            losses["magnetohydrostatic_equilibrium"] = (
                state.height_group_robust_loss(residual)
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
                    gravity_m_per_s2
                    + torch.linalg.vector_norm(rotating_frame_acceleration, dim=-1)
                )
                * self.normalization.length_m
                + magnetic_tesla.square().sum(dim=-1) / VACUUM_PERMEABILITY_H_PER_M
            )
            residual = state.normalize_height_groups(
                state.mass_density[:, None] * rotating_frame_acceleration
                + grad_pressure
                + state.mass_density[:, None]
                * gravity_m_per_s2[:, None]
                * state.radial_unit
                - lorentz_force,
                pressure_equivalent,
                self.normalization.length_m,
            )
            losses["momentum"] = state.height_group_robust_loss(residual)
        field_normalization = torch.linalg.vector_norm(
            state.magnetic_field_gauss, dim=-1
        )
        if "magnetic_divergence" in active:
            residual = state.normalize_height_groups(
                _divergence(state.derivative("magnetic")),
                field_normalization,
                self.normalization.length_m,
                scale_floor=self.normalization.magnetic_field_floor_gauss,
            )
            losses["magnetic_divergence"] = state.height_group_robust_loss(residual)
        if "radial_magnetic_energy_gradient" in active:
            radial_magnetic_derivative = torch.einsum(
                "nij,nj->ni",
                state.derivative("magnetic"),
                state.radial_unit,
            )
            radial_energy_gradient = 2.0 * (
                state.magnetic_field_gauss * radial_magnetic_derivative
            ).sum(dim=-1)
            magnetic_energy = state.magnetic_field_gauss.square().sum(dim=-1)
            mean_radial_energy_gradient = state.height_mean(radial_energy_gradient)
            energy_scale = state.height_mean(magnetic_energy)
            energy_floor = energy_scale.new_tensor(
                self.normalization.magnetic_field_floor_gauss**2
            )
            energy_scale = torch.sqrt(
                energy_scale.square() + energy_floor.square()
            ).detach()
            residual = (
                torch.relu(mean_radial_energy_gradient)
                * self.normalization.length_m
                / energy_scale
            )
            losses["radial_magnetic_energy_gradient"] = residual.square().mean()
        if "induction" in active:
            residual = state.normalize_height_groups(
                state.time_derivative("magnetic")
                - _curl(state.derivative("velocity_cross_magnetic")),
                field_normalization,
                self.normalization.transport_time_s,
                scale_floor=self.normalization.magnetic_field_floor_gauss,
            )
            losses["induction"] = state.height_group_robust_loss(residual)
        if "continuity" in active:
            log_density_gradient = state.derivative("log_density")[:, 0]
            residual = (
                state.time_derivative("log_density")[:, 0]
                + (log_density_gradient * state.velocity_field_m_per_s).sum(dim=-1)
                + _divergence(state.derivative("velocity"))
            )
            residual = state.grouped(residual * self.normalization.transport_time_s)
            losses["continuity"] = state.height_group_robust_loss(residual)
        if "adiabatic_pressure" in active:
            log_pressure_gradient = state.derivative("log_pressure")[:, 0]
            residual = (
                state.time_derivative("log_pressure")[:, 0]
                + (log_pressure_gradient * state.velocity_field_m_per_s).sum(dim=-1)
                + self.adiabatic_index * _divergence(state.derivative("velocity"))
            )
            residual = state.grouped(residual * self.normalization.transport_time_s)
            losses["adiabatic_pressure"] = state.height_group_robust_loss(residual)
        return PhysicsResult(losses, weights, state if return_state else None)

    def upper_domain(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        *,
        height_group_shape: tuple[int, int],
        create_graph: bool = True,
    ) -> PhysicsResult:
        """Regularize the spectrally invisible upper atmosphere."""

        weights = self.loss_weights
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        losses = {name: zero for name in UPPER_VOLUME_EQUATIONS}
        active = {name for name in UPPER_VOLUME_EQUATIONS if self.is_active(name)}
        if not active:
            return PhysicsResult(losses, weights)
        position = position_m.reshape(-1, 3)
        time = time_hours.to(parameter).reshape(-1, 1)
        if position.shape[0] < 1 or time.shape[0] != position.shape[0]:
            raise ValueError("Upper-domain regularization requires paired samples.")
        if (
            len(height_group_shape) != 2
            or any(type(value) is not int or value < 1 for value in height_group_shape)
            or math.prod(height_group_shape) != position.shape[0]
        ):
            raise ValueError(
                "Upper-domain height_group_shape must contain two positive integers "
                "whose product matches the sample count."
            )

        def grouped_mean_square(residual: torch.Tensor) -> torch.Tensor:
            grouped = residual.reshape(*height_group_shape, *residual.shape[1:])
            return grouped.square().mean(dim=tuple(range(1, grouped.ndim))).mean()

        del create_graph
        thermodynamic_active = {
            "upper_domain_microturbulence_prior",
            "upper_domain_temperature_prior",
        } & active
        if thermodynamic_active:
            position_rsun = (position / atmosphere_model.solar_radius_m).to(parameter)
            fields = atmosphere_model.evaluate_position_rsun(
                position_rsun,
                time_hours=time,
            )
            height_m = (
                torch.linalg.vector_norm(position_rsun, dim=-1)
                - position_rsun.new_tensor(1.0)
            ) * atmosphere_model.solar_radius_m
        if "upper_domain_microturbulence_prior" in active:
            reference_log_microturbulence = (
                atmosphere_model.reference_atmosphere.logs_at_height(height_m)[2]
            ).detach()
            residual = (
                torch.log(fields["microturbulence"]) - reference_log_microturbulence
            )
            losses["upper_domain_microturbulence_prior"] = grouped_mean_square(residual)
        if "upper_domain_temperature_prior" in active:
            reference_log_temperature = (
                atmosphere_model.reference_atmosphere.logs_at_height(height_m)[0]
            ).detach()
            residual = torch.log(fields["temperature"]) - reference_log_temperature
            losses["upper_domain_temperature_prior"] = grouped_mean_square(residual)
        return PhysicsResult(losses, weights)

    def upper_boundary(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> PhysicsResult:
        """Apply transmissive-velocity, current-free, and pressure conditions."""

        weights = self.loss_weights
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        losses = {name: zero for name in UPPER_BOUNDARY_EQUATIONS}
        active = {
            name for name in UPPER_BOUNDARY_EQUATIONS if self.is_active(name)
        }
        if not active:
            return PhysicsResult(losses, weights)
        position = position_m.reshape(-1, 3)
        time = time_hours.to(parameter).reshape(-1, 1)
        if position.shape[0] < 1 or time.shape[0] != position.shape[0]:
            raise ValueError("Upper-boundary constraints require paired samples.")

        differential_active = {
            "upper_boundary_open_velocity",
            "upper_boundary_current_free",
        } & active
        state = None
        if differential_active:
            state = self.build_state(
                atmosphere_model,
                position_m=position,
                time_hours=time,
                active_equations=differential_active,
                create_graph=create_graph,
                height_group_shape=(1, position.shape[0]),
            )

        if "upper_boundary_open_velocity" in active:
            radial_velocity_derivative = torch.einsum(
                "nij,nj->ni",
                state.derivative("velocity"),
                state.radial_unit,
            )
            residual = state.grouped(
                radial_velocity_derivative
                * self.normalization.length_m
                / self.normalization.velocity_scale_m_per_s
            )
            losses["upper_boundary_open_velocity"] = (
                state.height_group_robust_loss(residual)
            )

        if "upper_boundary_current_free" in active:
            curl = _curl(state.derivative("magnetic"))
            field_normalization = torch.linalg.vector_norm(
                state.magnetic_field_gauss, dim=-1
            )
            residual = state.normalize_height_groups(
                curl,
                field_normalization,
                self.normalization.length_m,
                scale_floor=self.normalization.magnetic_field_floor_gauss,
            )
            losses["upper_boundary_current_free"] = state.height_group_robust_loss(
                residual
            )

        if "upper_boundary_gas_pressure_prior" in active:
            position_rsun = (position / atmosphere_model.solar_radius_m).to(parameter)
            pressure = atmosphere_model.evaluate_position_rsun(
                position_rsun,
                time_hours=time,
            )["gas_pressure"]
            target_log_pressure = (
                atmosphere_model.top_boundary_reference_log_pressure.to(pressure)
            ).detach()
            residual = torch.log(pressure) - target_log_pressure
            losses["upper_boundary_gas_pressure_prior"] = residual.square().mean()
        return PhysicsResult(losses, weights)

    def side_boundary(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        normal: torch.Tensor,
        *,
        height_group_shape: tuple[int, int],
        create_graph: bool = True,
    ) -> PhysicsResult:
        """Apply open-velocity and current-free conditions on angular sides."""

        weights = self.loss_weights
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        losses = {name: zero for name in SIDE_BOUNDARY_EQUATIONS}
        active = {name for name in SIDE_BOUNDARY_EQUATIONS if self.is_active(name)}
        if not active:
            return PhysicsResult(losses, weights)
        position = position_m.reshape(-1, 3)
        time = time_hours.to(parameter).reshape(-1, 1)
        outward_normal = normal.to(parameter).reshape(-1, 3)
        if (
            position.shape[0] < 1
            or time.shape[0] != position.shape[0]
            or outward_normal.shape[0] != position.shape[0]
            or not torch.isfinite(outward_normal).all()
        ):
            raise ValueError(
                "Side-boundary constraints require paired positions, times, and "
                "finite normals."
            )
        normal_norm = torch.linalg.vector_norm(outward_normal, dim=-1, keepdim=True)
        if torch.any(normal_norm <= 0.0):
            raise ValueError("Side-boundary normals must be non-zero.")
        outward_normal = outward_normal / normal_norm

        state = self.build_state(
            atmosphere_model,
            position_m=position,
            time_hours=time,
            active_equations=active,
            create_graph=create_graph,
            height_group_shape=height_group_shape,
        )
        if "side_boundary_open_velocity" in active:
            normal_velocity_derivative = torch.einsum(
                "nij,nj->ni",
                state.derivative("velocity"),
                outward_normal,
            )
            residual = state.grouped(
                normal_velocity_derivative
                * self.normalization.length_m
                / self.normalization.velocity_scale_m_per_s
            )
            losses["side_boundary_open_velocity"] = (
                state.height_group_robust_loss(residual)
            )

        if "side_boundary_current_free" in active:
            field_normalization = torch.linalg.vector_norm(
                state.magnetic_field_gauss, dim=-1
            )
            residual = state.normalize_height_groups(
                _curl(state.derivative("magnetic")),
                field_normalization,
                self.normalization.length_m,
                scale_floor=self.normalization.magnetic_field_floor_gauss,
            )
            losses["side_boundary_current_free"] = (
                state.height_group_robust_loss(residual)
            )
        return PhysicsResult(losses, weights)

    def upper_boundary_gas_pressure_prior(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
    ) -> PhysicsResult:
        """Evaluate the fixed external-pressure member of :meth:`upper_boundary`."""

        result = self.upper_boundary(
            atmosphere_model,
            position_m,
            time_hours,
            create_graph=torch.is_grad_enabled(),
        )
        name = "upper_boundary_gas_pressure_prior"
        return PhysicsResult({name: result.losses[name]}, result.weights)

    def configuration(self) -> dict:
        return {
            "equations": self.equation_config,
            "gravity_m_per_s2": self.gravity_m_per_s2,
            "adiabatic_index": self.adiabatic_index,
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
    "SIDE_BOUNDARY_EQUATIONS",
    "UPPER_BOUNDARY_EQUATIONS",
    "UPPER_VOLUME_EQUATIONS",
    "VOLUME_EQUATIONS",
]
