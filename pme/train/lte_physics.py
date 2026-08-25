"""Fixed differential constraints for the spherical LTE inversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

import torch
from torch import nn


VOLUME_EQUATIONS = ("magnetohydrostatic_equilibrium", "magnetic_divergence")
BOUNDARY_EQUATIONS = (
    "mean_radial_optical_depth_anchor",
    "upper_boundary_gas_pressure_prior",
)
EQUATION_NAMES = (*VOLUME_EQUATIONS, *BOUNDARY_EQUATIONS)

GAUSS_TO_TESLA = 1.0e-4
VACUUM_PERMEABILITY_H_PER_M = 4.0 * math.pi * 1.0e-7


@dataclass(frozen=True)
class LTEPhysicsNormalization:
    """Independent scales used to nondimensionalize spatial residuals."""

    length_m: float = 1.0e6

    def __post_init__(self):
        for name, value in (("length_m", self.length_m),):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Physics normalization {name} must be finite and positive.")

    @classmethod
    def from_config(cls, config: Mapping | None) -> "LTEPhysicsNormalization":
        values = dict(config or {})
        unknown = set(values) - {"length_m"}
        if unknown:
            raise TypeError(f"Unknown physics normalization options: {sorted(unknown)}")
        return cls(**values)

    def configuration(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
        }


@dataclass(frozen=True)
class LTEPhysicsState:
    """Atmospheric values and shared Cartesian derivatives at sampled shell points."""

    position_m: torch.Tensor
    geometric_height_m: torch.Tensor
    temperature: torch.Tensor
    gas_pressure: torch.Tensor
    mass_density: torch.Tensor | None
    magnetic_field_gauss: torch.Tensor
    jacobian: torch.Tensor | None
    jacobian_slices: dict[str, slice]
    radial_unit: torch.Tensor

    def derivative(self, name: str) -> torch.Tensor:
        if self.jacobian is None or name not in self.jacobian_slices:
            raise KeyError(f"Primitive derivative {name!r} was not requested.")
        return self.jacobian[:, self.jacobian_slices[name], :]


@dataclass(frozen=True)
class LTEPhysicsResult:
    losses: dict[str, torch.Tensor]
    residual_norms: dict[str, torch.Tensor]
    weights: dict[str, float]
    state: LTEPhysicsState | None = None


def _joint_pointwise_jacobian(
    primitives: Mapping[str, torch.Tensor],
    position_rsun: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor | None, dict[str, slice]]:
    """Differentiate all requested primitive components with one input tensor."""

    if not primitives:
        return None, {}
    flattened = []
    slices = {}
    offset = 0
    for name, value in primitives.items():
        value = value.reshape(position_rsun.shape[0], -1)
        flattened.append(value)
        slices[name] = slice(offset, offset + value.shape[-1])
        offset += value.shape[-1]
    joined = torch.cat(flattened, dim=-1)
    rows = []
    for component in range(joined.shape[-1]):
        rows.append(torch.autograd.grad(
            joined[:, component],
            position_rsun,
            grad_outputs=torch.ones_like(joined[:, component]),
            create_graph=create_graph,
            retain_graph=True,
        )[0])
    return torch.stack(rows, dim=1), slices


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


def _surface_mean(values: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    """Broadcast the sampled spatial mean on each shared shell coordinate."""

    _, inverse = torch.unique(coordinate.detach(), sorted=False, return_inverse=True)
    sums = torch.zeros_like(values).scatter_add(0, inverse, values)
    counts = torch.zeros_like(values).scatter_add(0, inverse, torch.ones_like(values))
    return (sums / counts.clamp_min(1.0))[inverse]


class LTEPhysicsModule(nn.Module):
    """Evaluate magnetic and mean radial optical-depth constraints."""

    def __init__(
        self,
        equations: Mapping | None = None,
        *,
        gravity_m_per_s2: float | None = None,
        vector_basis_matches_spatial_coordinates: bool = False,
        normalization: Mapping | None = None,
    ):
        super().__init__()
        equations = dict(equations or {})
        unknown = set(equations) - set(EQUATION_NAMES)
        if unknown:
            raise KeyError(
                f"Unknown LTE physics equations {sorted(unknown)}; expected {EQUATION_NAMES}."
            )
        self.enabled: dict[str, bool] = {}
        self.fixed_weights: dict[str, float] = {}
        self.equation_config: dict[str, dict] = {}
        for name in EQUATION_NAMES:
            raw = equations.get(name, {})
            if isinstance(raw, Real) and not isinstance(raw, bool):
                raw = {"enabled": float(raw) != 0.0, "weight": float(raw)}
            if not isinstance(raw, Mapping):
                raise TypeError(f"Physics equation {name!r} must be a mapping or number.")
            config = dict(raw)
            enabled = bool(config.pop("enabled", False))
            weight = config.pop("weight", 1.0 if enabled else 0.0)
            if config:
                raise TypeError(f"Unknown options for physics equation {name!r}: {sorted(config)}")
            if not isinstance(weight, Real) or isinstance(weight, bool):
                raise TypeError(
                    f"Physics equation {name!r} requires a fixed numeric weight; schedules are unsupported."
                )
            weight = float(weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Physics equation {name!r} weight must be finite and non-negative.")
            self.enabled[name] = enabled
            self.fixed_weights[name] = weight if enabled else 0.0
            self.equation_config[name] = {"enabled": enabled, "weight": weight}

        self.gravity_m_per_s2 = None if gravity_m_per_s2 is None else float(gravity_m_per_s2)
        if self.enabled["magnetohydrostatic_equilibrium"] and (
            self.gravity_m_per_s2 is None or self.gravity_m_per_s2 <= 0
        ):
            raise ValueError(
                "magnetohydrostatic_equilibrium requires positive gravity_m_per_s2."
            )
        self.vector_basis_matches_spatial_coordinates = bool(
            vector_basis_matches_spatial_coordinates
        )
        vector_derivatives_enabled = (
            self.enabled["magnetohydrostatic_equilibrium"]
            or self.enabled["magnetic_divergence"]
        )
        if vector_derivatives_enabled and not self.vector_basis_matches_spatial_coordinates:
            raise ValueError(
                "Magnetic differential equations require "
                "vector_basis_matches_spatial_coordinates=true."
            )
        self.normalization = LTEPhysicsNormalization.from_config(normalization)

    @property
    def any_enabled(self) -> bool:
        return any(self.enabled.values())

    @property
    def volume_enabled(self) -> bool:
        return any(self.enabled[name] for name in VOLUME_EQUATIONS)

    def weights(self, global_step: int = 0, *, final: bool = False) -> dict[str, float]:
        del global_step, final
        return dict(self.fixed_weights)

    def build_state(
        self,
        atmosphere_model,
        continuum_opacity,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
        active_equations: set[str],
        *,
        create_graph: bool,
        position_m: torch.Tensor | None = None,
    ) -> LTEPhysicsState:
        parameter = next(atmosphere_model.parameters())
        coords = coords.to(parameter).reshape(-1, 3)
        geometric_height_m = geometric_height_m.to(parameter).reshape(-1)
        if coords.shape[0] != geometric_height_m.shape[0]:
            raise ValueError(
                "Physics coords and geometric_height_m must contain the same samples."
            )
        if position_m is None:
            position_m = atmosphere_model.position_from_coords_height(
                coords, geometric_height_m
            )
        else:
            position_m = position_m.to(parameter).reshape(-1, 3)
            if position_m.shape[0] != coords.shape[0]:
                raise ValueError("Physics positions and coordinates must contain the same samples.")

        # Carrington solar-radius coordinates remain O(1) in float32. The
        # chain rule below converts their derivatives to physical per-metre units.
        position_rsun = (position_m / atmosphere_model.solar_radius_m).detach().requires_grad_(True)
        fields = atmosphere_model.evaluate_position_rsun(position_rsun)
        temperature = fields["temperature"]
        pressure = fields["gas_pressure"]
        magnetic = fields["magnetic_field"]
        radial_unit = position_rsun / torch.linalg.vector_norm(
            position_rsun, dim=-1, keepdim=True
        )
        density = (
            continuum_opacity.reference_mass_density(temperature, pressure)
            if "magnetohydrostatic_equilibrium" in active_equations
            else None
        )
        primitives = {}
        if "magnetohydrostatic_equilibrium" in active_equations:
            primitives["pressure"] = pressure
        if (
            "magnetohydrostatic_equilibrium" in active_equations
            or "magnetic_divergence" in active_equations
        ):
            primitives["magnetic"] = magnetic
        jacobian, slices = _joint_pointwise_jacobian(
            primitives, position_rsun, create_graph=create_graph
        )
        if jacobian is not None:
            jacobian = jacobian / atmosphere_model.solar_radius_m
        return LTEPhysicsState(
            position_m=position_rsun * atmosphere_model.solar_radius_m,
            geometric_height_m=geometric_height_m,
            temperature=temperature,
            gas_pressure=pressure,
            mass_density=density,
            magnetic_field_gauss=magnetic,
            jacobian=jacobian,
            jacobian_slices=slices,
            radial_unit=radial_unit,
        )

    def volume(
        self,
        atmosphere_model,
        continuum_opacity,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
        *,
        global_step: int = 0,
        final_weights: bool = False,
        create_graph: bool = True,
        return_state: bool = False,
        position_m: torch.Tensor | None = None,
    ) -> LTEPhysicsResult:
        weights = self.weights(global_step, final=final_weights)
        active = {
            name for name in VOLUME_EQUATIONS
            if self.enabled[name] and weights[name] != 0.0
        }
        zero = next(atmosphere_model.parameters()).new_zeros(())
        losses = {name: zero for name in VOLUME_EQUATIONS}
        if not active:
            return LTEPhysicsResult(losses, {}, weights)
        state = self.build_state(
            atmosphere_model,
            continuum_opacity,
            coords,
            geometric_height_m,
            active,
            create_graph=create_graph,
            position_m=position_m,
        )
        residual_norms = {}
        if "magnetohydrostatic_equilibrium" in active:
            grad_pressure = state.derivative("pressure")[:, 0]
            magnetic_tesla = state.magnetic_field_gauss * GAUSS_TO_TESLA
            curl_magnetic_tesla = _curl(
                state.derivative("magnetic") * GAUSS_TO_TESLA
            )
            lorentz_force = torch.linalg.cross(
                curl_magnetic_tesla, magnetic_tesla, dim=-1
            ) / VACUUM_PERMEABILITY_H_PER_M
            # Compare the force imbalance with the mean pressure-equivalent
            # magnitude of all three MHS terms on the sampled height layer.
            # Multiplying rho*g by the fixed physical normalization length L0
            # gives pascals, while B^2/mu0 is magnetic pressure in the same
            # units. Every mean is formed from points on the same sampled
            # geometric-height layer; no optical-depth coordinate is involved.
            pressure_equivalent = (
                state.gas_pressure
                + state.mass_density
                * self.gravity_m_per_s2
                * self.normalization.length_m
                + magnetic_tesla.square().sum(dim=-1)
                / VACUUM_PERMEABILITY_H_PER_M
            )
            force_scale = _surface_mean(
                pressure_equivalent, state.geometric_height_m
            ).clamp_min(torch.finfo(state.gas_pressure.dtype).tiny).detach()
            residual = (
                grad_pressure
                + state.mass_density[:, None]
                * self.gravity_m_per_s2
                * state.radial_unit
                - lorentz_force
            ) * (self.normalization.length_m / force_scale)[:, None]
            losses["magnetohydrostatic_equilibrium"] = residual.square().sum(dim=-1).mean()
            residual_norms["magnetohydrostatic_equilibrium"] = torch.linalg.vector_norm(
                residual, dim=-1
            )
        if "magnetic_divergence" in active:
            layer_mean_field = _surface_mean(
                torch.linalg.vector_norm(state.magnetic_field_gauss, dim=-1),
                state.geometric_height_m,
            ).clamp_min(torch.finfo(state.magnetic_field_gauss.dtype).tiny).detach()
            residual = (
                _divergence(state.derivative("magnetic"))
                * self.normalization.length_m
                / layer_mean_field
            )
            losses["magnetic_divergence"] = residual.square().mean()
            residual_norms["magnetic_divergence"] = residual.abs()
        return LTEPhysicsResult(
            losses, residual_norms, weights, state if return_state else None
        )

    def mean_radial_optical_depth_anchor(
        self,
        atmosphere_model,
        continuum_opacity,
        coords: torch.Tensor,
        *,
        global_step: int = 0,
        final_weights: bool = False,
        depth_points: int = 65,
    ) -> LTEPhysicsResult:
        weights = self.weights(global_step, final=final_weights)
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        name = "mean_radial_optical_depth_anchor"
        if not self.enabled[name] or weights[name] == 0.0:
            return LTEPhysicsResult({name: zero}, {}, weights)
        if depth_points < 2:
            raise ValueError("The optical-depth anchor requires at least two radial points.")
        coords = coords.to(parameter).reshape(-1, 3)
        if coords.shape[0] < 1:
            raise ValueError("The optical-depth anchor requires surface samples.")
        q_top = atmosphere_model.log_tau500[0]
        if not q_top < 0:
            raise ValueError("The atmosphere must extend above log_tau500=0.")
        radial_q = torch.linspace(
            q_top,
            q_top.new_zeros(()),
            int(depth_points),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        height_m = atmosphere_model.depth_to_height(radial_q, (coords.shape[0],))
        fields = atmosphere_model.evaluate_at_height(coords, height_m)
        alpha500 = continuum_opacity.volume_extinction_at_5000(
            fields["temperature"], fields["gas_pressure"]
        )
        interval_m = height_m[..., :-1] - height_m[..., 1:]
        tau_surface = (
            0.5 * (alpha500[..., :-1] + alpha500[..., 1:]) * interval_m
        ).sum(dim=-1)
        if not torch.isfinite(tau_surface).all() or torch.any(tau_surface <= 0):
            raise FloatingPointError(
                "Radial optical depth at R_sun must be finite and positive."
            )
        log_tau_surface = torch.log10(tau_surface)
        mean_log_tau = log_tau_surface.mean()
        return LTEPhysicsResult(
            {name: mean_log_tau.square()},
            {
                name: log_tau_surface.abs(),
                "radial_log_tau500_at_solar_surface": log_tau_surface,
            },
            weights,
        )

    def upper_boundary_gas_pressure_prior(
        self,
        atmosphere_model,
        coords: torch.Tensor,
        *,
        global_step: int = 0,
        final_weights: bool = False,
    ) -> LTEPhysicsResult:
        """Weakly retain the radial-reference pressure at the shell top."""

        weights = self.weights(global_step, final=final_weights)
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        name = "upper_boundary_gas_pressure_prior"
        if not self.enabled[name] or weights[name] == 0.0:
            return LTEPhysicsResult({name: zero}, {}, weights)
        coords = coords.to(parameter).reshape(-1, 3)
        if coords.shape[0] < 1:
            raise ValueError("The upper-boundary pressure prior requires samples.")
        upper_height_m = parameter.new_full(
            (coords.shape[0], 1),
            atmosphere_model.shell_height_bounds_Mm[0] * 1.0e6,
        )
        pressure = atmosphere_model.evaluate_at_height(
            coords, upper_height_m
        )["gas_pressure"][..., 0]
        reference_log_pressure = atmosphere_model.reference_atmosphere.logs_at_height(
            upper_height_m[0, 0]
        )[1]
        residual = torch.log(pressure) - reference_log_pressure
        return LTEPhysicsResult(
            {name: residual.square().mean()},
            {name: residual.abs()},
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
    "LTEPhysicsModule",
    "LTEPhysicsNormalization",
    "LTEPhysicsResult",
    "LTEPhysicsState",
    "VOLUME_EQUATIONS",
]
