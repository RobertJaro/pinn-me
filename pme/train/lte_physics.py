"""Configurable differential physics for stratified LTE atmospheres.

Direct-``log_tau`` HSE evaluates only the pressure derivative with respect to
optical depth. Experimental geometric equations construct at most one joint
Jacobian of the primitive fields required by all active equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from pme.train.physics import PhysicsWeightSchedule


FOUR_PI = 4.0 * math.pi
CM_PER_M = 100.0
N_PER_M3_TO_DYN_PER_CM3 = 0.1
VOLUME_EQUATIONS = (
    "hse",
    "tau_mapping",
    "divergence_b",
    "mhs",
    "continuity",
    "induction",
    "momentum",
)
BOUNDARY_EQUATIONS = ("pressure_boundary",)
EQUATION_NAMES = (*VOLUME_EQUATIONS, *BOUNDARY_EQUATIONS)
MHD_EQUATIONS = {"divergence_b", "mhs", "continuity", "induction", "momentum"}


@dataclass(frozen=True)
class LTEPhysicsNormalization:
    """Gaussian-cgs normalization derived from independent ``L0``, ``t0``, ``B0``.

    Public atmosphere tensors retain their documented SI units (and gauss for
    magnetic field). Equation residuals are converted and nondimensionalized
    with the ideal-MHD scales derived here.
    """

    length_m: float = 1.0e6
    time_s: float = 1.0e3
    magnetic_field_gauss: float = 100.0

    def __post_init__(self):
        for name, value in (
            ("length_m", self.length_m),
            ("time_s", self.time_s),
            ("magnetic_field_gauss", self.magnetic_field_gauss),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Physics normalization {name} must be finite and positive."
                )

    @classmethod
    def from_config(cls, config: Mapping | None) -> LTEPhysicsNormalization:
        values = dict(config or {})
        unknown = set(values) - {"length_m", "time_s", "magnetic_field_gauss"}
        if unknown:
            raise TypeError(f"Unknown physics normalization options: {sorted(unknown)}")
        return cls(**values)

    @property
    def length_cm(self) -> float:
        return self.length_m * CM_PER_M

    @property
    def velocity_m_per_s(self) -> float:
        return self.length_m / self.time_s

    @property
    def pressure_dyn_per_cm2(self) -> float:
        return self.magnetic_field_gauss**2 / FOUR_PI

    @property
    def mass_density_g_per_cm3(self) -> float:
        velocity_cm_per_s = self.length_cm / self.time_s
        return self.pressure_dyn_per_cm2 / velocity_cm_per_s**2

    @property
    def gravity_m_per_s2(self) -> float:
        return self.length_m / self.time_s**2

    @property
    def force_density_dyn_per_cm3(self) -> float:
        return self.pressure_dyn_per_cm2 / self.length_cm

    def configuration(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "time_s": self.time_s,
            "magnetic_field_gauss": self.magnetic_field_gauss,
        }


@dataclass(frozen=True)
class LTEPhysicsState:
    """Physical values and their optional shared spatial Jacobian.

    Spatial derivatives, when present, use SI geometric coordinates
    ``[x_m, y_m, z_m]``. Direct-tau HSE instead stores only
    ``dP/dlog10(tau500)``.
    Geometric force balance differentiates gas pressure itself, not its
    logarithm. ``jacobian`` has shape ``[sample, primitive, xyz]`` and ``jacobian_slices``
    maps primitive names to its component slices. It is ``None`` when the
    active equations are algebraic and require no derivatives. Magnetic-field
    values and their Jacobian are in gauss and gauss per metre, respectively;
    force-balance equations convert derivatives to per centimetre and evaluate
    every force density in Gaussian cgs units (dyn cm^-3).
    """

    log_tau500: torch.Tensor
    position_m: torch.Tensor | None
    geometric_height_m: torch.Tensor | None
    metric_m_per_log_tau: torch.Tensor | None
    temperature: torch.Tensor
    gas_pressure: torch.Tensor
    mass_density: torch.Tensor | None
    alpha500: torch.Tensor | None
    magnetic_field_gauss: torch.Tensor
    velocity_field_m_per_s: torch.Tensor
    jacobian: torch.Tensor | None
    jacobian_slices: dict[str, slice]
    pressure_derivative_log_tau: torch.Tensor | None = None

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
    position_m: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor | None, dict[str, slice]]:
    """Differentiate selected primitives once into one shared matrix."""

    if not primitives:
        return None, {}
    flattened = []
    slices = {}
    offset = 0
    for name, value in primitives.items():
        value = value.reshape(position_m.shape[0], -1)
        flattened.append(value)
        slices[name] = slice(offset, offset + value.shape[-1])
        offset += value.shape[-1]
    joined = torch.cat(flattened, dim=-1)
    rows = []
    for component in range(joined.shape[-1]):
        derivative = torch.autograd.grad(
            joined[:, component],
            position_m,
            grad_outputs=torch.ones_like(joined[:, component]),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        rows.append(derivative)
    return torch.stack(rows, dim=1), slices


def _curl(jacobian: torch.Tensor) -> torch.Tensor:
    """Curl for a Jacobian arranged as ``[sample, component, x/y/z]``."""

    return torch.stack(
        (
            jacobian[:, 2, 1] - jacobian[:, 1, 2],
            jacobian[:, 0, 2] - jacobian[:, 2, 0],
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
        ),
        dim=-1,
    )


def _divergence(jacobian: torch.Tensor) -> torch.Tensor:
    return jacobian[:, 0, 0] + jacobian[:, 1, 1] + jacobian[:, 2, 2]


def _tau_surface_mean(
    values: torch.Tensor,
    log_tau500: torch.Tensor,
) -> torch.Tensor:
    """Return the sampled spatial mean at each shared optical depth."""

    _, inverse = torch.unique(
        log_tau500.detach(), sorted=False, return_inverse=True
    )
    # An upper-bound allocation avoids a device-to-host synchronization just
    # to discover the number of occupied groups.
    sums = torch.zeros_like(values).scatter_add(0, inverse, values)
    counts = torch.zeros_like(values).scatter_add(
        0, inverse, torch.ones_like(values)
    )
    return (sums / counts.clamp_min(1.0))[inverse]


def _direct_tau_hse_residual(
    pressure_derivative_log_tau: torch.Tensor,
    target_pressure_derivative_log_tau: torch.Tensor,
    gas_pressure: torch.Tensor,
    log_tau500: torch.Tensor,
) -> torch.Tensor:
    """Linear direct-tau HSE normalized by sampled mean surface pressure."""

    tiny = torch.finfo(gas_pressure.dtype).tiny
    mean_surface_pressure = _tau_surface_mean(
        gas_pressure, log_tau500
    ).clamp_min(tiny)
    # The surface mean defines residual units. Do not let its gradient create
    # an incentive to inflate pressure merely to shrink the HSE objective.
    return (
        pressure_derivative_log_tau - target_pressure_derivative_log_tau
    ) / mean_surface_pressure.detach()


def _lorentz_force_density_cgs(
    magnetic_field_gauss: torch.Tensor,
    magnetic_jacobian_gauss_per_m: torch.Tensor,
) -> torch.Tensor:
    """Return ``(curl(B) x B)/(4 pi)`` in dyn cm^-3 (Gaussian cgs)."""

    curl_gauss_per_cm = _curl(magnetic_jacobian_gauss_per_m) / CM_PER_M
    return torch.cross(curl_gauss_per_cm, magnetic_field_gauss, dim=-1) / FOUR_PI


class LTEPhysicsModule(nn.Module):
    """Evaluate selected static LTE/MHD equations with scheduled weights."""

    def __init__(
        self,
        equations: Mapping | None = None,
        *,
        gravity_m_per_s2: float | None = None,
        top_pressure_pa: float | None = None,
        vector_basis_matches_spatial_coordinates: bool = False,
        normalization: Mapping | None = None,
    ):
        super().__init__()
        equations = dict(equations or {})
        unknown = set(equations) - set(EQUATION_NAMES)
        if unknown:
            raise KeyError(
                f"Unknown LTE physics equations {sorted(unknown)}; "
                f"expected names from {EQUATION_NAMES}."
            )
        self.enabled = {}
        self.schedules = {}
        self.equation_config = {}
        for name in EQUATION_NAMES:
            raw = equations.get(name, {})
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                raw = {"enabled": float(raw) != 0.0, "weight": float(raw)}
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"Physics equation {name!r} must be a mapping or number."
                )
            config = dict(raw)
            enabled = bool(config.pop("enabled", False))
            weight = config.pop("weight", 1.0 if enabled else 0.0)
            if config:
                raise TypeError(
                    f"Unknown options for physics equation {name!r}: {sorted(config)}"
                )
            schedule = PhysicsWeightSchedule.from_config(weight)
            self.enabled[name] = enabled
            self.schedules[name] = schedule
            self.equation_config[name] = {
                "enabled": enabled,
                "weight": schedule.configuration(),
            }

        self.gravity_m_per_s2 = (
            None if gravity_m_per_s2 is None else float(gravity_m_per_s2)
        )
        self.top_pressure_pa = (
            None if top_pressure_pa is None else float(top_pressure_pa)
        )
        if any(self.enabled[name] for name in ("hse", "mhs", "momentum")):
            if self.gravity_m_per_s2 is None or not self.gravity_m_per_s2 > 0:
                raise ValueError(
                    "HSE/MHS/momentum physics requires positive gravity_m_per_s2."
                )
        force_balance_equations = [
            name for name in ("hse", "mhs", "momentum") if self.enabled[name]
        ]
        if len(force_balance_equations) > 1:
            raise ValueError(
                "Enable exactly one of hse, mhs, or momentum force balance; got "
                f"{force_balance_equations}. Combining them would impose additional "
                "unphysical zero-force constraints."
            )
        if self.enabled["pressure_boundary"]:
            if self.top_pressure_pa is None or not self.top_pressure_pa > 0:
                raise ValueError(
                    "pressure_boundary physics requires positive top_pressure_pa."
                )
        self.vector_basis_matches_spatial_coordinates = bool(
            vector_basis_matches_spatial_coordinates
        )
        self.normalization = LTEPhysicsNormalization.from_config(normalization)
        requested_mhd = sorted(name for name in MHD_EQUATIONS if self.enabled[name])
        if requested_mhd and not self.vector_basis_matches_spatial_coordinates:
            raise ValueError(
                "MHD equations require vector_basis_matches_spatial_coordinates=true "
                "after independently calibrating the Hinode transverse Stokes basis; "
                f"requested {requested_mhd}."
            )

    @property
    def any_enabled(self) -> bool:
        return any(self.enabled.values())

    @property
    def volume_enabled(self) -> bool:
        return any(self.enabled[name] for name in VOLUME_EQUATIONS)

    def weights(self, global_step: int, *, final: bool = False) -> dict[str, float]:
        return {
            name: (
                self.schedules[name].end
                if final
                else self.schedules[name].value_at(global_step)
            )
            if self.enabled[name]
            else 0.0
            for name in EQUATION_NAMES
        }

    def _active_volume_equations(
        self, global_step: int, *, final: bool
    ) -> tuple[set[str], dict[str, float]]:
        weights = self.weights(global_step, final=final)
        active = {
            name
            for name in VOLUME_EQUATIONS
            if self.enabled[name] and weights[name] != 0.0
        }
        return active, weights

    def build_state(
        self,
        atmosphere_model,
        continuum_opacity,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
        active_equations: set[str],
        *,
        create_graph: bool,
    ) -> LTEPhysicsState:
        parameter = next(atmosphere_model.parameters())
        coords = coords.to(device=parameter.device, dtype=parameter.dtype).reshape(
            -1, 3
        )
        q = log_tau500.to(device=parameter.device, dtype=parameter.dtype).reshape(-1, 1)
        if coords.shape[0] != q.shape[0]:
            raise ValueError(
                "Physics coords and log_tau500 must contain the same samples."
            )
        if getattr(atmosphere_model, "coordinate_mode", "geometric_height") == "log_tau":
            unsupported = active_equations & {
                "tau_mapping", "divergence_b", "mhs", "continuity",
                "induction", "momentum",
            }
            if unsupported:
                raise ValueError(
                    "A direct log_tau atmosphere cannot evaluate geometric spatial "
                    f"equations: {sorted(unsupported)}."
                )
            q = q.detach().requires_grad_("hse" in active_equations)
            fields = atmosphere_model.evaluate_points(coords, q)
            temperature = fields["temperature"][:, 0]
            pressure = fields["gas_pressure"][:, 0]
            density = (
                continuum_opacity.reference_mass_density(temperature, pressure)
                if "hse" in active_equations
                else None
            )
            alpha500 = (
                continuum_opacity.volume_extinction_at_5000(temperature, pressure)
                if "hse" in active_equations
                else None
            )
            pressure_derivative = None
            if "hse" in active_equations:
                pressure_derivative = torch.autograd.grad(
                    pressure,
                    q,
                    grad_outputs=torch.ones_like(pressure),
                    create_graph=create_graph,
                    retain_graph=True,
                )[0][:, 0]
            return LTEPhysicsState(
                log_tau500=q[:, 0],
                position_m=None,
                geometric_height_m=None,
                metric_m_per_log_tau=None,
                temperature=temperature,
                gas_pressure=pressure,
                mass_density=density,
                alpha500=alpha500,
                magnetic_field_gauss=fields["magnetic_field"][:, 0],
                velocity_field_m_per_s=fields["velocity_field"][:, 0],
                jacobian=None,
                jacobian_slices={},
                pressure_derivative_log_tau=pressure_derivative,
            )

        height, metric = atmosphere_model.height_mapping.height_and_metric(
            coords, q, create_graph=create_graph
        )
        z_m = height[:, 0]
        position_m = torch.stack(
            (coords[:, 1] * 1.0e6, coords[:, 2] * 1.0e6, z_m), dim=-1
        )
        model_coords = torch.stack(
            (
                coords[:, 0],
                position_m[:, 0] / 1.0e6,
                position_m[:, 1] / 1.0e6,
            ),
            dim=-1,
        )
        fields = atmosphere_model.evaluate_at_height(model_coords, position_m[:, 2:3])
        temperature = fields["temperature"][:, 0]
        pressure = fields["gas_pressure"][:, 0]
        velocity = fields["velocity_field"][:, 0]
        magnetic_gauss = fields["magnetic_field"][:, 0]

        needs_density = bool(
            active_equations & {"hse", "mhs", "continuity", "momentum"}
        )
        density = (
            continuum_opacity.reference_mass_density(temperature, pressure)
            if needs_density
            else None
        )
        alpha500 = (
            continuum_opacity.volume_extinction_at_5000(temperature, pressure)
            if "tau_mapping" in active_equations
            else None
        )
        derivative_names = set()
        if "hse" in active_equations:
            derivative_names.add("pressure")
        if "continuity" in active_equations:
            derivative_names.update(("log_density", "velocity"))
        if active_equations & {"divergence_b", "mhs", "induction", "momentum"}:
            derivative_names.add("magnetic")
        if active_equations & {"induction", "momentum"}:
            derivative_names.add("velocity")
        if active_equations & {"mhs", "momentum"}:
            derivative_names.add("pressure")

        candidates = {
            "pressure": pressure,
            "log_density": None if density is None else torch.log(density),
            "magnetic": magnetic_gauss,
            "velocity": velocity,
        }
        primitives = {
            name: candidates[name]
            for name in ("pressure", "log_density", "magnetic", "velocity")
            if name in derivative_names
        }
        jacobian, slices = _joint_pointwise_jacobian(
            primitives, position_m, create_graph=create_graph
        )
        return LTEPhysicsState(
            log_tau500=q[:, 0],
            position_m=position_m,
            geometric_height_m=z_m,
            metric_m_per_log_tau=metric[:, 0],
            temperature=temperature,
            gas_pressure=pressure,
            mass_density=density,
            alpha500=alpha500,
            magnetic_field_gauss=magnetic_gauss,
            velocity_field_m_per_s=velocity,
            jacobian=jacobian,
            jacobian_slices=slices,
        )

    def volume(
        self,
        atmosphere_model,
        continuum_opacity,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
        *,
        global_step: int,
        final_weights: bool = False,
        create_graph: bool = True,
        return_state: bool = False,
    ) -> LTEPhysicsResult:
        active, weights = self._active_volume_equations(
            global_step, final=final_weights
        )
        if not active:
            zero = next(atmosphere_model.parameters()).new_zeros(())
            return LTEPhysicsResult(
                {name: zero for name in VOLUME_EQUATIONS},
                {},
                weights,
                None,
            )
        state = self.build_state(
            atmosphere_model,
            continuum_opacity,
            coords,
            log_tau500,
            active,
            create_graph=create_graph,
        )
        zero = state.temperature.new_zeros(())
        losses = {name: zero for name in VOLUME_EQUATIONS}
        residual_norms = {}

        if "hse" in active:
            if state.pressure_derivative_log_tau is not None:
                tiny = torch.finfo(state.temperature.dtype).tiny
                tau500 = torch.pow(
                    state.log_tau500.new_tensor(10.0), state.log_tau500
                )
                target_pressure_derivative = (
                    state.log_tau500.new_tensor(math.log(10.0))
                    * tau500
                    * state.mass_density
                    * self.gravity_m_per_s2
                    / state.alpha500.clamp_min(tiny)
                )
                residual = _direct_tau_hse_residual(
                    state.pressure_derivative_log_tau,
                    target_pressure_derivative,
                    state.gas_pressure,
                    state.log_tau500,
                )
            else:
                pressure_gradient = state.derivative("pressure")[:, 0, 2]
                tiny = torch.finfo(state.gas_pressure.dtype).tiny
                surface_pressure = _tau_surface_mean(
                    state.gas_pressure, state.log_tau500
                ).clamp_min(tiny)
                predicted_pressure_derivative_log_tau = (
                    -pressure_gradient * state.metric_m_per_log_tau
                )
                required_pressure_derivative_log_tau = (
                    state.mass_density
                    * self.gravity_m_per_s2
                    * state.metric_m_per_log_tau
                )
                # Work in the retained tau coordinate while differentiating P
                # itself. The only normalization is the detached mean pressure
                # on each shared optical-depth surface.
                residual = (
                    predicted_pressure_derivative_log_tau
                    - required_pressure_derivative_log_tau
                ) / surface_pressure.detach()
            losses["hse"] = residual.square().mean()
            residual_norms["hse"] = residual.abs()

        if "tau_mapping" in active:
            tiny = torch.finfo(state.alpha500.dtype).tiny
            target = (
                state.log_tau500.new_tensor(math.log(10.0))
                * torch.pow(state.log_tau500.new_tensor(10.0), state.log_tau500)
            )
            signed_ratio = (
                state.alpha500 * state.metric_m_per_log_tau
            ) / target.clamp_min(tiny)
            # asinh preserves the sign (and therefore a useful gradient when
            # the unconstrained Z network initially has the wrong orientation)
            # while compressing very large dimensionless residuals.
            residual = torch.asinh(signed_ratio) - torch.asinh(
                torch.ones_like(signed_ratio)
            )
            losses["tau_mapping"] = residual.square().mean()
            residual_norms["tau_mapping"] = residual.abs()

        b_jac = (
            state.derivative("magnetic")
            if "magnetic" in state.jacobian_slices
            else None
        )
        v_jac = (
            state.derivative("velocity")
            if "velocity" in state.jacobian_slices
            else None
        )
        if "divergence_b" in active:
            divergence_unit = (
                self.normalization.magnetic_field_gauss / self.normalization.length_m
            )
            normalized = _divergence(b_jac) / divergence_unit
            losses["divergence_b"] = normalized.square().mean()
            residual_norms["divergence_b"] = normalized.abs()

        if "mhs" in active:
            b = state.magnetic_field_gauss
            rho = state.mass_density
            grad_p = (
                N_PER_M3_TO_DYN_PER_CM3
                * state.derivative("pressure")[:, 0]
                / self.normalization.force_density_dyn_per_cm3
            )
            gravity = torch.zeros_like(grad_p)
            gravity[:, 2] = (
                -N_PER_M3_TO_DYN_PER_CM3 * rho * self.gravity_m_per_s2
            ) / self.normalization.force_density_dyn_per_cm3
            lorentz = (
                _lorentz_force_density_cgs(b, b_jac)
                / self.normalization.force_density_dyn_per_cm3
            )
            residual = grad_p - gravity - lorentz
            losses["mhs"] = residual.square().sum(dim=-1).mean()
            residual_norms["mhs"] = torch.linalg.vector_norm(residual, dim=-1)

        if "continuity" in active:
            grad_log_rho = state.derivative("log_density")[:, 0]
            divergence_v = _divergence(v_jac)
            advection = (state.velocity_field_m_per_s * grad_log_rho).sum(dim=-1)
            normalized = (advection + divergence_v) * self.normalization.time_s
            losses["continuity"] = normalized.square().mean()
            residual_norms["continuity"] = normalized.abs()

        if "induction" in active:
            v = state.velocity_field_m_per_s
            b = state.magnetic_field_gauss
            # d_j(v x B)_i = eps_ikl[(d_j v_k)B_l + v_k(d_j B_l)]
            cross_jac = torch.stack(
                [
                    torch.cross(v_jac[:, :, axis], b, dim=-1)
                    + torch.cross(v, b_jac[:, :, axis], dim=-1)
                    for axis in range(3)
                ],
                dim=-1,
            )
            induction_unit_gauss_per_s = (
                self.normalization.magnetic_field_gauss / self.normalization.time_s
            )
            normalized = _curl(cross_jac) / induction_unit_gauss_per_s
            losses["induction"] = normalized.square().sum(dim=-1).mean()
            residual_norms["induction"] = torch.linalg.vector_norm(normalized, dim=-1)

        if "momentum" in active:
            v = state.velocity_field_m_per_s
            b = state.magnetic_field_gauss
            rho = state.mass_density
            grad_p = (
                N_PER_M3_TO_DYN_PER_CM3
                * state.derivative("pressure")[:, 0]
                / self.normalization.force_density_dyn_per_cm3
            )
            inertial = (
                N_PER_M3_TO_DYN_PER_CM3
                * (rho[:, None] * torch.einsum("nj,nij->ni", v, v_jac))
                / self.normalization.force_density_dyn_per_cm3
            )
            gravity = torch.zeros_like(inertial)
            gravity[:, 2] = (
                -N_PER_M3_TO_DYN_PER_CM3 * rho * self.gravity_m_per_s2
            ) / self.normalization.force_density_dyn_per_cm3
            lorentz = (
                _lorentz_force_density_cgs(b, b_jac)
                / self.normalization.force_density_dyn_per_cm3
            )
            residual = inertial + grad_p - gravity - lorentz
            losses["momentum"] = residual.square().sum(dim=-1).mean()
            residual_norms["momentum"] = torch.linalg.vector_norm(residual, dim=-1)

        return LTEPhysicsResult(
            losses,
            residual_norms,
            weights,
            state if return_state else None,
        )

    def pressure_boundary(
        self,
        atmosphere_model,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
        gas_pressure_pa: torch.Tensor | None,
        *,
        global_step: int,
        final_weights: bool = False,
    ) -> LTEPhysicsResult:
        weights = self.weights(global_step, final=final_weights)
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        if not self.enabled["pressure_boundary"] or weights["pressure_boundary"] == 0.0:
            return LTEPhysicsResult({"pressure_boundary": zero}, {}, weights)
        coords = coords.to(device=parameter.device, dtype=parameter.dtype)
        q = log_tau500.to(device=parameter.device, dtype=parameter.dtype).reshape(-1, 1)
        top = atmosphere_model.log_tau500[0]
        if not torch.allclose(q, top.expand_as(q)):
            raise ValueError(
                "Pressure-boundary samples must lie on the configured top face."
            )
        if getattr(atmosphere_model, "coordinate_mode", "geometric_height") == "log_tau":
            pressure = atmosphere_model.evaluate_points(coords, q)["gas_pressure"][:, 0]
        else:
            height = atmosphere_model.height_mapping(coords, q)
            pressure = atmosphere_model.evaluate_at_height(coords, height)["gas_pressure"][
                :, 0
            ]
        target = (
            pressure.new_full(pressure.shape, self.top_pressure_pa)
            if gas_pressure_pa is None
            else gas_pressure_pa.to(pressure).reshape_as(pressure)
        )
        if not torch.isfinite(target).all() or torch.any(target <= 0):
            raise ValueError("Pressure-boundary targets must be finite and positive.")
        residual = torch.log10(pressure / target)
        return LTEPhysicsResult(
            {"pressure_boundary": residual.square().mean()},
            {"pressure_boundary": residual.abs()},
            weights,
        )

    def configuration(self) -> dict:
        return {
            "equations": self.equation_config,
            "gravity_m_per_s2": self.gravity_m_per_s2,
            "top_pressure_pa": self.top_pressure_pa,
            "vector_basis_matches_spatial_coordinates": (
                self.vector_basis_matches_spatial_coordinates
            ),
            "normalization": self.normalization.configuration(),
        }


__all__ = [
    "BOUNDARY_EQUATIONS",
    "EQUATION_NAMES",
    "LTEPhysicsModule",
    "LTEPhysicsNormalization",
    "LTEPhysicsResult",
    "LTEPhysicsState",
    "MHD_EQUATIONS",
    "VOLUME_EQUATIONS",
]
