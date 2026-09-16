"""Fixed differential constraints for the spherical LTE inversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    "magnetic_force_free",
    "magnetic_current_free",
    "radial_magnetic_energy_gradient",
    "radial_magnetic_field",
    "induction",
    "continuity",
    "adiabatic_pressure",
)
UPPER_VOLUME_EQUATIONS = (
    "upper_domain_microturbulence_prior",
    "upper_domain_temperature_prior",
    "coronal_energy",
)
UPPER_BOUNDARY_EQUATIONS = (
    "upper_boundary_current_free",
    "upper_boundary_no_inflow",
    "upper_boundary_open_velocity",
    "upper_boundary_tangential_magnetic_neumann",
    "upper_boundary_gas_pressure_prior",
)
SIDE_BOUNDARY_EQUATIONS = (
    "side_boundary_current_free",
    "side_boundary_no_inflow",
    "side_boundary_open_velocity",
    "side_boundary_tangential_magnetic_neumann",
)
BOUNDARY_EQUATIONS = (*UPPER_BOUNDARY_EQUATIONS, *SIDE_BOUNDARY_EQUATIONS)
EQUATION_NAMES = (*VOLUME_EQUATIONS, *UPPER_VOLUME_EQUATIONS, *BOUNDARY_EQUATIONS)

GAUSS_TO_TESLA = 1.0e-4
VACUUM_PERMEABILITY_H_PER_M = 4.0 * math.pi * 1.0e-7


@dataclass(frozen=True)
class PhysicsNormalization:
    """Characteristic scales used to construct dimensionless residuals.

    The public configuration retains physical units so a run can state its
    characteristic length/time/field scales unambiguously.  For a model with
    the normalized atmosphere contract, these values are converted to fixed
    dimensionless coefficients before entering the residual graph.  Local
    height scales are detached, so normalization cannot become an optimizer
    shortcut through the current field magnitude or thermodynamic state.
    ``magnetic_field_scale_gauss`` fixes the magnetic residual denominator;
    otherwise the detached height-mean model-unit field magnitude is used.
    ``force_balance_pressure_scale_pa`` similarly fixes the common pressure
    denominator for HSE, MHS, and momentum; otherwise a detached local
    pressure-equivalent is used.  The transported field uses the common rate
    ``1 / T + V0 / L``.  ``detach_normalization_scale`` controls only the
    *adaptive* (height-mean or point-local) scale path used when no fixed
    scale is configured; when set to ``False`` that adaptive scale keeps a
    gradient, which removes the amplitude-shrink incentive documented above
    but introduces the opposite incentive (inflating the local field/pressure
    magnitude trivially shrinks the normalized residual). Prefer a fixed
    ``*_scale_*`` value where a physically motivated one is known; use
    ``detach_normalization_scale=False`` only as a deliberate experiment and
    monitor field/pressure amplitude for runaway growth.
    """

    length_m: float = 1.0e6
    time_s: float = 3_600.0
    magnetic_field_floor_gauss: float = 1.0
    velocity_scale_m_per_s: float = 1_000.0
    # Optional fixed scale for div B and curl-based magnetic residuals.
    magnetic_field_scale_gauss: float | None = None
    # Constant pressure-equivalent denominator for HSE, MHS, and momentum.
    force_balance_pressure_scale_pa: float | None = None
    # Smooth detached floor for local pressure-equivalent normalization.
    force_balance_pressure_floor_pa: float = 0.0
    # Whether the adaptive (non-fixed) normalization scale is detached from
    # autograd. True reproduces the original stop-gradient behavior.
    detach_normalization_scale: bool = True

    def __post_init__(self):
        if self.force_balance_pressure_scale_pa is not None:
            value = self.force_balance_pressure_scale_pa
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError("force_balance_pressure_scale_pa must be numeric.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("force_balance_pressure_scale_pa must be finite and positive.")
        if self.magnetic_field_scale_gauss is not None:
            value = self.magnetic_field_scale_gauss
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError("magnetic_field_scale_gauss must be numeric.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("magnetic_field_scale_gauss must be finite and positive.")
        if not isinstance(self.force_balance_pressure_floor_pa, Real) or isinstance(
            self.force_balance_pressure_floor_pa, bool
        ):
            raise TypeError("force_balance_pressure_floor_pa must be numeric.")
        if not math.isfinite(self.force_balance_pressure_floor_pa) or self.force_balance_pressure_floor_pa < 0:
            raise ValueError(
                "force_balance_pressure_floor_pa must be finite and non-negative."
            )
        if type(self.detach_normalization_scale) is not bool:
            raise TypeError("detach_normalization_scale must be boolean.")
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
            "magnetic_field_scale_gauss",
            "force_balance_pressure_scale_pa",
            "force_balance_pressure_floor_pa",
            "detach_normalization_scale",
        }
        if unknown:
            raise TypeError(f"Unknown physics normalization options: {sorted(unknown)}")
        return cls(**values)

    def configuration(self) -> dict[str, float | None]:
        configuration = {
            "length_m": self.length_m,
            "time_s": self.time_s,
            "magnetic_field_floor_gauss": self.magnetic_field_floor_gauss,
            "velocity_scale_m_per_s": self.velocity_scale_m_per_s,
            "magnetic_field_scale_gauss": self.magnetic_field_scale_gauss,
            "force_balance_pressure_scale_pa": self.force_balance_pressure_scale_pa,
        }
        if self.force_balance_pressure_floor_pa:
            configuration["force_balance_pressure_floor_pa"] = (
                self.force_balance_pressure_floor_pa
            )
        if not self.detach_normalization_scale:
            configuration["detach_normalization_scale"] = self.detach_normalization_scale
        return configuration

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
    # Normalized model-unit view used by the objective.  The physical fields
    # above are retained as a compatibility/diagnostic view for observation
    # and regression code; new physics code should use these fields and
    # ``model_derivative``.
    position_model: torch.Tensor | None = None
    gas_pressure_model: torch.Tensor | None = None
    mass_density_model: torch.Tensor | None = None
    velocity_field_model: torch.Tensor | None = None
    magnetic_field_model: torch.Tensor | None = None
    model_primitive_derivatives: "PrimitiveDerivatives | None" = None
    model_length_scale_m: float | None = None
    model_magnetic_scale_gauss: float | None = None

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

    def model_derivative(self, name: str) -> torch.Tensor:
        """Return d/dx_hat of a primitive in normalized model units."""
        derivatives = self.model_primitive_derivatives
        if derivatives is None:
            raise RuntimeError(
                "Normalized derivatives are unavailable for this physical-only state."
            )
        if name == "velocity_cross_magnetic":
            velocity_jacobian = self.model_derivative("velocity").transpose(1, 2)
            magnetic_jacobian = self.model_derivative("magnetic").transpose(1, 2)
            result = torch.linalg.cross(
                velocity_jacobian,
                self.magnetic_field_model[:, None, :],
                dim=-1,
            ) + torch.linalg.cross(
                self.velocity_field_model[:, None, :],
                magnetic_jacobian,
                dim=-1,
            )
            return result.transpose(1, 2)
        if name == "mass_flux":
            density_jacobian = self.model_derivative("density")[:, 0, :]
            return (
                self.mass_density_model[:, None, None]
                * self.model_derivative("velocity")
                + self.velocity_field_model[:, :, None]
                * density_jacobian[:, None, :]
            )
        return derivatives.spatial(name)

    def time_derivative(self, name: str) -> torch.Tensor:
        return self.primitive_derivatives.temporal(name)

    def model_time_derivative(self, name: str) -> torch.Tensor:
        """Return d/dt_hat of a primitive in normalized model units."""
        derivatives = self.model_primitive_derivatives
        if derivatives is None:
            raise RuntimeError(
                "Normalized derivatives are unavailable for this physical-only state."
            )
        return derivatives.temporal(name)

    def model_field(self, name: str) -> torch.Tensor:
        """Return a normalized primitive field from the shared state."""
        fields = {
            "gas_pressure": self.gas_pressure_model,
            "mass_density": self.mass_density_model,
            "velocity_field": self.velocity_field_model,
            "magnetic_field": self.magnetic_field_model,
        }
        value = fields.get(name)
        if value is None:
            raise RuntimeError(
                "Normalized fields are unavailable for this physical-only state."
            )
        return value

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
        detach_scale: bool = True,
    ) -> torch.Tensor:
        """Normalize a residual by its height-mean scale.

        ``normalization_quantity`` is the positive characteristic field in the
        same model-unit convention as the residual. Keeping one scale per
        sampled height prevents dense low-atmosphere values from suppressing
        the upper-atmosphere equations. By default the scale is detached so
        learned fields cannot alter a residual through its denominator;
        passing ``detach_scale=False`` keeps that gradient instead, which
        removes the amplitude-shrink incentive but introduces the opposite
        one (inflating the local scale trivially shrinks the normalized
        residual) -- see ``PhysicsNormalization.detach_normalization_scale``.
        ``scale_floor`` supplies a smooth floor before the optional detach.
        """

        grouped_residual = self.grouped(residual)
        scale = self.height_mean(normalization_quantity)
        if scale_floor > 0.0:
            floor = scale.new_tensor(float(scale_floor))
            scale = torch.sqrt(scale.square() + floor.square())
        scale = scale.clamp_min(torch.finfo(normalization_quantity.dtype).tiny)
        if detach_scale:
            scale = scale.detach()
        broadcast_shape = (scale.shape[0],) + (1,) * (grouped_residual.ndim - 1)
        return grouped_residual * (
            float(dimensional_factor) / scale.reshape(broadcast_shape)
        )

    def height_group_mse(
        self,
        grouped_residual: torch.Tensor,
        *,
        robust_delta: float = 0.0,
    ) -> torch.Tensor:
        """Average residual penalties equally over heights and components."""

        if tuple(grouped_residual.shape[:2]) != self.height_group_shape:
            raise ValueError("Normalized residual does not retain its height groups.")
        return _robust_penalty(grouped_residual, robust_delta).mean()

    def normalize_points(
        self,
        residual: torch.Tensor,
        normalization_quantity: torch.Tensor,
        dimensional_factor: float,
        *,
        scale_floor: float = 0.0,
        detach_scale: bool = True,
    ) -> torch.Tensor:
        """Normalize a residual by an independent per-point local scale.

        ``normalization_quantity`` carries one positive value per sampled
        point (no height/angular pooling): each point's residual is judged
        against its own local characteristic magnitude rather than a shared
        regional or height-mean value. This is the natural scale for a
        strictly local constraint (a force balance must hold pointwise), and
        it does not depend on an external reference or on where a "typical"
        photosphere/height boundary is assumed to be. ``residual`` may be
        scalar-per-point or carry trailing component dimensions (e.g. a
        3-vector force residual); ``normalization_quantity`` always has shape
        ``residual.shape[:1]`` matching the leading point dimension.

        Per-point granularity makes the optimizer-shortcut tradeoff described
        in ``normalize_height_groups`` *more* exploitable when not detached
        (a single point's own field/pressure fully determines its own
        denominator, with no pooling across neighbors to dilute the
        incentive), so prefer ``detach_scale=True`` here more strongly than
        for a pooled scale.
        """

        scale = normalization_quantity
        if scale_floor > 0.0:
            floor = scale.new_tensor(float(scale_floor))
            scale = torch.sqrt(scale.square() + floor.square())
        scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
        if detach_scale:
            scale = scale.detach()
        broadcast_shape = (scale.shape[0],) + (1,) * (residual.ndim - 1)
        return residual * (float(dimensional_factor) / scale.reshape(broadcast_shape))


@dataclass(frozen=True)
class PrimitiveDerivatives:
    """Spatial/temporal derivatives of base atmospheric outputs."""

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
    equation_diagnostics: dict | None = None


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


def _robust_penalty(residual: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """Return an MSE-equivalent pseudo-Huber penalty for normalized residuals.

    ``delta == 0`` deliberately retains the historical exact MSE path.  For a
    positive delta the smooth quadratic core has the same local curvature and
    value as MSE, while the linear tail prevents a few unresolved collocation
    points from dominating the update.
    """

    if delta <= 0.0:
        return residual.square()
    square = residual.square()
    # Algebraically this is 2*d^2*(sqrt(1 + (r/d)^2) - 1), but the rationalized
    # form avoids cancellation when the residual is in the quadratic core.
    return 2.0 * square / (torch.sqrt(1.0 + square / float(delta) ** 2) + 1.0)


def _mean_square(residual: torch.Tensor, *, robust_delta: float = 0.0) -> torch.Tensor:
    """Return an MSE or robust mean penalty for independent samples."""

    return _robust_penalty(residual, robust_delta).mean()


def _tangential_normal_derivative(
    jacobian: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    """Project a vector field's normal derivative into the boundary tangent."""

    normal_derivative = torch.einsum("nij,nj->ni", jacobian, normal)
    return (
        normal_derivative
        - (normal_derivative * normal).sum(dim=-1, keepdim=True) * normal
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
        robust_loss_delta: float = 0.0,
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
        self.coronal_energy = None
        for name in EQUATION_NAMES:
            raw = equations.get(name, {})
            if not isinstance(raw, Mapping):
                raise TypeError(f"Physics equation {name!r} must be a mapping.")
            config = dict(raw)
            if name == "coronal_energy":
                from prom3theus.rt.coronal_energy import (
                    CoronalEnergy,
                    CoronalEnergyOptions,
                )

                energy_options = CoronalEnergyOptions(
                    **{
                        key: value
                        for key, value in config.items()
                        if key not in {"enabled", "weight"}
                    }
                )
                config = {
                    key: value
                    for key, value in config.items()
                    if key in {"enabled", "weight"}
                }
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
            if name == "coronal_energy":
                self.equation_config[name].update(asdict(energy_options))
                if self.is_active(name):
                    self.coronal_energy = CoronalEnergy(energy_options)

        if self.is_active("coronal_energy") and self.is_active("adiabatic_pressure"):
            raise ValueError(
                "coronal_energy and global adiabatic_pressure are mutually exclusive"
            )

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
                "magnetic_force_free",
                "magnetic_current_free",
                "radial_magnetic_energy_gradient",
                "induction",
                "continuity",
                "adiabatic_pressure",
                "coronal_energy",
                "upper_boundary_current_free",
                "side_boundary_current_free",
                "upper_boundary_no_inflow",
                "side_boundary_no_inflow",
                "upper_boundary_open_velocity",
                "upper_boundary_tangential_magnetic_neumann",
                "side_boundary_open_velocity",
                "side_boundary_tangential_magnetic_neumann",
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
        if not isinstance(robust_loss_delta, Real) or isinstance(robust_loss_delta, bool):
            raise TypeError("robust_loss_delta must be numeric.")
        self.robust_loss_delta = float(robust_loss_delta)
        if not math.isfinite(self.robust_loss_delta) or self.robust_loss_delta < 0.0:
            raise ValueError("robust_loss_delta must be finite and non-negative.")

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
        has_model_units = all(
            hasattr(atmosphere_model, name)
            for name in (
                "temperature_scale_k",
                "gas_pressure_scale_pa",
                "density_scale_kg_m3",
                "solar_radius_model",
                "height_input_scale_m",
                "velocity_scale_m_per_s",
                "magnetic_scale_gauss",
                "normalize_atmosphere_fields",
                "denormalize_atmosphere_fields",
                "mass_density_normalized",
                "evaluate_position_rsun_normalized",
            )
        )
        if has_model_units:
            normalized_fields = atmosphere_model.evaluate_position_rsun_normalized(
                position_rsun, time_hours=time_hours
            )
            # Physical fields are retained only as compatibility aliases for
            # diagnostics/tests.  The derivative graph below consumes
            # normalized_fields directly.
            fields = atmosphere_model.denormalize_atmosphere_fields(
                normalized_fields
            )
        else:
            fields = atmosphere_model.evaluate_position_rsun(
                position_rsun, time_hours=time_hours
            )
            normalized_fields = None
        temperature = fields["temperature"]
        pressure = fields["gas_pressure"]
        magnetic = fields["magnetic_field"]
        velocity = fields["velocity_field"]
        uses_vector_potential = (
            getattr(atmosphere_model, "magnetic_representation", "direct")
            == "vector_potential"
        )
        radial_unit = position_rsun / torch.linalg.vector_norm(
            position_rsun, dim=-1, keepdim=True
        )
        density = None
        density_model = None
        if {
            "hydrostatic_equilibrium",
            "magnetohydrostatic_equilibrium",
            "momentum",
            "continuity",
        } & active_equations:
            if has_model_units:
                density_model = atmosphere_model.mass_density_normalized(
                    normalized_fields["temperature"],
                    normalized_fields["gas_pressure"],
                )
                density = density_model * atmosphere_model.density_scale_kg_m3
            else:
                density = atmosphere_model.thermodynamic_eos.mass_density(
                    temperature, pressure
                )
        # Select the representation that is differentiated.  For a real
        # atmosphere model this is exclusively the dimensionless contract;
        # the physical variables above are only compatibility/diagnostic
        # aliases and never enter the normalized objective graph.
        primitive_pressure = (
            normalized_fields["gas_pressure"] if has_model_units else pressure
        )
        primitive_magnetic = (
            normalized_fields["magnetic_field"] if has_model_units else magnetic
        )
        primitive_velocity = (
            normalized_fields["velocity_field"] if has_model_units else velocity
        )
        primitive_density = density_model if has_model_units else density
        primitives = {}
        if {
            "hydrostatic_equilibrium",
            "magnetohydrostatic_equilibrium",
            "momentum",
        } & active_equations:
            primitives["pressure"] = primitive_pressure
        if "adiabatic_pressure" in active_equations:
            primitives["log_pressure"] = torch.log(primitive_pressure)
        magnetic_derivative_equations = {
            "magnetohydrostatic_equilibrium",
            "momentum",
            "magnetic_divergence",
            "magnetic_force_free",
            "magnetic_current_free",
            "radial_magnetic_energy_gradient",
            "induction",
            "upper_boundary_tangential_magnetic_neumann",
            "side_boundary_tangential_magnetic_neumann",
            "upper_boundary_current_free",
            "side_boundary_current_free",
        } & active_equations
        if uses_vector_potential:
            # div B is identically zero when B is constructed as curl(A); no
            # magnetic Jacobian is needed just to evaluate that equation.
            magnetic_derivative_equations.discard("magnetic_divergence")
        if magnetic_derivative_equations:
            primitives["magnetic"] = primitive_magnetic
        if {
            "momentum",
            "induction",
            "continuity",
            "adiabatic_pressure",
            "upper_boundary_open_velocity",
            "side_boundary_open_velocity",
        } & active_equations:
            primitives["velocity"] = primitive_velocity
        if primitive_density is not None and "continuity" in active_equations:
            primitives["log_density"] = torch.log(primitive_density)
        if (
            primitive_density is not None
            and {
                "hydrostatic_equilibrium",
                "magnetohydrostatic_equilibrium",
                "momentum",
            }
            & active_equations
        ):
            primitives["density"] = primitive_density
        raw_derivatives = _pointwise_output_jacobian(
            primitives, coordinates, create_graph=create_graph
        )
        model_primitive_derivatives = None
        if has_model_units:
            model_jacobian_parts = []
            physical_jacobian_parts = []
            physical_field_scales = {
                "pressure": atmosphere_model.gas_pressure_scale_pa,
                "log_pressure": 1.0,
                "magnetic": atmosphere_model.magnetic_scale_gauss,
                "velocity": atmosphere_model.velocity_scale_m_per_s,
                "log_density": 1.0,
                "density": atmosphere_model.density_scale_kg_m3,
            }
            spatial_factor = (
                atmosphere_model.height_input_scale_m
                / atmosphere_model.solar_radius_m
            )
            temporal_factor = self.normalization.time_s / 3_600.0
            for name in primitives:
                selected = raw_derivatives.jacobian[:, raw_derivatives.slices[name], :]
                model_jacobian_parts.append(
                    torch.cat(
                        (
                            selected[..., :3] * spatial_factor,
                            selected[..., 3:] * temporal_factor,
                        ),
                        dim=-1,
                    )
                )
                physical_scale = selected.new_tensor(physical_field_scales[name])
                physical_jacobian_parts.append(
                    torch.cat(
                        (
                            selected[..., :3] * physical_scale
                            / atmosphere_model.solar_radius_m,
                            selected[..., 3:] * physical_scale / 3_600.0,
                        ),
                        dim=-1,
                    )
                )
            model_primitive_derivatives = PrimitiveDerivatives(
                torch.cat(model_jacobian_parts, dim=1)
                if model_jacobian_parts
                else raw_derivatives.jacobian.new_empty(
                    raw_derivatives.jacobian.shape
                ),
                raw_derivatives.slices,
            )
            physical_jacobian = (
                torch.cat(physical_jacobian_parts, dim=1)
                if physical_jacobian_parts
                else raw_derivatives.jacobian.new_empty(
                    raw_derivatives.jacobian.shape
                )
            )
        else:
            physical_jacobian = torch.cat(
                (
                    raw_derivatives.jacobian[..., :3]
                    / atmosphere_model.solar_radius_m,
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
            position_model=(
                position_rsun * atmosphere_model.solar_radius_model
                if has_model_units
                else None
            ),
            gas_pressure_model=(
                normalized_fields["gas_pressure"] if has_model_units else None
            ),
            mass_density_model=density_model,
            velocity_field_model=(
                normalized_fields["velocity_field"] if has_model_units else None
            ),
            magnetic_field_model=(
                normalized_fields["magnetic_field"] if has_model_units else None
            ),
            model_primitive_derivatives=model_primitive_derivatives,
            model_length_scale_m=(
                float(atmosphere_model.height_input_scale_m)
                if has_model_units
                else None
            ),
            model_magnetic_scale_gauss=(
                float(atmosphere_model.magnetic_scale_gauss)
                if has_model_units
                else None
            ),
        )

    def _uses_model_units(self, atmosphere_model) -> bool:
        return all(
            hasattr(atmosphere_model, name)
            for name in (
                "temperature_scale_k",
                "gas_pressure_scale_pa",
                "density_scale_kg_m3",
                "solar_radius_model",
                "height_input_scale_m",
                "velocity_scale_m_per_s",
                "magnetic_scale_gauss",
                "evaluate_position_rsun_normalized",
                "mass_density_normalized",
                "normalize_atmosphere_fields",
                "denormalize_atmosphere_fields",
            )
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
        magnetic_normalization_gauss: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> PhysicsResult:
        """Evaluate volume equations in normalized model units when available."""
        if self._uses_model_units(atmosphere_model):
            return self._volume_normalized(
                atmosphere_model,
                position_m,
                time_hours,
                height_group_shape=height_group_shape,
                create_graph=create_graph,
                return_state=return_state,
                magnetic_normalization_gauss=magnetic_normalization_gauss,
                return_diagnostics=return_diagnostics,
            )
        return self._volume_physical(
            atmosphere_model,
            position_m,
            time_hours,
            height_group_shape=height_group_shape,
            create_graph=create_graph,
            return_state=return_state,
            magnetic_normalization_gauss=magnetic_normalization_gauss,
            return_diagnostics=return_diagnostics,
        )

    def _volume_normalized(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        *,
        height_group_shape: tuple[int, int],
        create_graph: bool = True,
        return_state: bool = False,
        magnetic_normalization_gauss: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> PhysicsResult:
        """Evaluate the complete volume objective in dimensionless variables.

        The physical atmosphere API is sampled once at the adapter boundary;
        all derivatives and force terms below use x_hat, t_hat, rho_hat, p_hat,
        v_hat, and B_hat.  Coefficients such as magnetic pressure and gravity
        are fixed dimensionless numbers determined by the selected model-unit
        scales.
        """
        weights = self.loss_weights
        active = {
            name
            for name in VOLUME_EQUATIONS
            if self.enabled[name] and weights[name] != 0.0
        }
        uses_vector_potential = (
            getattr(atmosphere_model, "magnetic_representation", "direct")
            == "vector_potential"
        )
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
        diagnostics = {}

        # Global model-unit coefficients.  The numerical SI constants appear
        # only once here while constructing fixed coefficients; tensors in the
        # optimization graph are all dimensionless thereafter.
        length_scale_m = float(atmosphere_model.height_input_scale_m)
        time_scale_s = float(self.normalization.time_s)
        pressure_scale_pa = float(atmosphere_model.gas_pressure_scale_pa)
        density_scale = float(atmosphere_model.density_scale_kg_m3)
        velocity_scale = float(atmosphere_model.velocity_scale_m_per_s)
        magnetic_scale = float(atmosphere_model.magnetic_scale_gauss)
        # Magnetic-only equations do not require gravity.  Keep a neutral
        # coefficient for that case; force-balance equations are validated to
        # have a configured positive gravity in the constructor.
        gravity_scale = float(self.gravity_m_per_s2 or 1.0)
        force_length_factor = self.normalization.length_m / length_scale_m
        gravity_coefficient = (
            density_scale * gravity_scale * length_scale_m / pressure_scale_pa
        )
        magnetic_pressure_coefficient = (
            (magnetic_scale * GAUSS_TO_TESLA) ** 2
            / VACUUM_PERMEABILITY_H_PER_M
            / pressure_scale_pa
        )
        gravity_hat = (
            atmosphere_model.solar_radius_model.to(state.position_model)
            / torch.linalg.vector_norm(state.position_model, dim=-1)
        ).square()
        density_for_gravity = state.mass_density_model
        if density_for_gravity is None:
            density_for_gravity = state.position_m.new_zeros(state.position_m.shape[0])
        gravity_force = (
            gravity_coefficient
            * density_for_gravity
            * gravity_hat
        )

        def record(
            name: str,
            terms: Mapping[str, torch.Tensor],
            normalized_terms: Mapping[str, torch.Tensor] | None = None,
        ) -> None:
            if not return_diagnostics:
                return
            normalized_terms = normalized_terms or terms
            diagnostics[name] = {
                "units": "dimensionless model force/rate",
                "robust_loss_delta": self.robust_loss_delta,
                "terms": {key: value.detach() for key, value in terms.items()},
                "normalized_terms": {
                    key: value.detach() for key, value in normalized_terms.items()
                },
                "residual": sum(terms.values()).detach(),
                "normalized_residual": sum(normalized_terms.values()).detach(),
                "weight": weights[name],
            }

        def normalize_force_balance(residual, pressure_equivalent):
            fixed_scale = self.normalization.force_balance_pressure_scale_pa
            if fixed_scale is not None:
                fixed_hat = fixed_scale / pressure_scale_pa
                return state.grouped(residual * (force_length_factor / fixed_hat))
            return state.normalize_height_groups(
                residual,
                pressure_equivalent,
                force_length_factor,
                detach_scale=self.normalization.detach_normalization_scale,
                scale_floor=(
                    self.normalization.force_balance_pressure_floor_pa
                    / pressure_scale_pa
                ),
            )

        def normalize_force_term(term, pressure_equivalent):
            """Normalize one force term while retaining its flat shape for plots."""
            normalized = normalize_force_balance(term, pressure_equivalent)
            return normalized.reshape_as(term)

        reference = next(atmosphere_model.parameters())
        pressure = state.gas_pressure_model
        if pressure is None:
            pressure = reference.new_zeros(state.position_m.shape[0])
        density = state.mass_density_model
        if density is None:
            density = reference.new_zeros(state.position_m.shape[0])
        magnetic = state.magnetic_field_model
        if magnetic is None:
            magnetic = reference.new_zeros((state.position_m.shape[0], 3))
        velocity = state.velocity_field_model
        if velocity is None:
            velocity = reference.new_zeros((state.position_m.shape[0], 3))

        if "hydrostatic_equilibrium" in active:
            grad_pressure = state.model_derivative("pressure")[:, 0]
            radial_pressure_gradient = (grad_pressure * state.radial_unit).sum(dim=-1)
            pressure_equivalent = pressure + force_length_factor * gravity_force
            residual = normalize_force_balance(
                radial_pressure_gradient + gravity_force,
                pressure_equivalent,
            )
            losses["hydrostatic_equilibrium"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "hydrostatic_equilibrium",
                {"radial ∇p_hat": radial_pressure_gradient, "gravity": gravity_force},
                {
                    "radial ∇p_hat": normalize_force_term(
                        radial_pressure_gradient, pressure_equivalent
                    ),
                    "gravity": normalize_force_term(gravity_force, pressure_equivalent),
                },
            )

        curl_magnetic = None
        if {
            "magnetohydrostatic_equilibrium",
            "momentum",
            "magnetic_force_free",
            "magnetic_current_free",
            "induction",
            "radial_magnetic_energy_gradient",
        } & active:
            curl_magnetic = _curl(state.model_derivative("magnetic"))

        if "magnetohydrostatic_equilibrium" in active:
            grad_pressure = state.model_derivative("pressure")[:, 0]
            lorentz_force = magnetic_pressure_coefficient * torch.linalg.cross(
                curl_magnetic,
                magnetic,
                dim=-1,
            )
            pressure_equivalent = (
                pressure
                + force_length_factor * gravity_force
                + magnetic_pressure_coefficient * magnetic.square().sum(dim=-1)
            )
            gravity_vector = gravity_force[:, None] * state.radial_unit
            residual = normalize_force_balance(
                grad_pressure + gravity_vector - lorentz_force,
                pressure_equivalent,
            )
            losses["magnetohydrostatic_equilibrium"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "magnetohydrostatic_equilibrium",
                {"∇p_hat": grad_pressure, "gravity": gravity_vector, "−J×B": -lorentz_force},
                {
                    "∇p_hat": normalize_force_term(grad_pressure, pressure_equivalent),
                    "gravity": normalize_force_term(
                        gravity_vector, pressure_equivalent
                    ),
                    "−J×B": normalize_force_term(-lorentz_force, pressure_equivalent),
                },
            )

        inertial_force = None
        if "momentum" in active:
            grad_pressure = state.model_derivative("pressure")[:, 0]
            velocity_gradient = state.model_derivative("velocity")
            time_velocity = state.model_time_derivative("velocity")
            advective = torch.einsum("nij,nj->ni", velocity_gradient, velocity)
            omega_hat = velocity.new_tensor(
                [0.0, 0.0, CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S * time_scale_s]
            ).expand_as(velocity)
            coriolis = 2.0 * torch.linalg.cross(omega_hat, velocity, dim=-1)
            centrifugal = torch.linalg.cross(
                omega_hat,
                torch.linalg.cross(omega_hat, state.position_model, dim=-1),
                dim=-1,
            )
            time_coefficient = density_scale * velocity_scale * length_scale_m / (
                pressure_scale_pa * time_scale_s
            )
            advective_coefficient = density_scale * velocity_scale**2 / pressure_scale_pa
            centrifugal_coefficient = (
                density_scale
                * (CARRINGTON_ANGULAR_VELOCITY_RAD_PER_S**2)
                * length_scale_m**2
                / pressure_scale_pa
            )
            inertial_force = density[:, None] * (
                time_coefficient * time_velocity
                + advective_coefficient * advective
                + time_coefficient * coriolis
                + centrifugal_coefficient * centrifugal
            )
            lorentz_force = magnetic_pressure_coefficient * torch.linalg.cross(
                curl_magnetic,
                magnetic,
                dim=-1,
            )
            pressure_equivalent = (
                pressure
                + force_length_factor
                * (
                    gravity_force
                    + torch.linalg.vector_norm(inertial_force, dim=-1)
                )
                + magnetic_pressure_coefficient * magnetic.square().sum(dim=-1)
            )
            residual = normalize_force_balance(
                inertial_force
                + grad_pressure
                + gravity_force[:, None] * state.radial_unit
                - lorentz_force,
                pressure_equivalent,
            )
            losses["momentum"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "momentum",
                {
                    "inertial": inertial_force,
                    "∇p_hat": grad_pressure,
                    "gravity": gravity_force[:, None] * state.radial_unit,
                    "−J×B": -lorentz_force,
                },
                {
                    "inertial": normalize_force_term(inertial_force, pressure_equivalent),
                    "∇p_hat": normalize_force_term(grad_pressure, pressure_equivalent),
                    "gravity": normalize_force_term(
                        gravity_force[:, None] * state.radial_unit,
                        pressure_equivalent,
                    ),
                    "−J×B": normalize_force_term(-lorentz_force, pressure_equivalent),
                },
            )

        field_normalization = torch.linalg.vector_norm(magnetic, dim=-1)
        if magnetic_normalization_gauss is not None:
            field_normalization = (
                magnetic_normalization_gauss.detach().to(magnetic) / magnetic_scale
            )
            if field_normalization.shape != magnetic.shape[:1]:
                raise ValueError(
                    "Magnetic normalization must contain one value per volume point"
                )
        magnetic_floor = self.normalization.magnetic_field_floor_gauss / magnetic_scale

        def normalize_magnetic(residual, dimensional_factor):
            if self.normalization.magnetic_field_scale_gauss is not None:
                scale = self.normalization.magnetic_field_scale_gauss / magnetic_scale
                return state.grouped(residual * (dimensional_factor / scale))
            if magnetic_normalization_gauss is None:
                return state.normalize_height_groups(
                    residual,
                    field_normalization,
                    dimensional_factor,
                    detach_scale=self.normalization.detach_normalization_scale,
                    scale_floor=magnetic_floor,
                )
            scale = torch.sqrt(field_normalization.square() + magnetic_floor**2)
            shape = (len(scale),) + (1,) * (residual.ndim - 1)
            return state.grouped(residual * dimensional_factor / scale.reshape(shape))

        if "magnetic_divergence" in active:
            divergence = (
                magnetic[..., 0] * 0.0
                if uses_vector_potential
                else _divergence(state.model_derivative("magnetic"))
            )
            residual = normalize_magnetic(divergence, force_length_factor)
            losses["magnetic_divergence"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "magnetic_divergence",
                {"∇·B_hat": divergence},
                {"∇·B_hat": residual.reshape_as(divergence)},
            )

        if "magnetic_force_free" in active:
            residual = torch.linalg.cross(
                normalize_magnetic(curl_magnetic, force_length_factor),
                normalize_magnetic(magnetic, 1.0),
                dim=-1,
            )
            losses["magnetic_force_free"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "magnetic_force_free",
                {"curl(B_hat)×B_hat": residual.reshape(-1, 3)},
                {"normalized residual": residual.reshape(-1, 3)},
            )

        if "magnetic_current_free" in active:
            residual = normalize_magnetic(curl_magnetic, force_length_factor)
            losses["magnetic_current_free"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "magnetic_current_free",
                {"∇×B_hat": curl_magnetic},
                {"normalized residual": residual.reshape(-1, 3)},
            )

        if "radial_magnetic_field" in active:
            radial_component = (
                (magnetic * state.radial_unit).sum(dim=-1, keepdim=True)
                * state.radial_unit
            )
            tangential = magnetic - radial_component
            residual = normalize_magnetic(tangential, 1.0)
            losses["radial_magnetic_field"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "radial_magnetic_field",
                {"B_hat - (B_hat·r̂)r̂": tangential},
                {"normalized residual": residual.reshape(-1, 3)},
            )

        if "radial_magnetic_energy_gradient" in active:
            radial_magnetic_derivative = torch.einsum(
                "nij,nj->ni", state.model_derivative("magnetic"), state.radial_unit
            )
            radial_energy_gradient = 2.0 * (
                magnetic * radial_magnetic_derivative
            ).sum(dim=-1)
            magnetic_energy = (
                magnetic.square().sum(dim=-1)
                if magnetic_normalization_gauss is None
                else field_normalization.square()
            )
            mean_radial_energy_gradient = state.height_mean(radial_energy_gradient)
            energy_scale = state.height_mean(magnetic_energy)
            energy_floor = energy_scale.new_tensor(magnetic_floor**2)
            energy_scale = torch.sqrt(
                energy_scale.square() + energy_floor.square()
            ).detach()
            residual = (
                torch.relu(mean_radial_energy_gradient)
                * force_length_factor
                / energy_scale
            )
            losses["radial_magnetic_energy_gradient"] = _mean_square(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "radial_magnetic_energy_gradient",
                {"positive mean ∂r|B_hat|²": residual.reshape(-1)},
                {"normalized residual": residual.reshape(-1)},
            )

        if "induction" in active:
            transport_time_hat = self.normalization.transport_time_s / time_scale_s
            transport_advection_hat = (
                self.normalization.transport_time_s * velocity_scale / length_scale_m
            )
            induction = (
                transport_time_hat * state.model_time_derivative("magnetic")
                - transport_advection_hat
                * _curl(state.model_derivative("velocity_cross_magnetic"))
            )
            residual = normalize_magnetic(induction, 1.0)
            losses["induction"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "induction",
                {"normalized induction": induction},
                {"normalized residual": residual.reshape(-1, 3)},
            )

        transport_time_hat = self.normalization.transport_time_s / time_scale_s
        transport_advection_hat = (
            self.normalization.transport_time_s * velocity_scale / length_scale_m
        )
        if "continuity" in active:
            log_density_gradient = state.model_derivative("log_density")[:, 0]
            residual = (
                transport_time_hat * state.model_time_derivative("log_density")[:, 0]
                + transport_advection_hat
                * (
                    (log_density_gradient * velocity).sum(dim=-1)
                    + _divergence(state.model_derivative("velocity"))
                )
            )
            residual = state.grouped(residual)
            losses["continuity"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "continuity",
                {"normalized continuity": residual.reshape(-1)},
                {"normalized residual": residual.reshape(-1)},
            )

        if "adiabatic_pressure" in active:
            log_pressure_gradient = state.model_derivative("log_pressure")[:, 0]
            residual = (
                transport_time_hat * state.model_time_derivative("log_pressure")[:, 0]
                + transport_advection_hat
                * (
                    (log_pressure_gradient * velocity).sum(dim=-1)
                    + self.adiabatic_index
                    * _divergence(state.model_derivative("velocity"))
                )
            )
            residual = state.grouped(residual)
            losses["adiabatic_pressure"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record(
                "adiabatic_pressure",
                {"normalized adiabatic rate": residual.reshape(-1)},
                {"normalized residual": residual.reshape(-1)},
            )

        return PhysicsResult(
            losses,
            weights,
            state if return_state else None,
            diagnostics if return_diagnostics else None,
        )

    def _volume_physical(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        *,
        height_group_shape: tuple[int, int],
        create_graph: bool = True,
        return_state: bool = False,
        magnetic_normalization_gauss: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> PhysicsResult:
        weights = self.loss_weights
        active = {
            name
            for name in VOLUME_EQUATIONS
            if self.enabled[name] and weights[name] != 0.0
        }
        uses_vector_potential = (
            getattr(atmosphere_model, "magnetic_representation", "direct")
            == "vector_potential"
        )
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
        diagnostics = {}

        def record(name, terms, normalize, units):
            if not return_diagnostics:
                return
            raw = list(terms.values())
            normalized = [normalize(value) for value in raw]
            diagnostics[name] = {
                "units": units,
                "robust_loss_delta": self.robust_loss_delta,
                "terms": {key: value.detach() for key, value in terms.items()},
                "normalized_terms": {
                    key: value.reshape_as(raw_value).detach()
                    for (key, raw_value), value in zip(terms.items(), normalized)
                },
                "residual": sum(raw).detach(),
                "normalized_residual": sum(normalized).reshape_as(raw[0]).detach(),
                "weight": weights[name],
            }
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

        def normalize_force_balance(residual, pressure_equivalent):
            fixed_scale = self.normalization.force_balance_pressure_scale_pa
            if fixed_scale is not None:
                # Force density [Pa/m] times L/P0; the same denominator
                # multiplies every term, preserving the physical balance.
                return state.grouped(residual * (self.normalization.length_m / fixed_scale))
            return state.normalize_height_groups(
                residual,
                pressure_equivalent,
                self.normalization.length_m,
                detach_scale=self.normalization.detach_normalization_scale,
                scale_floor=self.normalization.force_balance_pressure_floor_pa,
            )

        if "hydrostatic_equilibrium" in active:
            grad_pressure = state.derivative("pressure")[:, 0]
            radial_pressure_gradient = (grad_pressure * state.radial_unit).sum(dim=-1)
            pressure_equivalent = (
                state.gas_pressure
                + state.mass_density * gravity_m_per_s2 * self.normalization.length_m
            )
            residual = normalize_force_balance(
                radial_pressure_gradient + state.mass_density * gravity_m_per_s2,
                pressure_equivalent,
            )
            losses["hydrostatic_equilibrium"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("hydrostatic_equilibrium", {
                "radial ∇P": radial_pressure_gradient,
                "ρg": state.mass_density * gravity_m_per_s2,
            }, lambda value: normalize_force_balance(value, pressure_equivalent), "Pa/m")
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
            residual = normalize_force_balance(
                grad_pressure
                + state.mass_density[:, None]
                * gravity_m_per_s2[:, None]
                * state.radial_unit
                - lorentz_force,
                pressure_equivalent,
            )
            losses["magnetohydrostatic_equilibrium"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("magnetohydrostatic_equilibrium", {
                "∇P": grad_pressure,
                "ρg": (state.mass_density * gravity_m_per_s2)[:, None] * state.radial_unit,
                "−J×B": -lorentz_force,
            }, lambda value: normalize_force_balance(value, pressure_equivalent), "Pa/m")
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
            residual = normalize_force_balance(
                state.mass_density[:, None] * rotating_frame_acceleration
                + grad_pressure
                + state.mass_density[:, None]
                * gravity_m_per_s2[:, None]
                * state.radial_unit
                - lorentz_force,
                pressure_equivalent,
            )
            losses["momentum"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("momentum", {
                "ρ ∂v/∂t": state.mass_density[:, None] * state.time_derivative("velocity"),
                "ρ(v·∇)v": state.mass_density[:, None] * torch.einsum("nij,nj->ni", state.derivative("velocity"), velocity),
                "Coriolis": state.mass_density[:, None] * coriolis_acceleration,
                "centrifugal": state.mass_density[:, None] * centrifugal_acceleration,
                "∇P": grad_pressure,
                "ρg": (state.mass_density * gravity_m_per_s2)[:, None] * state.radial_unit,
                "−J×B": -lorentz_force,
            }, lambda value: normalize_force_balance(value, pressure_equivalent), "Pa/m")
        field_normalization = torch.linalg.vector_norm(
            state.magnetic_field_gauss, dim=-1
        )
        if magnetic_normalization_gauss is not None:
            field_normalization = magnetic_normalization_gauss.detach().to(state.magnetic_field_gauss)
            if field_normalization.shape != state.magnetic_field_gauss.shape[:1]:
                raise ValueError("Magnetic normalization must contain one value per volume point")
        def normalize_magnetic(residual, dimensional_factor):
            if self.normalization.magnetic_field_scale_gauss is not None:
                return state.grouped(
                    residual * (dimensional_factor / self.normalization.magnetic_field_scale_gauss)
                )
            if magnetic_normalization_gauss is None:
                return state.normalize_height_groups(
                    residual, field_normalization, dimensional_factor,
                    detach_scale=self.normalization.detach_normalization_scale,
                    scale_floor=self.normalization.magnetic_field_floor_gauss,
                )
            scale = (field_normalization.square() + self.normalization.magnetic_field_floor_gauss**2).sqrt()
            shape = (len(scale),) + (1,) * (residual.ndim - 1)
            return state.grouped(residual * dimensional_factor / scale.reshape(shape))

        if "magnetic_divergence" in active:
            divergence = (
                state.magnetic_field_gauss[..., 0] * 0.0
                if uses_vector_potential
                else _divergence(state.derivative("magnetic"))
            )
            residual = normalize_magnetic(
                divergence,
                self.normalization.length_m,
            )
            losses["magnetic_divergence"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("magnetic_divergence", {"∇·B": divergence},
                   lambda value: normalize_magnetic(value, self.normalization.length_m), "G/m")
        if "magnetic_force_free" in active:
            # L (curl B x B) / B_scale^2, using the same fixed, height,
            # or cached normalization as div B.
            residual = torch.linalg.cross(
                normalize_magnetic(
                    _curl(state.derivative("magnetic")),
                    self.normalization.length_m,
                ),
                normalize_magnetic(state.magnetic_field_gauss, 1.0),
                dim=-1,
            )
            losses["magnetic_force_free"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            if return_diagnostics:
                force = torch.linalg.cross(_curl(state.derivative("magnetic")), state.magnetic_field_gauss, dim=-1)
                conversion = GAUSS_TO_TESLA**2 / VACUUM_PERMEABILITY_H_PER_M
                scale = normalize_magnetic(torch.ones_like(state.magnetic_field_gauss), 1.0)
                record("magnetic_force_free", {"J×B": force * conversion},
                       lambda value: state.grouped(value) * scale.square() * self.normalization.length_m / conversion, "Pa/m")
        if "magnetic_current_free" in active:
            residual = normalize_magnetic(
                _curl(state.derivative("magnetic")),
                self.normalization.length_m,
            )
            # A quadratic penalty makes concentrating a field change into a
            # thinner current sheet more expensive; a linear tail does not.
            losses["magnetic_current_free"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("magnetic_current_free", {"∇×B": _curl(state.derivative("magnetic"))},
                   lambda value: normalize_magnetic(value, self.normalization.length_m), "G/m")
        if "radial_magnetic_field" in active:
            radial_component = (
                (state.magnetic_field_gauss * state.radial_unit).sum(
                    dim=-1, keepdim=True
                )
                * state.radial_unit
            )
            tangential = state.magnetic_field_gauss - radial_component
            residual = normalize_magnetic(tangential, 1.0)
            losses["radial_magnetic_field"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("radial_magnetic_field", {"B - (B·r̂)r̂": tangential},
                   lambda value: normalize_magnetic(value, 1.0), "G")
        if "radial_magnetic_energy_gradient" in active:
            radial_magnetic_derivative = torch.einsum(
                "nij,nj->ni",
                state.derivative("magnetic"),
                state.radial_unit,
            )
            radial_energy_gradient = 2.0 * (
                state.magnetic_field_gauss * radial_magnetic_derivative
            ).sum(dim=-1)
            magnetic_energy = (
                state.magnetic_field_gauss.square().sum(dim=-1)
                if magnetic_normalization_gauss is None else field_normalization.square()
            )
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
            losses["radial_magnetic_energy_gradient"] = _mean_square(
                residual, robust_delta=self.robust_loss_delta
            )
            if return_diagnostics:
                # This objective penalizes the positive height-mean derivative,
                # not a pointwise derivative. Broadcast that mean along latitude.
                value = torch.relu(mean_radial_energy_gradient)[:, None].expand(state.height_group_shape).reshape(-1)
                record("radial_magnetic_energy_gradient", {"positive mean ∂r|B|²": value},
                       lambda x: state.grouped(x) * self.normalization.length_m / energy_scale[:, None], "G²/m")
        if "induction" in active:
            residual = normalize_magnetic(
                state.time_derivative("magnetic")
                - _curl(state.derivative("velocity_cross_magnetic")),
                self.normalization.transport_time_s,
            )
            losses["induction"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("induction", {"∂B/∂t": state.time_derivative("magnetic"),
                                 "−∇×(v×B)": -_curl(state.derivative("velocity_cross_magnetic"))},
                   lambda value: normalize_magnetic(value, self.normalization.transport_time_s), "G/s")
        if "continuity" in active:
            log_density_gradient = state.derivative("log_density")[:, 0]
            residual = (
                state.time_derivative("log_density")[:, 0]
                + (log_density_gradient * state.velocity_field_m_per_s).sum(dim=-1)
                + _divergence(state.derivative("velocity"))
            )
            residual = state.grouped(residual * self.normalization.transport_time_s)
            losses["continuity"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("continuity", {"∂lnρ/∂t": state.time_derivative("log_density")[:, 0],
                                  "v·∇lnρ": (log_density_gradient * state.velocity_field_m_per_s).sum(-1),
                                  "∇·v": _divergence(state.derivative("velocity"))},
                   lambda value: state.grouped(value * self.normalization.transport_time_s), "1/s")
        if "adiabatic_pressure" in active:
            log_pressure_gradient = state.derivative("log_pressure")[:, 0]
            residual = (
                state.time_derivative("log_pressure")[:, 0]
                + (log_pressure_gradient * state.velocity_field_m_per_s).sum(dim=-1)
                + self.adiabatic_index * _divergence(state.derivative("velocity"))
            )
            residual = state.grouped(residual * self.normalization.transport_time_s)
            losses["adiabatic_pressure"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )
            record("adiabatic_pressure", {"∂lnP/∂t": state.time_derivative("log_pressure")[:, 0],
                                          "v·∇lnP": (log_pressure_gradient * state.velocity_field_m_per_s).sum(-1),
                                          "γ∇·v": self.adiabatic_index * _divergence(state.derivative("velocity"))},
                   lambda value: state.grouped(value * self.normalization.transport_time_s), "1/s")
        return PhysicsResult(losses, weights, state if return_state else None,
                             diagnostics if return_diagnostics else None)

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
            return _robust_penalty(
                grouped, self.robust_loss_delta
            ).mean(dim=tuple(range(1, grouped.ndim))).mean()

        if "coronal_energy" in active:
            if not getattr(atmosphere_model, "time_dependent", False):
                raise ValueError(
                    "coronal_energy requires atmosphere.time_dependent=true"
                )
            minimum = self.coronal_energy.options.minimum_height_megameter * 1.0e6
            heights = position.norm(dim=-1) - atmosphere_model.solar_radius_m
            selected = heights >= minimum
            if selected.any():
                with torch.enable_grad():
                    residual = self.coronal_energy.residual(
                        atmosphere_model,
                        position[selected],
                        time[selected],
                        self.adiabatic_index,
                        self.normalization.transport_time_s,
                        create_graph=create_graph,
                        model_time_scale_s=self.normalization.time_s,
                    )
                    losses["coronal_energy"] = _mean_square(
                        residual, robust_delta=self.robust_loss_delta
                    )
        thermodynamic_active = {
            "upper_domain_microturbulence_prior",
            "upper_domain_temperature_prior",
        } & active
        if thermodynamic_active:
            position_rsun = (position / atmosphere_model.solar_radius_m).to(parameter)
            normalized_evaluator = getattr(
                atmosphere_model, "evaluate_position_rsun_normalized", None
            )
            if normalized_evaluator is None:
                fields = atmosphere_model.evaluate_position_rsun(
                    position_rsun,
                    time_hours=time,
                )
                fields_are_normalized = False
            else:
                fields = normalized_evaluator(position_rsun, time_hours=time)
                fields_are_normalized = True
            height_m = (
                torch.linalg.vector_norm(position_rsun, dim=-1)
                - position_rsun.new_tensor(1.0)
            ) * atmosphere_model.solar_radius_m
        if "upper_domain_microturbulence_prior" in active:
            reference_log_microturbulence = (
                atmosphere_model.reference_atmosphere.logs_at_height(height_m)[2]
            ).detach()
            if fields_are_normalized:
                reference_log_microturbulence = (
                    reference_log_microturbulence
                    - math.log(atmosphere_model.microturbulence_scale_m_per_s)
                )
            residual = (
                torch.log(fields["microturbulence"]) - reference_log_microturbulence
            )
            losses["upper_domain_microturbulence_prior"] = grouped_mean_square(residual)
        if "upper_domain_temperature_prior" in active:
            reference_log_temperature = (
                atmosphere_model.reference_atmosphere.logs_at_height(height_m)[0]
            ).detach()
            if fields_are_normalized:
                reference_log_temperature = (
                    reference_log_temperature
                    - math.log(atmosphere_model.temperature_scale_k)
                )
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
        """Apply configured top velocity, magnetic, and pressure conditions."""

        weights = self.loss_weights
        parameter = next(atmosphere_model.parameters())
        zero = parameter.new_zeros(())
        losses = {name: zero for name in UPPER_BOUNDARY_EQUATIONS}
        active = {name for name in UPPER_BOUNDARY_EQUATIONS if self.is_active(name)}
        if not active:
            return PhysicsResult(losses, weights)
        position = position_m.reshape(-1, 3)
        time = time_hours.to(parameter).reshape(-1, 1)
        if position.shape[0] < 1 or time.shape[0] != position.shape[0]:
            raise ValueError("Upper-boundary constraints require paired samples.")

        differential_active = {
            "upper_boundary_current_free",
            "upper_boundary_no_inflow",
            "upper_boundary_open_velocity",
            "upper_boundary_tangential_magnetic_neumann",
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
            velocity_derivative = (
                state.model_derivative("velocity")
                if state.model_primitive_derivatives is not None
                else state.derivative("velocity")
            )
            radial_velocity_derivative = torch.einsum(
                "nij,nj->ni",
                velocity_derivative,
                state.radial_unit,
            )
            if state.model_primitive_derivatives is not None:
                residual = state.grouped(
                    radial_velocity_derivative
                    * self.normalization.length_m
                    / atmosphere_model.height_input_scale_m
                )
            else:
                residual = state.grouped(
                    radial_velocity_derivative
                    * self.normalization.length_m
                    / self.normalization.velocity_scale_m_per_s
                )
            losses["upper_boundary_open_velocity"] = state.height_group_mse(
                residual, robust_delta=self.robust_loss_delta
            )

        if "upper_boundary_no_inflow" in active:
            boundary_velocity = (
                state.velocity_field_model
                if state.velocity_field_model is not None
                else state.velocity_field_m_per_s
            )
            inward_speed = (
                -(boundary_velocity * state.radial_unit).sum(-1)
            ).relu()
            losses["upper_boundary_no_inflow"] = _mean_square(
                inward_speed
                / (
                    1.0
                    if state.velocity_field_model is not None
                    else self.normalization.velocity_scale_m_per_s
                ),
                robust_delta=self.robust_loss_delta,
            )
        if "upper_boundary_current_free" in active:
            losses["upper_boundary_current_free"] = self._boundary_current_loss(state)

        if "upper_boundary_tangential_magnetic_neumann" in active:
            magnetic_derivative = (
                state.model_derivative("magnetic")
                if state.model_primitive_derivatives is not None
                else state.derivative("magnetic")
            )
            residual = _tangential_normal_derivative(
                magnetic_derivative, state.radial_unit
            )
            field_normalization = torch.linalg.vector_norm(
                state.magnetic_field_model
                if state.magnetic_field_model is not None
                else state.magnetic_field_gauss,
                dim=-1,
            )
            residual = state.normalize_points(
                residual,
                field_normalization,
                (
                    self.normalization.length_m / atmosphere_model.height_input_scale_m
                    if state.model_primitive_derivatives is not None
                    else self.normalization.length_m
                ),
                scale_floor=(
                    self.normalization.magnetic_field_floor_gauss
                    / atmosphere_model.magnetic_scale_gauss
                    if state.model_primitive_derivatives is not None
                    else self.normalization.magnetic_field_floor_gauss
                ),
                detach_scale=self.normalization.detach_normalization_scale,
            )
            losses["upper_boundary_tangential_magnetic_neumann"] = _mean_square(
                residual, robust_delta=self.robust_loss_delta
            )

        if "upper_boundary_gas_pressure_prior" in active:
            position_rsun = (position / atmosphere_model.solar_radius_m).to(parameter)
            normalized_evaluator = getattr(
                atmosphere_model, "evaluate_position_rsun_normalized", None
            )
            if normalized_evaluator is None:
                pressure = atmosphere_model.evaluate_position_rsun(
                    position_rsun,
                    time_hours=time,
                )["gas_pressure"]
                target_log_pressure = (
                    atmosphere_model.top_boundary_reference_log_pressure.to(pressure)
                ).detach()
            else:
                pressure = normalized_evaluator(
                    position_rsun,
                    time_hours=time,
                )["gas_pressure"]
                target_log_pressure = (
                    atmosphere_model.top_boundary_reference_log_pressure.to(pressure)
                    - math.log(atmosphere_model.gas_pressure_scale_pa)
                ).detach()
            residual = torch.log(pressure) - target_log_pressure
            losses["upper_boundary_gas_pressure_prior"] = _mean_square(
                residual, robust_delta=self.robust_loss_delta
            )
        return PhysicsResult(losses, weights)

    def side_boundary(
        self,
        atmosphere_model,
        position_m: torch.Tensor,
        time_hours: torch.Tensor,
        normal: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> PhysicsResult:
        """Apply configured velocity and magnetic conditions on angular sides."""

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
            height_group_shape=(1, position.shape[0]),
        )
        if "side_boundary_no_inflow" in active:
            boundary_velocity = (
                state.velocity_field_model
                if state.velocity_field_model is not None
                else state.velocity_field_m_per_s
            )
            inward_speed = (
                -(boundary_velocity * outward_normal).sum(-1)
            ).relu()
            losses["side_boundary_no_inflow"] = _mean_square(
                inward_speed
                / (
                    1.0
                    if state.velocity_field_model is not None
                    else self.normalization.velocity_scale_m_per_s
                ),
                robust_delta=self.robust_loss_delta,
            )
        if "side_boundary_current_free" in active:
            losses["side_boundary_current_free"] = self._boundary_current_loss(state)
        if "side_boundary_open_velocity" in active:
            velocity_derivative = (
                state.model_derivative("velocity")
                if state.model_primitive_derivatives is not None
                else state.derivative("velocity")
            )
            normal_velocity_derivative = torch.einsum(
                "nij,nj->ni",
                velocity_derivative,
                outward_normal,
            )
            residual = normal_velocity_derivative * (
                self.normalization.length_m
                / (
                    atmosphere_model.height_input_scale_m
                    if state.model_primitive_derivatives is not None
                    else self.normalization.velocity_scale_m_per_s
                )
            )
            losses["side_boundary_open_velocity"] = _mean_square(
                residual, robust_delta=self.robust_loss_delta
            )

        if "side_boundary_tangential_magnetic_neumann" in active:
            magnetic_derivative = (
                state.model_derivative("magnetic")
                if state.model_primitive_derivatives is not None
                else state.derivative("magnetic")
            )
            residual = _tangential_normal_derivative(
                magnetic_derivative, outward_normal
            )
            field_normalization = torch.linalg.vector_norm(
                state.magnetic_field_model
                if state.magnetic_field_model is not None
                else state.magnetic_field_gauss,
                dim=-1,
            )
            residual = state.normalize_points(
                residual,
                field_normalization,
                (
                    self.normalization.length_m / atmosphere_model.height_input_scale_m
                    if state.model_primitive_derivatives is not None
                    else self.normalization.length_m
                ),
                scale_floor=(
                    self.normalization.magnetic_field_floor_gauss
                    / atmosphere_model.magnetic_scale_gauss
                    if state.model_primitive_derivatives is not None
                    else self.normalization.magnetic_field_floor_gauss
                ),
                detach_scale=self.normalization.detach_normalization_scale,
            )
            losses["side_boundary_tangential_magnetic_neumann"] = _mean_square(
                residual, robust_delta=self.robust_loss_delta
            )
        return PhysicsResult(losses, weights)

    def _boundary_current_loss(self, state: PhysicsState) -> torch.Tensor:
        """Dimensionless mu0*J*L/<|B|>, with a detached boundary-mean scale.

        Curl and field both use gauss, so the gauss-to-tesla and mu0 factors
        cancel. The magnitude is averaged, not the signed field vector.
        """
        if state.model_primitive_derivatives is not None:
            residual = state.normalize_height_groups(
                _curl(state.model_derivative("magnetic")),
                torch.linalg.vector_norm(state.magnetic_field_model, dim=-1),
                self.normalization.length_m / state.model_length_scale_m,
                scale_floor=self.normalization.magnetic_field_floor_gauss
                / state.model_magnetic_scale_gauss,
                detach_scale=self.normalization.detach_normalization_scale,
            )
        else:
            residual = state.normalize_height_groups(
                _curl(state.derivative("magnetic")),
                torch.linalg.vector_norm(state.magnetic_field_gauss, dim=-1),
                self.normalization.length_m,
                scale_floor=self.normalization.magnetic_field_floor_gauss,
                detach_scale=self.normalization.detach_normalization_scale,
            )
        return state.height_group_mse(
            residual, robust_delta=self.robust_loss_delta
        )

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
            "robust_loss_delta": self.robust_loss_delta,
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
