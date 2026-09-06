"""Smooth neural representation of a spherical LTE atmosphere.

The operational representation is a heliocentric Carrington atmosphere
``F(x, y, r - R_sun, t)``. Thermodynamics are learned as unbounded linear
residuals in natural-log space around a radial reference. Observer rays are sampled
inside a fixed spherical shell and optical depth is obtained by integrating
absolute opacity along those physical paths. Static runs omit the time channel;
time-dependent runs use each HMI acquisition or Hinode scan ``DATE_OBS``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping

import torch
from torch import nn

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN, MLPModel, SPEED_OF_LIGHT
from prom3theus.resources import verify_manifest_resource
from .atomic import AtomicDatabase
from .plasma import SolarPlasmaTable
from .geometry import (
    RayTraceResult,
    validate_scene_basis,
)


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


def _load_falc_reference_document(atomic: AtomicDatabase) -> tuple[dict, str]:
    """Load the checksum-pinned complete FALC reference atmosphere."""

    resource = atomic.data_root.joinpath("falc_reference_atmosphere.json")
    digest = verify_manifest_resource(
        resource,
        atomic.source_manifest,
        resource_name="common/falc_reference_atmosphere.json",
        kind="radial-reference",
    )
    with _open_text(resource) as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise RuntimeError("Invalid pinned FALC radial-reference resource.")
    source = document.get("source")
    if not isinstance(source, Mapping) or not isinstance(
        source.get("source_sha256"), Mapping
    ):
        raise RuntimeError("Invalid pinned FALC radial-reference resource.")
    if (
        document.get("schema_version") != 1
        or document.get("model") != "FALC_82"
        or document.get("native_depth_count") != 82
        or source.get("commit") != "18cda77d038a97f007a783dcb61ea9a9a1244bf7"
        or source["source_sha256"].get("stic_falc_82")
        != atomic.source_manifest.get("sources", {})
        .get("stic_falc_82", {})
        .get("sha256")
    ):
        raise RuntimeError("Invalid pinned FALC radial-reference resource.")
    return document, digest


def _linear_interpolate(
    nodes: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    *,
    extrapolate: bool = False,
) -> torch.Tensor:
    """Piecewise-linear interpolation, differentiable inside each interval."""

    if not extrapolate:
        query = query.clamp(nodes[0], nodes[-1])
    upper = torch.searchsorted(nodes, query.detach().contiguous(), right=True).clamp(
        1, nodes.numel() - 1
    )
    lower = upper - 1
    x0, x1 = nodes[lower], nodes[upper]
    fraction = (query - x0) / (x1 - x0)
    fraction = fraction[(...,) + (None,) * (values.ndim - 1)]
    return values[lower] + fraction * (values[upper] - values[lower])


class RadialReferenceAtmosphere(nn.Module):
    """One-dimensional atmosphere supplying a physical radial baseline."""

    def __init__(
        self,
        config: str | Mapping = "falc_82",
        *,
        upper_atmosphere_config: Mapping | None = None,
        maximum_height_m: float | None = None,
        solar_radius_m: float | None = None,
        thermodynamic_eos: SolarPlasmaTable | None = None,
        atomic_database: AtomicDatabase | None = None,
    ):
        super().__init__()
        native_falc = None
        self.falc_resource_sha256 = None
        if isinstance(config, str):
            if config.lower() != "falc_82":
                raise ValueError(
                    "reference_atmosphere_config must be 'falc_82' or a mapping."
                )
            atomic = atomic_database or AtomicDatabase()
            native_falc, self.falc_resource_sha256 = _load_falc_reference_document(
                atomic
            )
            line_reference = native_falc["line_formation_reference"]
            data = {
                "log_tau500": line_reference["log10_tau500"],
                "height_m": line_reference["height_m"],
                "temperature_k": line_reference["temperature_k"],
                "gas_pressure_pa": line_reference["gas_pressure_pa"],
                "microturbulence_m_per_s": line_reference["microturbulence_m_per_s"],
            }
            self.name = "complete pinned STiC FALC_82"
        else:
            data = dict(config)
            self.name = str(data.pop("name", "configured radial reference"))
        required = {
            "log_tau500",
            "height_m",
            "temperature_k",
            "gas_pressure_pa",
            "microturbulence_m_per_s",
        }
        unknown, missing = set(data) - required, required - set(data)
        if unknown or missing:
            raise TypeError(
                f"Invalid radial-reference fields; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}."
            )
        q = _as_reference_tensor(data["log_tau500"], name="reference log_tau500")
        height = _as_reference_tensor(data["height_m"], name="reference height_m")
        temperature = _as_reference_tensor(
            data["temperature_k"], name="reference temperature_k"
        )
        pressure = _as_reference_tensor(
            data["gas_pressure_pa"], name="reference gas_pressure_pa"
        )
        micro = _as_reference_tensor(
            data["microturbulence_m_per_s"], name="reference microturbulence_m_per_s"
        )
        if not all(
            value.shape == q.shape for value in (height, temperature, pressure, micro)
        ):
            raise ValueError("All radial-reference arrays must have the same shape.")
        if q.ndim != 1 or q.numel() < 2 or not torch.all(q[1:] > q[:-1]):
            raise ValueError(
                "Reference log_tau500 must be a strictly increasing vector."
            )
        if not torch.all(height[1:] < height[:-1]):
            raise ValueError(
                "Reference height must decrease as optical depth increases."
            )
        if (
            torch.any(temperature <= 0)
            or torch.any(pressure <= 0)
            or torch.any(micro <= 0)
        ):
            raise ValueError("Reference thermodynamic quantities must be positive.")
        thermodynamic_height = height.flip(0)
        thermodynamic_logs = torch.stack(
            (
                torch.log(temperature.flip(0)),
                torch.log(pressure.flip(0)),
                torch.log(micro.flip(0)),
            ),
            dim=-1,
        )
        native_depth_count = int(q.numel())
        if native_falc is not None:
            native_coordinates = native_falc["coordinates"]
            native_log_tau500 = _as_reference_tensor(
                native_coordinates["log10_tau500"],
                name="native FALC log10_tau500",
            )
            native_height = _as_reference_tensor(
                native_coordinates["height_m"], name="native FALC height_m"
            )
            native_temperature = _as_reference_tensor(
                native_falc["temperature_k"], name="native FALC temperature_k"
            )
            native_pressure = _as_reference_tensor(
                native_falc["gas_pressure_pa"], name="native FALC gas_pressure_pa"
            )
            native_micro = _as_reference_tensor(
                native_falc["microturbulence_m_per_s"],
                name="native FALC microturbulence_m_per_s",
            )
            if not all(
                value.shape == native_log_tau500.shape
                for value in (
                    native_height,
                    native_temperature,
                    native_pressure,
                    native_micro,
                )
            ) or native_log_tau500.shape != (82,):
                raise RuntimeError("Invalid complete native FALC reference arrays.")
            restored = native_log_tau500 < q[0]
            if int(restored.sum()) != 37:
                raise RuntimeError(
                    "The pinned FALC reference must restore 37 nodes above logtau=-5."
                )
            restored_height = native_height[restored].flip(0)
            restored_logs = torch.stack(
                (
                    torch.log(native_temperature[restored].flip(0)),
                    torch.log(native_pressure[restored].flip(0)),
                    torch.log(native_micro[restored].flip(0)),
                ),
                dim=-1,
            )
            if restored_height[0] <= thermodynamic_height[-1] or not torch.all(
                restored_height[1:] > restored_height[:-1]
            ):
                raise RuntimeError("The restored native FALC heights are not monotone.")
            thermodynamic_height = torch.cat((thermodynamic_height, restored_height))
            thermodynamic_logs = torch.cat((thermodynamic_logs, restored_logs))
            native_depth_count = int(native_log_tau500.numel())

        self.native_atmosphere_top_m = float(thermodynamic_height[-1])
        self.native_atmosphere_top_temperature_k = (
            float(native_temperature[0])
            if native_falc is not None
            else float(torch.exp(thermodynamic_logs[-1, 0]))
        )
        self.upper_atmosphere_metadata = None
        if upper_atmosphere_config is not None:
            options = dict(upper_atmosphere_config)
            expected = {
                "type",
                "transition_region_top_megameter",
                "coronal_temperature_k",
                "reference_grid_points",
            }
            if set(options) != expected:
                raise TypeError(
                    f"upper_atmosphere_config must contain exactly {sorted(expected)}."
                )
            if options["type"] != "hydrostatic_corona":
                raise ValueError(
                    "Only a hydrostatic_corona upper atmosphere is supported."
                )
            if (
                maximum_height_m is None
                or solar_radius_m is None
                or thermodynamic_eos is None
            ):
                raise ValueError(
                    "A hydrostatic upper atmosphere requires its maximum height, "
                    "solar radius, and thermodynamic EoS."
                )
            base_top_m = float(thermodynamic_height[-1])
            maximum_height_m = float(maximum_height_m)
            transition_top_m = float(options["transition_region_top_megameter"]) * 1.0e6
            coronal_temperature_k = float(options["coronal_temperature_k"])
            point_count = options["reference_grid_points"]
            if isinstance(point_count, bool) or not isinstance(point_count, int):
                raise TypeError("reference_grid_points must be an integer.")
            if point_count < 2:
                raise ValueError("reference_grid_points must be at least 2.")
            if not maximum_height_m >= transition_top_m > base_top_m:
                raise ValueError(
                    "The transition-region top must be above the FALC top and below "
                    "the full-domain top."
                )
            if not math.isfinite(coronal_temperature_k) or (
                coronal_temperature_k < self.native_atmosphere_top_temperature_k
            ):
                raise ValueError(
                    "The coronal temperature must not be below the native FALC-top "
                    "temperature and must be finite."
                )
            if maximum_height_m == transition_top_m:
                transition_intervals = point_count
            else:
                transition_fraction_of_domain = (transition_top_m - base_top_m) / (
                    maximum_height_m - base_top_m
                )
                transition_intervals = min(
                    point_count - 1,
                    max(1, round(point_count * transition_fraction_of_domain)),
                )
            transition_height = torch.linspace(
                base_top_m,
                transition_top_m,
                transition_intervals + 1,
                dtype=height.dtype,
            )[1:]
            coronal_intervals = point_count - transition_intervals
            coronal_height = torch.linspace(
                transition_top_m,
                maximum_height_m,
                coronal_intervals + 1,
                dtype=height.dtype,
            )[1:]
            upper_height = torch.cat((transition_height, coronal_height))
            transition_fraction = (
                (upper_height - base_top_m) / (transition_top_m - base_top_m)
            ).clamp(0.0, 1.0)
            smooth = transition_fraction.pow(3) * (
                transition_fraction * (transition_fraction * 6.0 - 15.0) + 10.0
            )
            base_log_temperature = thermodynamic_logs[-1, 0]
            upper_log_temperature = torch.lerp(
                base_log_temperature,
                upper_height.new_tensor(math.log(coronal_temperature_k)),
                smooth,
            )
            upper_temperature = torch.exp(upper_log_temperature)
            upper_log_pressure_values = []
            previous_height = thermodynamic_height[-1]
            previous_temperature = torch.exp(base_log_temperature)
            previous_log_pressure = thermodynamic_logs[-1, 1]
            reference_gravity = thermodynamic_eos.reference_gravity_m_per_s2
            solar_radius = float(solar_radius_m)
            for next_height, next_temperature in zip(
                upper_height, upper_temperature, strict=True
            ):
                midpoint_height = 0.5 * (previous_height + next_height)
                midpoint_temperature = torch.sqrt(
                    previous_temperature * next_temperature
                )
                gravity = (
                    reference_gravity
                    * (solar_radius / (solar_radius + float(midpoint_height))) ** 2
                )
                previous_pressure = torch.exp(previous_log_pressure)
                mean_particle_mass_u = thermodynamic_eos.mean_molecular_weight(
                    midpoint_temperature, previous_pressure
                )
                predicted_delta_log_pressure = -(
                    mean_particle_mass_u
                    * ATOMIC_MASS_UNIT
                    * gravity
                    * (next_height - previous_height)
                    / (K_BOLTZMANN * midpoint_temperature)
                )
                midpoint_pressure = torch.exp(
                    previous_log_pressure + 0.5 * predicted_delta_log_pressure
                )
                mean_particle_mass_u = thermodynamic_eos.mean_molecular_weight(
                    midpoint_temperature, midpoint_pressure
                )
                delta_log_pressure = -(
                    mean_particle_mass_u
                    * ATOMIC_MASS_UNIT
                    * gravity
                    * (next_height - previous_height)
                    / (K_BOLTZMANN * midpoint_temperature)
                )
                next_log_pressure = previous_log_pressure + delta_log_pressure
                upper_log_pressure_values.append(next_log_pressure)
                previous_height = next_height
                previous_temperature = next_temperature
                previous_log_pressure = next_log_pressure
            upper_log_pressure = torch.stack(upper_log_pressure_values)
            upper_log_micro = thermodynamic_logs[-1, 2].expand_as(upper_height)
            thermodynamic_height = torch.cat((thermodynamic_height, upper_height))
            thermodynamic_logs = torch.cat(
                (
                    thermodynamic_logs,
                    torch.stack(
                        (upper_log_temperature, upper_log_pressure, upper_log_micro),
                        dim=-1,
                    ),
                )
            )
            self.upper_atmosphere_metadata = {
                "type": "hydrostatic_corona",
                "native_atmosphere_top_m": base_top_m,
                "native_atmosphere_top_temperature_k": (
                    self.native_atmosphere_top_temperature_k
                ),
                "transition_region_top_m": transition_top_m,
                "coronal_temperature_k": coronal_temperature_k,
                "maximum_height_m": maximum_height_m,
                "reference_grid_points": point_count,
                "temperature": (
                    "log-temperature smootherstep from the native atmosphere top "
                    "to the configured isothermal corona"
                ),
                "gravity": "spherical inverse-square",
                "pressure": "hydrostatic integral using the hybrid solar EoS",
                "microturbulence": "native FALC-top value held constant",
            }
        self.register_buffer("log_tau500", q)
        self.register_buffer("height_descending_m", height)
        self.register_buffer("height_ascending_m", height.flip(0))
        self.register_buffer("thermodynamic_height_ascending_m", thermodynamic_height)
        self.register_buffer(
            "thermodynamic_logs_ascending",
            thermodynamic_logs,
        )
        self.native_depth_count = native_depth_count

    def height_from_log_tau(self, q: torch.Tensor) -> torch.Tensor:
        return _linear_interpolate(
            self.log_tau500.to(q), self.height_descending_m.to(q), q
        )

    def log_tau_from_height(self, height_m: torch.Tensor) -> torch.Tensor:
        """Invert the reference height relation, including its top extrapolation."""

        return _linear_interpolate(
            self.height_ascending_m.to(height_m),
            self.log_tau500.flip(0).to(height_m),
            height_m,
            extrapolate=True,
        )

    def logs_at_height(
        self, height_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = self.thermodynamic_height_ascending_m.to(height_m)
        interpolated = _linear_interpolate(
            nodes,
            self.thermodynamic_logs_ascending.to(height_m),
            height_m,
            extrapolate=True,
        )
        return interpolated.unbind(dim=-1)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "coordinate": "radius minus solar radius in metres",
            "tau_one_height_m": 0.0,
            "height_bounds_m": [
                float(self.thermodynamic_height_ascending_m[0]),
                float(self.thermodynamic_height_ascending_m[-1]),
            ],
            "parameterization": "tabulated natural-log thermodynamic reference",
            "native_depth_count": self.native_depth_count,
            "native_atmosphere_top_m": self.native_atmosphere_top_m,
            "native_atmosphere_top_temperature_k": (
                self.native_atmosphere_top_temperature_k
            ),
            "falc_resource_sha256": self.falc_resource_sha256,
            "upper_atmosphere": self.upper_atmosphere_metadata,
        }


def _initialize_smooth_coordinate_mlp(model: MLPModel, activation: str) -> None:
    """Initialize a deep smooth MLP without suppressing coordinate variance.

    PyTorch's default linear-layer initialization is too contractive when it is
    repeated through a deep SiLU/GELU coordinate network.  The network remains
    random, but its output is then dominated by the final bias and initially
    appears spatially uniform.  Use activation-aware, variance-preserving
    hidden weights and an ordinary Xavier readout.  Biases intentionally remain
    independent random variables; no physical output channel is zeroed.
    """

    activation = str(activation).lower()
    hidden_layers = (model.in_layer, *model.hidden_layers)
    with torch.no_grad():
        for layer in hidden_layers:
            if activation == "tanh":
                nn.init.xavier_uniform_(
                    layer.weight,
                    gain=nn.init.calculate_gain("tanh"),
                )
            else:
                # PyTorch has no dedicated SiLU/GELU gain.  The ReLU gain is
                # the standard variance-preserving approximation for these
                # smooth, non-saturating activations.
                nn.init.kaiming_uniform_(
                    layer.weight,
                    a=0.0,
                    nonlinearity="relu",
                )
            bound = 1.0 / math.sqrt(layer.in_features)
            layer.bias.uniform_(-bound, bound)

        nn.init.xavier_uniform_(model.out_layer.weight)
        output_bound = 1.0 / math.sqrt(model.out_layer.in_features)
        model.out_layer.bias.uniform_(-output_bound, output_bound)


def _as_float_tensor(value, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_reference_tensor(value, *, name: str) -> torch.Tensor:
    """Keep checksum-pinned reference coordinates in generation precision."""

    tensor = torch.as_tensor(value, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _coordinate_pair(value, *, name: str, positive: bool) -> torch.Tensor:
    tensor = _as_float_tensor(value, name=name)
    if tensor.shape != (2,):
        raise ValueError(f"{name} must contain [spatial-X, spatial-Y].")
    if positive and torch.any(tensor <= 0):
        raise ValueError(f"{name} values must be positive.")
    return tensor


@dataclass(frozen=True)
class StratifiedAtmosphere:
    """Physical atmosphere sampled on a common optical-depth grid.

    Parameters
    ----------
    log_tau500:
        One-dimensional ``log10(tau_500)`` grid, strictly increasing from the
        top of the atmosphere to the bottom, shape ``[depth]``.
    temperature:
        Temperature in kelvin, shape ``[..., depth]``.
    velocity_field:
        Velocity vector in m/s and with shape ``[..., depth, 3]`` in the
        Heliographic Carrington Cartesian frame when spherical geometry is enabled.
        The synthesis adapter projects this vector into the observer frame;
        positive-redshift velocity is then the negative observer-directed
        component. The transverse components are not constrained by a
        single-view Stokes spectrum without additional dynamical physics.
    microturbulence:
        Microturbulent speed in m/s, shape ``[..., depth]``.
    magnetic_field:
        Magnetic vector in gauss and with shape ``[..., depth, 3]``. It is in
        the Heliographic Carrington Cartesian frame when spherical geometry is
        enabled, and is projected to ``[B_Q, B_U, B_LOS]`` immediately before
        polarized synthesis.
    gas_pressure:
        Gas pressure in pascal, shape ``[..., depth]``.  In an inversion this
        is predicted by the coordinate network and may be constrained by
        hydrostatic equilibrium or time-dependent momentum balance; it is not
        solved iteratively inside radiative transfer.
    """

    log_tau500: torch.Tensor
    temperature: torch.Tensor
    velocity_field: torch.Tensor
    microturbulence: torch.Tensor
    magnetic_field: torch.Tensor
    gas_pressure: torch.Tensor | None = None
    geometric_height_m: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.log_tau500.ndim != 1:
            raise ValueError("log_tau500 must be one-dimensional.")
        if self.log_tau500.numel() < 2:
            raise ValueError("log_tau500 must contain at least two depth points.")
        if self.log_tau500.dtype not in (torch.float32, torch.float64):
            raise TypeError("log_tau500 must use float32 or float64.")
        if not torch.isfinite(self.log_tau500).all() or not torch.all(
            self.log_tau500[1:] > self.log_tau500[:-1]
        ):
            raise ValueError(
                "log_tau500 must be finite and strictly increase from top to bottom."
            )
        if self.temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError("Atmospheric fields must use float32 or float64.")

        depth = self.log_tau500.numel()
        scalar_fields = {
            "temperature": self.temperature,
            "microturbulence": self.microturbulence,
        }
        if self.gas_pressure is not None:
            scalar_fields["gas_pressure"] = self.gas_pressure
        if self.geometric_height_m is not None:
            scalar_fields["geometric_height_m"] = self.geometric_height_m
        for name, value in scalar_fields.items():
            if value.shape[-1:] != (depth,):
                raise ValueError(
                    f"{name} must end in depth dimension {depth}; got {tuple(value.shape)}."
                )
            if value.shape[:-1] != self.temperature.shape[:-1]:
                raise ValueError(
                    f"{name} and temperature must have matching batch dimensions."
                )
            if value.device != self.temperature.device:
                raise ValueError(f"{name} and temperature must be on the same device.")
            if value.dtype != self.temperature.dtype:
                raise TypeError(f"{name} and temperature must use the same dtype.")

        expected_vector_shape = (*self.temperature.shape, 3)
        if self.velocity_field.shape != expected_vector_shape:
            raise ValueError(
                "velocity_field must have shape [..., depth, 3]; "
                f"expected {expected_vector_shape}, got {tuple(self.velocity_field.shape)}."
            )
        if self.velocity_field.device != self.temperature.device:
            raise ValueError(
                "velocity_field and temperature must be on the same device."
            )
        if self.velocity_field.dtype != self.temperature.dtype:
            raise TypeError("velocity_field and temperature must use the same dtype.")
        expected_magnetic_shape = expected_vector_shape
        if self.magnetic_field.shape != expected_magnetic_shape:
            raise ValueError(
                "magnetic_field must have shape [..., depth, 3]; "
                f"expected {expected_magnetic_shape}, got {tuple(self.magnetic_field.shape)}."
            )
        if self.magnetic_field.device != self.temperature.device:
            raise ValueError(
                "magnetic_field and temperature must be on the same device."
            )
        if self.magnetic_field.dtype != self.temperature.dtype:
            raise TypeError("magnetic_field and temperature must use the same dtype.")
        if self.log_tau500.device != self.temperature.device:
            raise ValueError(
                "log_tau500 and atmospheric fields must be on the same device."
            )
        if self.log_tau500.dtype != self.temperature.dtype:
            raise TypeError(
                "log_tau500 and atmospheric fields must use the same dtype."
            )

    @property
    def depth(self) -> int:
        return int(self.log_tau500.numel())

    @property
    def batch_shape(self) -> torch.Size:
        return self.temperature.shape[:-1]

    @property
    def v_los(self) -> torch.Tensor:
        """Positive-redshift LOS speed used by the radiative-transfer solver."""

        return -self.velocity_field[..., 2]


class StratifiedAtmosphereModel(nn.Module):
    """Coordinate MLP producing a smooth stratified atmosphere.

    The MLP uses activation-aware variance-preserving random initialization.
    Thermodynamic readout rows start at zero so the initial state equals the
    radial reference; vector readout rows remain random.
    Positive thermodynamic variables use unbounded, reference-anchored linear
    residuals in natural-log space.  Finite-domain radiative-transfer tables
    validate their own inputs independently of this primary atmosphere model.
    Cartesian magnetic vectors use a direct unit-scaled linear decoder.
    Velocity components use a smooth arctangent bound with the configured
    local linear scale, preventing invalid relativistic Doppler factors while
    retaining nonzero gradients outside the ordinary photospheric range.
    """

    output_names = (
        "temperature",
        "v_x",
        "v_y",
        "v_z",
        "b_x",
        "b_y",
        "b_z",
        "microturbulence",
        "gas_pressure",
    )

    def __init__(
        self,
        log_tau500,
        temperature_log_scale: float = 0.2,
        velocity_scale_m_per_s: float = 1_000.0,
        velocity_max_m_per_s: float = 100_000.0,
        magnetic_scale_gauss: float = 100.0,
        gas_pressure_log_scale: float = 1.0,
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(1.0, 1.0),
        height_input_scale_m: float = 1_000_000.0,
        time_dependent: bool = False,
        time_coordinate_center_hours: float = 0.0,
        time_coordinate_scale_hours: float = 1.0,
        shell_height_bounds_Mm: tuple[float, float] | None = None,
        line_formation_height_bounds_Mm: tuple[float, float] | None = None,
        tangent_margin_m: float = 1_000.0,
        reference_atmosphere_config: str | Mapping | None = "falc_82",
        upper_atmosphere_config: Mapping | None = None,
        microturbulence_log_scale: float = 0.2,
        scene_geometry_config: Mapping | None = None,
        model_config: Mapping | None = None,
    ):
        super().__init__()
        log_tau500 = _as_float_tensor(log_tau500, name="log_tau500")
        if log_tau500.ndim != 1 or log_tau500.numel() < 2:
            raise ValueError(
                "log_tau500 must be a one-dimensional grid with at least two points."
            )
        if not torch.all(log_tau500[1:] > log_tau500[:-1]):
            raise ValueError("log_tau500 must increase strictly from top to bottom.")

        scales = {
            "temperature_log_scale": temperature_log_scale,
            "velocity_scale_m_per_s": velocity_scale_m_per_s,
            "velocity_max_m_per_s": velocity_max_m_per_s,
            "magnetic_scale_gauss": magnetic_scale_gauss,
            "gas_pressure_log_scale": gas_pressure_log_scale,
            "microturbulence_log_scale": microturbulence_log_scale,
        }
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in scales.values()
        ):
            raise ValueError(
                f"Atmosphere physical scales must be positive; got {scales}."
            )
        if math.sqrt(3.0) * float(velocity_max_m_per_s) >= SPEED_OF_LIGHT:
            raise ValueError(
                "sqrt(3)*velocity_max_m_per_s must remain below the speed of light "
                "so every projected velocity is strictly subluminal."
            )

        self.register_buffer("log_tau500", log_tau500)
        self.temperature_log_scale = float(temperature_log_scale)
        self.velocity_scale_m_per_s = float(velocity_scale_m_per_s)
        self.velocity_max_m_per_s = float(velocity_max_m_per_s)
        self.magnetic_scale_gauss = float(magnetic_scale_gauss)
        self.gas_pressure_log_scale = float(gas_pressure_log_scale)
        self.microturbulence_log_scale = float(microturbulence_log_scale)
        self.register_buffer(
            "spatial_coordinate_center_mm",
            _coordinate_pair(
                spatial_coordinate_center_mm,
                name="spatial_coordinate_center_mm",
                positive=False,
            ),
        )
        self.register_buffer(
            "spatial_coordinate_scale_mm",
            _coordinate_pair(
                spatial_coordinate_scale_mm,
                name="spatial_coordinate_scale_mm",
                positive=True,
            ),
        )
        if not float(height_input_scale_m) > 0:
            raise ValueError("height_input_scale_m must be positive.")
        self.height_input_scale_m = float(height_input_scale_m)
        if type(time_dependent) is not bool:
            raise TypeError("time_dependent must be boolean.")
        self.time_dependent = time_dependent
        self.time_coordinate_center_hours = float(time_coordinate_center_hours)
        self.time_coordinate_scale_hours = float(time_coordinate_scale_hours)
        if not math.isfinite(self.time_coordinate_center_hours):
            raise ValueError("time_coordinate_center_hours must be finite.")
        if (
            not math.isfinite(self.time_coordinate_scale_hours)
            or self.time_coordinate_scale_hours <= 0
        ):
            raise ValueError("time_coordinate_scale_hours must be finite and positive.")
        if reference_atmosphere_config is None:
            raise ValueError(
                "The spherical LTE atmosphere requires a radial reference."
            )
        atomic_database = AtomicDatabase()
        self.thermodynamic_eos = SolarPlasmaTable(atomic_database)
        self.reference_atmosphere = RadialReferenceAtmosphere(
            reference_atmosphere_config,
            atomic_database=atomic_database,
        )
        tangent_margin_m = float(tangent_margin_m)
        if not math.isfinite(tangent_margin_m) or tangent_margin_m <= 0:
            raise ValueError("tangent_margin_m must be finite and positive.")
        self.tangent_margin_m = tangent_margin_m
        if scene_geometry_config is None:
            raise ValueError(
                "The spherical LTE atmosphere requires scene_geometry_config."
            )
        scene_config = dict(scene_geometry_config)
        unknown_scene = set(scene_config) - {"solar_radius_m", "scene_basis"}
        if unknown_scene:
            raise TypeError(f"Unknown scene-geometry options: {sorted(unknown_scene)}")
        solar_radius_m = float(scene_config["solar_radius_m"])
        if not math.isfinite(solar_radius_m) or solar_radius_m <= 0:
            raise ValueError("scene_geometry_config.solar_radius_m must be positive.")
        self.register_buffer("solar_radius_m", log_tau500.new_tensor(solar_radius_m))
        self.solar_radius_value_m = solar_radius_m
        reference_endpoint_heights = self.reference_atmosphere.height_from_log_tau(
            self.log_tau500[[0, -1]]
        )
        reference_height_bounds_Mm = tuple(
            float(height) / 1.0e6 for height in reference_endpoint_heights
        )
        if shell_height_bounds_Mm is None:
            shell_height_bounds_Mm = reference_height_bounds_Mm
        shell_height_bounds_Mm = tuple(map(float, shell_height_bounds_Mm))
        if len(shell_height_bounds_Mm) != 2 or not (
            math.isfinite(shell_height_bounds_Mm[0])
            and math.isfinite(shell_height_bounds_Mm[1])
            and shell_height_bounds_Mm[0] > shell_height_bounds_Mm[1]
        ):
            raise ValueError(
                "shell_height_bounds_Mm must be [outer_height, inner_height] "
                "as offsets from R_sun in Mm, with outer_height > inner_height."
            )
        if self.log_tau500[0] < 0 < self.log_tau500[-1] and not (
            shell_height_bounds_Mm[0] > 0 > shell_height_bounds_Mm[1]
        ):
            raise ValueError(
                "A shell whose depth grid crosses log_tau500=0 must have a positive "
                "outer height and negative inner height."
            )
        self.shell_height_bounds_Mm = shell_height_bounds_Mm
        if line_formation_height_bounds_Mm is None:
            line_formation_height_bounds_Mm = shell_height_bounds_Mm
        line_formation_height_bounds_Mm = tuple(
            map(float, line_formation_height_bounds_Mm)
        )
        if len(line_formation_height_bounds_Mm) != 2 or not (
            line_formation_height_bounds_Mm[1] == shell_height_bounds_Mm[1]
            and shell_height_bounds_Mm[1]
            < line_formation_height_bounds_Mm[0]
            <= shell_height_bounds_Mm[0]
        ):
            raise ValueError(
                "line_formation_height_bounds_Mm must share the shell inner "
                "height and end at or below the full-domain outer height."
            )
        self.line_formation_height_bounds_Mm = line_formation_height_bounds_Mm
        self.extrapolation_enabled = (
            line_formation_height_bounds_Mm[0] < shell_height_bounds_Mm[0]
        )
        if self.extrapolation_enabled:
            if upper_atmosphere_config is None:
                raise ValueError(
                    "An extrapolated shell requires upper_atmosphere_config."
                )
            self.reference_atmosphere = RadialReferenceAtmosphere(
                reference_atmosphere_config,
                upper_atmosphere_config=upper_atmosphere_config,
                maximum_height_m=shell_height_bounds_Mm[0] * 1.0e6,
                solar_radius_m=solar_radius_m,
                thermodynamic_eos=self.thermodynamic_eos,
                atomic_database=atomic_database,
            )
            self.transition_region_top_m = (
                float(upper_atmosphere_config["transition_region_top_megameter"])
                * 1.0e6
            )
        else:
            if upper_atmosphere_config is not None:
                raise ValueError(
                    "upper_atmosphere_config requires an extrapolated shell."
                )
            self.transition_region_top_m = None
        reference_outer_m, reference_inner_m = (
            float(value) * 1.0e6 for value in reference_height_bounds_Mm
        )
        self.outer_height_scale = (
            line_formation_height_bounds_Mm[0] * 1.0e6 / reference_outer_m
        )
        self.inner_height_scale = (
            line_formation_height_bounds_Mm[1] * 1.0e6 / reference_inner_m
        )
        top_height = log_tau500.new_tensor(shell_height_bounds_Mm[0] * 1.0e6)
        top_reference_logs = self.reference_atmosphere.logs_at_height(top_height)
        if top_reference_logs[1] <= math.log(torch.finfo(log_tau500.dtype).tiny):
            raise ValueError(
                "The FALC pressure at the configured domain top is not "
                "representable in the atmosphere dtype."
            )
        self.register_buffer(
            "top_boundary_reference_log_pressure",
            top_reference_logs[1].detach().clone(),
        )
        self.scene_basis_values = tuple(
            tuple(float(component) for component in row)
            for row in scene_config["scene_basis"]
        )
        self.register_buffer(
            "scene_basis", validate_scene_basis(scene_config["scene_basis"])
        )
        self.network_input_names = (
            ("x", "y", "z", "time") if self.time_dependent else ("x", "y", "z")
        )

        if not isinstance(model_config, Mapping):
            raise TypeError("model_config must be a mapping.")
        config = dict(model_config)
        expected_model_fields = {
            "type",
            "dim",
            "n_layers",
            "activation",
            "encoding_config",
        }
        if set(config) != expected_model_fields:
            raise TypeError(
                "model_config must contain exactly type, dim, n_layers, activation, "
                "and encoding_config."
            )
        model_type = config.pop("type")
        if model_type != "mlp":
            raise ValueError(
                "StratifiedAtmosphereModel currently supports model type 'mlp' only."
            )
        activation = config["activation"]
        if not isinstance(activation, str):
            raise TypeError("model_config.activation must be a string.")
        smooth_activations = {"silu", "gelu", "tanh"}
        if activation not in smooth_activations:
            raise ValueError(
                "The LTE atmosphere must use a smooth activation; choose one of "
                f"{sorted(smooth_activations)}, got {activation!r}."
            )
        self.network = MLPModel(
            in_dim=len(self.network_input_names),
            out_dim=len(self.output_names),
            **config,
        )
        _initialize_smooth_coordinate_mlp(self.network, activation=activation)
        # Exact initial radial thermodynamics; velocity and magnetic rows retain
        # random readouts so polarized gradients do not start at a symmetry point.
        with torch.no_grad():
            for index in (0, 7, 8):
                self.network.out_layer.weight[index].zero_()
                self.network.out_layer.bias[index].zero_()

    def _evaluation_grid(self, log_tau500=None) -> torch.Tensor:
        """Return a validated evaluation grid in the model's dtype/device.

        ``self.log_tau500`` defines the represented optical-depth interval and
        the reference quadrature grid.  It does not define trainable nodes: the
        coordinate MLP may be evaluated at any strictly increasing points
        inside the same interval.
        """

        if log_tau500 is None:
            return self.log_tau500
        grid = torch.as_tensor(
            log_tau500,
            dtype=self.log_tau500.dtype,
            device=self.log_tau500.device,
        )
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError(
                "log_tau500 must be a one-dimensional grid with at least two points."
            )
        if not torch.isfinite(grid).all() or not torch.all(grid[1:] > grid[:-1]):
            raise ValueError(
                "log_tau500 must be finite and strictly increase from top to bottom."
            )
        tolerance = 2.0 * torch.finfo(grid.dtype).eps
        if grid[0] < self.log_tau500[0] - tolerance or (
            grid[-1] > self.log_tau500[-1] + tolerance
        ):
            raise ValueError(
                "Evaluation log_tau500 must stay inside the represented depth interval."
            )
        return grid

    def _network_inputs(
        self,
        coords: torch.Tensor,
        log_tau500: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coords.ndim < 1 or coords.shape[-1] != 3:
            raise ValueError(
                f"coords must end in [x, y, time]; got {tuple(coords.shape)}."
            )
        if not torch.is_floating_point(coords):
            coords = coords.float()
        geometric_height = self.depth_to_height(log_tau500, coords.shape[:-1])
        return self._network_inputs_at_height(
            coords, geometric_height
        ), geometric_height

    def _network_inputs_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> torch.Tensor:
        normalized_xy = (
            coords[..., :2] - self.spatial_coordinate_center_mm.to(coords)
        ) / self.spatial_coordinate_scale_mm.to(coords)
        if geometric_height_m.shape[:-1] != coords.shape[:-1]:
            raise ValueError(
                "geometric_height_m must have shape coords.shape[:-1]+[points]."
            )
        depth = geometric_height_m.shape[-1]
        spatial = normalized_xy.unsqueeze(-2).expand(
            *coords.shape[:-1], depth, normalized_xy.shape[-1]
        )
        normalized_height = geometric_height_m / self.height_input_scale_m
        represented = torch.cat((spatial, normalized_height[..., None]), dim=-1)
        if self.time_dependent:
            normalized_time = (
                coords[..., 2:3] - self.time_coordinate_center_hours
            ) / self.time_coordinate_scale_hours
            temporal = normalized_time.unsqueeze(-2).expand(
                *coords.shape[:-1], depth, 1
            )
            represented = torch.cat((represented, temporal), dim=-1)
        return represented

    def depth_to_height(
        self,
        depth_coordinate: torch.Tensor,
        leading_shape=(),
    ) -> torch.Tensor:
        """Map the shell parameter monotonically from outer to inner height.

        In ``physical_shell`` mode the ``log_tau500`` sampling grid
        is only a dimensionless, ordered quadrature parameter.  It has no
        optical-depth meaning; tau500 is derived from the predicted opacity.
        """

        q = torch.as_tensor(
            depth_coordinate, device=self.log_tau500.device, dtype=self.log_tau500.dtype
        )
        reference_height = self.reference_atmosphere.height_from_log_tau(q)
        height = torch.where(
            q <= 0,
            reference_height * self.outer_height_scale,
            reference_height * self.inner_height_scale,
        )
        if q.ndim == 1 and leading_shape:
            height = height.reshape(*([1] * len(leading_shape)), q.numel()).expand(
                *leading_shape, q.numel()
            )
        return height

    def forward(self, coords: torch.Tensor, log_tau500=None) -> StratifiedAtmosphere:
        """Evaluate the continuous atmosphere at an arbitrary depth grid."""

        grid = self._evaluation_grid(log_tau500)
        inputs, geometric_height = self._network_inputs(coords, grid)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        fields = self._decode_raw(raw, geometric_height_m=geometric_height)
        return StratifiedAtmosphere(
            log_tau500=grid,
            geometric_height_m=geometric_height,
            **fields,
        )

    def _decode_raw(
        self,
        raw: torch.Tensor,
        *,
        geometric_height_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode network channels into physical quantities."""

        if geometric_height_m is None:
            raise RuntimeError("Radial-reference decoding requires geometric height.")
        reference_logs = self.reference_atmosphere.logs_at_height(
            geometric_height_m.to(raw)
        )
        temperature_log = reference_logs[0] + (self.temperature_log_scale * raw[..., 0])
        velocity_argument = raw[..., 1:4] * (
            0.5 * math.pi * self.velocity_scale_m_per_s / self.velocity_max_m_per_s
        )
        velocity_field = (
            self.velocity_max_m_per_s * (2.0 / math.pi) * torch.atan(velocity_argument)
        )
        magnetic_field = self.magnetic_scale_gauss * raw[..., 4:7]
        microturbulence_log = reference_logs[2] + (
            self.microturbulence_log_scale * raw[..., 7]
        )
        gas_pressure_log = reference_logs[1] + (
            self.gas_pressure_log_scale * raw[..., 8]
        )
        return {
            "temperature": torch.exp(temperature_log),
            "velocity_field": velocity_field,
            "microturbulence": torch.exp(microturbulence_log),
            "magnetic_field": magnetic_field,
            "gas_pressure": torch.exp(gas_pressure_log),
        }

    def evaluate_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``F(x,y,z)`` at paired physical heights in metres."""

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        inputs = self._network_inputs_at_height(coords, height)
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        return self._decode_raw(raw, geometric_height_m=height)

    def evaluate_chart_height_points(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate paired chart/time coordinates and physical heights.

        Unlike :meth:`evaluate_at_height`, this method does not introduce a
        Cartesian product between coordinates and a trailing height grid.
        Each coordinate is evaluated at exactly its paired height.
        """

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        if coords.shape[-1] != 3 or height.shape != coords.shape[:-1]:
            raise ValueError(
                "Paired chart coordinates must end in three values and height "
                "must match their leading dimensions."
            )
        inputs = self._network_inputs_at_height(coords, height[..., None])[..., 0, :]
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        return self._decode_raw(raw, geometric_height_m=height)

    def position_from_coords_height(
        self, coords: torch.Tensor, geometric_height_m: torch.Tensor
    ) -> torch.Tensor:
        """Map chart coordinates and radial height directly to Cartesian points."""

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        return (
            self._chart_direction(coords[..., :2])
            * (self.solar_radius_m + height)[..., None]
        )

    def _chart_direction(self, chart_xy_mm: torch.Tensor) -> torch.Tensor:
        """Fast inverse gnomonic map for already validated model coordinates."""

        local = torch.cat(
            (
                chart_xy_mm * (1.0e6 / self.solar_radius_m),
                torch.ones_like(chart_xy_mm[..., :1]),
            ),
            dim=-1,
        )
        direction = torch.matmul(local, self.scene_basis.to(local))
        return direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

    def _position_chart_height(
        self, position_rsun: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast chart/height map for trusted nonzero solar-local positions."""

        radius_rsun = torch.linalg.vector_norm(position_rsun, dim=-1)
        direction = position_rsun / radius_rsun[..., None]
        local = torch.matmul(direction, self.scene_basis.to(direction).T)
        chart = self.solar_radius_m * local[..., :2] / local[..., 2:3] / 1.0e6
        height = (radius_rsun - 1.0) * self.solar_radius_m
        return chart, height

    @staticmethod
    def _near_sphere_offset(
        point_rsun: torch.Tensor,
        ray: torch.Tensor,
        radius_rsun: torch.Tensor,
    ) -> torch.Tensor:
        projection = (point_rsun * ray).sum(dim=-1)
        impact = torch.linalg.vector_norm(
            torch.linalg.cross(point_rsun, ray, dim=-1), dim=-1
        )
        return -projection - torch.sqrt(
            (radius_rsun.square() - impact.square()).clamp_min(0.0)
        )

    def evaluate_position_rsun(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the field at Carrington Cartesian positions in solar radii."""

        position = position_rsun.to(
            device=self.log_tau500.device, dtype=self.log_tau500.dtype
        )
        chart, height = self._position_chart_height(position)
        if time_hours is None:
            time = torch.full_like(chart[..., :1], self.time_coordinate_center_hours)
        else:
            time = torch.as_tensor(time_hours, dtype=chart.dtype, device=chart.device)
            if time.ndim == chart.ndim - 1:
                time = time[..., None]
            try:
                time = torch.broadcast_to(time, (*chart.shape[:-1], 1))
            except RuntimeError as error:
                raise ValueError(
                    "time_hours must broadcast to the position leading dimensions."
                ) from error
        coords = torch.cat((chart, time), dim=-1)
        inputs = self._network_inputs_at_height(coords, height[..., None])
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )[..., 0, :]
        return self._decode_raw(raw, geometric_height_m=height)

    def evaluate_position_points(
        self, position_m: torch.Tensor, time_hours: torch.Tensor | float | None = None
    ) -> dict[str, torch.Tensor]:
        """Evaluate the unique solar-Cartesian field at physical 3-D points."""

        position = position_m.to(
            device=self.log_tau500.device, dtype=self.log_tau500.dtype
        )
        return self.evaluate_position_rsun(
            position / self.solar_radius_m, time_hours=time_hours
        )

    def trace_rays(
        self,
        coords: torch.Tensor,
        ray_direction: torch.Tensor,
        log_tau500: torch.Tensor,
    ) -> tuple[StratifiedAtmosphere, RayTraceResult]:
        """Sample physical points along observer rays through the atmosphere."""

        coords = coords.to(device=self.log_tau500.device, dtype=self.log_tau500.dtype)
        if coords.ndim < 1 or coords.shape[-1] != 3 or not torch.isfinite(coords).all():
            raise ValueError("coords must end in finite [x, y, time] values.")
        direction = ray_direction.to(device=coords.device, dtype=coords.dtype)
        direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if not torch.isfinite(direction).all() or torch.any(direction_norm <= 0):
            raise ValueError("Ray directions must be finite non-zero vectors.")
        direction = direction / direction_norm
        grid = self._evaluation_grid(log_tau500)
        if direction.shape != (*coords.shape[:-1], 3):
            raise ValueError("Ray directions must match coords leading dimensions.")

        # coords[..., :2] identifies the ray at the photosphere. In this local
        # Carrington parameterization, positions are O(1) R_sun and offsets are
        # O(shell thickness / R_sun), preserving both in float32.
        surface_reference_rsun = self._chart_direction(coords[..., :2])

        requested_height = self.depth_to_height(grid, coords.shape[:-1])
        # The impact parameter belongs to the ray, not to a depth sample.
        # Compute it once before broadcasting the ray across the shell grid.
        impact = torch.linalg.vector_norm(
            torch.linalg.cross(surface_reference_rsun, direction, dim=-1), dim=-1
        )
        requested_radius = 1.0 + requested_height / self.solar_radius_m
        outer_radius = requested_radius[..., :1]
        inner_radius = requested_radius[..., -1:]
        effective_inner = torch.maximum(
            inner_radius,
            impact[..., None] + self.tangent_margin_m / self.solar_radius_m,
        )
        if torch.any(effective_inner >= outer_radius):
            raise ValueError(
                "At least one ray does not traverse a finite atmospheric shell "
                "inside the configured tangent margin."
            )
        outer_offset = self._near_sphere_offset(
            surface_reference_rsun, direction, outer_radius[..., 0]
        )
        inner_offset = self._near_sphere_offset(
            surface_reference_rsun, direction, effective_inner[..., 0]
        )
        # Parameterize the ray in height units before adding the O(1) solar
        # radius.  Forming ``outer_radius - requested_radius`` and later
        # subtracting ``outer_offset`` again loses precision in float32 when
        # two jittered depth points lie close together.  The algebra below is
        # identical, but every subtraction remains on the atmospheric-shell
        # scale and therefore preserves strictly ordered ray distances.
        outer_height = requested_height[..., :1]
        inner_height = requested_height[..., -1:]
        fraction = (outer_height - requested_height) / (outer_height - inner_height)
        ray_span_rsun = inner_offset - outer_offset
        local_offset_rsun = fraction * ray_span_rsun[..., None]
        offset_rsun = outer_offset[..., None] + local_offset_rsun
        local_distance_m = local_offset_rsun * self.solar_radius_m
        expanded_reference = surface_reference_rsun.unsqueeze(-2).expand(
            *requested_height.shape, 3
        )
        expanded_direction = direction.unsqueeze(-2).expand_as(expanded_reference)
        position_rsun = expanded_reference + offset_rsun[..., None] * expanded_direction
        position = position_rsun * self.solar_radius_m
        chart, actual_height = self._position_chart_height(position_rsun)
        ray_time = coords[..., 2].unsqueeze(-1).expand(position_rsun.shape[:-1])
        evaluation_coords = torch.cat((chart, ray_time[..., None]), dim=-1)
        fields = self.evaluate_chart_height_points(evaluation_coords, actual_height)
        atmosphere = StratifiedAtmosphere(
            log_tau500=grid,
            geometric_height_m=actual_height.to(coords),
            **fields,
        )
        return atmosphere, RayTraceResult(
            position_m=position,
            distance_m=local_distance_m,
            chart_xy_mm=chart,
            geometric_height_m=actual_height,
        )

    def reference_metadata(self) -> dict:
        radial_reference = self.reference_atmosphere.metadata()
        return {
            "name": "radial reference atmosphere plus learned perturbations",
            "reference": radial_reference,
            "provenance_note": (
                "pinned STiC FALC_82 thermodynamics; no magnetic or velocity seed"
            ),
            "network_initialization": (
                "activation-aware variance-preserving random hidden weights, "
                "zero thermodynamic perturbation readouts and random vector readouts"
            ),
            "log_tau500": self.log_tau500.detach().cpu().tolist(),
            "temperature_log_scale": self.temperature_log_scale,
            "temperature_parameterization": (
                "unbounded linear natural-log residual around radial reference"
            ),
            "thermodynamic_eos": self.thermodynamic_eos.metadata(),
            "microturbulence_log_scale": self.microturbulence_log_scale,
            "microturbulence_parameterization": (
                "unbounded linear natural-log residual around radial reference"
            ),
            "gas_pressure_log_scale": self.gas_pressure_log_scale,
            "gas_pressure_parameterization": (
                "unbounded linear natural-log residual around radial reference"
            ),
            "line_formation_height_bounds_Mm": list(
                self.line_formation_height_bounds_Mm
            ),
            "full_domain_height_bounds_Mm": list(self.shell_height_bounds_Mm),
            "top_boundary_reference": {
                "source": (
                    "hydrostatic FALC-to-corona reference"
                    if self.extrapolation_enabled
                    else "FALC radial reference"
                ),
                "gas_pressure_pa": float(
                    torch.exp(self.top_boundary_reference_log_pressure)
                ),
            },
            "velocity_scale_m_per_s": self.velocity_scale_m_per_s,
            "velocity_max_m_per_s": self.velocity_max_m_per_s,
            "magnetic_scale_gauss": self.magnetic_scale_gauss,
            "vector_decoder": (
                "velocity uses a smoothly bounded component-wise arctangent output "
                "with the configured local linear scale and nonzero tail gradients; "
                "magnetic field remains an unbounded component-wise linear output; "
                "output weights and biases remain randomly initialized; both vectors "
                "always have three components"
            ),
            "velocity_observability": (
                "single-view LTE Stokes synthesis constrains only the negative "
                "projection onto the toward-observer axis (positive redshift); "
                "the two observer-transverse combinations require additional "
                "dynamical physics"
            ),
            "velocity_frame_convention": (
                "physical velocity projected to the observer, with components "
                "expressed in the instantaneous Carrington Cartesian basis; it is "
                "not a Carrington coordinate time derivative"
            ),
            "depth_coordinate_role": "ordered computational shell parameter; tau500 is derived from opacity",
            "shell_height_bounds_Mm": list(self.shell_height_bounds_Mm),
            "tangent_margin_m": self.tangent_margin_m,
            "spherical_geometry": {
                "enabled": True,
                "solar_radius_m": self.solar_radius_value_m,
                "reference_frame": "heliocentric Heliographic Carrington Cartesian (co-rotating)",
                "ray_geometry_units": (
                    "dimensionless solar radii with signed offsets from the "
                    "photospheric ray point"
                ),
                "scene_basis_rows_solar_cartesian": self.scene_basis.detach()
                .cpu()
                .tolist(),
                "field_vector_basis": "Heliographic Carrington Cartesian [Xc,Yc,Zc]",
            },
            "spatial_coordinates": {
                "public_input": "observer-independent Carrington gnomonic chart in Mm",
                "network_center_mm": self.spatial_coordinate_center_mm.detach()
                .cpu()
                .tolist(),
                "network_scale_mm": self.spatial_coordinate_scale_mm.detach()
                .cpu()
                .tolist(),
                "network_transform": "(xy_mm - center_mm) / scale_mm",
            },
            "network_inputs": list(self.network_input_names),
            "time_dependent": self.time_dependent,
            "time_network_transform": (
                "(time_hours-time_coordinate_center_hours)/time_coordinate_scale_hours"
                if self.time_dependent
                else "time coordinate omitted (static inversion)"
            ),
            "time_coordinate_center_hours": self.time_coordinate_center_hours,
            "time_coordinate_scale_hours": self.time_coordinate_scale_hours,
            "network_outputs": list(self.output_names),
            "depth_network_transform": "z_normalized = z_m / height_input_scale_m",
            "height_input_scale_m": self.height_input_scale_m,
        }
