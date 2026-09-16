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
from numbers import Real
from typing import Mapping

import torch
from torch import nn

from prom3theus.core import (
    ATOMIC_MASS_UNIT,
    ELECTRON_VOLT,
    H_PLANCK,
    K_BOLTZMANN,
    M_ELECTRON,
    MLPModel,
    SIRENModel,
    SPEED_OF_LIGHT,
)
from prom3theus.resources import verify_manifest_resource
from .atomic import AtomicDatabase
from .plasma import SolarPlasmaTable
from .geometry import (
    RayTraceResult,
    validate_scene_basis,
)


# Hydrogen ionization energy expressed as a temperature, for the partial
# ionization adiabatic gradient below.
HYDROGEN_IONIZATION_K = 13.598_434_599_702 * ELECTRON_VOLT / K_BOLTZMANN

# Height step of the adiabatic interior continuation.  Two kilometres resolves
# the sub-photospheric adiabat far past convergence while keeping the one-time
# construction of even a several-megametre shell inexpensive.
INTERIOR_CONTINUATION_STEP_M = 2.0e3


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


def _hydrogen_ionization_fraction(
    thermodynamic_eos: SolarPlasmaTable,
    temperature: torch.Tensor,
    gas_pressure: torch.Tensor,
) -> torch.Tensor:
    """Return the LTE hydrogen ionization degree.

    Saha, closed with the hydrogen nucleus density from the shared EoS mass
    density.  Inside the STiC table this reproduces the tabulated electron
    density to better than one percent, so it is not a competing ionization
    balance: it is that same LTE physics continued past the table's 10 kK
    ceiling.  Reading the runtime electron density instead would be wrong
    exactly where the interior continuation needs it, because above 10 kK the
    EoS blends toward CHIANTI coronal equilibrium -- right for the corona, and
    several-fold too neutral for a dense convective interior, which would
    leave the adiabat far too steep.

    That ceiling cannot be lifted by regenerating the table.  It is set by the
    STiC/CHIANTI charge bridge, not by the continuum package: at the table's
    thin, hot corner LTE is already fully ionized while coronal equilibrium is
    not, so the bridge's monotone logit margin peaks at exactly 10 kK and turns
    negative by 12.6 kK.  This closure is therefore permanent, not a stopgap.
    """

    mass_density = thermodynamic_eos.mass_density(temperature, gas_pressure)
    hydrogen_density = mass_density / (
        thermodynamic_eos.eos.mass_u_per_h_nucleus * ATOMIC_MASS_UNIT
    )
    # Saha with the ground-state partition functions U(H II)/U(H I) = 1/2
    # cancelling the ionized-state spin degeneracy of two.
    equilibrium = (
        (2.0 * math.pi * M_ELECTRON * K_BOLTZMANN / H_PLANCK**2) * temperature
    ).pow(1.5) * torch.exp(-HYDROGEN_IONIZATION_K / temperature)
    ratio = equilibrium / hydrogen_density
    return 0.5 * (torch.sqrt(ratio.square() + 4.0 * ratio) - ratio)


def _adiabatic_gradient(
    thermodynamic_eos: SolarPlasmaTable,
    temperature: torch.Tensor,
    gas_pressure: torch.Tensor,
) -> torch.Tensor:
    """Return ``dlnT/dlnP`` along an adiabat for a partly ionized hydrogen gas.

    This is the Kippenhahn & Weigert result: latent heat of ionization flattens
    the adiabat wherever hydrogen is partly ionized, and the expression returns
    to the ideal ``2/5`` where the gas is either neutral or fully ionized.
    Assuming the ideal ``2/5`` instead would make the continuation roughly
    twice too steep through the hydrogen ionization zone, which is exactly the
    region a sub-photospheric shell floor reaches.
    """

    fraction = _hydrogen_ionization_fraction(
        thermodynamic_eos, temperature, gas_pressure
    )
    neutral_ionized_product = fraction * (1.0 - fraction)
    excitation = 2.5 + HYDROGEN_IONIZATION_K / temperature
    return (2.0 + neutral_ionized_product * excitation) / (
        5.0 + neutral_ionized_product * excitation.square()
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
        minimum_height_m: float | None = None,
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
        self.lower_atmosphere_metadata = None
        base_bottom_m = float(thermodynamic_height[0])
        if minimum_height_m is not None and float(minimum_height_m) < base_bottom_m:
            if solar_radius_m is None or thermodynamic_eos is None:
                raise ValueError(
                    "An adiabatic interior continuation requires the solar radius "
                    "and thermodynamic EoS."
                )
            minimum_height_m = float(minimum_height_m)
            if not math.isfinite(minimum_height_m):
                raise ValueError("minimum_height_m must be finite.")
            if minimum_height_m <= -float(solar_radius_m):
                raise ValueError("minimum_height_m must stay above the solar centre.")
            interval_count = max(
                1,
                math.ceil(
                    (base_bottom_m - minimum_height_m) / INTERIOR_CONTINUATION_STEP_M
                ),
            )
            interior_height = torch.linspace(
                base_bottom_m,
                minimum_height_m,
                interval_count + 1,
                dtype=height.dtype,
            )
            reference_gravity = thermodynamic_eos.reference_gravity_m_per_s2
            solar_radius = float(solar_radius_m)
            log_temperature = thermodynamic_logs[0, 0]
            log_pressure = thermodynamic_logs[0, 1]
            interior_logs = []

            def interior_derivatives(log_t, log_p, gravity):
                """Return ``(dlnT/dh, dlnP/dh)`` for a hydrostatic adiabat."""

                temperature_k = torch.exp(log_t)[None]
                pressure_pa = torch.exp(log_p)[None]
                mean_particle_mass_u = thermodynamic_eos.mean_molecular_weight(
                    temperature_k, pressure_pa
                )[0]
                pressure_slope = -(
                    mean_particle_mass_u
                    * ATOMIC_MASS_UNIT
                    * gravity
                    / (K_BOLTZMANN * temperature_k[0])
                )
                gradient = _adiabatic_gradient(
                    thermodynamic_eos, temperature_k, pressure_pa
                )[0]
                return gradient * pressure_slope, pressure_slope

            # Midpoint marching, matching the hydrostatic upper extension: the
            # gradient and mean particle mass both vary strongly through the
            # hydrogen ionization zone, so a single endpoint evaluation per
            # step would bias the whole stratification.
            for index in range(interval_count):
                step_m = interior_height[index + 1] - interior_height[index]
                midpoint_height = 0.5 * (
                    interior_height[index] + interior_height[index + 1]
                )
                gravity = (
                    reference_gravity
                    * (solar_radius / (solar_radius + float(midpoint_height))) ** 2
                )
                predicted_t, predicted_p = interior_derivatives(
                    log_temperature, log_pressure, gravity
                )
                midpoint_t, midpoint_p = interior_derivatives(
                    log_temperature + 0.5 * predicted_t * step_m,
                    log_pressure + 0.5 * predicted_p * step_m,
                    gravity,
                )
                log_temperature = log_temperature + midpoint_t * step_m
                log_pressure = log_pressure + midpoint_p * step_m
                interior_logs.append(
                    torch.stack(
                        (log_temperature, log_pressure, thermodynamic_logs[0, 2])
                    )
                )
            # Built from the top down, but the stored table ascends in height.
            interior_logs = torch.stack(interior_logs).flip(0)
            interior_height = interior_height[1:].flip(0)
            thermodynamic_height = torch.cat((interior_height, thermodynamic_height))
            thermodynamic_logs = torch.cat((interior_logs, thermodynamic_logs))
            self.lower_atmosphere_metadata = {
                "type": "adiabatic_interior",
                "native_atmosphere_bottom_m": base_bottom_m,
                "native_atmosphere_bottom_temperature_k": float(
                    torch.exp(thermodynamic_logs[interval_count, 0])
                ),
                "minimum_height_m": minimum_height_m,
                "height_step_m": INTERIOR_CONTINUATION_STEP_M,
                "interval_count": interval_count,
                "temperature": (
                    "adiabatic dlnT/dlnP for a partly ionized hydrogen gas "
                    "(Kippenhahn & Weigert), with the LTE Saha ionization "
                    "degree closed on the shared EoS mass density"
                ),
                "gravity": "spherical inverse-square",
                "pressure": "hydrostatic integral using the hybrid solar EoS",
                "microturbulence": "native FALC-bottom value held constant",
                "limitation": (
                    "a strict adiabat; the superadiabatic excess of the upper "
                    "convection zone is not reproduced"
                ),
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
            "lower_atmosphere": self.lower_atmosphere_metadata,
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


def _batched_vector_jacobian(
    vector: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Differentiate one three-vector output independently at each point."""

    if (
        vector.shape != coordinates.shape
        or coordinates.ndim != 2
        or coordinates.shape[-1] != 3
    ):
        raise ValueError("Vector-potential differentiation requires [point, 3] tensors.")
    if not vector.requires_grad:
        return coordinates.new_zeros((coordinates.shape[0], 3, 3))
    basis = torch.eye(3, dtype=vector.dtype, device=vector.device)
    grad_outputs = basis[:, None, :].expand(3, coordinates.shape[0], 3)
    jacobian = torch.autograd.grad(
        vector,
        coordinates,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
        is_grads_batched=True,
    )[0]
    if jacobian is None:
        return coordinates.new_zeros((coordinates.shape[0], 3, 3))
    return jacobian.movedim(0, 1)


def _scalar_gradient(
    scalar: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Differentiate one scalar output independently at each point."""

    if (
        scalar.shape != coordinates.shape[:-1]
        or coordinates.ndim != 2
        or coordinates.shape[-1] != 3
    ):
        raise ValueError(
            "Scalar-potential differentiation requires a [point] scalar and a "
            "[point, 3] coordinate tensor."
        )
    if not scalar.requires_grad:
        return coordinates.new_zeros((coordinates.shape[0], 3))
    gradient = torch.autograd.grad(
        scalar,
        coordinates,
        grad_outputs=torch.ones_like(scalar),
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )[0]
    if gradient is None:
        return coordinates.new_zeros((coordinates.shape[0], 3))
    return gradient


def _curl_from_jacobian(jacobian: torch.Tensor) -> torch.Tensor:
    """Return curl for a ``[..., vector component, coordinate]`` Jacobian."""

    return torch.stack(
        (
            jacobian[..., 2, 1] - jacobian[..., 1, 2],
            jacobian[..., 0, 2] - jacobian[..., 2, 0],
            jacobian[..., 1, 0] - jacobian[..., 0, 1],
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class StratifiedAtmosphere:
    """Physical atmosphere sampled at ordered points along a path.

    Parameters
    ----------
    depth_coordinate:
        Ordered labels along the path, shape ``[depth]``. Physical shell rays
        use sample indices; only an explicit optical-depth transfer path gives
        these labels the meaning ``log10(tau_500)``.
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
    vector_potential:
        Optional magnetic vector potential in gauss-metres, shape
        ``[..., depth, 3]``. It is present when the atmosphere model uses the
        vector-potential magnetic representation.
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

    depth_coordinate: torch.Tensor
    temperature: torch.Tensor
    velocity_field: torch.Tensor
    microturbulence: torch.Tensor
    magnetic_field: torch.Tensor
    vector_potential: torch.Tensor | None = None
    gas_pressure: torch.Tensor | None = None
    geometric_height_m: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.depth_coordinate.ndim != 1:
            raise ValueError("depth_coordinate must be one-dimensional.")
        if self.depth_coordinate.numel() < 2:
            raise ValueError("depth_coordinate must contain at least two depth points.")
        if self.depth_coordinate.dtype not in (torch.float32, torch.float64):
            raise TypeError("depth_coordinate must use float32 or float64.")
        if not torch.isfinite(self.depth_coordinate).all() or not torch.all(
            self.depth_coordinate[1:] > self.depth_coordinate[:-1]
        ):
            raise ValueError(
                "depth_coordinate must be finite and strictly increase from top to bottom."
            )
        if self.temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError("Atmospheric fields must use float32 or float64.")

        depth = self.depth_coordinate.numel()
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
        if self.vector_potential is not None:
            if self.vector_potential.shape != expected_vector_shape:
                raise ValueError(
                    "vector_potential must have shape [..., depth, 3]; "
                    f"expected {expected_vector_shape}, got {tuple(self.vector_potential.shape)}."
                )
            if self.vector_potential.device != self.temperature.device:
                raise ValueError(
                    "vector_potential and temperature must be on the same device."
                )
            if self.vector_potential.dtype != self.temperature.dtype:
                raise TypeError(
                    "vector_potential and temperature must use the same dtype."
                )
        if self.depth_coordinate.device != self.temperature.device:
            raise ValueError(
                "depth_coordinate and atmospheric fields must be on the same device."
            )
        if self.depth_coordinate.dtype != self.temperature.dtype:
            raise TypeError(
                "depth_coordinate and atmospheric fields must use the same dtype."
            )

    @property
    def depth(self) -> int:
        return int(self.depth_coordinate.numel())

    @property
    def batch_shape(self) -> torch.Size:
        return self.temperature.shape[:-1]

    @property
    def v_los(self) -> torch.Tensor:
        """Positive-redshift LOS speed used by the radiative-transfer solver."""

        return -self.velocity_field[..., 2]


class StratifiedAtmosphereModel(nn.Module):
    """Coordinate network producing a smooth stratified atmosphere.

    The operational SIREN uses sinusoidal activations and canonical SIREN
    initialization, with non-radial coordinate bandwidth tapered by radius.
    Thermodynamic readout rows start at zero so the initial state equals the
    radial reference; vector readout rows remain random.
    Positive thermodynamic variables use unbounded, reference-anchored linear
    residuals in natural-log space.  Finite-domain radiative-transfer tables
    validate their own inputs independently of this primary atmosphere model.
    Training-side ``evaluate_position_rsun_normalized`` exposes the fixed
    model-unit contract; the physical ``evaluate_position_*`` methods are
    adapters for radiative transfer and external observations.
    Magnetic vectors use either a direct Cartesian decoder or a Cartesian
    vector-potential decoder. In the latter mode, the three magnetic readout
    channels are an ``A`` field in G m and the exposed magnetic field is the
    physical Cartesian curl ``B = curl(A)`` in G.
    Velocity components use a smooth arctangent bound with the configured
    local linear scale, preventing invalid relativistic Doppler factors while
    retaining nonzero gradients outside the ordinary photospheric range.
    """

    _DIRECT_OUTPUT_NAMES = (
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
    _VECTOR_POTENTIAL_OUTPUT_NAMES = (
        "temperature",
        "v_x",
        "v_y",
        "v_z",
        "a_x",
        "a_y",
        "a_z",
        "microturbulence",
        "gas_pressure",
    )
    # Same layout as the direct representation (b_x, b_y, b_z is read as the
    # delta-field contribution), plus one trailing scalar-potential channel.
    _POTENTIAL_DELTA_OUTPUT_NAMES = _DIRECT_OUTPUT_NAMES + ("psi",)
    output_names = _DIRECT_OUTPUT_NAMES

    def __init__(
        self,
        *,
        shell_height_bounds_Mm: tuple[float, float],
        temperature_log_scale: float = 0.2,
        velocity_scale_m_per_s: float = 1_000.0,
        velocity_max_m_per_s: float = 100_000.0,
        magnetic_scale_gauss: float = 100.0,
        magnetic_representation: str = "direct",
        magnetic_reference_height_megameter: float | None = None,
        magnetic_potential_delta_cool_steps: int = 0,
        magnetic_potential_delta_ramp_steps: int = 0,
        gas_pressure_log_scale: float = 1.0,
        spatial_coordinate_center_mm=(0.0, 0.0),
        spatial_coordinate_scale_mm=(1.0, 1.0),
        height_input_scale_m: float = 1_000_000.0,
        uniform_spatial_scaling: bool = False,
        time_dependent: bool = False,
        time_coordinate_center_hours: float = 0.0,
        time_coordinate_scale_hours: float = 1.0,
        line_formation_height_bounds_Mm: tuple[float, float] | None = None,
        tangent_margin_m: float = 1_000.0,
        reference_atmosphere_config: str | Mapping | None = "falc_82",
        upper_atmosphere_config: Mapping | None = None,
        microturbulence_log_scale: float = 0.2,
        scene_geometry_config: Mapping | None = None,
        model_config: Mapping | None = None,
    ):
        super().__init__()
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

        self.temperature_log_scale = float(temperature_log_scale)
        self.velocity_scale_m_per_s = float(velocity_scale_m_per_s)
        self.velocity_max_m_per_s = float(velocity_max_m_per_s)
        self.magnetic_scale_gauss = float(magnetic_scale_gauss)
        if magnetic_representation not in {
            "direct",
            "vector_potential",
            "potential_delta",
        }:
            raise ValueError(
                "magnetic_representation must be 'direct', 'vector_potential', "
                "or 'potential_delta'."
            )
        self.magnetic_representation = magnetic_representation
        self.output_names = {
            "vector_potential": self._VECTOR_POTENTIAL_OUTPUT_NAMES,
            "potential_delta": self._POTENTIAL_DELTA_OUTPUT_NAMES,
        }.get(magnetic_representation, self._DIRECT_OUTPUT_NAMES)
        for name, value in (
            ("magnetic_potential_delta_cool_steps", magnetic_potential_delta_cool_steps),
            ("magnetic_potential_delta_ramp_steps", magnetic_potential_delta_ramp_steps),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        self.magnetic_potential_delta_cool_steps = magnetic_potential_delta_cool_steps
        self.magnetic_potential_delta_ramp_steps = magnetic_potential_delta_ramp_steps
        self.register_buffer(
            "current_step", torch.zeros((), dtype=torch.long), persistent=False
        )
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
        self.vector_potential_scale_gauss_m = (
            self.magnetic_scale_gauss * self.height_input_scale_m
        )
        if type(uniform_spatial_scaling) is not bool:
            raise TypeError("uniform_spatial_scaling must be boolean")
        self.uniform_spatial_scaling = uniform_spatial_scaling
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
        self.register_buffer(
            "solar_radius_m",
            torch.tensor(solar_radius_m, dtype=torch.get_default_dtype()),
        )
        # Geometry used by the normalized training representation.  Operators
        # such as div/curl differentiate with respect to x_hat = x / L0, so
        # the only spherical conversion they need is the dimensionless ratio
        # R_sun / L0.
        self.register_buffer(
            "solar_radius_model",
            self.solar_radius_m / self.height_input_scale_m,
        )
        self.solar_radius_value_m = solar_radius_m
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
        self.shell_height_bounds_Mm = shell_height_bounds_Mm
        if (
            self.magnetic_representation in ("vector_potential", "potential_delta")
            and magnetic_reference_height_megameter is not None
        ):
            raise ValueError(
                "magnetic_reference_height_megameter is incompatible with the "
                f"{self.magnetic_representation!r} magnetic representation."
            )
        if magnetic_reference_height_megameter is not None:
            if isinstance(magnetic_reference_height_megameter, bool) or not isinstance(
                magnetic_reference_height_megameter, Real
            ):
                raise TypeError("magnetic_reference_height_megameter must be numeric or null")
            magnetic_reference_height_megameter = float(magnetic_reference_height_megameter)
            if not math.isfinite(magnetic_reference_height_megameter) or not (
                shell_height_bounds_Mm[1]
                <= magnetic_reference_height_megameter
                <= shell_height_bounds_Mm[0]
            ):
                raise ValueError("magnetic_reference_height_megameter must be finite and lie within the shell")
        self.magnetic_reference_height_megameter = magnetic_reference_height_megameter
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
        if self.extrapolation_enabled and upper_atmosphere_config is None:
            raise ValueError("An extrapolated shell requires upper_atmosphere_config.")
        if not self.extrapolation_enabled and upper_atmosphere_config is not None:
            raise ValueError("upper_atmosphere_config requires an extrapolated shell.")
        # The shell floor routinely sits below the FALC table, whose deepest
        # node is only about 69 km under the surface.  Hand the floor to the
        # reference so it continues adiabatically instead of extrapolating the
        # tabulated gradient, which reaches absurd interior temperatures within
        # a few hundred kilometres.
        self.reference_atmosphere = RadialReferenceAtmosphere(
            reference_atmosphere_config,
            upper_atmosphere_config=upper_atmosphere_config,
            maximum_height_m=(
                shell_height_bounds_Mm[0] * 1.0e6
                if self.extrapolation_enabled
                else None
            ),
            minimum_height_m=shell_height_bounds_Mm[1] * 1.0e6,
            solar_radius_m=solar_radius_m,
            thermodynamic_eos=self.thermodynamic_eos,
            atomic_database=atomic_database,
        )
        self.transition_region_top_m = (
            float(upper_atmosphere_config["transition_region_top_megameter"]) * 1.0e6
            if self.extrapolation_enabled
            else None
        )

        # Fixed scales for the dimensionless training contract.  These are
        # global reference scales, not local base-profile normalizers: actual
        # learned T and P are divided by these constants, and the EOS/table
        # adapter converts them to SI only while performing its query.
        reference_scale_logs = self.reference_atmosphere.logs_at_height(
            self.solar_radius_m.new_zeros(())
        )
        self.temperature_scale_k = float(torch.exp(reference_scale_logs[0]).item())
        self.gas_pressure_scale_pa = float(torch.exp(reference_scale_logs[1]).item())
        self.microturbulence_scale_m_per_s = float(
            torch.exp(reference_scale_logs[2]).item()
        )
        with torch.no_grad():
            scale_temperature = torch.as_tensor(
                self.temperature_scale_k,
                dtype=self.solar_radius_m.dtype,
            )
            scale_pressure = torch.as_tensor(
                self.gas_pressure_scale_pa,
                dtype=self.solar_radius_m.dtype,
            )
            density_scale = self.thermodynamic_eos.mass_density(
                scale_temperature,
                scale_pressure,
            )
            electron_density_scale = self.thermodynamic_eos.electron_density(
                scale_temperature,
                scale_pressure,
            )
        self.density_scale_kg_m3 = float(density_scale.detach().item())
        self.electron_density_scale_m3 = float(electron_density_scale.detach().item())
        top_height = self.solar_radius_m.new_tensor(shell_height_bounds_Mm[0] * 1.0e6)
        top_reference_logs = self.reference_atmosphere.logs_at_height(top_height)
        if top_reference_logs[1] <= math.log(
            torch.finfo(self.solar_radius_m.dtype).tiny
        ):
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
        model_type = config.pop("type", None)
        self.network_type = model_type
        if model_type == "mlp":
            expected_fields = {"dim", "n_layers", "activation", "encoding_config"}
            if set(config) != expected_fields:
                raise TypeError(
                    "MLP model_config must contain exactly dim, n_layers, activation, "
                    "and encoding_config."
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
        elif model_type == "siren":
            expected_fields = {
                "dim",
                "n_layers",
                "first_omega_0",
                "hidden_omega_0",
                "radial_weighting_config",
            }
            if set(config) != expected_fields:
                raise TypeError(
                    "SIREN model_config must contain exactly dim, n_layers, "
                    "first_omega_0, hidden_omega_0, and radial_weighting_config."
                )
            radial_weighting = config.pop("radial_weighting_config")
            if radial_weighting is not None:
                if not isinstance(radial_weighting, Mapping):
                    raise TypeError("radial_weighting_config must be a mapping or null.")
                radial_weighting = dict(radial_weighting)
                radial_weighting.update(
                    radial_dimension=2,
                    radial_bounds=(
                        shell_height_bounds_Mm[1] * 1.0e6 / self.height_input_scale_m,
                        shell_height_bounds_Mm[0] * 1.0e6 / self.height_input_scale_m,
                    ),
                )
            self.network = SIRENModel(
                in_dim=len(self.network_input_names),
                out_dim=len(self.output_names),
                radial_weighting_config=radial_weighting,
                **config,
            )
        else:
            raise ValueError(
                "StratifiedAtmosphereModel supports model types 'siren' and legacy "
                "'mlp'."
            )
        # Exact initial radial thermodynamics; velocity and magnetic rows retain
        # random readouts so polarized gradients do not start at a symmetry point.
        with torch.no_grad():
            for index in (0, 7, 8):
                self.network.out_layer.weight[index].zero_()
                self.network.out_layer.bias[index].zero_()

    def set_step(self, step: int) -> None:
        """Set the current training step for the ``potential_delta`` alpha ramp."""

        if type(step) is not int or step < 0:
            raise ValueError("Atmosphere model step must be a non-negative integer.")
        self.current_step.fill_(step)

    def _potential_delta_alpha(self) -> float:
        """Return the current delta-field contribution weight in [0, 1].

        Zero for ``magnetic_potential_delta_cool_steps``, then ramps linearly
        to one over the following ``magnetic_potential_delta_ramp_steps``.
        """

        step = int(self.current_step)
        cool_steps = self.magnetic_potential_delta_cool_steps
        if step < cool_steps:
            return 0.0
        ramp_steps = self.magnetic_potential_delta_ramp_steps
        if ramp_steps <= 0:
            return 1.0
        return min(1.0, (step - cool_steps) / ramp_steps)

    def _height_grid(self, geometric_height_m) -> torch.Tensor:
        """Validate physical sampling heights, ordered from outer to inner."""
        height = torch.as_tensor(geometric_height_m).to(self.solar_radius_m)
        if height.ndim != 1 or height.numel() < 2:
            raise ValueError(
                "geometric_height_m must be a one-dimensional grid with at least two points."
            )
        if not torch.isfinite(height).all() or not torch.all(height[1:] < height[:-1]):
            raise ValueError(
                "geometric_height_m must be finite and strictly decrease inward."
            )
        outer, inner = self.shell_height_bounds_Mm
        tolerance = (
            2 * torch.finfo(height.dtype).eps * max(abs(outer), abs(inner)) * 1e6
        )
        if height[0] > outer * 1e6 + tolerance or height[-1] < inner * 1e6 - tolerance:
            raise ValueError("Evaluation heights must stay inside the spherical shell.")
        return height

    def _network_inputs_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> torch.Tensor:
        if coords.ndim < 1 or coords.shape[-1] != 3:
            raise ValueError("coords must end in [x, y, time].")
        # Scene chart geometry is unchanged. Only neural input normalization
        # uses the same physical length for x, y, and radial height.
        spatial_scale_mm = (
            self.height_input_scale_m / 1e6
            if getattr(self, "uniform_spatial_scaling", False)
            else self.spatial_coordinate_scale_mm.to(coords)
        )
        normalized_xy = (
            coords[..., :2] - self.spatial_coordinate_center_mm.to(coords)
        ) / spatial_scale_mm
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

    def _network_raw(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate stratified channels and an optional fixed-height magnetic readout.

        The magnetic reference preserves the input angular position and time;
        only its height coordinate is replaced. Thus the native Cartesian field
        is constant along a radial column, while oblique rays can still cross
        angular structure. The default path performs the unchanged single query.
        """
        raw = self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(
            *inputs.shape[:-1], -1
        )
        if self.magnetic_reference_height_megameter is None:
            return raw
        reference_height = self.magnetic_reference_height_megameter * 1e6 / self.height_input_scale_m
        magnetic_inputs = torch.cat((
            inputs[..., :2],
            torch.full_like(inputs[..., 2:3], reference_height),
            inputs[..., 3:],
        ), dim=-1)
        magnetic_raw = self.network(magnetic_inputs.reshape(-1, magnetic_inputs.shape[-1])).reshape(
            *magnetic_inputs.shape[:-1], -1
        )
        return torch.cat((raw[..., :4], magnetic_raw[..., 4:7], raw[..., 7:]), dim=-1)

    def forward(self, coords: torch.Tensor, geometric_height_m) -> StratifiedAtmosphere:
        """Evaluate the continuous atmosphere at explicitly supplied physical heights."""
        coords = coords.to(self.solar_radius_m)
        height = self._height_grid(geometric_height_m)
        geometric_height = height.expand(*coords.shape[:-1], height.numel())
        fields = self.evaluate_at_height(coords, geometric_height)
        return StratifiedAtmosphere(
            depth_coordinate=torch.arange(
                height.numel(), device=height.device, dtype=height.dtype
            ),
            geometric_height_m=geometric_height,
            **fields,
        )

    def _decode_raw(
        self,
        raw: torch.Tensor,
        *,
        geometric_height_m: torch.Tensor | None = None,
        magnetic_field_gauss: torch.Tensor | None = None,
        vector_potential: torch.Tensor | None = None,
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
        if self.magnetic_representation == "vector_potential":
            if magnetic_field_gauss is None or vector_potential is None:
                raise RuntimeError(
                    "Vector-potential decoding requires the derived magnetic field "
                    "and vector potential."
                )
            magnetic_field = magnetic_field_gauss
        elif self.magnetic_representation == "potential_delta":
            if magnetic_field_gauss is None:
                raise RuntimeError(
                    "Potential-plus-delta decoding requires the derived magnetic field."
                )
            magnetic_field = magnetic_field_gauss
        else:
            magnetic_field = self.magnetic_scale_gauss * raw[..., 4:7]
        microturbulence_log = reference_logs[2] + (
            self.microturbulence_log_scale * raw[..., 7]
        )
        gas_pressure_log = reference_logs[1] + (
            self.gas_pressure_log_scale * raw[..., 8]
        )
        fields = {
            "temperature": torch.exp(temperature_log),
            "velocity_field": velocity_field,
            "microturbulence": torch.exp(microturbulence_log),
            "magnetic_field": magnetic_field,
            "gas_pressure": torch.exp(gas_pressure_log),
        }
        if vector_potential is not None:
            fields["vector_potential"] = vector_potential
        return fields

    def _decode_raw_normalized(
        self,
        raw: torch.Tensor,
        *,
        geometric_height_m: torch.Tensor | None = None,
        magnetic_field_model: torch.Tensor | None = None,
        vector_potential_model: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode network channels directly into dimensionless model units."""

        if geometric_height_m is None:
            raise RuntimeError("Radial-reference decoding requires geometric height.")
        reference_logs = self.reference_atmosphere.logs_at_height(
            geometric_height_m.to(raw)
        )
        temperature = (
            torch.exp(
                reference_logs[0]
                - math.log(self.temperature_scale_k)
                + self.temperature_log_scale * raw[..., 0]
            )
        )
        velocity_argument = raw[..., 1:4] * (
            0.5 * math.pi * self.velocity_scale_m_per_s / self.velocity_max_m_per_s
        )
        velocity = (
            self.velocity_max_m_per_s
            / self.velocity_scale_m_per_s
            * (2.0 / math.pi)
            * torch.atan(velocity_argument)
        )
        if self.magnetic_representation == "vector_potential":
            if magnetic_field_model is None or vector_potential_model is None:
                raise RuntimeError(
                    "Vector-potential decoding requires the derived normalized "
                    "magnetic field and vector potential."
                )
            magnetic = magnetic_field_model
        elif self.magnetic_representation == "potential_delta":
            if magnetic_field_model is None:
                raise RuntimeError(
                    "Potential-plus-delta decoding requires the derived normalized "
                    "magnetic field."
                )
            magnetic = magnetic_field_model
        else:
            magnetic = raw[..., 4:7]
        microturbulence = (
            torch.exp(
                reference_logs[2]
                - math.log(self.microturbulence_scale_m_per_s)
                + self.microturbulence_log_scale * raw[..., 7]
            )
        )
        gas_pressure = (
            torch.exp(
                reference_logs[1]
                - math.log(self.gas_pressure_scale_pa)
                + self.gas_pressure_log_scale * raw[..., 8]
            )
        )
        fields = {
            "temperature": temperature,
            "velocity_field": velocity,
            "microturbulence": microturbulence,
            "magnetic_field": magnetic,
            "gas_pressure": gas_pressure,
            "geometric_height_m": geometric_height_m / self.height_input_scale_m,
        }
        if vector_potential_model is not None:
            fields["vector_potential"] = vector_potential_model
        return fields

    def evaluate_at_height(
        self,
        coords: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``F(x,y,z)`` at paired physical heights in metres."""

        coords = coords.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        if self.magnetic_representation in ("vector_potential", "potential_delta"):
            if height.ndim < 1 or height.shape[:-1] != coords.shape[:-1]:
                raise ValueError(
                    "geometric_height_m must have shape coords.shape[:-1]+[points]."
                )
            depth = height.shape[-1]
            paired_coords = coords.unsqueeze(-2).expand(
                *coords.shape[:-1], depth, coords.shape[-1]
            )
            fields = self.evaluate_chart_height_points(
                paired_coords.reshape(-1, paired_coords.shape[-1]),
                height.reshape(-1),
            )
            return {
                name: value.reshape(
                    (*coords.shape[:-1], depth, *value.shape[1:])
                )
                for name, value in fields.items()
            }
        inputs = self._network_inputs_at_height(coords, height)
        raw = self._network_raw(inputs)
        return self._decode_raw(
            raw,
            geometric_height_m=height,
        )

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

        coords = coords.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        height = geometric_height_m.to(device=coords.device, dtype=coords.dtype)
        if coords.shape[-1] != 3 or height.shape != coords.shape[:-1]:
            raise ValueError(
                "Paired chart coordinates must end in three values and height "
                "must match their leading dimensions."
            )
        if self.magnetic_representation in ("vector_potential", "potential_delta"):
            position = self.position_from_coords_height(coords, height)
            return self.evaluate_position_rsun(
                position / self.solar_radius_m,
                time_hours=coords[..., 2],
            )
        inputs = self._network_inputs_at_height(coords, height[..., None])[..., 0, :]
        raw = self._network_raw(inputs)
        return self._decode_raw(
            raw,
            geometric_height_m=height,
        )

    def position_from_coords_height(
        self, coords: torch.Tensor, geometric_height_m: torch.Tensor
    ) -> torch.Tensor:
        """Map chart coordinates and radial height directly to Cartesian points."""

        coords = coords.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
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

    def _evaluate_vector_potential_only(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Evaluate the physical vector potential without taking its curl."""

        if self.magnetic_representation != "vector_potential":
            raise RuntimeError(
                "Vector-potential evaluation requires the vector_potential representation."
            )
        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        output_shape = position.shape[:-1]
        need_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            flat_position = position.reshape(-1, 3)
            if not flat_position.requires_grad:
                flat_position = flat_position.detach().clone().requires_grad_(True)
            chart, height = self._position_chart_height(flat_position)
            if time_hours is None:
                time = torch.full(
                    (*output_shape, 1),
                    self.time_coordinate_center_hours,
                    dtype=position.dtype,
                    device=position.device,
                )
            else:
                time = torch.as_tensor(
                    time_hours, dtype=position.dtype, device=position.device
                )
                if time.ndim == len(output_shape):
                    time = time[..., None]
                try:
                    time = torch.broadcast_to(time, (*output_shape, 1))
                except RuntimeError as error:
                    raise ValueError(
                        "time_hours must broadcast to the position leading dimensions."
                    ) from error
            coordinates = torch.cat((chart, time.reshape(-1, 1)), dim=-1)
            inputs = self._network_inputs_at_height(
                coordinates, height.reshape(-1, 1)
            )[..., 0, :]
            raw = self._network_raw(inputs)
            vector_potential = self.vector_potential_scale_gauss_m * raw[..., 4:7]
        if not need_graph:
            vector_potential = vector_potential.detach()
        return vector_potential.reshape((*output_shape, 3))

    def _evaluate_vector_potential_only_normalized(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Evaluate A_hat directly, without constructing a physical A tensor."""

        if self.magnetic_representation != "vector_potential":
            raise RuntimeError(
                "Vector-potential evaluation requires the vector_potential representation."
            )
        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        output_shape = position.shape[:-1]
        need_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            flat_position = position.reshape(-1, 3)
            if not flat_position.requires_grad:
                flat_position = flat_position.detach().clone().requires_grad_(True)
            chart, height = self._position_chart_height(flat_position)
            if time_hours is None:
                time = torch.full(
                    (*output_shape, 1),
                    self.time_coordinate_center_hours,
                    dtype=position.dtype,
                    device=position.device,
                )
            else:
                time = torch.as_tensor(
                    time_hours, dtype=position.dtype, device=position.device
                )
                if time.ndim == len(output_shape):
                    time = time[..., None]
                try:
                    time = torch.broadcast_to(time, (*output_shape, 1))
                except RuntimeError as error:
                    raise ValueError(
                        "time_hours must broadcast to the position leading dimensions."
                    ) from error
            coordinates = torch.cat((chart, time.reshape(-1, 1)), dim=-1)
            inputs = self._network_inputs_at_height(
                coordinates, height.reshape(-1, 1)
            )[..., 0, :]
            raw = self._network_raw(inputs)
            vector_potential = raw[..., 4:7]
        if not need_graph:
            vector_potential = vector_potential.detach()
        return vector_potential.reshape((*output_shape, 3))

    def _evaluate_vector_potential_position_normalized(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate normalized A and B = curl_xhat(A_hat)."""

        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        output_shape = position.shape[:-1]
        need_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            flat_position = position.reshape(-1, 3)
            if not flat_position.requires_grad:
                flat_position = flat_position.detach().clone().requires_grad_(True)
            chart, height = self._position_chart_height(flat_position)
            if time_hours is None:
                time = torch.full(
                    (*output_shape, 1),
                    self.time_coordinate_center_hours,
                    dtype=position.dtype,
                    device=position.device,
                )
            else:
                time = torch.as_tensor(
                    time_hours, dtype=position.dtype, device=position.device
                )
                if time.ndim == len(output_shape):
                    time = time[..., None]
                try:
                    time = torch.broadcast_to(time, (*output_shape, 1))
                except RuntimeError as error:
                    raise ValueError(
                        "time_hours must broadcast to the position leading dimensions."
                    ) from error
            coordinates = torch.cat((chart, time.reshape(-1, 1)), dim=-1)
            inputs = self._network_inputs_at_height(
                coordinates, height.reshape(-1, 1)
            )[..., 0, :]
            raw = self._network_raw(inputs)
            vector_potential = raw[..., 4:7]
            jacobian = _batched_vector_jacobian(
                vector_potential,
                flat_position,
                create_graph=need_graph,
            )
            magnetic_field = _curl_from_jacobian(jacobian) / self.solar_radius_model
            fields = self._decode_raw_normalized(
                raw,
                geometric_height_m=height,
                magnetic_field_model=magnetic_field,
                vector_potential_model=vector_potential,
            )
        if not need_graph:
            fields = {name: value.detach() for name, value in fields.items()}
        return {
            name: value.reshape((*output_shape, *value.shape[1:]))
            for name, value in fields.items()
        }

    def _evaluate_vector_potential_position(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``A`` and derive ``B = curl(A)`` in physical Cartesian space."""

        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        output_shape = position.shape[:-1]
        need_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            flat_position = position.reshape(-1, 3)
            if not flat_position.requires_grad:
                # ``@torch.inference_mode`` inputs cannot be made grad-bearing
                # by detaching alone; clone into an ordinary autograd tensor.
                flat_position = flat_position.detach().clone().requires_grad_(True)
            chart, height = self._position_chart_height(flat_position)
            if time_hours is None:
                time = torch.full(
                    (*output_shape, 1),
                    self.time_coordinate_center_hours,
                    dtype=position.dtype,
                    device=position.device,
                )
            else:
                time = torch.as_tensor(
                    time_hours, dtype=position.dtype, device=position.device
                )
                if time.ndim == len(output_shape):
                    time = time[..., None]
                try:
                    time = torch.broadcast_to(time, (*output_shape, 1))
                except RuntimeError as error:
                    raise ValueError(
                        "time_hours must broadcast to the position leading dimensions."
                    ) from error
            coordinates = torch.cat((chart, time.reshape(-1, 1)), dim=-1)
            inputs = self._network_inputs_at_height(
                coordinates, height.reshape(-1, 1)
            )[..., 0, :]
            raw = self._network_raw(inputs)
            vector_potential = (
                self.vector_potential_scale_gauss_m * raw[..., 4:7]
            )
            jacobian = _batched_vector_jacobian(
                vector_potential,
                flat_position,
                create_graph=need_graph,
            )
            magnetic_field = _curl_from_jacobian(jacobian) / self.solar_radius_m
            fields = self._decode_raw(
                raw,
                geometric_height_m=height,
                magnetic_field_gauss=magnetic_field,
                vector_potential=vector_potential,
            )
        if not need_graph:
            fields = {name: value.detach() for name, value in fields.items()}
        return {
            name: value.reshape((*output_shape, *value.shape[1:]))
            for name, value in fields.items()
        }

    def _evaluate_potential_delta_position(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``B = grad(psi) + alpha * B_delta`` in physical Cartesian space.

        ``psi`` uses the same physical scale as the vector potential (it is a
        first spatial derivative away from a field, exactly like ``curl(A)``),
        so its gradient is directly comparable in Gauss.  ``B_delta`` reuses
        the direct representation's own field scale and channels.  ``alpha``
        is a step-scheduled scalar (see ``_potential_delta_alpha``), not a
        learned quantity: at alpha=0 the field is exactly curl-free by
        construction, regardless of what the network has learned.
        """

        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        output_shape = position.shape[:-1]
        need_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            flat_position = position.reshape(-1, 3)
            if not flat_position.requires_grad:
                flat_position = flat_position.detach().clone().requires_grad_(True)
            chart, height = self._position_chart_height(flat_position)
            if time_hours is None:
                time = torch.full(
                    (*output_shape, 1),
                    self.time_coordinate_center_hours,
                    dtype=position.dtype,
                    device=position.device,
                )
            else:
                time = torch.as_tensor(
                    time_hours, dtype=position.dtype, device=position.device
                )
                if time.ndim == len(output_shape):
                    time = time[..., None]
                try:
                    time = torch.broadcast_to(time, (*output_shape, 1))
                except RuntimeError as error:
                    raise ValueError(
                        "time_hours must broadcast to the position leading dimensions."
                    ) from error
            coordinates = torch.cat((chart, time.reshape(-1, 1)), dim=-1)
            inputs = self._network_inputs_at_height(
                coordinates, height.reshape(-1, 1)
            )[..., 0, :]
            raw = self._network_raw(inputs)
            psi = self.vector_potential_scale_gauss_m * raw[..., 9]
            gradient = _scalar_gradient(psi, flat_position, create_graph=need_graph)
            potential_field = gradient / self.solar_radius_m
            delta_field = self.magnetic_scale_gauss * raw[..., 4:7]
            magnetic_field = (
                potential_field + self._potential_delta_alpha() * delta_field
            )
            fields = self._decode_raw(
                raw,
                geometric_height_m=height,
                magnetic_field_gauss=magnetic_field,
            )
        if not need_graph:
            fields = {name: value.detach() for name, value in fields.items()}
        return {
            name: value.reshape((*output_shape, *value.shape[1:]))
            for name, value in fields.items()
        }

    def _evaluate_potential_delta_position_normalized(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``B_hat = grad(psi_hat) + alpha * B_delta_hat`` in model units."""

        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        output_shape = position.shape[:-1]
        need_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            flat_position = position.reshape(-1, 3)
            if not flat_position.requires_grad:
                flat_position = flat_position.detach().clone().requires_grad_(True)
            chart, height = self._position_chart_height(flat_position)
            if time_hours is None:
                time = torch.full(
                    (*output_shape, 1),
                    self.time_coordinate_center_hours,
                    dtype=position.dtype,
                    device=position.device,
                )
            else:
                time = torch.as_tensor(
                    time_hours, dtype=position.dtype, device=position.device
                )
                if time.ndim == len(output_shape):
                    time = time[..., None]
                try:
                    time = torch.broadcast_to(time, (*output_shape, 1))
                except RuntimeError as error:
                    raise ValueError(
                        "time_hours must broadcast to the position leading dimensions."
                    ) from error
            coordinates = torch.cat((chart, time.reshape(-1, 1)), dim=-1)
            inputs = self._network_inputs_at_height(
                coordinates, height.reshape(-1, 1)
            )[..., 0, :]
            raw = self._network_raw(inputs)
            psi = raw[..., 9]
            gradient = _scalar_gradient(psi, flat_position, create_graph=need_graph)
            potential_field = gradient / self.solar_radius_model
            delta_field = raw[..., 4:7]
            magnetic_field = (
                potential_field + self._potential_delta_alpha() * delta_field
            )
            fields = self._decode_raw_normalized(
                raw,
                geometric_height_m=height,
                magnetic_field_model=magnetic_field,
            )
        if not need_graph:
            fields = {name: value.detach() for name, value in fields.items()}
        return {
            name: value.reshape((*output_shape, *value.shape[1:]))
            for name, value in fields.items()
        }

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

        if self.magnetic_representation == "vector_potential":
            return self._evaluate_vector_potential_position(position_rsun, time_hours)
        if self.magnetic_representation == "potential_delta":
            return self._evaluate_potential_delta_position(position_rsun, time_hours)

        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
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
        raw = self._network_raw(inputs)[..., 0, :]
        return self._decode_raw(
            raw,
            geometric_height_m=height,
        )

    def normalize_atmosphere_fields(
        self,
        fields: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert physical atmosphere outputs to dimensionless model units.

        ``evaluate_position_*`` remains a physical adapter for radiative
        transfer and observation synthesis.  Differentiable training terms
        should use this normalized representation instead.
        """
        normalized = dict(fields)
        normalized["temperature"] = fields["temperature"] / self.temperature_scale_k
        normalized["gas_pressure"] = (
            fields["gas_pressure"] / self.gas_pressure_scale_pa
        )
        normalized["microturbulence"] = (
            fields["microturbulence"] / self.microturbulence_scale_m_per_s
        )
        normalized["velocity_field"] = (
            fields["velocity_field"] / self.velocity_scale_m_per_s
        )
        normalized["magnetic_field"] = (
            fields["magnetic_field"] / self.magnetic_scale_gauss
        )
        if "vector_potential" in fields:
            normalized["vector_potential"] = (
                fields["vector_potential"] / self.vector_potential_scale_gauss_m
            )
        if "geometric_height_m" in fields:
            normalized["geometric_height_m"] = (
                fields["geometric_height_m"] / self.height_input_scale_m
            )
        return normalized

    def denormalize_atmosphere_fields(
        self,
        fields: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert a normalized atmosphere to the physical RT adapter view."""
        physical = dict(fields)
        physical["temperature"] = fields["temperature"] * self.temperature_scale_k
        physical["gas_pressure"] = (
            fields["gas_pressure"] * self.gas_pressure_scale_pa
        )
        physical["microturbulence"] = (
            fields["microturbulence"] * self.microturbulence_scale_m_per_s
        )
        physical["velocity_field"] = (
            fields["velocity_field"] * self.velocity_scale_m_per_s
        )
        physical["magnetic_field"] = (
            fields["magnetic_field"] * self.magnetic_scale_gauss
        )
        if "vector_potential" in fields:
            physical["vector_potential"] = (
                fields["vector_potential"] * self.vector_potential_scale_gauss_m
            )
        if "geometric_height_m" in fields:
            physical["geometric_height_m"] = (
                fields["geometric_height_m"] * self.height_input_scale_m
            )
        return physical

    def evaluate_position_rsun_normalized(
        self,
        position_rsun: torch.Tensor,
        time_hours: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the atmosphere directly in dimensionless model units."""

        if self.magnetic_representation == "vector_potential":
            return self._evaluate_vector_potential_position_normalized(
                position_rsun, time_hours
            )
        if self.magnetic_representation == "potential_delta":
            return self._evaluate_potential_delta_position_normalized(
                position_rsun, time_hours
            )
        position = position_rsun.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
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
        raw = self._network_raw(inputs)[..., 0, :]
        return self._decode_raw_normalized(
            raw,
            geometric_height_m=height,
        )

    def mass_density_normalized(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Query the EOS in SI and return the density in model units.

        Inputs and output are dimensionless.  The two conversions surrounding
        the table call are deliberately kept in this adapter rather than in
        the physics residuals.
        """
        temperature_si = temperature * self.temperature_scale_k
        gas_pressure_si = gas_pressure * self.gas_pressure_scale_pa
        density_si = self.thermodynamic_eos.mass_density(
            temperature_si,
            gas_pressure_si,
        )
        return density_si / self.density_scale_kg_m3

    def electron_density_normalized(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Query electron density through the SI table adapter and normalize it."""
        temperature_si = temperature * self.temperature_scale_k
        gas_pressure_si = gas_pressure * self.gas_pressure_scale_pa
        density_si = self.thermodynamic_eos.electron_density(
            temperature_si,
            gas_pressure_si,
        )
        return density_si / self.electron_density_scale_m3

    def evaluate_position_points(
        self, position_m: torch.Tensor, time_hours: torch.Tensor | float | None = None
    ) -> dict[str, torch.Tensor]:
        """Evaluate the unique solar-Cartesian field at physical 3-D points."""

        position = position_m.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        return self.evaluate_position_rsun(
            position / self.solar_radius_m, time_hours=time_hours
        )

    def trace_rays(
        self,
        coords: torch.Tensor,
        ray_direction: torch.Tensor,
        geometric_height_m: torch.Tensor,
    ) -> tuple[StratifiedAtmosphere, RayTraceResult]:
        """Sample physical points along observer rays through the atmosphere."""

        coords = coords.to(
            device=self.solar_radius_m.device, dtype=self.solar_radius_m.dtype
        )
        if coords.ndim < 1 or coords.shape[-1] != 3 or not torch.isfinite(coords).all():
            raise ValueError("coords must end in finite [x, y, time] values.")
        direction = ray_direction.to(device=coords.device, dtype=coords.dtype)
        direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if not torch.isfinite(direction).all() or torch.any(direction_norm <= 0):
            raise ValueError("Ray directions must be finite non-zero vectors.")
        direction = direction / direction_norm
        height = self._height_grid(geometric_height_m)
        grid = torch.arange(height.numel(), device=height.device, dtype=height.dtype)
        if direction.shape != (*coords.shape[:-1], 3):
            raise ValueError("Ray directions must match coords leading dimensions.")

        # coords[..., :2] identifies the ray at the photosphere. In this local
        # Carrington parameterization, positions are O(1) R_sun and offsets are
        # O(shell thickness / R_sun), preserving both in float32.
        surface_reference_rsun = self._chart_direction(coords[..., :2])

        requested_height = height.expand(*coords.shape[:-1], height.numel())
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
            depth_coordinate=grid,
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
                ("SIREN initialization with unwarped coordinates, "
                 "zero thermodynamic perturbation readouts and random vector readouts"
                 if isinstance(self.network.coordinate_weighting, nn.Identity)
                 else "SIREN initialization with radius-weighted non-radial input "
                 "bandwidth, zero thermodynamic perturbation readouts and "
                 "random vector readouts")
                if self.network_type == "siren"
                else "activation-aware variance-preserving random hidden weights, "
                "zero thermodynamic perturbation readouts and random vector readouts"
            ),
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
            "model_units": {
                "contract": (
                    "x_hat=x/L0, t_hat=t/T0, T_hat=T/T0, P_hat=P/P0, "
                    "rho_hat=rho/rho0, v_hat=v/V0, B_hat=B/B0, "
                    "A_hat=A/(B0 L0)"
                ),
                "length_scale_m": self.height_input_scale_m,
                "temperature_scale_k": self.temperature_scale_k,
                "gas_pressure_scale_pa": self.gas_pressure_scale_pa,
                "density_scale_kg_m3": self.density_scale_kg_m3,
                "electron_density_scale_m3": self.electron_density_scale_m3,
                "velocity_scale_m_per_s": self.velocity_scale_m_per_s,
                "magnetic_scale_gauss": self.magnetic_scale_gauss,
                "vector_potential_scale_gauss_m": self.vector_potential_scale_gauss_m,
                "training_physical_units": (
                    "SI is restricted to table/forward-model adapters; residuals "
                    "consume dimensionless model-unit fields"
                ),
            },
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
            "magnetic_representation": self.magnetic_representation,
            **(
                {
                    "vector_potential_scale_gauss_m": self.vector_potential_scale_gauss_m,
                    "magnetic_parameterization": (
                        "network outputs Cartesian A in G m; physical Cartesian "
                        "B in G is derived as curl(A)"
                    ),
                }
                if self.magnetic_representation == "vector_potential"
                else {
                    "vector_potential_scale_gauss_m": self.vector_potential_scale_gauss_m,
                    "magnetic_parameterization": (
                        "network outputs a scalar potential psi (G m) and a "
                        "direct delta field B_delta (G); physical Cartesian B "
                        "in G is grad(psi) + alpha*B_delta, alpha ramping "
                        f"linearly from 0 to 1 over steps "
                        f"[{self.magnetic_potential_delta_cool_steps}, "
                        f"{self.magnetic_potential_delta_cool_steps + self.magnetic_potential_delta_ramp_steps}]"
                    ),
                }
                if self.magnetic_representation == "potential_delta"
                else {
                    "magnetic_parameterization": (
                        "network outputs physical Cartesian B directly in G"
                    )
                }
            ),
            **({
                "magnetic_reference_height_megameter": self.magnetic_reference_height_megameter,
                "magnetic_height_dependence": (
                    "native Cartesian B readout evaluated at the reference height for the "
                    "same angular coordinates and time; constant along radial columns"
                ),
            } if self.magnetic_reference_height_megameter is not None else {}),
            "vector_decoder": (
                "velocity uses a smoothly bounded component-wise arctangent output "
                "with the configured local linear scale and nonzero tail gradients; "
                + (
                    "magnetic field is the physical Cartesian curl of an unbounded "
                    "Cartesian vector-potential decoder; "
                    if self.magnetic_representation == "vector_potential"
                    else (
                        "magnetic field is the gradient of a scalar potential "
                        "decoder plus a step-scheduled unbounded Cartesian delta "
                        "decoder; "
                        if self.magnetic_representation == "potential_delta"
                        else "magnetic field uses an unbounded Cartesian decoder; "
                    )
                )
                + "output weights and biases remain randomly initialized; both vectors "
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
            "coordinate_system": "physical spherical shell; heights in metres above R_sun",
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
                "network_scale_mm": (
                    [self.height_input_scale_m / 1e6] * 2
                    if getattr(self, "uniform_spatial_scaling", False)
                    else self.spatial_coordinate_scale_mm.detach().cpu().tolist()
                ),
                "network_transform": "(xy_mm - center_mm) / scale_mm",
            },
            "network_inputs": list(self.network_input_names),
            "network_type": self.network_type,
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
