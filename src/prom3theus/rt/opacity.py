"""Differentiable LTE continuum and bound-bound opacity ingredients.

Continuum and Planck wavelength arguments are physical vacuum Angstrom.
Spectral-line laboratory wavelengths remain standard-air Angstrom and are
converted internally. All returned coefficients and rates use SI units unless
a docstring states otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math

import torch
from torch import nn

from prom3theus.core import (
    ATOMIC_MASS_UNIT,
    ELECTRON_VOLT,
    H_PLANCK,
    K_BOLTZMANN,
    M_ELECTRON,
    SPEED_OF_LIGHT,
)
from prom3theus.resources import verify_manifest_resource
from .atomic import AtomicDatabase, SpectralLine
from .wavelength import air_to_vacuum_angstrom


EPSILON_0 = 8.854_187_8128e-12  # F m^-1
ELEMENTARY_CHARGE = 1.602_176_634e-19  # C
BOHR_RADIUS = 5.291_772_109_03e-11  # m
HC_OVER_K = H_PLANCK * SPEED_OF_LIGHT / K_BOLTZMANN  # m K

# Public unit contract at the lookup/model boundary.  The pinned Wittmann
# generator works in Gaussian-cgs internally, but converts every stored value
# to these units before taking log10.  Keeping the contract next to the runtime
# reader makes artifact provenance explicit and prevents model-normalization
# scales from being mistaken for table units.
STIC_LOOKUP_UNITS = {
    "temperature": "K",
    "gas_pressure": "Pa",
    "wavelength": "vacuum Angstrom",
    "true_absorption": "m^-1",
    "scattering_extinction": "m^-1",
    "mass_density": "kg m^-3",
    "electron_density": "m^-3",
    "neutral_hydrogen_density": "m^-3",
    "fe_i_population_over_partition": "m^-3",
}


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


@dataclass(frozen=True)
class STICLookupState:
    """Reusable cubic thermodynamic interpolation coordinates."""

    temperature: torch.Tensor
    t_indices: torch.Tensor
    p_indices: torch.Tensor
    t_weight: torch.Tensor
    p_weight: torch.Tensor


@dataclass(frozen=True)
class STICSpectralState:
    """Reusable wavelength-table interpolation coordinates."""

    lower_indices: torch.Tensor
    upper_indices: torch.Tensor
    weight: torch.Tensor


def _linear_coordinates(grid: torch.Tensor, samples: torch.Tensor):
    bounded = samples.clamp(min=grid[0], max=grid[-1])
    upper = torch.searchsorted(grid, bounded, right=False).clamp(1, grid.numel() - 1)
    lower = upper - 1
    weight = (bounded - grid[lower]) / (grid[upper] - grid[lower])
    return lower, upper, weight


def _validated_stic_domains(declared_domains) -> tuple[tuple[float, float], ...]:
    """Validate and return exactly the wavelength domains declared by STiC."""

    domains = []
    for interval in declared_domains:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("STiC validated wavelength domains must be pairs.")
        lower, upper = map(float, interval)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(
                "STiC validated wavelength domains must be finite and ordered."
            )
        domains.append((lower, upper))
    if not domains:
        raise ValueError("STiC must declare at least one validated wavelength domain.")
    return tuple(domains)


class ContinuumOpacity(nn.Module):
    """Pinned STiC total continuum and thermodynamic populations.

    Runtime total extinction is a differentiable, non-iterative lookup of the
    STiC Wittmann EOS/background-continuum calculation.  It includes true
    absorption and scattering at vacuum 5000 Angstrom and across the HMI
    6173-Angstrom and Hinode 630-nm windows when their stored nodes are present.
    A missing or checksum-invalid STiC table is a hard error;
    physical optical-depth/geometric-height work has no analytic fallback.
    """

    def __init__(
        self,
        atomic_database: AtomicDatabase | None = None,
    ):
        super().__init__()
        self.atomic = atomic_database or AtomicDatabase()
        stic_table_file = self.atomic.data_root.joinpath("stic_continuum_table.json")
        self.stic_table_sha256 = verify_manifest_resource(
            stic_table_file,
            self.atomic.source_manifest,
            resource_name="common/stic_continuum_table.json",
            kind="STiC-total-continuum",
        )
        with _open_text(stic_table_file) as handle:
            stic_document = json.load(handle)
        if stic_document.get("schema_version") != 2:
            raise ValueError("Unsupported STiC continuum table schema.")
        axes = stic_document.get("axes", {})
        log_temperature = torch.tensor(
            axes.get("log10_temperature_k", ()), dtype=torch.float64
        )
        log_pressure = torch.tensor(
            axes.get("log10_gas_pressure_pa", ()), dtype=torch.float64
        )
        wavelength = torch.tensor(
            axes.get("wavelength_vacuum_angstrom", ()), dtype=torch.float64
        )
        log_absorption = torch.tensor(
            stic_document.get("log10_true_absorption_m1", ()), dtype=torch.float64
        )
        log_scattering = torch.tensor(
            stic_document.get("log10_scattering_extinction_m1", ()),
            dtype=torch.float64,
        )
        log_density = torch.tensor(
            stic_document.get("log10_mass_density_kg_m3", ()), dtype=torch.float64
        )
        log_electron_density = torch.tensor(
            stic_document.get("log10_electron_density_m3", ()), dtype=torch.float64
        )
        log_hydrogen_neutral_density = torch.tensor(
            stic_document.get("log10_neutral_hydrogen_density_m3", ()),
            dtype=torch.float64,
        )
        log_fe_i_population_over_partition = torch.tensor(
            stic_document.get("log10_fe_i_population_over_partition_m3", ()),
            dtype=torch.float64,
        )
        expected_thermodynamic_shape = (
            log_temperature.numel(),
            log_pressure.numel(),
        )
        expected_continuum_shape = (*expected_thermodynamic_shape, wavelength.numel())
        if log_temperature.numel() < 4 or log_pressure.numel() < 4:
            raise ValueError("STiC continuum axes require at least four nodes.")
        if (
            wavelength.numel() < 2
            or not torch.isfinite(wavelength).all()
            or torch.any(wavelength[1:] <= wavelength[:-1])
        ):
            raise ValueError("STiC continuum wavelengths must be strictly increasing.")
        if log_absorption.shape != expected_continuum_shape or (
            log_scattering.shape != expected_continuum_shape
        ):
            raise ValueError("STiC continuum coefficients have incompatible shapes.")
        if (
            log_density.shape != expected_thermodynamic_shape
            or (log_electron_density.shape != expected_thermodynamic_shape)
            or (log_hydrogen_neutral_density.shape != expected_thermodynamic_shape)
            or (
                log_fe_i_population_over_partition.shape != expected_thermodynamic_shape
            )
        ):
            raise ValueError(
                "STiC thermodynamic reference tables have incompatible shapes."
            )
        if (
            not torch.isfinite(log_temperature).all()
            or not torch.isfinite(log_pressure).all()
            or torch.any(log_temperature[1:] <= log_temperature[:-1])
            or torch.any(log_pressure[1:] <= log_pressure[:-1])
            or not torch.allclose(
                torch.diff(log_temperature),
                torch.diff(log_temperature)[:1],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            or not torch.allclose(
                torch.diff(log_pressure),
                torch.diff(log_pressure)[:1],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        ):
            raise ValueError("STiC thermodynamic axes must be uniform in log10.")
        for name, values in (
            ("true absorption", log_absorption),
            ("scattering", log_scattering),
            ("mass density", log_density),
            ("electron density", log_electron_density),
            ("neutral atomic-hydrogen density", log_hydrogen_neutral_density),
            ("Fe-I population over partition", log_fe_i_population_over_partition),
        ):
            if not torch.isfinite(values).all():
                raise ValueError(f"STiC table contains non-finite {name} values.")
        self.register_buffer("stic_log_temperature", log_temperature, persistent=False)
        self.register_buffer("stic_log_pressure", log_pressure, persistent=False)
        self.register_buffer(
            "_stic_cubic_offsets",
            torch.arange(-1, 3, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("stic_wavelength_angstrom", wavelength, persistent=False)
        self.register_buffer(
            "stic_log_true_absorption", log_absorption, persistent=False
        )
        self.register_buffer("stic_log_scattering", log_scattering, persistent=False)
        # Keep the two physical components separate through logarithmic
        # interpolation, but gather their shared thermodynamic neighborhood
        # once.  The component axis precedes wavelength so the existing
        # spectral interpolation continues to operate on the final axis.
        self.register_buffer(
            "stic_log_continuum_components",
            torch.stack((log_absorption, log_scattering), dim=-2),
            persistent=False,
        )
        self.register_buffer("stic_log_mass_density", log_density, persistent=False)
        self.register_buffer(
            "stic_log_electron_density", log_electron_density, persistent=False
        )
        self.register_buffer(
            "stic_log_hydrogen_neutral_density",
            log_hydrogen_neutral_density,
            persistent=False,
        )
        self.register_buffer(
            "stic_log_fe_i_population_over_partition",
            log_fe_i_population_over_partition,
            persistent=False,
        )
        self.register_buffer(
            "stic_log_line_thermodynamics",
            torch.stack(
                (
                    log_electron_density,
                    log_hydrogen_neutral_density,
                    log_fe_i_population_over_partition,
                ),
                dim=-1,
            ),
            persistent=False,
        )
        self.stic_interpolation = str(stic_document.get("interpolation", "unknown"))
        self.validated_wavelength_domains_angstrom = _validated_stic_domains(
            stic_document.get("validated_wavelength_domains_angstrom", ()),
        )
        self.reference_solver = dict(stic_document.get("reference_solver", {}))
        self.stic_table_schema_version = int(stic_document["schema_version"])
        raw_contract = stic_document.get("continuum_contract", {})
        contract_fields = (
            "tau500_quantity",
            "scattering_is_stored_separately",
            "physical_height_qualified",
            "runtime_fallback",
            "source_function_limitation",
            "thermodynamic_population_contract",
        )
        try:
            self.continuum_contract = {
                field: raw_contract[field] for field in contract_fields
            }
        except (KeyError, TypeError) as error:
            raise RuntimeError("The STiC continuum contract is incomplete.") from error
        self.reference_top_boundary = dict(stic_document.get("falc_top_boundary", {}))
        if (
            self.reference_solver.get("commit")
            != ("18cda77d038a97f007a783dcb61ea9a9a1244bf7")
            or self.reference_solver.get("runtime_iterations") != 0
        ):
            raise RuntimeError(
                "The total-continuum table is not the pinned offline-only STiC reference."
            )
        if self.continuum_contract.get("physical_height_qualified") is not True:
            raise RuntimeError(
                "The STiC resource does not qualify total tau500 for physical height."
            )
        if self.continuum_contract.get("runtime_fallback") != (
            "none; missing or invalid STiC table is a hard error"
        ):
            raise RuntimeError(
                "The STiC resource permits an unsupported opacity fallback."
            )

    def prepare_stic_lookup(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
    ) -> STICLookupState:
        """Prepare differentiable `(T, P)` coordinates once per atmosphere."""

        if temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError("STiC lookup supports float32 and float64 tensors only")
        if (
            gas_pressure.dtype != temperature.dtype
            or gas_pressure.device != temperature.device
        ):
            raise TypeError(
                "temperature and gas_pressure must use one dtype and device"
            )
        temperature, gas_pressure = torch.broadcast_tensors(temperature, gas_pressure)
        if (
            not torch.isfinite(temperature).all()
            or not torch.isfinite(gas_pressure).all()
            or torch.any(temperature <= 0)
            or torch.any(gas_pressure <= 0)
        ):
            raise ValueError(
                "temperature and gas_pressure must be finite and strictly positive"
            )
        log_temperature = torch.log10(temperature)
        log_pressure = torch.log10(gas_pressure)
        t_axis = self.stic_log_temperature.to(temperature)
        p_axis = self.stic_log_pressure.to(temperature)
        tolerance = 16.0 * torch.finfo(temperature.dtype).eps
        if torch.any(log_temperature < t_axis[0] - tolerance) or torch.any(
            log_temperature > t_axis[-1] + tolerance
        ):
            raise ValueError(
                "temperature lies outside the prepared STiC continuum table"
            )
        if torch.any(log_pressure < p_axis[0] - tolerance) or torch.any(
            log_pressure > p_axis[-1] + tolerance
        ):
            raise ValueError(
                "gas_pressure lies outside the prepared STiC continuum table"
            )
        t_coordinate = (log_temperature - t_axis[0]) / (t_axis[1] - t_axis[0])
        p_coordinate = (log_pressure - p_axis[0]) / (p_axis[1] - p_axis[0])
        t_lower = torch.floor(t_coordinate).long().clamp(0, t_axis.numel() - 2)
        p_lower = torch.floor(p_coordinate).long().clamp(0, p_axis.numel() - 2)
        offsets = self._stic_cubic_offsets.to(device=temperature.device)
        return STICLookupState(
            temperature=temperature,
            t_indices=(t_lower[..., None] + offsets).clamp(0, t_axis.numel() - 1),
            p_indices=(p_lower[..., None] + offsets).clamp(0, p_axis.numel() - 1),
            t_weight=(t_coordinate - t_lower).clamp(0.0, 1.0),
            p_weight=(p_coordinate - p_lower).clamp(0.0, 1.0),
        )

    def _interpolate_stic_log_table(
        self,
        values: torch.Tensor,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Tensor-product cubic interpolation of a positive log10 table."""

        if state is None:
            state = self.prepare_stic_lookup(temperature, gas_pressure)
        temperature = state.temperature
        values = values.to(temperature)
        neighborhood = values[
            state.t_indices[..., :, None], state.p_indices[..., None, :]
        ]
        trailing_dimensions = values.ndim - 2
        interpolation_dimension = -(trailing_dimensions + 1)

        def cubic(samples: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            p0, p1, p2, p3 = samples.unbind(dim=interpolation_dimension)
            while weight.ndim < p1.ndim:
                weight = weight[..., None]
            return p1 + 0.5 * weight * (
                p2
                - p0
                + weight
                * (
                    2.0 * p0
                    - 5.0 * p1
                    + 4.0 * p2
                    - p3
                    + weight * (3.0 * (p1 - p2) + p3 - p0)
                )
            )

        pressure_interpolated = cubic(neighborhood, state.p_weight)
        return cubic(pressure_interpolated, state.t_weight)

    def _validate_stic_wavelengths(self, wavelength: torch.Tensor) -> None:
        if not self.validated_wavelength_domains_angstrom:
            raise RuntimeError(
                "STiC continuum table has no validated wavelength domain."
            )
        valid = torch.zeros_like(wavelength, dtype=torch.bool)
        tolerance = 8.0 * torch.finfo(wavelength.dtype).eps * wavelength
        for lower, upper in self.validated_wavelength_domains_angstrom:
            valid = valid | (
                (wavelength >= wavelength.new_tensor(lower) - tolerance)
                & (wavelength <= wavelength.new_tensor(upper) + tolerance)
            )
        if not torch.all(valid):
            invalid = wavelength[~valid].detach().cpu().tolist()
            domains = ", ".join(
                (
                    f"{lower:g} Angstrom"
                    if lower == upper
                    else f"{lower:g}--{upper:g} Angstrom"
                )
                for lower, upper in self.validated_wavelength_domains_angstrom
            )
            raise ValueError(
                f"STiC total continuum is only validated within {domains}; "
                f"invalid values: {invalid}. Supply a checksum-verified STiC "
                "resource generated for the requested spectral window."
            )

    def _stic_component(
        self,
        values: torch.Tensor,
        wavelength_vacuum_angstrom,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
        *,
        spectral_state: STICSpectralState | None = None,
    ) -> torch.Tensor:
        wavelength = self._wavelength_tensor(wavelength_vacuum_angstrom, temperature)
        if spectral_state is None:
            self._validate_stic_wavelengths(wavelength)
            spectral_state = self._stic_wavelength_coordinates(wavelength, temperature)
        log_values = self._interpolate_stic_log_table(
            values, temperature, gas_pressure, state
        )
        lower = spectral_state.lower_indices.to(device=temperature.device)
        upper = spectral_state.upper_indices.to(device=temperature.device)
        weight = spectral_state.weight.to(temperature)
        log_interpolated = log_values[..., lower] + weight * (
            log_values[..., upper] - log_values[..., lower]
        )
        return torch.pow(temperature.new_tensor(10.0), log_interpolated)

    def _stic_wavelength_coordinates(
        self,
        wavelength: torch.Tensor,
        temperature: torch.Tensor,
    ) -> STICSpectralState:
        """Return spectral interpolation coordinates for one wavelength grid."""

        wavelength_grid = self.stic_wavelength_angstrom.to(temperature)
        lower, upper, weight = _linear_coordinates(wavelength_grid, wavelength)
        return STICSpectralState(lower, upper, weight)

    def prepare_stic_spectral_state(
        self,
        wavelength_vacuum_angstrom,
    ) -> STICSpectralState:
        """Validate and prepare fixed STiC wavelength interpolation indices.

        The returned tensors contain no atmospheric state and may be reused for
        every atmosphere synthesized on the same wavelength grid.  Callers that
        supply changing grids can omit this state from :meth:`forward`; the
        existing dynamic path validates and constructs it on demand.
        """

        wavelength = torch.as_tensor(wavelength_vacuum_angstrom)
        if not wavelength.is_floating_point():
            wavelength = wavelength.to(torch.get_default_dtype())
        if wavelength.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "STiC spectral lookup supports float32 and float64 tensors only"
            )
        if wavelength.ndim == 0:
            wavelength = wavelength[None]
        if wavelength.ndim != 1:
            raise ValueError(
                "wavelength_vacuum_angstrom must be scalar or one-dimensional"
            )
        self._validate_stic_wavelengths(wavelength)
        return self._stic_wavelength_coordinates(wavelength, wavelength)

    def true_absorption(
        self,
        wavelength_vacuum_angstrom,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Return STiC true continuum absorption in m⁻¹."""

        return self._stic_component(
            self.stic_log_true_absorption,
            wavelength_vacuum_angstrom,
            temperature,
            gas_pressure,
            state,
        )

    def scattering_extinction(
        self,
        wavelength_vacuum_angstrom,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Return STiC continuum scattering extinction in m⁻¹."""

        return self._stic_component(
            self.stic_log_scattering,
            wavelength_vacuum_angstrom,
            temperature,
            gas_pressure,
            state,
        )

    def reference_thermodynamics(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return interpolated STiC ``rho`` and ``ne`` reference quantities."""

        return {
            "mass_density": self.reference_mass_density(
                temperature, gas_pressure, state
            ),
            "electron_density": self.reference_electron_density(
                temperature, gas_pressure, state
            ),
        }

    def reference_mass_density(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Return the pinned STiC mass density in kg m⁻³."""

        log_density = self._interpolate_stic_log_table(
            self.stic_log_mass_density, temperature, gas_pressure, state
        )
        return torch.pow(temperature.new_tensor(10.0), log_density)

    def reference_electron_density(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Return the pinned STiC electron density in m⁻³.

        This narrow helper avoids interpolating the independently stored mass
        density when synthesis needs only the electron perturber density for
        Stark broadening.
        """

        log_electron_density = self._interpolate_stic_log_table(
            self.stic_log_line_thermodynamics[..., 0],
            temperature,
            gas_pressure,
            state,
        )
        return torch.pow(temperature.new_tensor(10.0), log_electron_density)

    def reference_neutral_hydrogen_density(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Return pinned-STiC physical neutral atomic H density in m⁻³.

        The offline generator stores the undivided ``n(H I)`` returned by the
        Wittmann hydrogen equilibrium.  This is deliberately not the
        ``n(H I)/U(H I)`` partial used internally by continuum-opacity code and
        is therefore the correct perturber density for ABO broadening.
        """

        log_hydrogen_neutral = self._interpolate_stic_log_table(
            self.stic_log_line_thermodynamics[..., 1],
            temperature,
            gas_pressure,
            state,
        )
        return torch.pow(temperature.new_tensor(10.0), log_hydrogen_neutral)

    def reference_fe_i_population_over_partition(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Return pinned-STiC ``n(Fe I) / U(Fe I)`` in m⁻³."""

        log_population = self._interpolate_stic_log_table(
            self.stic_log_line_thermodynamics[..., 2],
            temperature,
            gas_pressure,
            state,
        )
        return torch.pow(temperature.new_tensor(10.0), log_population)

    def reference_line_thermodynamics(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
        *,
        include_electron_density: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Interpolate every line-synthesis reservoir in one table gather."""

        names = ["hydrogen_neutral", "fe_i_population_over_partition"]
        tables = self.stic_log_line_thermodynamics[..., 1:]
        if include_electron_density:
            names.insert(0, "electron_density")
            tables = self.stic_log_line_thermodynamics
        logs = self._interpolate_stic_log_table(
            tables,
            temperature,
            gas_pressure,
            state,
        )
        values = torch.pow(temperature.new_tensor(10.0), logs)
        return dict(zip(names, values.unbind(dim=-1)))

    @staticmethod
    def _fe_i_lower_level_population_from_reservoir(
        line: SpectralLine,
        temperature: torch.Tensor,
        fe_i_population_over_partition: torch.Tensor,
    ) -> torch.Tensor:
        if line.element != "Fe" or line.ion_stage != 1:
            raise ValueError(
                "The pinned STiC line-population resource currently supports Fe I only."
            )
        temperature, reservoir = torch.broadcast_tensors(
            temperature, fe_i_population_over_partition
        )
        excitation_temperature = line.lower_excitation_ev * ELECTRON_VOLT / K_BOLTZMANN
        log_population = (
            torch.log(reservoir)
            + math.log(line.lower_statistical_weight)
            - excitation_temperature / temperature
        )
        return torch.exp(log_population)

    def reference_lower_level_populations(
        self,
        lines: Sequence[SpectralLine],
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
        *,
        fe_i_population_over_partition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return STiC-consistent LTE lower populations for reviewed Fe-I lines.

        The shared ``n(Fe I)/U(Fe I)`` reservoir is interpolated once, then each
        level receives the differentiable Boltzmann factor formed from its
        reviewed excitation energy and lower ``J``.  Values use m⁻³ and no
        iterative solver runs at inference or training time.
        """

        lines = tuple(lines)
        if not lines:
            raise ValueError("At least one spectral line is required")
        reservoir = fe_i_population_over_partition
        if reservoir is None:
            reservoir = self.reference_fe_i_population_over_partition(
                temperature, gas_pressure, state
            )
        return {
            line.id: self._fe_i_lower_level_population_from_reservoir(
                line, temperature, reservoir
            )
            for line in lines
        }

    @staticmethod
    def _wavelength_tensor(
        wavelength_vacuum_angstrom, temperature: torch.Tensor
    ) -> torch.Tensor:
        wavelength = torch.as_tensor(
            wavelength_vacuum_angstrom,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        if wavelength.ndim == 0:
            wavelength = wavelength[None]
        if wavelength.ndim != 1:
            raise ValueError(
                "wavelength_vacuum_angstrom must be scalar or one-dimensional"
            )
        return wavelength

    def forward(
        self,
        wavelength_vacuum_angstrom,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
        *,
        spectral_state: STICSpectralState | None = None,
    ) -> torch.Tensor:
        """Return STiC total extinction, shape ``temperature.shape + (W,)``.

        Total extinction is true absorption plus scattering.  The current
        synthesis source remains thermal (``B_lambda``) for both components;
        :attr:`continuum_contract` records that explicit approximation.
        """

        wavelength = self._wavelength_tensor(wavelength_vacuum_angstrom, temperature)
        if spectral_state is None:
            spectral_state = self.prepare_stic_spectral_state(wavelength)
        elif any(
            values.shape != wavelength.shape
            for values in (
                spectral_state.lower_indices,
                spectral_state.upper_indices,
                spectral_state.weight,
            )
        ):
            raise ValueError(
                "Prepared STiC spectral state does not match the wavelength grid."
            )
        components = self._stic_component(
            self.stic_log_continuum_components,
            wavelength,
            temperature,
            gas_pressure,
            state,
            spectral_state=spectral_state,
        )
        return components.sum(dim=-2)

    def volume_extinction_at_5000(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: STICLookupState | None = None,
    ) -> torch.Tensor:
        """Return STiC total 5000-Angstrom volume extinction in m⁻¹."""

        return self(temperature.new_tensor([5000.0]), temperature, gas_pressure, state)[
            ..., 0
        ]

    @property
    def reference_top_pressure_pa(self) -> float:
        """FALC/STiC gas pressure at the reference top optical depth."""

        return float(self.reference_top_boundary["gas_pressure_pa"])

    @property
    def reference_top_log_tau500(self) -> float:
        """FALC/STiC optical-depth coordinate used for the top pressure."""

        return float(self.reference_top_boundary["target_log10_tau500"])

    @property
    def reference_gravity_m_per_s2(self) -> float:
        """FALC gravity paired with the reference atmosphere, in m s⁻²."""

        gravity = float(self.reference_top_boundary["gravity_cm_s2"]) / 100.0
        if not math.isfinite(gravity) or gravity <= 0.0:
            raise RuntimeError("The pinned FALC resource has no valid gravity.")
        return gravity

    def validate_physical_height_contract(
        self,
        log_tau500_top: float,
        top_pressure_pa: float,
        *,
        pressure_relative_tolerance: float = 0.05,
        log_tau_absolute_tolerance: float = 1.0e-6,
    ) -> None:
        """Reject a top boundary inconsistent with the pinned FALC/STiC pair."""

        if pressure_relative_tolerance < 0.0 or log_tau_absolute_tolerance < 0.0:
            raise ValueError("physical-height contract tolerances must be non-negative")
        if abs(float(log_tau500_top) - self.reference_top_log_tau500) > (
            log_tau_absolute_tolerance
        ):
            raise RuntimeError(
                "Physical-height top log(tau500) is inconsistent with the pinned "
                f"FALC/STiC reference ({self.reference_top_log_tau500:g})."
            )
        relative_error = abs(
            float(top_pressure_pa) / self.reference_top_pressure_pa - 1.0
        )
        if relative_error > pressure_relative_tolerance:
            raise RuntimeError(
                "Physical-height top pressure is inconsistent with the pinned "
                f"FALC/STiC reference ({self.reference_top_pressure_pa:.9g} Pa at "
                f"log(tau500)={self.reference_top_log_tau500:g}); got "
                f"{float(top_pressure_pa):.9g} Pa."
            )

    def metadata(self) -> dict:
        """Return artifact-safe continuum provenance and limitations."""

        return {
            "lookup_units": dict(STIC_LOOKUP_UNITS),
            "stic_table_schema_version": self.stic_table_schema_version,
            "stic_table_sha256": self.stic_table_sha256,
            "reference_solver": self.reference_solver,
            "interpolation": self.stic_interpolation,
            "validated_wavelength_domains_angstrom": [
                list(interval)
                for interval in self.validated_wavelength_domains_angstrom
            ],
            "continuum_contract": self.continuum_contract,
            "falc_top_boundary": self.reference_top_boundary,
            "thermodynamic_population_fields": {
                "neutral_hydrogen_density": "m^-3, physical n(H I)",
                "fe_i_population_over_partition": "m^-3, n(Fe I)/U(Fe I)",
                "line_lower_population": (
                    "m^-3, [n(Fe I)/U(Fe I)] (2J_l+1) exp(-E_l/k_B T)"
                ),
            },
        }


def planck_lambda(
    temperature: torch.Tensor,
    wavelength_vacuum_angstrom,
    *,
    radiance_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Planck radiance, optionally normalized directly in log space.

    Without ``radiance_scale`` the result is physical ``B_lambda`` in
    W m^-3 sr^-1. With a positive scalar scale, return the dimensionless ratio
    ``B_lambda / radiance_scale`` without ever materializing the large physical
    radiance in the autograd graph. The result has shape
    ``temperature.shape + (W,)``.
    """

    if temperature.dtype not in (torch.float32, torch.float64):
        raise TypeError("Planck radiance supports float32 and float64 tensors only")
    if not torch.isfinite(temperature).all() or torch.any(temperature <= 0):
        raise ValueError("temperature must be finite and strictly positive")
    wavelength = torch.as_tensor(
        wavelength_vacuum_angstrom,
        dtype=temperature.dtype,
        device=temperature.device,
    )
    if wavelength.ndim == 0:
        wavelength = wavelength[None]
    if (
        wavelength.ndim != 1
        or not torch.isfinite(wavelength).all()
        or torch.any(wavelength <= 0)
    ):
        raise ValueError(
            "wavelength_vacuum_angstrom must be finite, positive, and one-dimensional"
        )
    wavelength_m = wavelength * 1.0e-10
    exponent = HC_OVER_K / temperature[..., None] / wavelength_m
    # Evaluate the lambda^-5 prefactor and Bose denominator in log space.
    # Direct float32 division by lambda^5 has a finite forward value near
    # 630 nm but an overflowing backward pass because lambda^10 underflows.
    log_expm1 = exponent + torch.log(-torch.expm1(-exponent))
    log_radiance = (
        temperature.new_tensor(math.log(2.0 * H_PLANCK * SPEED_OF_LIGHT**2))
        - 5.0 * torch.log(wavelength_m)
        - log_expm1
    )
    if radiance_scale is not None:
        if not isinstance(radiance_scale, torch.Tensor):
            scalar_scale = float(radiance_scale)
            if not math.isfinite(scalar_scale) or scalar_scale <= 0:
                raise ValueError("radiance_scale must be one finite positive scalar")
        scale = torch.as_tensor(
            radiance_scale,
            dtype=log_radiance.dtype,
            device=log_radiance.device,
        )
        if scale.numel() != 1:
            raise ValueError("radiance_scale must be one finite positive scalar")
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("radiance_scale must be one finite positive scalar")
        log_radiance = log_radiance - torch.log(scale)
    return torch.exp(log_radiance)


def integrated_line_opacity(
    line: SpectralLine,
    lower_population: torch.Tensor,
    temperature: torch.Tensor,
    *,
    rest_frequency_hz: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Frequency-integrated line extinction in m⁻¹ Hz.

    Multiplication by a profile ``phi_nu`` normalized to unit integral over Hz
    yields the monochromatic line extinction in m⁻¹.
    """

    if temperature.dtype not in (torch.float32, torch.float64):
        raise TypeError("Line opacity supports float32 and float64 tensors only")
    if (
        lower_population.dtype != temperature.dtype
        or lower_population.device != temperature.device
    ):
        raise TypeError(
            "lower_population and temperature must use one dtype and device"
        )
    lower_population, temperature = torch.broadcast_tensors(
        lower_population, temperature
    )
    if (
        not torch.isfinite(temperature).all()
        or not torch.isfinite(lower_population).all()
        or torch.any(temperature <= 0)
        or torch.any(lower_population < 0)
    ):
        raise ValueError(
            "temperature must be finite and positive and lower_population must "
            "be finite and non-negative"
        )
    if rest_frequency_hz is None:
        wavelength_air = temperature.new_tensor(line.wavelength_air_angstrom)
        wavelength_vacuum = air_to_vacuum_angstrom(wavelength_air)
        rest_frequency = SPEED_OF_LIGHT / (wavelength_vacuum * 1.0e-10)
    else:
        rest_frequency = torch.as_tensor(
            rest_frequency_hz,
            dtype=temperature.dtype,
            device=temperature.device,
        )
    if (
        rest_frequency.numel() != 1
        or not torch.isfinite(rest_frequency).all()
        or torch.any(rest_frequency <= 0)
    ):
        raise ValueError("rest_frequency_hz must be one finite positive scalar")
    photon_temperature = H_PLANCK * rest_frequency / K_BOLTZMANN
    stimulated_emission = -torch.expm1(-photon_temperature / temperature)
    coefficient = ELEMENTARY_CHARGE**2 / (4.0 * EPSILON_0 * M_ELECTRON * SPEED_OF_LIGHT)
    return (
        coefficient * line.oscillator_strength * lower_population * stimulated_emission
    )


def doppler_velocity(
    temperature: torch.Tensor,
    microturbulence_m_s: torch.Tensor,
    atomic_mass_u: float,
) -> torch.Tensor:
    """Return ``sqrt(2 k T / m + xi^2)`` in m s⁻¹."""

    temperature, microturbulence_m_s = torch.broadcast_tensors(
        temperature, microturbulence_m_s
    )
    if temperature.dtype not in (torch.float32, torch.float64):
        raise TypeError("Doppler widths support float32 and float64 tensors only")
    if microturbulence_m_s.dtype != temperature.dtype or (
        microturbulence_m_s.device != temperature.device
    ):
        raise TypeError(
            "temperature and microturbulence_m_s must use one dtype and device"
        )
    if not math.isfinite(float(atomic_mass_u)) or atomic_mass_u <= 0:
        raise ValueError("atomic_mass_u must be finite and positive")
    if (
        not torch.isfinite(temperature).all()
        or not torch.isfinite(microturbulence_m_s).all()
        or torch.any(temperature <= 0)
        or torch.any(microturbulence_m_s < 0)
    ):
        raise ValueError(
            "temperature must be finite and positive and microturbulence must "
            "be finite and non-negative"
        )
    mass = temperature.new_tensor(atomic_mass_u * ATOMIC_MASS_UNIT)
    return torch.sqrt(
        2.0 * K_BOLTZMANN * temperature / mass + microturbulence_m_s.square()
    )


def doppler_width_frequency(
    line: SpectralLine,
    temperature: torch.Tensor,
    microturbulence_m_s: torch.Tensor,
    atomic_mass_u: float,
) -> torch.Tensor:
    """Return the 1/e Doppler width ``Delta nu_D`` in Hz."""

    wavelength_air = temperature.new_tensor(line.wavelength_air_angstrom)
    wavelength_vacuum = air_to_vacuum_angstrom(wavelength_air)
    rest_frequency = SPEED_OF_LIGHT / (wavelength_vacuum * 1.0e-10)
    return (
        rest_frequency
        * doppler_velocity(temperature, microturbulence_m_s, atomic_mass_u)
        / SPEED_OF_LIGHT
    )


def damping_rate(
    line: SpectralLine,
    temperature: torch.Tensor,
    electron_density: torch.Tensor,
    hydrogen_neutral: torch.Tensor,
) -> torch.Tensor:
    """Return total radiative + Stark + ABO damping ``Gamma`` in s⁻¹.

    The corresponding Voigt parameter is ``a = Gamma/(4*pi*Delta nu_D)``.
    The ABO Maxwellian average follows Anstee & O'Mara and includes the factor
    of two used by RH/Lightweaver to convert the tabulated half-width.
    """

    temperature, electron_density, hydrogen_neutral = torch.broadcast_tensors(
        temperature, electron_density, hydrogen_neutral
    )
    if temperature.dtype not in (torch.float32, torch.float64):
        raise TypeError("Damping rates support float32 and float64 tensors only")
    if any(
        value.dtype != temperature.dtype or value.device != temperature.device
        for value in (electron_density, hydrogen_neutral)
    ):
        raise TypeError(
            "temperature and perturber densities must use one dtype and device"
        )
    if (
        not torch.isfinite(temperature).all()
        or not torch.isfinite(electron_density).all()
        or not torch.isfinite(hydrogen_neutral).all()
        or torch.any(temperature <= 0)
        or torch.any(electron_density < 0)
        or torch.any(hydrogen_neutral < 0)
    ):
        raise ValueError(
            "temperature must be finite and positive and number densities must "
            "be finite and non-negative"
        )
    if line.element != "Fe":
        raise ValueError(
            "The current ABO damping implementation supports Fe lines only"
        )
    if line.log_gamma_stark_s_cm3 is not None and (
        line.stark_temperature_exponent is None
        or not math.isfinite(line.stark_temperature_exponent)
    ):
        raise ValueError("Stark damping requires a finite temperature exponent")
    if (
        not math.isfinite(line.abo_sigma_a0_squared)
        or line.abo_sigma_a0_squared <= 0
        or not math.isfinite(line.abo_alpha)
        or not 0 <= line.abo_alpha < 4
    ):
        raise ValueError("ABO sigma must be positive and 0 <= alpha < 4")
    gamma_radiative = temperature.new_tensor(
        0.0 if line.log_gamma_rad_s is None else 10.0**line.log_gamma_rad_s
    )
    electron_density_cm3 = electron_density * 1.0e-6
    if line.log_gamma_stark_s_cm3 is None:
        gamma_stark = torch.zeros_like(temperature)
    else:
        gamma_stark = (
            10.0**line.log_gamma_stark_s_cm3
            * electron_density_cm3
            * (temperature / 10_000.0).pow(line.stark_temperature_exponent)
        )

    hydrogen_mass_u = 1.008
    iron_mass_u = 55.845
    reduced_mass = (
        hydrogen_mass_u
        * iron_mass_u
        / (hydrogen_mass_u + iron_mass_u)
        * ATOMIC_MASS_UNIT
    )
    mean_relative_velocity = torch.sqrt(
        8.0 * K_BOLTZMANN * temperature / (torch.pi * reduced_mass)
    )
    sigma_reference = line.abo_sigma_a0_squared * BOHR_RADIUS**2
    alpha = line.abo_alpha
    maxwell_factor = (4.0 / math.pi) ** (0.5 * alpha) * math.gamma(2.0 - 0.5 * alpha)
    gamma_neutral_h = (
        2.0
        * maxwell_factor
        * hydrogen_neutral
        * sigma_reference
        * mean_relative_velocity
        * (mean_relative_velocity / 1.0e4).pow(-alpha)
    )
    return gamma_radiative + gamma_stark + gamma_neutral_h


__all__ = [
    "ContinuumOpacity",
    "damping_rate",
    "doppler_velocity",
    "doppler_width_frequency",
    "integrated_line_opacity",
    "planck_lambda",
]
