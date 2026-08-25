"""Differentiable tabulated LTE equation of state."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping

import torch
from torch import nn

from pme.lte.atomic import AtomicDatabase, SpectralLine, verify_manifest_resource


K_BOLTZMANN = 1.380_649e-23  # J K^-1
H_PLANCK = 6.626_070_15e-34  # J s
M_ELECTRON = 9.109_383_7015e-31  # kg
ATOMIC_MASS_UNIT = 1.660_539_066_60e-27  # kg
ELECTRON_VOLT = 1.602_176_634e-19  # J
H_MINUS_AFFINITY_EV = 0.754_195
_LOG_SAHA_PREFACTOR = math.log(2.0) + 1.5 * math.log(
    2.0 * math.pi * M_ELECTRON * K_BOLTZMANN / H_PLANCK**2
)
_LOG_H_MINUS_TRANSLATIONAL = math.log(0.25) + 1.5 * math.log(
    H_PLANCK**2 / (2.0 * math.pi * M_ELECTRON * K_BOLTZMANN)
)


@dataclass(frozen=True)
class EOSState:
    """LTE populations and thermodynamic densities in SI units.

    Every tensor has the broadcast shape of the input temperature and gas
    pressure.  Population values and ``electron_density`` are in inverse
    cubic metres, pressures in Pa, and ``mass_density`` in kg m⁻³.
    """

    gas_pressure: torch.Tensor
    electron_pressure: torch.Tensor
    electron_density: torch.Tensor
    hydrogen_total: torch.Tensor
    hydrogen_neutral: torch.Tensor
    hydrogen_ionized: torch.Tensor
    hydrogen_minus: torch.Tensor
    mass_density: torch.Tensor
    populations: Mapping[str, torch.Tensor]
    lower_level_populations: Mapping[str, torch.Tensor]

class LTEEOS(nn.Module):
    """Tabulated three-ion-stage LTE EOS with differentiable interpolation.

    All configured elements contribute to gas pressure, mass density, and the
    electron balance. During resource preparation, charge neutrality and the
    hydrogen, relevant ion-stage, and configured line-level fractions are
    solved on a dense ``(log10(T), log10(Pgas))`` grid. Runtime synthesis uses
    differentiable tensor-product cubic lookup in logarithmic quantities, renormalizes
    conserved fractions, and performs no iterative charge or population solve.

    This photospheric EOS intentionally excludes molecules and ionization
    stages above III; callers working in cool molecular atmospheres or above
    roughly 10000 K should use a more complete chemical EOS.
    """

    def __init__(
        self,
        atomic_database: AtomicDatabase | None = None,
        *,
        table_file=None,
    ):
        super().__init__()
        self.atomic = atomic_database or AtomicDatabase()
        self.element_symbols = tuple(self.atomic.elements)
        if "H" not in self.element_symbols:
            raise ValueError("The LTE EOS requires hydrogen metadata")
        for symbol in self.element_symbols:
            suffixes = ("I", "II") if symbol == "H" else ("I", "II", "III")
            for suffix in suffixes:
                key = f"{symbol}_{suffix}"
                if key not in self.atomic.partition_table:
                    raise ValueError(f"The LTE EOS requires partition function {key!r}")
            if symbol != "H" and self.atomic.element(symbol).second_ionization_ev is None:
                raise ValueError(f"The LTE EOS requires a second ionization energy for {symbol!r}")

        table_resource = (
            self.atomic.data_root.joinpath("eos_table.json")
            if table_file is None
            else table_file
        )
        self.table_sha256 = verify_manifest_resource(
            table_resource, self.atomic.source_manifest, kind="EOS-table"
        )
        with open(table_resource, encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("schema_version") != 2:
            raise ValueError("Unsupported LTE EOS table schema.")
        axes = document.get("axes", {})
        log_temperature = torch.tensor(
            axes.get("log10_temperature_k", ()), dtype=torch.float64
        )
        log_pressure = torch.tensor(
            axes.get("log10_gas_pressure_pa", ()), dtype=torch.float64
        )
        log_electron_density = torch.tensor(
            document.get("log10_electron_density_m3", ()), dtype=torch.float64
        )
        hydrogen_fraction_document = document.get("log10_hydrogen_fractions", {})
        hydrogen_fraction_names = ("H_I", "H_II", "H_minus")
        log_hydrogen_fractions = torch.stack([
            torch.tensor(hydrogen_fraction_document.get(name, ()), dtype=torch.float64)
            for name in hydrogen_fraction_names
        ])
        ion_fraction_document = document.get("log10_ion_fractions", {})
        ion_species = tuple(sorted(ion_fraction_document))
        log_ion_fractions = torch.stack([
            torch.tensor(ion_fraction_document[name], dtype=torch.float64)
            for name in ion_species
        ]) if ion_species else torch.empty(0, *log_electron_density.shape, dtype=torch.float64)
        lower_document = document.get("log10_lower_level_fraction_of_element", {})
        lower_level_ids = tuple(sorted(lower_document))
        log_lower_level_fractions = torch.stack([
            torch.tensor(lower_document[line_id], dtype=torch.float64)
            for line_id in lower_level_ids
        ]) if lower_level_ids else torch.empty(0, *log_electron_density.shape, dtype=torch.float64)
        if log_temperature.numel() < 2 or log_pressure.numel() < 2:
            raise ValueError("LTE EOS table axes must each contain at least two points.")
        if log_electron_density.shape != (log_temperature.numel(), log_pressure.numel()):
            raise ValueError("LTE EOS electron-density table has an incompatible shape.")
        expected_grid = log_electron_density.shape
        if log_hydrogen_fractions.shape != (3, *expected_grid):
            raise ValueError("LTE EOS hydrogen-fraction table has an incompatible shape.")
        if log_ion_fractions.shape[1:] != expected_grid:
            raise ValueError("LTE EOS ion-fraction table has an incompatible shape.")
        if log_lower_level_fractions.shape[1:] != expected_grid:
            raise ValueError("LTE EOS lower-level table has an incompatible shape.")
        temperature_steps = torch.diff(log_temperature)
        pressure_steps = torch.diff(log_pressure)
        if not torch.allclose(temperature_steps, temperature_steps[:1], rtol=1e-10, atol=1e-12):
            raise ValueError("LTE EOS temperature axis must be uniform in log10(T).")
        if not torch.allclose(pressure_steps, pressure_steps[:1], rtol=1e-10, atol=1e-12):
            raise ValueError("LTE EOS pressure axis must be uniform in log10(Pgas).")
        for name, values in (
            ("electron density", log_electron_density),
            ("hydrogen fractions", log_hydrogen_fractions),
            ("ion fractions", log_ion_fractions),
            ("lower-level fractions", log_lower_level_fractions),
        ):
            if not torch.isfinite(values).all():
                raise ValueError(f"LTE EOS table contains non-finite {name}.")
        self.register_buffer("table_log_temperature", log_temperature)
        self.register_buffer("table_log_pressure", log_pressure)
        self.register_buffer("table_log_electron_density", log_electron_density)
        self.register_buffer("table_log_hydrogen_fractions", log_hydrogen_fractions)
        self.register_buffer("table_log_ion_fractions", log_ion_fractions)
        self.register_buffer("table_log_lower_level_fractions", log_lower_level_fractions)
        self.hydrogen_fraction_names = hydrogen_fraction_names
        self.ion_species = ion_species
        self.lower_level_ids = lower_level_ids
        self.table_metadata = document.get("reference_solver", {})
        self.table_interpolation = str(document.get("interpolation", "unknown"))
        self.table_schema_version = int(document["schema_version"])

    @staticmethod
    def generate_table_document(
        atomic: AtomicDatabase,
        *,
        log_temperature_bounds=(3.4, 4.0),
        log_pressure_bounds=(-1.5, 6.0),
        temperature_count: int = 257,
        pressure_count: int = 277,
        iterations: int = 80,
    ) -> dict:
        """Generate the offline charge-neutrality reference table."""

        if temperature_count < 2 or pressure_count < 2:
            raise ValueError("EOS table axes require at least two points.")
        if iterations < 1:
            raise ValueError("Invalid offline EOS reference-solver settings.")
        log_temperature = torch.linspace(
            *log_temperature_bounds, temperature_count, dtype=torch.float64
        )
        log_pressure = torch.linspace(
            *log_pressure_bounds, pressure_count, dtype=torch.float64
        )
        temperature = torch.pow(10.0, log_temperature[:, None])
        gas_pressure = torch.pow(10.0, log_pressure[None, :])
        temperature, gas_pressure = torch.broadcast_tensors(temperature, gas_pressure)
        total_particle_density = torch.exp(
            torch.log(gas_pressure) - math.log(K_BOLTZMANN) - torch.log(temperature)
        )
        symbols = tuple(atomic.elements)
        abundances = {
            symbol: temperature.new_tensor(atomic.abundance_ratio(symbol))
            for symbol in symbols
        }
        nuclei_per_hydrogen = sum(abundances.values())

        def log_saha(symbol: str, stage: int) -> torch.Tensor:
            metadata = atomic.element(symbol)
            ionization_ev = (
                metadata.ionization_ev if stage == 1 else metadata.second_ionization_ev
            )
            suffixes = ("I", "II", "III")
            lower = atomic.partition_function(f"{symbol}_{suffixes[stage - 1]}", temperature)
            upper = atomic.partition_function(f"{symbol}_{suffixes[stage]}", temperature)
            return (
                temperature.new_tensor(_LOG_SAHA_PREFACTOR)
                + 1.5 * torch.log(temperature)
                + torch.log(upper)
                - torch.log(lower)
                - (ionization_ev * ELECTRON_VOLT / K_BOLTZMANN) / temperature
            )

        log_saha_first = {symbol: log_saha(symbol, 1) for symbol in symbols}
        log_saha_second = {
            symbol: log_saha(symbol, 2) for symbol in symbols if symbol != "H"
        }
        h_minus_factor = LTEEOS.h_minus_equilibrium_factor(temperature)
        # A bracketed charge solve is slower than Newton but is performed only
        # during offline table preparation and is robust in the cool,
        # high-pressure corner where an undamped Newton step can leave the
        # physical electron-density interval.
        log_total = torch.log(total_particle_density)
        lower = log_total - 60.0
        upper = log_total + math.log1p(-1.0e-12)
        for _ in range(iterations):
            midpoint = 0.5 * (lower + upper)
            net_charge, _ = LTEEOS._charge_and_log_derivative(
                midpoint,
                log_saha_first,
                log_saha_second,
                abundances,
                h_minus_factor,
                symbols,
            )
            electron_density = torch.exp(midpoint)
            residual = (
                electron_density
                * nuclei_per_hydrogen
                / (total_particle_density - electron_density).clamp_min(
                    torch.finfo(electron_density.dtype).tiny
                )
                - net_charge
            )
            lower = torch.where(residual < 0.0, midpoint, lower)
            upper = torch.where(residual >= 0.0, midpoint, upper)
        log_electron_density = 0.5 * (lower + upper)
        electron_density = torch.exp(log_electron_density)
        hydrogen_neutral_f, hydrogen_ionized_f, hydrogen_minus_f = (
            LTEEOS._hydrogen_fractions(
                log_saha_first["H"], electron_density, h_minus_factor
            )
        )
        tiny = torch.finfo(temperature.dtype).tiny
        log10_hydrogen_fractions = {
            name: torch.log10(values.clamp_min(tiny)).tolist()
            for name, values in zip(
                ("H_I", "H_II", "H_minus"),
                (hydrogen_neutral_f, hydrogen_ionized_f, hydrogen_minus_f),
            )
        }
        line_elements = tuple(sorted({line.element for line in atomic.lines if line.element != "H"}))
        log10_ion_fractions = {}
        element_fractions = {}
        for symbol in line_elements:
            first = log_saha_first[symbol] - log_electron_density
            second = log_saha_second[symbol] - log_electron_density
            fractions = torch.softmax(
                torch.stack((torch.zeros_like(first), first, first + second), dim=-1),
                dim=-1,
            )
            element_fractions[symbol] = fractions
            for index, suffix in enumerate(("I", "II", "III")):
                log10_ion_fractions[f"{symbol}_{suffix}"] = torch.log10(
                    fractions[..., index].clamp_min(tiny)
                ).tolist()
        log10_lower_level_fractions = {}
        for line in atomic.lines:
            ion_fraction = element_fractions[line.element][..., line.ion_stage - 1]
            partition = atomic.partition_function(
                f"{line.element}_{('I', 'II', 'III')[line.ion_stage - 1]}", temperature
            )
            excitation_temperature = line.lower_excitation_ev * ELECTRON_VOLT / K_BOLTZMANN
            fraction = (
                ion_fraction
                * line.lower_statistical_weight
                / partition
                * torch.exp(-excitation_temperature / temperature)
            )
            log10_lower_level_fractions[line.id] = torch.log10(
                fraction.clamp_min(tiny)
            ).tolist()

        return {
            "schema_version": 2,
            "axes": {
                "log10_temperature_k": log_temperature.tolist(),
                "log10_gas_pressure_pa": log_pressure.tolist(),
            },
            "log10_electron_density_m3": (
                log_electron_density / math.log(10.0)
            ).tolist(),
            "log10_hydrogen_fractions": log10_hydrogen_fractions,
            "log10_ion_fractions": log10_ion_fractions,
            "log10_lower_level_fraction_of_element": log10_lower_level_fractions,
            "interpolation": "differentiable tensor-product cubic lookup in log10(T) and log10(Pgas); logarithmic quantities interpolated before exponentiation and fraction renormalization",
            "reference_solver": {
                "equations": "Saha-Boltzmann particle and charge conservation",
                "method": "bracketed bisection in log electron density",
                "iterations": int(iterations),
                "runtime_iterations": 0,
            },
        }

    def _interpolate_log_table(
        self,
        values: torch.Tensor,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
    ) -> torch.Tensor:
        log_temperature = torch.log10(temperature)
        log_pressure = torch.log10(gas_pressure)
        t_axis = self.table_log_temperature.to(temperature)
        p_axis = self.table_log_pressure.to(temperature)
        tolerance = 16.0 * torch.finfo(temperature.dtype).eps
        if torch.any(log_temperature < t_axis[0] - tolerance) or torch.any(
            log_temperature > t_axis[-1] + tolerance
        ):
            raise ValueError("temperature lies outside the prepared LTE EOS table.")
        if torch.any(log_pressure < p_axis[0] - tolerance) or torch.any(
            log_pressure > p_axis[-1] + tolerance
        ):
            raise ValueError("gas pressure lies outside the prepared LTE EOS table.")
        t_coordinate = (log_temperature - t_axis[0]) / (t_axis[1] - t_axis[0])
        p_coordinate = (log_pressure - p_axis[0]) / (p_axis[1] - p_axis[0])
        t_lower = torch.floor(t_coordinate).long().clamp(0, t_axis.numel() - 2)
        p_lower = torch.floor(p_coordinate).long().clamp(0, p_axis.numel() - 2)
        t_weight = (t_coordinate - t_lower).clamp(0.0, 1.0)
        p_weight = (p_coordinate - p_lower).clamp(0.0, 1.0)
        values = values.to(temperature)
        offsets = torch.arange(-1, 3, device=temperature.device)
        t_indices = (t_lower[..., None] + offsets).clamp(0, t_axis.numel() - 1)
        p_indices = (p_lower[..., None] + offsets).clamp(0, p_axis.numel() - 1)
        neighborhood = values[t_indices[..., :, None], p_indices[..., None, :]]

        def cubic(samples: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            p0, p1, p2, p3 = samples.unbind(dim=-1)
            return p1 + 0.5 * weight * (
                p2 - p0
                + weight * (
                    2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                    + weight * (3.0 * (p1 - p2) + p3 - p0)
                )
            )

        pressure_interpolated = cubic(neighborhood, p_weight[..., None])
        return cubic(pressure_interpolated, t_weight)

    def _electron_density_from_table(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> torch.Tensor:
        log_electron_density = self._interpolate_log_table(
            self.table_log_electron_density, temperature, gas_pressure
        )
        return torch.pow(temperature.new_tensor(10.0), log_electron_density)

    def _partition(self, element: str, stage: int, temperature: torch.Tensor) -> torch.Tensor:
        if stage not in (1, 2, 3):
            raise ValueError("The photospheric EOS supports ion stages I, II, and III")
        suffix = ("I", "II", "III")[stage - 1]
        return self.atomic.partition_function(f"{element}_{suffix}", temperature)

    def log_saha_factor(
        self,
        element: str,
        temperature: torch.Tensor,
        ionization_stage: int = 1,
    ) -> torch.Tensor:
        """Return ``log(S_i)`` for the Saha ratio ``n_(i+1)/n_i=S_i/n_e``."""

        temperature = torch.as_tensor(temperature)
        if torch.any(temperature <= 0):
            raise ValueError("temperature must be strictly positive")
        if ionization_stage not in (1, 2):
            raise ValueError("ionization_stage must be 1 (I->II) or 2 (II->III)")
        metadata = self.atomic.element(element)
        ionization_ev = (
            metadata.ionization_ev
            if ionization_stage == 1
            else metadata.second_ionization_ev
        )
        if ionization_ev is None:
            raise ValueError(f"No stage-{ionization_stage} ionization energy for {element!r}")
        ionization_temperature = ionization_ev * ELECTRON_VOLT / K_BOLTZMANN
        u_lower = self._partition(element, ionization_stage, temperature)
        u_upper = self._partition(element, ionization_stage + 1, temperature)
        return (
            temperature.new_tensor(_LOG_SAHA_PREFACTOR)
            + 1.5 * torch.log(temperature)
            + torch.log(u_upper)
            - torch.log(u_lower)
            - ionization_temperature / temperature
        )

    def saha_ratio(
        self,
        element: str,
        temperature: torch.Tensor,
        electron_density: torch.Tensor,
        ionization_stage: int = 1,
    ) -> torch.Tensor:
        """Return the LTE ratio ``n_(i+1)/n_i`` for stage I or II."""

        temperature, electron_density = torch.broadcast_tensors(temperature, electron_density)
        if torch.any(electron_density <= 0):
            raise ValueError("electron_density must be strictly positive")
        return torch.exp(
            self.log_saha_factor(element, temperature, ionization_stage)
            - torch.log(electron_density)
        )

    @staticmethod
    def h_minus_equilibrium_factor(temperature: torch.Tensor) -> torch.Tensor:
        """Return ``n(H-)/(n(H I) n_e)`` in m³.

        The factor follows the detailed-balance expression used by RH and
        Lightweaver, with partition weights ``U(H-)=1`` and ``U(H I)=2``.
        """

        temperature = torch.as_tensor(temperature)
        if torch.any(temperature <= 0):
            raise ValueError("temperature must be strictly positive")
        return torch.exp(
            temperature.new_tensor(_LOG_H_MINUS_TRANSLATIONAL)
            - 1.5 * torch.log(temperature)
            + (H_MINUS_AFFINITY_EV * ELECTRON_VOLT / K_BOLTZMANN)
            / temperature
        )

    @classmethod
    def h_minus_population(
        cls,
        temperature: torch.Tensor,
        electron_density: torch.Tensor,
        hydrogen_neutral: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``n(H-)`` in m⁻³ from neutral-H and electron densities."""

        temperature, electron_density, hydrogen_neutral = torch.broadcast_tensors(
            temperature, electron_density, hydrogen_neutral
        )
        return (
            hydrogen_neutral
            * electron_density
            * cls.h_minus_equilibrium_factor(temperature)
        )

    @staticmethod
    def _hydrogen_fractions(
        log_saha_h: torch.Tensor,
        electron_density: torch.Tensor,
        h_minus_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio_ionized = torch.exp(log_saha_h - torch.log(electron_density))
        ratio_minus = electron_density * h_minus_factor
        denominator = 1.0 + ratio_ionized + ratio_minus
        neutral = 1.0 / denominator
        return neutral, ratio_ionized / denominator, ratio_minus / denominator

    @staticmethod
    def _charge_and_log_derivative(
        log_electron_density: torch.Tensor,
        log_saha_first: Mapping[str, torch.Tensor],
        log_saha_second: Mapping[str, torch.Tensor],
        abundances: Mapping[str, torch.Tensor],
        h_minus_factor: torch.Tensor,
        element_symbols: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Net charge per H nucleus and its derivative with respect to log(ne)."""

        log_ratio_h_ionized = log_saha_first["H"] - log_electron_density
        log_ratio_h_minus = log_electron_density + torch.log(h_minus_factor)
        log_weights = torch.stack(
            (
                torch.zeros_like(log_electron_density),
                log_ratio_h_ionized,
                log_ratio_h_minus,
            ),
            dim=-1,
        )
        h_neutral_f, h_ionized_f, h_minus_f = torch.softmax(log_weights, dim=-1).unbind(-1)
        del h_neutral_f
        # d log(weight)/d log(ne) is [0, -1, +1].
        mean_log_weight_derivative = -h_ionized_f + h_minus_f
        d_h_ionized = h_ionized_f * (-1.0 - mean_log_weight_derivative)
        d_h_minus = h_minus_f * (1.0 - mean_log_weight_derivative)
        charge = h_ionized_f - h_minus_f
        derivative = d_h_ionized - d_h_minus

        # Treat all metals as one tensor dimension. The previous scalar loop
        # launched one softmax and several pointwise kernels per element for
        # every Newton iteration, which dominated EOS/MHS runtime on both CPU
        # and accelerators. This is algebraically the same three-stage Saha
        # sum, with element as the penultimate axis.
        metals = tuple(symbol for symbol in element_symbols if symbol != "H")
        if metals:
            log_ratio_first = torch.stack(
                [log_saha_first[symbol] for symbol in metals], dim=-1
            ) - log_electron_density[..., None]
            log_ratio_second = torch.stack(
                [log_saha_second[symbol] for symbol in metals], dim=-1
            ) - log_electron_density[..., None]
            fractions = torch.softmax(
                torch.stack(
                    (
                        torch.zeros_like(log_ratio_first),
                        log_ratio_first,
                        log_ratio_first + log_ratio_second,
                    ),
                    dim=-1,
                ),
                dim=-1,
            )
            singly_ionized = fractions[..., 1]
            doubly_ionized = fractions[..., 2]
            mean_weight_derivative = -singly_ionized - 2.0 * doubly_ionized
            charge_per_nucleus = singly_ionized + 2.0 * doubly_ionized
            charge_derivative = (
                singly_ionized * (-1.0 - mean_weight_derivative)
                + 2.0 * doubly_ionized * (-2.0 - mean_weight_derivative)
            )
            abundance = torch.stack([abundances[symbol] for symbol in metals])
            charge = charge + (abundance * charge_per_nucleus).sum(dim=-1)
            derivative = derivative + (abundance * charge_derivative).sum(dim=-1)
        return charge, derivative

    def forward(self, temperature: torch.Tensor, gas_pressure: torch.Tensor) -> EOSState:
        """Look up the LTE state for ``temperature`` [K] and ``Pgas`` [Pa]."""

        temperature, gas_pressure = torch.broadcast_tensors(temperature, gas_pressure)
        if not temperature.is_floating_point():
            temperature = temperature.to(torch.get_default_dtype())
            gas_pressure = gas_pressure.to(temperature)
        if torch.any(temperature <= 0):
            raise ValueError("temperature must be strictly positive")
        if torch.any(gas_pressure <= 0):
            raise ValueError("gas_pressure must be strictly positive")

        # Form the ideal-gas density in log space.  The mathematically
        # equivalent direct division ``P / (k_B T)`` has an ill-conditioned
        # float32 backward pass: autograd squares the ~1e-19 denominator and
        # underflows it before taking the quotient.  Exponentiating this log
        # expression keeps both the value and its derivative representable.
        log_total_particle_density = (
            torch.log(gas_pressure)
            - math.log(K_BOLTZMANN)
            - torch.log(temperature)
        )
        total_particle_density = torch.exp(log_total_particle_density)
        abundances = {
            symbol: temperature.new_tensor(self.atomic.abundance_ratio(symbol))
            for symbol in self.element_symbols
        }
        nuclei_per_hydrogen = sum(abundances.values())
        electron_density = self._electron_density_from_table(temperature, gas_pressure)

        hydrogen_total = (total_particle_density - electron_density) / nuclei_per_hydrogen
        hydrogen_fraction_values = torch.stack([
            torch.pow(
                temperature.new_tensor(10.0),
                self._interpolate_log_table(
                    table, temperature, gas_pressure
                ),
            )
            for table in self.table_log_hydrogen_fractions
        ], dim=-1)
        hydrogen_fraction_values = hydrogen_fraction_values / hydrogen_fraction_values.sum(
            dim=-1, keepdim=True
        )
        h_neutral_f, h_ionized_f, h_minus_f = hydrogen_fraction_values.unbind(-1)
        hydrogen_neutral = hydrogen_total * h_neutral_f
        hydrogen_ionized = hydrogen_total * h_ionized_f
        hydrogen_minus = hydrogen_total * h_minus_f

        populations: dict[str, torch.Tensor] = {
            "H_I": hydrogen_neutral,
            "H_II": hydrogen_ionized,
            "H_minus": hydrogen_minus,
        }
        table_species = {
            name: table
            for name, table in zip(self.ion_species, self.table_log_ion_fractions)
        }
        table_elements = tuple(sorted({name.rsplit("_", 1)[0] for name in self.ion_species}))
        for symbol in table_elements:
            names = tuple(f"{symbol}_{suffix}" for suffix in ("I", "II", "III"))
            fractions = torch.stack([
                torch.pow(
                    temperature.new_tensor(10.0),
                    self._interpolate_log_table(
                        table_species[name], temperature, gas_pressure
                    ),
                )
                for name in names
            ], dim=-1)
            fractions = fractions / fractions.sum(dim=-1, keepdim=True)
            total = hydrogen_total * abundances[symbol]
            for index, name in enumerate(names):
                populations[name] = total * fractions[..., index]

        lower_level_populations = {}
        for line_id, table in zip(
            self.lower_level_ids, self.table_log_lower_level_fractions
        ):
            line = self.atomic.get_line(line_id)
            fraction = torch.pow(
                temperature.new_tensor(10.0),
                self._interpolate_log_table(table, temperature, gas_pressure),
            )
            lower_level_populations[line_id] = (
                hydrogen_total * abundances[line.element] * fraction
            )

        mass_per_hydrogen = sum(
            abundances[symbol] * self.atomic.element(symbol).atomic_mass_u
            for symbol in self.element_symbols
        ) * ATOMIC_MASS_UNIT
        mass_density = hydrogen_total * mass_per_hydrogen
        return EOSState(
            gas_pressure=gas_pressure,
            electron_pressure=electron_density * K_BOLTZMANN * temperature,
            electron_density=electron_density,
            hydrogen_total=hydrogen_total,
            hydrogen_neutral=hydrogen_neutral,
            hydrogen_ionized=hydrogen_ionized,
            hydrogen_minus=hydrogen_minus,
            mass_density=mass_density,
            populations=populations,
            lower_level_populations=lower_level_populations,
        )

    def lower_level_population(
        self,
        line: SpectralLine,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        eos_state: EOSState | None = None,
    ) -> torch.Tensor:
        """Return the Boltzmann lower-level population in m⁻³."""

        state = self(temperature, gas_pressure) if eos_state is None else eos_state
        if line.id in state.lower_level_populations:
            return state.lower_level_populations[line.id]
        if line.ion_stage not in (1, 2, 3):
            raise ValueError("Line ionization stage must be I, II, or III")
        suffix = ("I", "II", "III")[line.ion_stage - 1]
        population_key = f"{line.element}_{suffix}"
        if population_key not in state.populations:
            raise KeyError(
                f"EOS table has no population for {population_key!r}; regenerate it "
                f"with line {line.id!r} included in the atomic database."
            )
        ion_population = state.populations[population_key]
        partition = self._partition(line.element, line.ion_stage, temperature)
        excitation_temperature = (
            line.lower_excitation_ev * ELECTRON_VOLT / K_BOLTZMANN
        )
        return (
            ion_population
            * line.lower_statistical_weight
            / partition
            * torch.exp(-excitation_temperature / temperature)
        )

__all__ = [
    "ATOMIC_MASS_UNIT",
    "ELECTRON_VOLT",
    "EOSState",
    "H_PLANCK",
    "K_BOLTZMANN",
    "LTEEOS",
    "M_ELECTRON",
]
