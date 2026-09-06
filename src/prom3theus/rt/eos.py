"""Thermodynamic equations of state independent of radiative opacity."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import torch
from torch import nn

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN
from prom3theus.resources import verify_manifest_resource

from ._interpolation import clamped_catmull_rom, uniform_axis_coordinate
from .atomic import AtomicDatabase


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


@dataclass(frozen=True)
class EOSLookupState:
    """Reusable interpolation coordinates on the photospheric STiC table."""

    temperature: torch.Tensor
    t_indices: torch.Tensor
    p_indices: torch.Tensor
    t_weight: torch.Tensor
    p_weight: torch.Tensor


class HybridSolarEOS(nn.Module):
    """STiC, CHIANTI-equilibrium, and fully ionized solar EoS closure.

    Radiative coefficients remain the responsibility of :class:`ContinuumOpacity`.
    This object supplies only thermodynamic closure for magnetofluid equations and
    hydrostatic reference construction. It preserves the packaged STiC states over
    their native temperature interval, freezes composition below the cold endpoint,
    joins the hot endpoint to the default CHIANTI coronal-equilibrium charge
    distribution, and finally approaches the fully ionized ideal-mixture limit.
    """

    def __init__(self, atomic_database: AtomicDatabase | None = None):
        super().__init__()
        atomic = atomic_database or AtomicDatabase()
        table_resource = atomic.data_root.joinpath("stic_continuum_table.json")
        self.stic_table_sha256 = verify_manifest_resource(
            table_resource,
            atomic.source_manifest,
            resource_name="common/stic_continuum_table.json",
            kind="thermodynamic-EoS",
        )
        with _open_text(table_resource) as handle:
            document = json.load(handle)
        if document.get("schema_version") != 2:
            raise ValueError("Unsupported STiC thermodynamic table schema.")
        axes = document.get("axes", {})
        log_temperature = torch.tensor(
            axes.get("log10_temperature_k", ()), dtype=torch.float64
        )
        log_pressure = torch.tensor(
            axes.get("log10_gas_pressure_pa", ()), dtype=torch.float64
        )
        log_density = torch.tensor(
            document.get("log10_mass_density_kg_m3", ()), dtype=torch.float64
        )
        log_electron_density = torch.tensor(
            document.get("log10_electron_density_m3", ()), dtype=torch.float64
        )
        expected = (log_temperature.numel(), log_pressure.numel())
        if (
            min(expected) < 4
            or log_density.shape != expected
            or log_electron_density.shape != expected
            or not torch.isfinite(log_density).all()
            or not torch.isfinite(log_electron_density).all()
        ):
            raise ValueError("Invalid STiC thermodynamic EoS arrays.")
        self.register_buffer("stic_log_temperature", log_temperature, persistent=False)
        self.register_buffer("stic_log_pressure", log_pressure, persistent=False)
        self.register_buffer("stic_log_mass_density", log_density, persistent=False)
        self.register_buffer(
            "stic_log_electron_density", log_electron_density, persistent=False
        )
        self.register_buffer(
            "_cubic_offsets", torch.arange(-1, 3, dtype=torch.long), persistent=False
        )

        chianti_resource = atomic.data_root.joinpath("chianti_thermodynamic_table.json")
        self.chianti_table_sha256 = verify_manifest_resource(
            chianti_resource,
            atomic.source_manifest,
            resource_name="common/chianti_thermodynamic_table.json",
            kind="thermodynamic-EoS",
        )
        with _open_text(chianti_resource) as handle:
            chianti_document = json.load(handle)
        if chianti_document.get("schema_version") != 1:
            raise ValueError("Unsupported CHIANTI thermodynamic table schema.")
        chianti_log_temperature = torch.tensor(
            chianti_document.get("axes", {}).get("log10_temperature_k", ()),
            dtype=torch.float64,
        )
        chianti_log_electrons = torch.tensor(
            chianti_document.get("log10_free_electrons_per_h_nucleus", ()),
            dtype=torch.float64,
        )
        chianti_slopes = torch.tensor(
            chianti_document.get(
                "pchip_d_log10_free_electrons_per_h_d_log10_temperature", ()
            ),
            dtype=torch.float64,
        )
        if (
            chianti_log_temperature.ndim != 1
            or chianti_log_temperature.numel() < 4
            or chianti_log_electrons.shape != chianti_log_temperature.shape
            or chianti_slopes.shape != chianti_log_temperature.shape
            or not torch.isfinite(chianti_log_temperature).all()
            or not torch.isfinite(chianti_log_electrons).all()
            or not torch.isfinite(chianti_slopes).all()
            or torch.any(chianti_slopes < 0.0)
            or not torch.all(chianti_log_temperature[1:] > chianti_log_temperature[:-1])
            or not torch.all(chianti_log_electrons[1:] > chianti_log_electrons[:-1])
        ):
            raise ValueError("Invalid CHIANTI thermodynamic EoS arrays.")

        composition = chianti_document.get("composition", {})
        fully_ionized = chianti_document.get("fully_ionized_limit", {})
        try:
            self.mass_u_per_h_nucleus = float(composition["mass_u_per_h_nucleus"])
            self.nuclei_per_h_nucleus = float(composition["nuclei_per_h_nucleus"])
            self.fully_ionized_electrons_per_h_nucleus = float(
                composition["fully_ionized_electrons_per_h_nucleus"]
            )
            self.mean_molecular_weight_fully_ionized = float(
                fully_ionized["mean_molecular_weight"]
            )
            self.mean_molecular_weight_per_electron = float(
                fully_ionized["mean_molecular_weight_per_electron"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid CHIANTI composition metadata.") from error
        if (
            min(
                self.mass_u_per_h_nucleus,
                self.nuclei_per_h_nucleus,
                self.fully_ionized_electrons_per_h_nucleus,
            )
            <= 0.0
            or not math.isclose(
                self.mean_molecular_weight_fully_ionized,
                self.mass_u_per_h_nucleus
                / (
                    self.nuclei_per_h_nucleus
                    + self.fully_ionized_electrons_per_h_nucleus
                ),
                rel_tol=1.0e-12,
            )
            or not math.isclose(
                self.mean_molecular_weight_per_electron,
                self.mass_u_per_h_nucleus / self.fully_ionized_electrons_per_h_nucleus,
                rel_tol=1.0e-12,
            )
        ):
            raise ValueError("Inconsistent CHIANTI fully ionized limit.")

        source = chianti_document.get("reference_ionization_equilibrium", {})
        source_records = atomic.source_manifest.get("sources", {})
        if (
            source.get("database_release") != "11.0.2"
            or source.get("source_sha256")
            != source_records.get("chianti_ioneq_v11_0_2", {}).get("sha256")
            or source.get("version_source_sha256")
            != source_records.get("chianti_database_version_11_0_2", {}).get("sha256")
        ):
            raise ValueError("CHIANTI thermodynamic provenance is inconsistent.")

        contract = chianti_document.get("runtime_contract", {})
        try:
            self.minimum_temperature_k = float(contract["minimum_temperature_k"])
            self.stic_exact_max_temperature_k = float(
                contract["stic_exact_max_temperature_k"]
            )
            self.transition_temperature_bounds_k = tuple(
                float(value)
                for value in contract["stic_to_chianti_log_blend_temperature_k"]
            )
            self.ideal_transition_temperature_bounds_k = tuple(
                float(value)
                for value in contract[
                    "chianti_to_fully_ionized_log_blend_temperature_k"
                ]
            )
            lower_transition = contract["lower_transition"]
            self.lower_tangent_match_width_log10_temperature = float(
                lower_transition["endpoint_tangent_match_width_log10_temperature"]
            )
            self.lower_tangent_ramp_power = int(
                lower_transition["endpoint_tangent_ramp_power"]
            )
            pressure_continuation = contract["pressure_continuation"]
            self.pressure_tangent_decay_width_log10_pressure = float(
                pressure_continuation["tangent_decay_width_log10_pressure"]
            )
            self.pressure_tangent_decay_power = int(
                pressure_continuation["tangent_decay_power"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid thermodynamic EoS runtime contract.") from error
        if (
            len(self.transition_temperature_bounds_k) != 2
            or len(self.ideal_transition_temperature_bounds_k) != 2
            or not math.isclose(
                self.minimum_temperature_k,
                10.0 ** float(log_temperature[0]),
                rel_tol=1.0e-12,
            )
            or not math.isclose(
                self.stic_exact_max_temperature_k,
                10.0 ** float(log_temperature[-1]),
                rel_tol=1.0e-12,
            )
            or not math.isclose(
                self.transition_temperature_bounds_k[0],
                self.stic_exact_max_temperature_k,
                rel_tol=1.0e-12,
            )
            or not (
                self.transition_temperature_bounds_k[0]
                < self.transition_temperature_bounds_k[1]
                < self.ideal_transition_temperature_bounds_k[0]
                < self.ideal_transition_temperature_bounds_k[1]
                <= 10.0 ** float(chianti_log_temperature[-1])
            )
            or not (
                0.0
                < self.lower_tangent_match_width_log10_temperature
                <= 0.5
                * math.log10(
                    self.transition_temperature_bounds_k[1]
                    / self.transition_temperature_bounds_k[0]
                )
            )
            or self.lower_tangent_ramp_power < 2
            or self.pressure_tangent_decay_width_log10_pressure <= 0.0
            or self.pressure_tangent_decay_power < 1
            or not isinstance(lower_transition, dict)
            or not isinstance(pressure_continuation, dict)
            or lower_transition.get("electron_coordinate")
            != "logit(free_electrons_per_H / fully_ionized_electrons_per_H)"
            or lower_transition.get("non_electron_coordinate")
            != "log10(non_electron_particles_per_H)"
            or pressure_continuation.get("coordinates")
            != (
                "logit(electrons_per_H / fully_ionized_electrons_per_H) and "
                "log10(non_electron_particles_per_H)"
            )
            or pressure_continuation.get("derivative_ramp") != "(1 - u)^p"
        ):
            raise ValueError("Thermodynamic EoS transition bounds are inconsistent.")

        reconstructed_chianti_slopes = self._pchip_slopes(
            chianti_log_temperature, chianti_log_electrons
        )
        if not torch.allclose(
            chianti_slopes,
            reconstructed_chianti_slopes,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError("Inconsistent CHIANTI PCHIP interpolation slopes.")
        self.register_buffer(
            "chianti_log_temperature",
            chianti_log_temperature,
            persistent=False,
        )
        self.register_buffer(
            "chianti_log_electrons_per_h",
            chianti_log_electrons,
            persistent=False,
        )
        self.register_buffer(
            "chianti_log_electron_slopes",
            chianti_slopes,
            persistent=False,
        )
        self.chianti_equilibrium_model = str(source["model"])
        top = document.get("falc_top_boundary", {})
        self.reference_gravity_m_per_s2 = float(top["gravity_cm_s2"]) / 100.0

    def _validate_state(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = torch.as_tensor(temperature)
        pressure = torch.as_tensor(
            gas_pressure, dtype=temperature.dtype, device=temperature.device
        )
        if temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError("Thermodynamic EoS supports float32 and float64 only.")
        temperature, pressure = torch.broadcast_tensors(temperature, pressure)
        if (
            not torch.isfinite(temperature).all()
            or not torch.isfinite(pressure).all()
            or torch.any(temperature <= 0.0)
            or torch.any(pressure <= 0.0)
        ):
            raise ValueError(
                "temperature and gas_pressure must be finite and strictly positive"
            )
        return temperature, pressure

    def _coordinates(
        self, temperature: torch.Tensor, pressure: torch.Tensor
    ) -> tuple[EOSLookupState, torch.Tensor, torch.Tensor]:
        t_axis = self.stic_log_temperature.to(temperature)
        p_axis = self.stic_log_pressure.to(temperature)
        anchor_log_temperature = torch.log10(temperature).clamp(t_axis[0], t_axis[-1])
        anchor_log_pressure = torch.log10(pressure).clamp(p_axis[0], p_axis[-1])
        t_coordinate = uniform_axis_coordinate(anchor_log_temperature, t_axis)
        p_coordinate = uniform_axis_coordinate(anchor_log_pressure, p_axis)
        t_lower = torch.floor(t_coordinate).long().clamp(0, t_axis.numel() - 2)
        p_lower = torch.floor(p_coordinate).long().clamp(0, p_axis.numel() - 2)
        offsets = self._cubic_offsets.to(temperature.device)
        state = EOSLookupState(
            temperature=temperature,
            t_indices=(t_lower[..., None] + offsets).clamp(0, t_axis.numel() - 1),
            p_indices=(p_lower[..., None] + offsets).clamp(0, p_axis.numel() - 1),
            t_weight=(t_coordinate - t_lower).clamp(0.0, 1.0),
            p_weight=(p_coordinate - p_lower).clamp(0.0, 1.0),
        )
        return state, anchor_log_temperature, anchor_log_pressure

    @staticmethod
    def _cubic(
        samples: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return clamped_catmull_rom(samples, weight)

    def _interpolate(self, table: torch.Tensor, state: EOSLookupState) -> torch.Tensor:
        values = table.to(state.temperature)
        neighborhood = values[
            state.t_indices[..., :, None], state.p_indices[..., None, :]
        ]
        pressure_interpolated = self._cubic(
            neighborhood,
            state.p_weight[..., None],
        )
        return self._cubic(
            pressure_interpolated,
            state.t_weight,
        )

    def _pressure_continued_states(
        self,
        state: EOSLookupState,
        log_pressure: torch.Tensor,
        anchor_log_pressure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate STiC states with a bounded, C1 pressure continuation.

        The table-edge derivatives are carried in the bounded physical coordinates
        ``logit(e/E_inf)`` and ``log10(h)`` and smoothly decay to zero.  The
        resulting particle mapping is therefore constant sufficiently far outside
        the table, so both density and electron density recover their required
        linear pressure scaling without permitting unphysical charge or a negative
        heavy-particle count.
        """

        interior_log_density = self._interpolate(self.stic_log_mass_density, state)
        interior_log_electron_density = self._interpolate(
            self.stic_log_electron_density, state
        )
        pressure_axis = self.stic_log_pressure.to(state.temperature)
        below = log_pressure < pressure_axis[0]
        above = log_pressure > pressure_axis[-1]
        outside = below | above

        # Continue every temperature-neighborhood node before the final cubic
        # interpolation in T. This preserves the same one-sided temperature
        # derivative at the hot edge that is handed to the STiC/CHIANTI bridge.
        log_density_table = self.stic_log_mass_density.to(state.temperature)
        log_electron_density_table = self.stic_log_electron_density.to(
            state.temperature
        )
        pressure_cell_left = torch.where(
            below,
            torch.zeros_like(state.t_weight, dtype=torch.long),
            torch.full_like(
                state.t_weight,
                pressure_axis.numel() - 2,
                dtype=torch.long,
            ),
        )
        pressure_cell_right = pressure_cell_left + 1
        pressure_edge = torch.where(below, pressure_cell_left, pressure_cell_right)
        edge_log_density = log_density_table[state.t_indices, pressure_edge[..., None]]
        edge_log_electron_density = log_electron_density_table[
            state.t_indices, pressure_edge[..., None]
        ]
        pressure_cell_width = (
            pressure_axis[pressure_cell_right] - pressure_axis[pressure_cell_left]
        )[..., None]
        density_slope = (
            0.5
            * (
                log_density_table[state.t_indices, pressure_cell_right[..., None]]
                - log_density_table[state.t_indices, pressure_cell_left[..., None]]
            )
            / pressure_cell_width
        )
        electron_density_slope = (
            0.5
            * (
                log_electron_density_table[
                    state.t_indices, pressure_cell_right[..., None]
                ]
                - log_electron_density_table[
                    state.t_indices, pressure_cell_left[..., None]
                ]
            )
            / pressure_cell_width
        )

        log_mass_per_h_kg = math.log10(self.mass_u_per_h_nucleus * ATOMIC_MASS_UNIT)
        edge_log_temperature = self.stic_log_temperature.to(state.temperature)[
            state.t_indices
        ]
        edge_log_pressure = pressure_axis[pressure_edge][..., None]
        log_particles_per_h = (
            edge_log_pressure
            + log_mass_per_h_kg
            - math.log10(K_BOLTZMANN)
            - edge_log_temperature
            - edge_log_density
        )
        log_electrons_per_h = (
            edge_log_electron_density + log_mass_per_h_kg - edge_log_density
        )
        ten = state.temperature.new_tensor(10.0)
        particles_per_h = torch.pow(ten, log_particles_per_h)
        electrons_per_h = torch.pow(ten, log_electrons_per_h)
        fully_ionized_electrons = state.temperature.new_tensor(
            self.fully_ionized_electrons_per_h_nucleus
        )

        # Clamps make the unused branch total for arbitrary vectorized queries;
        # the packaged STiC edge states lie strictly inside both bounds, so these
        # operations do not alter values or derivatives on the supported table.
        finfo = torch.finfo(state.temperature.dtype)
        electron_fraction = (electrons_per_h / fully_ionized_electrons).clamp(
            finfo.tiny, 1.0 - finfo.eps
        )
        non_electron_particles = (particles_per_h - electrons_per_h).clamp_min(
            finfo.tiny
        )
        charge_coordinate = torch.logit(electron_fraction)
        log_non_electron_particles = torch.log10(non_electron_particles)

        particles_slope = 1.0 - density_slope
        electrons_slope = electron_density_slope - density_slope
        charge_slope = math.log(10.0) * electrons_slope / (1.0 - electron_fraction)
        non_electron_slope = (
            particles_per_h * particles_slope - electrons_per_h * electrons_slope
        ) / non_electron_particles

        distance = torch.abs(log_pressure - anchor_log_pressure)[..., None]
        width = self.pressure_tangent_decay_width_log10_pressure
        fraction = (distance / width).clamp(0.0, 1.0)
        integrated_ramp = (
            width
            / (self.pressure_tangent_decay_power + 1)
            * (1.0 - torch.pow(1.0 - fraction, self.pressure_tangent_decay_power + 1))
        )
        signed_integrated_ramp = torch.where(
            below[..., None],
            -integrated_ramp,
            torch.where(
                above[..., None],
                integrated_ramp,
                torch.zeros_like(integrated_ramp),
            ),
        )
        continued_charge_coordinate = (
            charge_coordinate + charge_slope * signed_integrated_ramp
        )
        continued_log_non_electron_particles = (
            log_non_electron_particles + non_electron_slope * signed_integrated_ramp
        )
        continued_electrons_per_h = fully_ionized_electrons * torch.sigmoid(
            continued_charge_coordinate
        )
        continued_non_electron_particles = torch.pow(
            ten, continued_log_non_electron_particles
        )
        continued_particles_per_h = (
            continued_non_electron_particles + continued_electrons_per_h
        )
        continued_log_density = (
            log_pressure[..., None]
            + log_mass_per_h_kg
            - math.log10(K_BOLTZMANN)
            - edge_log_temperature
            - torch.log10(continued_particles_per_h)
        )
        continued_log_electron_density = (
            torch.log10(continued_electrons_per_h)
            + continued_log_density
            - log_mass_per_h_kg
        )
        continued_log_density = self._cubic(
            continued_log_density,
            state.t_weight,
        )
        continued_log_electron_density = self._cubic(
            continued_log_electron_density,
            state.t_weight,
        )
        return (
            torch.where(outside, continued_log_density, interior_log_density),
            torch.where(
                outside,
                continued_log_electron_density,
                interior_log_electron_density,
            ),
        )

    @staticmethod
    def _pchip_slopes(axis: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Return shape-preserving cubic slopes for one strictly monotone table."""

        spacing = axis[1:] - axis[:-1]
        secant = (values[1:] - values[:-1]) / spacing
        slopes = torch.zeros_like(values)
        previous, following = secant[:-1], secant[1:]
        weight_previous = 2.0 * spacing[1:] + spacing[:-1]
        weight_following = spacing[1:] + 2.0 * spacing[:-1]
        harmonic = (weight_previous + weight_following) / (
            weight_previous / previous + weight_following / following
        )
        slopes[1:-1] = torch.where(previous * following > 0.0, harmonic, 0.0)

        first = (
            (2.0 * spacing[0] + spacing[1]) * secant[0] - spacing[0] * secant[1]
        ) / (spacing[0] + spacing[1])
        first = torch.where(first * secant[0] <= 0.0, 0.0, first)
        first = torch.where(
            (secant[0] * secant[1] < 0.0)
            & (torch.abs(first) > 3.0 * torch.abs(secant[0])),
            3.0 * secant[0],
            first,
        )
        last = (
            (2.0 * spacing[-1] + spacing[-2]) * secant[-1] - spacing[-1] * secant[-2]
        ) / (spacing[-1] + spacing[-2])
        last = torch.where(last * secant[-1] <= 0.0, 0.0, last)
        last = torch.where(
            (secant[-1] * secant[-2] < 0.0)
            & (torch.abs(last) > 3.0 * torch.abs(secant[-1])),
            3.0 * secant[-1],
            last,
        )
        slopes[0], slopes[-1] = first, last
        return slopes

    def _chianti_log_electrons_and_slope(
        self, log_temperature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate log10(electrons/H) and its log-temperature derivative."""

        axis = self.chianti_log_temperature.to(log_temperature)
        values = self.chianti_log_electrons_per_h.to(log_temperature)
        slopes = self.chianti_log_electron_slopes.to(log_temperature)
        query = log_temperature.clamp(axis[0], axis[-1])
        upper = torch.searchsorted(axis, query.detach().contiguous(), right=True).clamp(
            1, axis.numel() - 1
        )
        lower = upper - 1
        x0, x1 = axis[lower], axis[upper]
        width = x1 - x0
        fraction = (query - x0) / width
        fraction2 = fraction.square()
        fraction3 = fraction2 * fraction
        interpolated = (
            (2.0 * fraction3 - 3.0 * fraction2 + 1.0) * values[lower]
            + (fraction3 - 2.0 * fraction2 + fraction) * width * slopes[lower]
            + (-2.0 * fraction3 + 3.0 * fraction2) * values[upper]
            + (fraction3 - fraction2) * width * slopes[upper]
        )
        derivative = (
            (6.0 * fraction2 - 6.0 * fraction) * values[lower] / width
            + (3.0 * fraction2 - 4.0 * fraction + 1.0) * slopes[lower]
            + (-6.0 * fraction2 + 6.0 * fraction) * values[upper] / width
            + (3.0 * fraction2 - 2.0 * fraction) * slopes[upper]
        )
        inside = (log_temperature >= axis[0]) & (log_temperature <= axis[-1])
        return interpolated, torch.where(inside, derivative, 0.0)

    def _chianti_log_electrons(self, log_temperature: torch.Tensor) -> torch.Tensor:
        return self._chianti_log_electrons_and_slope(log_temperature)[0]

    @staticmethod
    def _transition_weight(
        log_temperature: torch.Tensor, bounds_k: tuple[float, float]
    ) -> torch.Tensor:
        lower, upper = (math.log10(value) for value in bounds_k)
        fraction = ((log_temperature - lower) / (upper - lower)).clamp(0.0, 1.0)
        return fraction.pow(3) * (10.0 + fraction * (-15.0 + 6.0 * fraction))

    def _stic_log_mappings_with_endpoint(
        self, temperature: torch.Tensor, pressure: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return STiC mappings plus their value and slope at the hot edge."""

        log_temperature = torch.log10(temperature)
        # The STiC table contains no colder composition states.  Freeze only the
        # particles-per-H mapping at its cold endpoint while retaining the model's
        # actual T and P in `_states`; density and electron density then preserve
        # their ideal-gas P/T scaling without bounding the primary atmosphere.
        composition_log_temperature = log_temperature.clamp_min(
            self.stic_log_temperature[0].to(temperature)
        )
        log_pressure = torch.log10(pressure)
        lookup, anchor_log_temperature, anchor_log_pressure = self._coordinates(
            temperature, pressure
        )
        log_density, log_electron_density = self._pressure_continued_states(
            lookup,
            log_pressure,
            anchor_log_pressure,
        )

        top_log_temperature = self.stic_log_temperature[-1].to(temperature)
        previous_log_temperature = self.stic_log_temperature[-2].to(temperature)
        top_temperature = torch.full_like(temperature, 10.0).pow(top_log_temperature)
        previous_temperature = torch.full_like(temperature, 10.0).pow(
            previous_log_temperature
        )
        top_lookup, _, top_anchor_log_pressure = self._coordinates(
            top_temperature, pressure
        )
        previous_lookup, _, previous_anchor_log_pressure = self._coordinates(
            previous_temperature, pressure
        )
        top_log_density, top_log_electron_density = self._pressure_continued_states(
            top_lookup,
            log_pressure,
            top_anchor_log_pressure,
        )
        (
            previous_log_density,
            previous_log_electron_density,
        ) = self._pressure_continued_states(
            previous_lookup,
            log_pressure,
            previous_anchor_log_pressure,
        )
        inverse_width = 1.0 / (top_log_temperature - previous_log_temperature)
        density_slope = 0.5 * (top_log_density - previous_log_density) * inverse_width
        electron_density_slope = (
            0.5
            * (top_log_electron_density - previous_log_electron_density)
            * inverse_width
        )
        temperature_extrapolation = torch.where(
            log_temperature > top_log_temperature,
            log_temperature - anchor_log_temperature,
            torch.zeros_like(log_temperature),
        )
        continued_log_density = log_density + density_slope * temperature_extrapolation
        continued_log_electron_density = (
            log_electron_density + electron_density_slope * temperature_extrapolation
        )

        log_mass_per_h_kg = math.log10(self.mass_u_per_h_nucleus * ATOMIC_MASS_UNIT)
        log_particles_per_h = (
            log_pressure
            + log_mass_per_h_kg
            - math.log10(K_BOLTZMANN)
            - composition_log_temperature
            - continued_log_density
        )
        log_electrons_per_h = (
            continued_log_electron_density + log_mass_per_h_kg - continued_log_density
        )
        particle_slope = -density_slope - 1.0
        electron_slope = electron_density_slope - density_slope
        endpoint_log_particles_per_h = (
            log_pressure
            + log_mass_per_h_kg
            - math.log10(K_BOLTZMANN)
            - top_log_temperature
            - top_log_density
        )
        endpoint_log_electrons_per_h = (
            top_log_electron_density + log_mass_per_h_kg - top_log_density
        )
        return (
            log_particles_per_h,
            log_electrons_per_h,
            endpoint_log_particles_per_h,
            endpoint_log_electrons_per_h,
            particle_slope,
            electron_slope,
        )

    def _stic_log_mappings(
        self, temperature: torch.Tensor, pressure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tangent-continued log10 particles/H and electrons/H mappings."""

        mappings = self._stic_log_mappings_with_endpoint(temperature, pressure)
        return mappings[0], mappings[1]

    def _stic_to_chianti_bridge(
        self,
        log_temperature: torch.Tensor,
        stic_endpoint_log_particles: torch.Tensor,
        stic_endpoint_log_electrons: torch.Tensor,
        stic_endpoint_particle_slope: torch.Tensor,
        stic_endpoint_electron_slope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a bounded, composition-consistent C1 lower transition."""

        chianti_log_electrons, _ = self._chianti_log_electrons_and_slope(
            log_temperature
        )
        lower_log_temperature, upper_log_temperature = (
            math.log10(value) for value in self.transition_temperature_bounds_k
        )
        upper_coordinate = torch.full_like(log_temperature, upper_log_temperature)
        chianti_endpoint_log_electrons, chianti_endpoint_electron_slope = (
            self._chianti_log_electrons_and_slope(upper_coordinate)
        )

        ten = log_temperature.new_tensor(10.0)
        fully_ionized_electrons = log_temperature.new_tensor(
            self.fully_ionized_electrons_per_h_nucleus
        )
        stic_particles = torch.pow(ten, stic_endpoint_log_particles)
        stic_electrons = torch.pow(ten, stic_endpoint_log_electrons)
        stic_non_electron_particles = stic_particles - stic_electrons
        chianti_endpoint_electrons = torch.pow(ten, chianti_endpoint_log_electrons)
        if (
            torch.any(stic_electrons <= 0.0)
            or torch.any(stic_electrons >= fully_ionized_electrons)
            or torch.any(stic_non_electron_particles <= 0.0)
            or torch.any(chianti_endpoint_electrons <= 0.0)
            or torch.any(chianti_endpoint_electrons >= fully_ionized_electrons)
        ):
            raise RuntimeError(
                "The STiC/CHIANTI join is outside its physical composition bounds."
            )

        # z = logit(e / E_inf) keeps the bridged charge strictly between zero
        # and the fully ionized composition without clipping its derivatives.
        stic_charge_coordinate = torch.log(
            stic_electrons / (fully_ionized_electrons - stic_electrons)
        )
        chianti_charge_coordinate = torch.log(
            chianti_endpoint_electrons
            / (fully_ionized_electrons - chianti_endpoint_electrons)
        )
        stic_charge_slope = (
            math.log(10.0)
            * stic_endpoint_electron_slope
            * fully_ionized_electrons
            / (fully_ionized_electrons - stic_electrons)
        )
        chianti_charge_slope = (
            math.log(10.0)
            * chianti_endpoint_electron_slope
            * fully_ionized_electrons
            / (fully_ionized_electrons - chianti_endpoint_electrons)
        )
        stic_non_electron_slope = (
            stic_particles * stic_endpoint_particle_slope
            - stic_electrons * stic_endpoint_electron_slope
        ) / stic_non_electron_particles

        width = self.lower_tangent_match_width_log10_temperature
        power = self.lower_tangent_ramp_power
        transition_width = upper_log_temperature - lower_log_temperature
        fraction = ((log_temperature - lower_log_temperature) / transition_width).clamp(
            0.0, 1.0
        )
        smootherstep = fraction.pow(3) * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        lower_fraction = ((log_temperature - lower_log_temperature) / width).clamp(
            0.0, 1.0
        )
        upper_fraction = (
            (log_temperature - (upper_log_temperature - width)) / width
        ).clamp(0.0, 1.0)
        lower_ramp = 1.0 - torch.pow(1.0 - lower_fraction, power)
        upper_ramp = torch.pow(upper_fraction, power)

        # The monotone endpoint ramps carry the exact one-sided slopes. Their
        # integrated changes are removed from the smootherstep interior, so
        # values and first derivatives match both source regimes exactly.
        lower_charge_change = stic_charge_slope * width / power
        upper_charge_change = chianti_charge_slope * width / power
        middle_charge_change = (
            chianti_charge_coordinate
            - stic_charge_coordinate
            - lower_charge_change
            - upper_charge_change
        )
        if (
            torch.any(stic_charge_slope < 0.0)
            or torch.any(chianti_charge_slope < 0.0)
            or torch.any(middle_charge_change < 0.0)
        ):
            raise RuntimeError(
                "The STiC/CHIANTI charge bridge is not monotone for this state."
            )
        charge_coordinate = (
            stic_charge_coordinate
            + lower_charge_change * lower_ramp
            + middle_charge_change * smootherstep
            + upper_charge_change * upper_ramp
        )
        bridge_electrons = fully_ionized_electrons * torch.sigmoid(charge_coordinate)

        # Heavy-particle pressure is carried independently only across the
        # splice; total particles are always reconstructed as h + e.
        stic_log_non_electron_particles = torch.log10(stic_non_electron_particles)
        chianti_log_non_electron_particles = math.log10(self.nuclei_per_h_nucleus)
        lower_non_electron_change = stic_non_electron_slope * width / power
        middle_non_electron_change = (
            chianti_log_non_electron_particles
            - stic_log_non_electron_particles
            - lower_non_electron_change
        )
        bridge_log_non_electron_particles = (
            stic_log_non_electron_particles
            + lower_non_electron_change * lower_ramp
            + middle_non_electron_change * smootherstep
        )
        bridge_non_electron_particles = torch.pow(
            ten, bridge_log_non_electron_particles
        )
        bridge_log_electrons = torch.log10(bridge_electrons)
        bridge_log_particles = torch.log10(
            bridge_non_electron_particles + bridge_electrons
        )
        return bridge_log_particles, bridge_log_electrons, chianti_log_electrons

    def _log_mappings(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature, pressure = self._validate_state(temperature, gas_pressure)
        log_temperature = torch.log10(temperature)
        (
            stic_log_particles,
            stic_log_electrons,
            stic_endpoint_log_particles,
            stic_endpoint_log_electrons,
            stic_endpoint_particle_slope,
            stic_endpoint_electron_slope,
        ) = self._stic_log_mappings_with_endpoint(temperature, pressure)
        (
            bridge_log_particles,
            bridge_log_electrons,
            chianti_log_electrons,
        ) = self._stic_to_chianti_bridge(
            log_temperature,
            stic_endpoint_log_particles,
            stic_endpoint_log_electrons,
            stic_endpoint_particle_slope,
            stic_endpoint_electron_slope,
        )
        chianti_electrons = torch.pow(
            temperature.new_tensor(10.0), chianti_log_electrons
        )
        chianti_log_particles = torch.log10(
            chianti_electrons + self.nuclei_per_h_nucleus
        )
        lower_log_temperature, upper_log_temperature = (
            math.log10(value) for value in self.transition_temperature_bounds_k
        )
        log_particles = torch.where(
            log_temperature < lower_log_temperature,
            stic_log_particles,
            torch.where(
                log_temperature < upper_log_temperature,
                bridge_log_particles,
                chianti_log_particles,
            ),
        )
        log_electrons = torch.where(
            log_temperature < lower_log_temperature,
            stic_log_electrons,
            torch.where(
                log_temperature < upper_log_temperature,
                bridge_log_electrons,
                chianti_log_electrons,
            ),
        )

        ideal_log_electrons = math.log10(self.fully_ionized_electrons_per_h_nucleus)
        upper_weight = self._transition_weight(
            log_temperature, self.ideal_transition_temperature_bounds_k
        )
        transitioned_log_electrons = torch.lerp(
            log_electrons,
            torch.full_like(log_electrons, ideal_log_electrons),
            upper_weight,
        )
        transitioned_electrons = torch.pow(
            temperature.new_tensor(10.0), transitioned_log_electrons
        )
        transitioned_log_particles = torch.log10(
            transitioned_electrons + self.nuclei_per_h_nucleus
        )
        ideal_start_log_temperature = math.log10(
            self.ideal_transition_temperature_bounds_k[0]
        )
        in_ideal_transition = log_temperature > ideal_start_log_temperature
        log_particles = torch.where(
            in_ideal_transition, transitioned_log_particles, log_particles
        )
        log_electrons = torch.where(
            in_ideal_transition, transitioned_log_electrons, log_electrons
        )
        return temperature, pressure, log_particles, log_electrons

    def _states(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature, pressure, log_particles, log_electrons = self._log_mappings(
            temperature, gas_pressure
        )
        log_hydrogen_density = (
            torch.log10(pressure)
            - math.log10(K_BOLTZMANN)
            - torch.log10(temperature)
            - log_particles
        )
        ten = temperature.new_tensor(10.0)
        density = torch.pow(
            ten,
            log_hydrogen_density
            + math.log10(self.mass_u_per_h_nucleus * ATOMIC_MASS_UNIT),
        )
        electron_density = torch.pow(ten, log_hydrogen_density + log_electrons)
        return density, electron_density

    def mass_density(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> torch.Tensor:
        """Return mass density in kg m^-3 across photosphere and corona."""

        return self._states(temperature, gas_pressure)[0]

    def electron_density(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> torch.Tensor:
        """Return electron number density in m^-3 across the complete domain."""

        return self._states(temperature, gas_pressure)[1]

    def mean_molecular_weight(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> torch.Tensor:
        """Return the local mass per pressure-carrying particle in atomic units."""

        temperature, _, log_particles, _ = self._log_mappings(temperature, gas_pressure)
        return self.mass_u_per_h_nucleus / torch.pow(
            temperature.new_tensor(10.0), log_particles
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "stic_chianti_coronal_equilibrium_fully_ionized",
            "stic_table_sha256": self.stic_table_sha256,
            "chianti_table_sha256": self.chianti_table_sha256,
            "minimum_temperature_k": self.minimum_temperature_k,
            "cold_temperature_policy": (
                "freeze STiC composition at the minimum table temperature and "
                "retain actual positive T and P in the ideal-gas density closure"
            ),
            "stic_exact_max_temperature_k": self.stic_exact_max_temperature_k,
            "stic_to_chianti_transition_temperature_bounds_k": list(
                self.transition_temperature_bounds_k
            ),
            "chianti_to_fully_ionized_transition_temperature_bounds_k": list(
                self.ideal_transition_temperature_bounds_k
            ),
            "chianti_equilibrium_model": self.chianti_equilibrium_model,
            "lower_tangent_match_width_log10_temperature": (
                self.lower_tangent_match_width_log10_temperature
            ),
            "pressure_tangent_decay_width_log10_pressure": (
                self.pressure_tangent_decay_width_log10_pressure
            ),
            "pressure_tangent_decay_power": self.pressure_tangent_decay_power,
            "mass_u_per_h_nucleus": self.mass_u_per_h_nucleus,
            "nuclei_per_h_nucleus": self.nuclei_per_h_nucleus,
            "fully_ionized_electrons_per_h_nucleus": (
                self.fully_ionized_electrons_per_h_nucleus
            ),
            "mean_molecular_weight_fully_ionized": (
                self.mean_molecular_weight_fully_ionized
            ),
            "mean_molecular_weight_per_electron": (
                self.mean_molecular_weight_per_electron
            ),
            "thermodynamic_role": (
                "full-domain density and charge closure shared by MHS and the "
                "combined LTE plasma provider"
            ),
        }


__all__ = ["EOSLookupState", "HybridSolarEOS"]
