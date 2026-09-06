"""Unified photospheric-to-coronal plasma-state lookup.

The runtime provider combines the pinned STiC radiative table with the
STiC/CHIANTI/fully-ionized thermodynamic closure.  Neural atmosphere outputs
remain unrestricted: only the finite photospheric contribution is tapered away
where its source physics ceases to apply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .atomic import AtomicDatabase, SpectralLine
from .eos import HybridSolarEOS
from .opacity import ContinuumOpacity, STICLookupState, STICSpectralState


# CODATA 2018 Thomson cross-section in m^2.  Retaining this analytic hot
# scattering contribution keeps the reference extinction strictly positive
# without pretending that a hot coronal cell has the 10-kK STiC continuum.
THOMSON_CROSS_SECTION_M2 = 6.652_458_732_1e-29


@dataclass(frozen=True)
class SolarPlasmaState:
    """One consistent thermodynamic state evaluated at unrestricted ``T, P``."""

    temperature: torch.Tensor
    gas_pressure: torch.Tensor
    stic_lookup: STICLookupState
    photospheric_weight: torch.Tensor
    mass_density: torch.Tensor
    electron_density: torch.Tensor
    pressure_density_scale: torch.Tensor


class SolarPlasmaTable(ContinuumOpacity):
    """Combined STiC radiative, CHIANTI, and ideal-mixture plasma provider.

    STiC values are reproduced exactly over their native temperature and
    pressure rectangle.  From 10 kK to 31.6 kK their optical populations and
    true continuum fade smoothly to zero while scattering transitions to the
    analytic Thomson value.  Thermodynamic density and charge use
    :class:`HybridSolarEOS` over the complete positive ``T, P`` domain.
    """

    def __init__(
        self,
        atomic_database: AtomicDatabase | None = None,
        *,
        thermodynamic_eos: HybridSolarEOS | None = None,
    ):
        atomic = atomic_database or AtomicDatabase()
        super().__init__(atomic)
        self.eos = thermodynamic_eos or HybridSolarEOS(atomic)
        if self.eos.transition_temperature_bounds_k[0] != (
            10.0 ** float(self.stic_log_temperature[-1])
        ):
            raise RuntimeError(
                "The STiC radiative and hybrid-EOS hot boundaries do not meet."
            )

    @staticmethod
    def _smootherstep(fraction: torch.Tensor) -> torch.Tensor:
        fraction = fraction.clamp(0.0, 1.0)
        return fraction.pow(3) * (10.0 + fraction * (-15.0 + 6.0 * fraction))

    def _photospheric_weight(self, temperature: torch.Tensor) -> torch.Tensor:
        lower, upper = self.eos.transition_temperature_bounds_k
        log_temperature = torch.log10(temperature)
        fraction = (log_temperature - temperature.new_tensor(lower).log10()) / (
            temperature.new_tensor(upper).log10()
            - temperature.new_tensor(lower).log10()
        )
        return 1.0 - self._smootherstep(fraction)

    def prepare_plasma_state(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
    ) -> SolarPlasmaState:
        """Evaluate the shared composition and reusable STiC coordinates."""

        temperature, gas_pressure = self.eos._validate_state(temperature, gas_pressure)
        stic_lookup = super().prepare_stic_lookup(temperature, gas_pressure)
        mass_density = self.eos.mass_density(temperature, gas_pressure)
        electron_density = self.eos.electron_density(temperature, gas_pressure)

        # Only pressure excursions scale the edge radiative reservoirs here.
        # Temperature behavior is handled by the endpoint-tangent continuation
        # and photospheric fade below.  In the far pressure tails the hybrid EOS
        # composition is constant, so this ratio becomes exactly linear in P.
        pressure_anchor_density = self.eos.mass_density(
            temperature,
            stic_lookup.gas_pressure,
        )
        pressure_density_scale = mass_density / pressure_anchor_density
        return SolarPlasmaState(
            temperature=temperature,
            gas_pressure=gas_pressure,
            stic_lookup=stic_lookup,
            photospheric_weight=self._photospheric_weight(temperature),
            mass_density=mass_density,
            electron_density=electron_density,
            pressure_density_scale=pressure_density_scale,
        )

    @staticmethod
    def _expand_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while value.ndim < target.ndim:
            value = value[..., None]
        return value

    def _hot_continued_stic_logs(
        self,
        table: torch.Tensor,
        state: SolarPlasmaState,
    ) -> torch.Tensor:
        """Carry the exact STiC hot-edge value and tangent into the fade."""

        base = super()._interpolate_stic_log_table(
            table,
            state.temperature,
            state.gas_pressure,
            state.stic_lookup,
        )
        log_temperature = torch.log10(state.temperature)
        top_log_temperature = self.stic_log_temperature[-1].to(state.temperature)
        previous_log_temperature = self.stic_log_temperature[-2].to(state.temperature)
        top_temperature = torch.full_like(state.temperature, 10.0).pow(
            top_log_temperature
        )
        previous_temperature = torch.full_like(state.temperature, 10.0).pow(
            previous_log_temperature
        )
        top_lookup = super().prepare_stic_lookup(top_temperature, state.gas_pressure)
        previous_lookup = super().prepare_stic_lookup(
            previous_temperature, state.gas_pressure
        )
        top = super()._interpolate_stic_log_table(
            table,
            top_temperature,
            state.gas_pressure,
            top_lookup,
        )
        previous = super()._interpolate_stic_log_table(
            table,
            previous_temperature,
            state.gas_pressure,
            previous_lookup,
        )
        slope = (
            0.5 * (top - previous) / (top_log_temperature - previous_log_temperature)
        )
        fade_top = state.temperature.new_tensor(
            self.eos.transition_temperature_bounds_k[1]
        ).log10()
        excursion = (log_temperature - top_log_temperature).clamp(
            0.0,
            fade_top - top_log_temperature,
        )
        return base + slope * self._expand_like(excursion, base)

    def _stic_radiative_component(
        self,
        table: torch.Tensor,
        wavelength_vacuum_angstrom,
        state: SolarPlasmaState,
        *,
        spectral_state: STICSpectralState | None = None,
    ) -> torch.Tensor:
        wavelength = self._wavelength_tensor(
            wavelength_vacuum_angstrom, state.temperature
        )
        if spectral_state is None:
            self._validate_stic_wavelengths(wavelength)
            spectral_state = self._stic_wavelength_coordinates(
                wavelength, state.temperature
            )
        logs = self._hot_continued_stic_logs(table, state)
        lower = spectral_state.lower_indices.to(device=state.temperature.device)
        upper = spectral_state.upper_indices.to(device=state.temperature.device)
        weight = spectral_state.weight.to(state.temperature)
        interpolated = logs[..., lower] + weight * (logs[..., upper] - logs[..., lower])
        return torch.pow(state.temperature.new_tensor(10.0), interpolated)

    def continuum_components(
        self,
        wavelength_vacuum_angstrom,
        state: SolarPlasmaState,
        *,
        spectral_state: STICSpectralState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return true absorption and scattering over the complete domain."""

        components = self._stic_radiative_component(
            self.stic_log_continuum_components,
            wavelength_vacuum_angstrom,
            state,
            spectral_state=spectral_state,
        )
        photospheric_weight = self._expand_like(
            state.photospheric_weight, components[..., 0, :]
        )
        density_scale = self._expand_like(
            state.pressure_density_scale, components[..., 0, :]
        )
        absorption = (
            photospheric_weight * components[..., 0, :] * density_scale.square()
        )
        stic_scattering = components[..., 1, :] * density_scale
        thomson = self._expand_like(
            state.electron_density
            * state.temperature.new_tensor(THOMSON_CROSS_SECTION_M2),
            stic_scattering,
        )
        scattering = (
            photospheric_weight * stic_scattering
            + (1.0 - photospheric_weight) * thomson
        )
        return absorption, scattering

    def forward(
        self,
        wavelength_vacuum_angstrom,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
        *,
        spectral_state: STICSpectralState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        absorption, scattering = self.continuum_components(
            wavelength_vacuum_angstrom,
            state,
            spectral_state=spectral_state,
        )
        return absorption + scattering

    def true_absorption(
        self,
        wavelength_vacuum_angstrom,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return self.continuum_components(wavelength_vacuum_angstrom, state)[0]

    def scattering_extinction(
        self,
        wavelength_vacuum_angstrom,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return self.continuum_components(wavelength_vacuum_angstrom, state)[1]

    def volume_extinction_at_5000(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return self(
            temperature.new_tensor([5000.0]),
            temperature,
            gas_pressure,
            state,
        )[..., 0]

    def reference_thermodynamics(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> dict[str, torch.Tensor]:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return {
            "mass_density": state.mass_density,
            "electron_density": state.electron_density,
        }

    def reference_mass_density(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return state.mass_density

    def reference_electron_density(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return state.electron_density

    def _photospheric_line_reservoirs(
        self,
        state: SolarPlasmaState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logs = self._hot_continued_stic_logs(
            self.stic_log_line_thermodynamics[..., 1:], state
        )
        values = torch.pow(state.temperature.new_tensor(10.0), logs)
        weight = self._expand_like(state.photospheric_weight, values)
        scale = self._expand_like(state.pressure_density_scale, values)
        faded = weight * values * scale
        return faded[..., 0], faded[..., 1]

    def reference_neutral_hydrogen_density(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return self._photospheric_line_reservoirs(state)[0]

    def reference_fe_i_population_over_partition(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
    ) -> torch.Tensor:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        return self._photospheric_line_reservoirs(state)[1]

    def reference_line_thermodynamics(
        self,
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
        *,
        include_electron_density: bool = False,
    ) -> dict[str, torch.Tensor]:
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        hydrogen_neutral, fe_i = self._photospheric_line_reservoirs(state)
        result = {
            "hydrogen_neutral": hydrogen_neutral,
            "fe_i_population_over_partition": fe_i,
        }
        if include_electron_density:
            result = {"electron_density": state.electron_density, **result}
        return result

    def reference_lower_level_populations(
        self,
        lines: Sequence[SpectralLine],
        temperature: torch.Tensor,
        gas_pressure: torch.Tensor,
        state: SolarPlasmaState | None = None,
        *,
        fe_i_population_over_partition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        lines = tuple(lines)
        if not lines:
            raise ValueError("At least one spectral line is required")
        state = state or self.prepare_plasma_state(temperature, gas_pressure)
        reservoir = fe_i_population_over_partition
        if reservoir is None:
            reservoir = self.reference_fe_i_population_over_partition(
                temperature, gas_pressure, state
            )
        return {
            line.id: self._fe_i_lower_level_population_from_reservoir(
                line, state.temperature, reservoir
            )
            for line in lines
        }

    def mass_density(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> torch.Tensor:
        return self.eos.mass_density(temperature, gas_pressure)

    def electron_density(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> torch.Tensor:
        return self.eos.electron_density(temperature, gas_pressure)

    def mean_molecular_weight(
        self, temperature: torch.Tensor, gas_pressure: torch.Tensor
    ) -> torch.Tensor:
        return self.eos.mean_molecular_weight(temperature, gas_pressure)

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata.update(
            {
                "type": "combined_stic_chianti_ideal_plasma_table",
                "thermodynamic_lookup_policy": (
                    "unrestricted positive T and P; exact STiC in its native domain, "
                    "C1 photospheric fade to CHIANTI, and analytic ideal tails"
                ),
                "photospheric_fade_temperature_bounds_k": list(
                    self.eos.transition_temperature_bounds_k
                ),
                "hot_true_absorption": "zero above the photospheric fade",
                "hot_scattering": "Thomson extinction n_e * sigma_T",
                "pressure_tail": (
                    "hybrid-EOS density scaling outside the native STiC pressure axis"
                ),
                "thermodynamic_eos": self.eos.metadata(),
            }
        )
        return metadata


__all__ = [
    "SolarPlasmaState",
    "SolarPlasmaTable",
    "THOMSON_CROSS_SECTION_M2",
]
