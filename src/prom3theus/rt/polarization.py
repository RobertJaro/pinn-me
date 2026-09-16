"""Depth-dependent polarized line opacity and propagation matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN, SPEED_OF_LIGHT

from .atomic import AtomicDatabase, SpectralLine
from .opacity import damping_rate, integrated_line_opacity
from .profiles import VoigtFaraday
from .zeeman import ZeemanPattern

# Exact CODATA/SI constants used at the tensor boundary.  Wavelength and field
# remain in Angstrom and gauss only where explicitly indicated.
ZEEMAN_HZ_PER_GAUSS = 1.399_624_493_61e6


@dataclass(frozen=True)
class PropagationDiagnostics:
    """Intermediate opacity quantities useful in validation."""

    continuum_extinction_ratio: torch.Tensor
    eta_i: torch.Tensor
    line_absorption: dict[str, torch.Tensor]
    damping: dict[str, torch.Tensor]
    doppler_width_hz: dict[str, torch.Tensor]
    damping_electron_density: torch.Tensor | None
    damping_hydrogen_neutral: torch.Tensor
    lower_level_populations: dict[str, torch.Tensor]


def _atomic_mass_u(database: AtomicDatabase, element: str) -> float:
    return float(database.element(element).atomic_mass_u)


class PolarizedLineOpacity(nn.Module):
    """Construct the LTE Stokes propagation matrix for one or more lines.

    All line profiles are normalized in frequency.  The supplied continuum
    extinction coefficients are per length, and ``alpha500`` is the total continuum
    extinction at 5000 vacuum Angstrom. With ``normalize_to_alpha500=True`` the
    resulting matrix is normalized per unit vertical ``tau_500``. Exact-ray
    synthesis instead leaves it in inverse metres so the formal solver can apply
    traced layer lengths directly.
    """

    def __init__(
        self,
        lines: Sequence[SpectralLine],
        atomic_database: AtomicDatabase | None = None,
        faddeeva_coefficients: int = 32,
        zero_field_regularization_gauss: float = 1.0e-3,
    ):
        super().__init__()
        if not lines:
            raise ValueError("At least one spectral line is required")
        if zero_field_regularization_gauss <= 0:
            raise ValueError("zero_field_regularization_gauss must be positive")
        self.lines = tuple(lines)
        self.atomic = atomic_database or AtomicDatabase()
        self.requires_stark_electron_density = any(
            line.log_gamma_stark_s_cm3 is not None for line in self.lines
        )
        self.profile = VoigtFaraday(n_coefs=faddeeva_coefficients)
        self.patterns = nn.ModuleList(
            ZeemanPattern(
                line.j_lower,
                line.j_upper,
                line.lande_lower,
                line.lande_upper,
            )
            for line in self.lines
        )
        # Atomic metadata are constants, not model tensors. All registered
        # quantities use the default model dtype; no float64 promotion is needed.
        vacuum = []
        for line in self.lines:
            reference_um = line.wavelength_air_angstrom / 1e4  # micrometres
            s0 = 1 / reference_um**2
            n0 = (
                1
                + 0.00008336624212083
                + 0.02408926869968 / (130.1065924522 - s0)
                + 0.0001599740894897 / (38.92568793293 - s0)
            )
            vacuum.append(line.wavelength_air_angstrom * n0)
        self._rest_frequencies_hz = tuple(
            SPEED_OF_LIGHT / (value * 1e-10) for value in vacuum
        )
        self.register_buffer(
            "line_wavelength_vacuum_angstrom", torch.tensor(vacuum), persistent=False
        )
        self.register_buffer(
            "line_rest_frequency_hz",
            torch.tensor(self._rest_frequencies_hz),
            persistent=False,
        )
        self.register_buffer(
            "line_atomic_mass_kg",
            torch.tensor(
                [
                    _atomic_mass_u(self.atomic, line.element) * ATOMIC_MASS_UNIT
                    for line in self.lines
                ]
            ),
            persistent=False,
        )
        self.zero_field_regularization_gauss = float(zero_field_regularization_gauss)

    @staticmethod
    def _validate_inputs(
        wavelength_angstrom: torch.Tensor,
        temperature: torch.Tensor,
        velocity_los: torch.Tensor,
        microturbulence: torch.Tensor,
        magnetic_field: torch.Tensor,
        continuum_extinction: torch.Tensor,
        alpha500: torch.Tensor,
    ) -> None:
        if wavelength_angstrom.ndim != 1 or wavelength_angstrom.numel() == 0:
            raise ValueError(
                "wavelength_angstrom must be a non-empty one-dimensional grid"
            )
        if temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "Polarized opacity supports float32 and float64 tensors only"
            )
        if (
            temperature.shape != velocity_los.shape
            or temperature.shape != microturbulence.shape
        ):
            raise ValueError(
                "temperature, velocity_los, and microturbulence must share shape [..., depth]"
            )
        if magnetic_field.shape != (*temperature.shape, 3):
            raise ValueError("magnetic_field must have shape [..., depth, 3]")
        if continuum_extinction.shape != (
            *temperature.shape,
            wavelength_angstrom.numel(),
        ):
            raise ValueError(
                "continuum_extinction must have shape [..., depth, wavelength]"
            )
        if alpha500.shape not in (temperature.shape, (*temperature.shape, 1)):
            raise ValueError("alpha500 must have shape [..., depth] or [..., depth, 1]")
        inputs = {
            "wavelength_angstrom": wavelength_angstrom,
            "temperature": temperature,
            "velocity_los": velocity_los,
            "microturbulence": microturbulence,
            "magnetic_field": magnetic_field,
            "continuum_extinction": continuum_extinction,
            "alpha500": alpha500,
        }
        for name, values in inputs.items():
            if values.device != temperature.device or values.dtype != temperature.dtype:
                raise TypeError(
                    f"{name} must use the temperature tensor's dtype and device"
                )
        # Transfer the small vector of validation flags once. Inspecting each
        # reduction in Python separately serializes CPU and GPU execution.
        checks = [
            (torch.isfinite(values).all(), f"{name} must contain only finite values")
            for name, values in inputs.items()
        ]
        checks.extend(
            [
                (
                    (
                        (wavelength_angstrom >= 2000.0)
                        & (wavelength_angstrom <= 100000.0)
                    ).all(),
                    "Standard-air wavelengths must lie in the supported 2000--100000 Angstrom range",
                ),
                ((temperature > 0).all(), "temperature must be strictly positive"),
                (
                    (velocity_los.abs() < SPEED_OF_LIGHT).all(),
                    "velocity_los must remain strictly subluminal",
                ),
                ((microturbulence >= 0).all(), "microturbulence must be non-negative"),
                (
                    (continuum_extinction > 0).all() & (alpha500 > 0).all(),
                    "continuum_extinction and alpha500 must be strictly positive",
                ),
            ]
        )
        valid = torch.stack([flag for flag, _ in checks]).cpu().tolist()
        for passed, (_, message) in zip(valid, checks, strict=True):
            if not passed:
                raise ValueError(message)

    def _orientation(self, magnetic_field: torch.Tensor):
        bx, by, b_los = magnetic_field.unbind(dim=-1)
        epsilon2 = self.zero_field_regularization_gauss**2
        field2 = bx.square() + by.square() + b_los.square()
        regularized2 = field2 + epsilon2
        field_strength = torch.sqrt(regularized2)

        # Assign the infinitesimal regularizing field to the LOS direction.  At
        # exactly B=0 this makes eta_I orientation independent, while the
        # combinations entering eta_V/Q/U remain differentiable in Cartesian B.
        cos_inclination = b_los / field_strength
        cos_inclination2 = (b_los.square() + epsilon2) / regularized2
        sin_inclination2 = (bx.square() + by.square()) / regularized2
        sin2_cos2azimuth = (bx.square() - by.square()) / regularized2
        sin2_sin2azimuth = 2.0 * bx * by / regularized2
        return (
            field_strength,
            cos_inclination,
            cos_inclination2,
            sin_inclination2,
            sin2_cos2azimuth,
            sin2_sin2azimuth,
        )

    def velocity_offsets_km_s(self, wavelength_angstrom):
        """Rest-minus-observed frequency as an equivalent velocity in km/s.

        Evaluate c*(lambda_vac-lambda0_vac)/lambda_vac without subtracting
        absolute optical frequencies or nearly equal refractive indices. Air
        wavelength residuals use a split line centre, and the refractive-index
        difference is factored algebraically. Every tensor stays in input dtype.
        """
        wavelength = torch.as_tensor(wavelength_angstrom)
        offsets = []
        wavelength_um = wavelength / 1e4  # micrometres, order unity
        for line in self.lines:
            origin = int(line.wavelength_air_angstrom)
            delta_l = (
                (wavelength - origin) - (line.wavelength_air_angstrom - origin)
            ) / 1e4
            reference_um = line.wavelength_air_angstrom / 1e4
            s, s0 = wavelength_um.reciprocal().square(), 1 / reference_um**2
            delta_s = (
                -delta_l
                * (wavelength_um + reference_um)
                / (wavelength_um.square() * reference_um**2)
            )
            n0 = (
                1
                + 0.00008336624212083
                + 0.02408926869968 / (130.1065924522 - s0)
                + 0.0001599740894897 / (38.92568793293 - s0)
            )
            delta_n = delta_s * (
                0.02408926869968 / ((130.1065924522 - s) * (130.1065924522 - s0))
                + 0.0001599740894897 / ((38.92568793293 - s) * (38.92568793293 - s0))
            )
            relative = (delta_l * n0 + wavelength_um * delta_n) / (
                wavelength_um * (n0 + delta_n)
            )
            offsets.append((SPEED_OF_LIGHT / 1000) * relative)
        return torch.stack(offsets)

    def line_center_extinction(
        self,
        temperature: torch.Tensor,
        microturbulence: torch.Tensor,
        *,
        lower_level_populations: Mapping[str, torch.Tensor],
        damping_electron_density: torch.Tensor | None,
        damping_hydrogen_neutral: torch.Tensor,
    ) -> torch.Tensor:
        """Return the summed rest-frame line-centre extinction in m^-1.

        Every line is evaluated at its own rest frequency with no magnetic
        splitting and no bulk velocity, so this is the largest monochromatic
        line extinction the configured window can reach at each depth. It is a
        depth-sampling quantity used to locate the layers where the line cores
        form; it never enters the emergent Stokes vector, where the field,
        velocity, and Zeeman pattern all matter.
        """

        total = torch.zeros_like(temperature)
        rest_frequencies = temperature.new_tensor(self._rest_frequencies_hz)
        atomic_masses = self.line_atomic_mass_kg.to(temperature)
        electron_density = (
            torch.zeros_like(temperature)
            if damping_electron_density is None
            else damping_electron_density
        )
        for line_index, line in enumerate(self.lines):
            rest_frequency = rest_frequencies[line_index]
            doppler_velocity = torch.sqrt(
                (2.0 * K_BOLTZMANN / atomic_masses[line_index]) * temperature
                + microturbulence.square()
            )
            doppler_width = rest_frequency * (doppler_velocity / SPEED_OF_LIGHT)
            damping_parameter = damping_rate(
                line,
                temperature,
                electron_density,
                damping_hydrogen_neutral,
            ) / (4.0 * torch.pi * doppler_width)
            peak_profile, _ = self.profile.dimensionless(
                torch.zeros_like(damping_parameter),
                damping_parameter,
                assume_nonnegative_damping=True,
            )
            integrated = integrated_line_opacity(
                line,
                lower_level_populations[line.id],
                temperature,
                rest_frequency_hz=rest_frequency,
            )
            total = total + integrated * peak_profile / doppler_width
        return total

    def _group_profiles(
        self, pattern, reduced_offset, zeeman_splitting, damping, doppler_width_hz
    ):
        shift_factors, strengths, group_slices = pattern.profile_tensors(
            like=reduced_offset
        )
        # The profile receives only dimensionless Doppler-width coordinates.
        offset = (
            reduced_offset[..., :, None]
            - zeeman_splitting[..., None, None] * shift_factors
        )
        absorption, dispersion = self.profile.dimensionless(
            offset,
            damping[..., None, None],
            assume_nonnegative_damping=True,
        )
        scale = strengths.reshape(*([1] * zeeman_splitting.ndim), 1, -1)
        scaled_absorption = absorption * scale
        scaled_dispersion = dispersion * scale
        return {
            group: (
                scaled_absorption[..., component_slice].sum(dim=-1)
                / doppler_width_hz[..., None],
                scaled_dispersion[..., component_slice].sum(dim=-1)
                / doppler_width_hz[..., None],
            )
            for group, component_slice in zip(pattern.groups, group_slices)
        }

    def forward(
        self,
        wavelength_angstrom: torch.Tensor,
        temperature: torch.Tensor,
        velocity_los: torch.Tensor,
        microturbulence: torch.Tensor,
        magnetic_field: torch.Tensor,
        continuum_extinction: torch.Tensor,
        alpha500: torch.Tensor,
        *,
        lower_level_populations: Mapping[str, torch.Tensor],
        damping_electron_density: torch.Tensor | None,
        damping_hydrogen_neutral: torch.Tensor,
        normalize_to_alpha500: bool,
        return_diagnostics: bool = False,
        line_velocity_offsets_km_s: torch.Tensor | None = None,
    ):
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        alpha500 = torch.as_tensor(
            alpha500, dtype=temperature.dtype, device=temperature.device
        )
        if alpha500.shape == (*temperature.shape, 1):
            alpha500 = alpha500[..., 0]
        self._validate_inputs(
            wavelength,
            temperature,
            velocity_los,
            microturbulence,
            magnetic_field,
            continuum_extinction,
            alpha500,
        )

        def validated_density(
            name: str,
            values: torch.Tensor,
            *,
            allow_zero: bool = False,
        ) -> torch.Tensor:
            values = torch.as_tensor(
                values,
                dtype=temperature.dtype,
                device=temperature.device,
            )
            if values.shape != temperature.shape:
                raise ValueError(f"{name} must share the temperature shape")
            invalid = values < 0 if allow_zero else values <= 0
            if not torch.isfinite(values).all() or torch.any(invalid):
                qualifier = "non-negative" if allow_zero else "strictly positive"
                raise ValueError(f"{name} must be finite and {qualifier}")
            return values

        diagnostic_electron_density = damping_electron_density
        if damping_electron_density is None:
            if self.requires_stark_electron_density:
                raise ValueError(
                    "A plasma-state damping_electron_density is required when any "
                    "selected line has Stark broadening."
                )
            electron_density_for_rate = torch.zeros_like(temperature)
        else:
            damping_electron_density = validated_density(
                "damping_electron_density", damping_electron_density
            )
            diagnostic_electron_density = damping_electron_density
            electron_density_for_rate = damping_electron_density
        damping_hydrogen_neutral = validated_density(
            "damping_hydrogen_neutral",
            damping_hydrogen_neutral,
            allow_zero=True,
        )
        validated_lower_populations = {}
        for line in self.lines:
            if line.id not in lower_level_populations:
                raise KeyError(f"Missing lower-level population for {line.id!r}")
            validated_lower_populations[line.id] = validated_density(
                f"lower_level_populations[{line.id!r}]",
                lower_level_populations[line.id],
                allow_zero=True,
            )

        if line_velocity_offsets_km_s is None:
            velocity_offsets = self.velocity_offsets_km_s(wavelength)
        else:
            velocity_offsets = torch.as_tensor(
                line_velocity_offsets_km_s,
                device=temperature.device,
                dtype=temperature.dtype,
            )
            if (
                velocity_offsets.shape != (len(self.lines), len(wavelength))
                or not torch.isfinite(velocity_offsets).all()
            ):
                raise ValueError(
                    "line_velocity_offsets_km_s must be finite [line,wavelength]"
                )
        continuum_extinction_ratio = None
        if normalize_to_alpha500:
            continuum_extinction_ratio = continuum_extinction / alpha500[..., None]
            eta_i = continuum_extinction_ratio
        else:
            eta_i = continuum_extinction
        zeros = torch.zeros_like(eta_i)
        eta_q = zeros
        eta_u = zeros
        eta_v = zeros
        rho_q = zeros
        rho_u = zeros
        rho_v = zeros

        (
            field_strength,
            cos_inclination,
            cos_inclination2,
            sin_inclination2,
            sin2_cos2azimuth,
            sin2_sin2azimuth,
        ) = self._orientation(magnetic_field)
        beta = velocity_los / SPEED_OF_LIGHT
        bulk_doppler_factor = torch.sqrt((1.0 - beta) / (1.0 + beta))

        line_absorption = {}
        damping_values = {}
        doppler_widths = {}
        rest_frequencies = temperature.new_tensor(self._rest_frequencies_hz)
        # Rationalized D-1 avoids cancellation around zero velocity.
        bulk_shift_km_s = (
            -2 * (velocity_los / 1000) / ((1 + beta) * (bulk_doppler_factor + 1))
        )
        atomic_masses = self.line_atomic_mass_kg.to(temperature)
        for line_index, (line, pattern) in enumerate(zip(self.lines, self.patterns)):
            rest_frequency = rest_frequencies[line_index]
            mass = atomic_masses[line_index]
            doppler_velocity_km_s = torch.sqrt(
                (2.0 * K_BOLTZMANN / mass / 1e6) * temperature
                + (microturbulence / 1000).square()
            )
            width_km_s = bulk_doppler_factor * doppler_velocity_km_s
            # Hz occurs only at the physical opacity/damping unit boundary.
            doppler_width = rest_frequency * (width_km_s / (SPEED_OF_LIGHT / 1000))
            total_damping = damping_rate(
                line,
                temperature,
                electron_density_for_rate,
                damping_hydrogen_neutral,
            )
            damping_parameter = total_damping / (4.0 * torch.pi * doppler_width)
            reduced_offset = (
                velocity_offsets[line_index] + bulk_shift_km_s[..., None]
            ) / width_km_s[..., None]
            # kG -> km/s, then divide by thermal/turbulent width. Doppler
            # factors cancel in the Zeeman displacement in line-width units.
            zeeman_km_s_per_kG = (
                ZEEMAN_HZ_PER_GAUSS
                * SPEED_OF_LIGHT
                / self._rest_frequencies_hz[line_index]
            )
            splitting = (
                (field_strength / 1000) * zeeman_km_s_per_kG / doppler_velocity_km_s
            )
            groups = self._group_profiles(
                pattern, reduced_offset, splitting, damping_parameter, doppler_width
            )
            phi_b, psi_b = groups["blue"]
            phi_p, psi_p = groups["pi"]
            phi_r, psi_r = groups["red"]

            lower_population = validated_lower_populations[line.id]
            integrated = integrated_line_opacity(
                line,
                lower_population,
                temperature,
                rest_frequency_hz=rest_frequency,
            )
            line_scale = integrated / alpha500 if normalize_to_alpha500 else integrated
            coefficient = 0.5 * line_scale[..., None]

            average_sigma_phi = 0.5 * (phi_b + phi_r)
            average_sigma_psi = 0.5 * (psi_b + psi_r)
            eta_i = eta_i + coefficient * (
                phi_p * sin_inclination2[..., None]
                + average_sigma_phi * (1.0 + cos_inclination2[..., None])
            )
            difference_phi = phi_p - average_sigma_phi
            difference_psi = psi_p - average_sigma_psi
            eta_q = eta_q + coefficient * difference_phi * sin2_cos2azimuth[..., None]
            eta_u = eta_u + coefficient * difference_phi * sin2_sin2azimuth[..., None]
            eta_v = eta_v + coefficient * (phi_r - phi_b) * cos_inclination[..., None]
            rho_q = rho_q + coefficient * difference_psi * sin2_cos2azimuth[..., None]
            rho_u = rho_u + coefficient * difference_psi * sin2_sin2azimuth[..., None]
            rho_v = rho_v + coefficient * (psi_r - psi_b) * cos_inclination[..., None]

            if return_diagnostics:
                diagnostic_scale = integrated / alpha500
                line_absorption[line.id] = (
                    diagnostic_scale[..., None] * (phi_p + phi_b + phi_r) / 3.0
                )
                damping_values[line.id] = damping_parameter
                doppler_widths[line.id] = doppler_width

        row_i = torch.stack((eta_i, eta_q, eta_u, eta_v), dim=-1)
        row_q = torch.stack((eta_q, eta_i, rho_v, -rho_u), dim=-1)
        row_u = torch.stack((eta_u, -rho_v, eta_i, rho_q), dim=-1)
        row_v = torch.stack((eta_v, rho_u, -rho_q, eta_i), dim=-1)
        propagation = torch.stack((row_i, row_q, row_u, row_v), dim=-2)

        if not return_diagnostics:
            return propagation
        if continuum_extinction_ratio is None:
            continuum_extinction_ratio = continuum_extinction / alpha500[..., None]
        return propagation, PropagationDiagnostics(
            continuum_extinction_ratio=continuum_extinction_ratio,
            eta_i=eta_i,
            line_absorption=line_absorption,
            damping=damping_values,
            doppler_width_hz=doppler_widths,
            damping_electron_density=diagnostic_electron_density,
            damping_hydrogen_neutral=damping_hydrogen_neutral,
            lower_level_populations=validated_lower_populations,
        )
