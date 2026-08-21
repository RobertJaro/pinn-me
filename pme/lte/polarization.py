"""Depth-dependent polarized line opacity and propagation matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from pme.lte.atomic import AtomicDatabase, SpectralLine
from pme.lte.opacity import damping_rate, integrated_line_opacity
from pme.lte.profiles import VoigtFaraday
from pme.lte.zeeman import ZeemanPattern
from pme.lte.wavelength import air_to_vacuum_angstrom


# Exact CODATA/SI constants used at the tensor boundary.  Wavelength and field
# remain in Angstrom and gauss only where explicitly indicated.
C_LIGHT = 299_792_458.0
K_BOLTZMANN = 1.380_649e-23
ATOMIC_MASS_UNIT = 1.660_539_066_60e-27
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
    data = database.element(element)
    if hasattr(data, "atomic_mass_u"):
        return float(data.atomic_mass_u)
    if isinstance(data, dict):
        for key in ("atomic_mass_u", "mass_u", "atomic_mass"):
            if key in data:
                return float(data[key])
    raise KeyError(f"Atomic mass is unavailable for {element!r}")


class PolarizedLineOpacity(nn.Module):
    """Construct the LTE Stokes propagation matrix for one or more lines.

    All line profiles are normalized in frequency.  The supplied continuum
    extinction coefficients are per length, and ``alpha500`` is the total continuum
    extinction at 5000 vacuum Angstrom. The resulting matrix is normalized per unit
    vertical ``tau_500`` and can be passed directly to the formal solver.
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
        if wavelength_angstrom.ndim != 1:
            raise ValueError("wavelength_angstrom must be one-dimensional")
        if temperature.shape != velocity_los.shape or temperature.shape != microturbulence.shape:
            raise ValueError("temperature, velocity_los, and microturbulence must share shape [..., depth]")
        if magnetic_field.shape != (*temperature.shape, 3):
            raise ValueError("magnetic_field must have shape [..., depth, 3]")
        if continuum_extinction.shape != (*temperature.shape, wavelength_angstrom.numel()):
            raise ValueError("continuum_extinction must have shape [..., depth, wavelength]")
        if alpha500.shape not in (temperature.shape, (*temperature.shape, 1)):
            raise ValueError("alpha500 must have shape [..., depth] or [..., depth, 1]")
        inputs = {
            "temperature": temperature,
            "velocity_los": velocity_los,
            "microturbulence": microturbulence,
            "magnetic_field": magnetic_field,
            "continuum_extinction": continuum_extinction,
            "alpha500": alpha500,
        }
        for name, values in inputs.items():
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
        if torch.any(temperature <= 0):
            raise ValueError("temperature must be positive")
        if torch.any(microturbulence < 0):
            raise ValueError("microturbulence cannot be negative")
        if torch.any(wavelength_angstrom <= 0):
            raise ValueError("wavelengths must be positive")
        if torch.any(continuum_extinction <= 0) or torch.any(alpha500 <= 0):
            raise ValueError("continuum extinction and alpha500 must be strictly positive")

    def _orientation(self, magnetic_field: torch.Tensor):
        bx, by, b_los = magnetic_field.unbind(dim=-1)
        epsilon2 = self.zero_field_regularization_gauss ** 2
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

    def _group_profiles(
        self,
        pattern: ZeemanPattern,
        frequency_hz: torch.Tensor,
        central_frequency_hz: torch.Tensor,
        doppler_width_hz: torch.Tensor,
        damping: torch.Tensor,
        field_strength_gauss: torch.Tensor,
        bulk_doppler_factor: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        output = {}
        for group in pattern.groups:
            shift_factors, strengths = pattern.group_tensors(group, like=frequency_hz)
            component_frequency = (
                central_frequency_hz[..., None]
                - ZEEMAN_HZ_PER_GAUSS
                * field_strength_gauss[..., None]
                * shift_factors
                * bulk_doppler_factor[..., None]
            )
            # Use the solar-spectropolarimetry convention in which the reduced
            # coordinate increases toward redder wavelength.  In frequency
            # units this is (nu_component - nu), not (nu - nu_component).
            # Voigt absorption is even, but the Faraday--Voigt profile is odd;
            # reversing this subtraction would therefore invert every rho
            # coefficient while leaving Stokes I and the eta terms unchanged.
            offset = (
                component_frequency[..., None, :]
                - frequency_hz.reshape(*([1] * central_frequency_hz.ndim), -1, 1)
            ) / doppler_width_hz[..., None, None]
            absorption, dispersion = self.profile.dimensionless(
                offset,
                damping[..., None, None],
            )
            scale = strengths.reshape(*([1] * central_frequency_hz.ndim), 1, -1)
            absorption = (absorption * scale).sum(dim=-1) / doppler_width_hz[..., None]
            dispersion = (dispersion * scale).sum(dim=-1) / doppler_width_hz[..., None]
            output[group] = absorption, dispersion
        return output

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
        return_diagnostics: bool = False,
    ):
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        alpha500 = torch.as_tensor(alpha500, dtype=temperature.dtype, device=temperature.device)
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
        def validated_density(name: str, values: torch.Tensor) -> torch.Tensor:
            values = torch.as_tensor(
                values,
                dtype=temperature.dtype,
                device=temperature.device,
            )
            if values.shape != temperature.shape:
                raise ValueError(f"{name} must share the temperature shape")
            if not torch.isfinite(values).all() or torch.any(values <= 0):
                raise ValueError(f"{name} must be finite and strictly positive")
            return values

        diagnostic_electron_density = damping_electron_density
        if damping_electron_density is None:
            if self.requires_stark_electron_density:
                raise ValueError(
                    "A pinned-STiC damping_electron_density is required when any "
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
            "damping_hydrogen_neutral", damping_hydrogen_neutral
        )
        validated_lower_populations = {}
        for line in self.lines:
            if line.id not in lower_level_populations:
                raise KeyError(f"Missing lower-level population for {line.id!r}")
            validated_lower_populations[line.id] = validated_density(
                f"lower_level_populations[{line.id!r}]",
                lower_level_populations[line.id],
            )

        wavelength_vacuum = air_to_vacuum_angstrom(wavelength)
        frequency = C_LIGHT / (wavelength_vacuum * 1.0e-10)
        continuum_extinction_ratio = continuum_extinction / alpha500[..., None]
        eta_i = continuum_extinction_ratio
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

        line_absorption = {}
        damping_values = {}
        doppler_widths = {}
        for line, pattern in zip(self.lines, self.patterns):
            line_wavelength_air = temperature.new_tensor(
                line.wavelength_air_angstrom
            )
            line_wavelength_vacuum = air_to_vacuum_angstrom(
                line_wavelength_air
            )
            rest_frequency = C_LIGHT / (line_wavelength_vacuum * 1.0e-10)
            mass = temperature.new_tensor(_atomic_mass_u(self.atomic, line.element) * ATOMIC_MASS_UNIT)
            beta = velocity_los / C_LIGHT
            if torch.any(beta.abs() >= 1.0):
                raise ValueError("velocity_los must remain strictly subluminal")
            bulk_doppler_factor = torch.sqrt((1.0 - beta) / (1.0 + beta))
            doppler_velocity = torch.sqrt(
                2.0 * K_BOLTZMANN * temperature / mass
                + microturbulence.square()
            )
            central_frequency = rest_frequency * bulk_doppler_factor
            doppler_width = central_frequency * doppler_velocity / C_LIGHT
            total_damping = damping_rate(
                line,
                temperature,
                electron_density_for_rate,
                damping_hydrogen_neutral,
            )
            damping_parameter = total_damping / (4.0 * torch.pi * doppler_width)
            groups = self._group_profiles(
                pattern,
                frequency,
                central_frequency,
                doppler_width,
                damping_parameter,
                field_strength,
                bulk_doppler_factor,
            )
            phi_b, psi_b = groups["blue"]
            phi_p, psi_p = groups["pi"]
            phi_r, psi_r = groups["red"]

            lower_population = validated_lower_populations[line.id]
            integrated = integrated_line_opacity(line, lower_population, temperature)
            line_scale = integrated / alpha500
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
                line_absorption[line.id] = line_scale[..., None] * (
                    phi_p + phi_b + phi_r
                ) / 3.0
                damping_values[line.id] = damping_parameter
                doppler_widths[line.id] = doppler_width

        row_i = torch.stack((eta_i, eta_q, eta_u, eta_v), dim=-1)
        row_q = torch.stack((eta_q, eta_i, rho_v, -rho_u), dim=-1)
        row_u = torch.stack((eta_u, -rho_v, eta_i, rho_q), dim=-1)
        row_v = torch.stack((eta_v, rho_u, -rho_q, eta_i), dim=-1)
        propagation = torch.stack((row_i, row_q, row_u, row_v), dim=-2)

        if not return_diagnostics:
            return propagation
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
