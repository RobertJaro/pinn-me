"""Depth-dependent polarized line opacity and propagation matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from prom3theus.core import ATOMIC_MASS_UNIT, K_BOLTZMANN, SPEED_OF_LIGHT
from prom3theus.core.validation import require_tensor

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


@dataclass(frozen=True)
class PreparedLine:
    """Wavelength-independent line physics, with one value per atmosphere point."""

    width_km_s: torch.Tensor
    doppler_width_hz: torch.Tensor
    damping: torch.Tensor
    splitting: torch.Tensor
    integrated_opacity: torch.Tensor


@dataclass(frozen=True)
class PreparedLineOpacity:
    """Live differentiable local state reused by every spectral block."""

    cos_inclination: torch.Tensor
    cos_inclination_squared: torch.Tensor
    sin_inclination_squared: torch.Tensor
    sin_squared_cos_two_azimuth: torch.Tensor
    sin_squared_sin_two_azimuth: torch.Tensor
    bulk_shift_km_s: torch.Tensor
    lines: tuple[PreparedLine, ...]
    electron_density: torch.Tensor | None
    hydrogen_neutral: torch.Tensor
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
    def _validate_spectral_inputs(wavelength, temperature, continuum, alpha500):
        if temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError("Polarized opacity supports float32 and float64 tensors only")
        if wavelength.ndim != 1 or wavelength.numel() == 0:
            raise ValueError("wavelength_angstrom must be a non-empty one-dimensional grid")
        if continuum.shape != (*temperature.shape, wavelength.numel()):
            raise ValueError("continuum_extinction must have shape [..., depth, wavelength]")
        if alpha500.shape != temperature.shape:
            raise ValueError("alpha500 must have shape [..., depth] or [..., depth, 1]")
        for name, value in (("wavelength_angstrom", wavelength),
                            ("continuum_extinction", continuum), ("alpha500", alpha500)):
            if value.dtype != temperature.dtype or value.device != temperature.device:
                raise TypeError(f"{name} must use the temperature tensor's dtype and device")
            require_tensor(torch.isfinite(value).all(), f"{name} must contain only finite values")
        require_tensor(((wavelength >= 2000.) & (wavelength <= 100000.)).all(),
                       "Standard-air wavelengths must lie in the supported 2000--100000 Angstrom range")
        require_tensor((continuum > 0).all() & (alpha500 > 0).all(),
                       "continuum_extinction and alpha500 must be strictly positive")

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

        zero = torch.zeros_like(temperature)
        prepared = self.prepare(
            temperature, zero, microturbulence,
            zero[..., None].expand(*temperature.shape, 3),
            lower_level_populations=lower_level_populations,
            damping_electron_density=damping_electron_density,
            damping_hydrogen_neutral=damping_hydrogen_neutral,
        )
        total = zero
        for line in prepared.lines:
            peak, _ = self.profile.dimensionless(zero, line.damping,
                                                assume_nonnegative_damping=True)
            total = total + line.integrated_opacity * peak / line.doppler_width_hz
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

    def prepare(
        self, temperature, velocity_los, microturbulence, magnetic_field, *,
        lower_level_populations, damping_electron_density, damping_hydrogen_neutral,
        validate=True,
    ) -> PreparedLineOpacity:
        """Compute line strengths, widths, damping and geometry once per point.

        The returned tensors retain first and higher-order gradients. They must
        be recomputed after every atmosphere update, never cached across steps.
        """
        if validate:
            if temperature.dtype not in (torch.float32, torch.float64):
                raise TypeError("Polarized opacity supports float32 and float64 tensors only")
            for name, value in (("temperature", temperature), ("velocity_los", velocity_los),
                                ("microturbulence", microturbulence), ("magnetic_field", magnetic_field)):
                expected = (*temperature.shape, 3) if name == "magnetic_field" else temperature.shape
                if value.shape != expected:
                    raise ValueError(f"{name} must share the temperature point shape (plus 3 for magnetic field)")
                if value.device != temperature.device or value.dtype != temperature.dtype:
                    raise TypeError(f"{name} must use the temperature tensor's dtype and device")
                require_tensor(torch.isfinite(value).all(), f"{name} must contain only finite values")
            require_tensor((temperature > 0).all(), "temperature must be strictly positive")
            require_tensor((velocity_los.abs() < SPEED_OF_LIGHT).all(), "velocity_los must remain strictly subluminal")
            require_tensor((microturbulence >= 0).all(), "microturbulence must be non-negative")

        def density(name, value, *, allow_zero=False):
            value = torch.as_tensor(value, dtype=temperature.dtype, device=temperature.device)
            if value.shape != temperature.shape:
                raise ValueError(f"{name} must share the temperature shape")
            if validate:
                positive = value >= 0 if allow_zero else value > 0
                qualifier = "non-negative" if allow_zero else "strictly positive"
                require_tensor(torch.isfinite(value).all() & positive.all(),
                               f"{name} must be finite and {qualifier}")
            return value

        electrons = damping_electron_density
        if electrons is None:
            if self.requires_stark_electron_density:
                raise ValueError("A plasma-state damping_electron_density is required when any selected line has Stark broadening.")
            rate_electrons = torch.zeros_like(temperature)
        else:
            electrons = density("damping_electron_density", electrons)
            rate_electrons = electrons
        hydrogen = density("damping_hydrogen_neutral", damping_hydrogen_neutral, allow_zero=True)
        populations = {}
        for line in self.lines:
            if line.id not in lower_level_populations:
                raise KeyError(f"Missing lower-level population for {line.id!r}")
            populations[line.id] = density(f"lower_level_populations[{line.id!r}]",
                                           lower_level_populations[line.id], allow_zero=True)

        field_strength, cos_i, cos_i2, sin_i2, sin2_cos2a, sin2_sin2a = self._orientation(magnetic_field)
        beta = velocity_los / SPEED_OF_LIGHT
        doppler_factor = torch.sqrt((1. - beta) / (1. + beta))
        # Rationalized D-1 avoids cancellation around zero velocity.
        shift = -2 * (velocity_los / 1000) / ((1 + beta) * (doppler_factor + 1))
        frequencies = temperature.new_tensor(self._rest_frequencies_hz)
        masses = self.line_atomic_mass_kg.to(temperature)
        prepared = []
        for index, line in enumerate(self.lines):
            thermal_width = torch.sqrt((2 * K_BOLTZMANN / masses[index] / 1e6) * temperature
                                       + (microturbulence / 1000).square())
            width = doppler_factor * thermal_width
            width_hz = frequencies[index] * (width / (SPEED_OF_LIGHT / 1000))
            damping = damping_rate(line, temperature, rate_electrons, hydrogen) / (4 * torch.pi * width_hz)
            # Doppler factors cancel in Zeeman displacement in line-width units.
            zeeman_per_kG = ZEEMAN_HZ_PER_GAUSS * SPEED_OF_LIGHT / self._rest_frequencies_hz[index]
            splitting = (field_strength / 1000) * zeeman_per_kG / thermal_width
            integrated = integrated_line_opacity(line, populations[line.id], temperature,
                                                 rest_frequency_hz=frequencies[index])
            prepared.append(PreparedLine(width, width_hz, damping, splitting, integrated))
        return PreparedLineOpacity(cos_i, cos_i2, sin_i2, sin2_cos2a, sin2_sin2a,
                                   shift, tuple(prepared), electrons, hydrogen, populations)

    def evaluate_prepared(self, prepared, velocity_offsets, continuum_extinction, *,
                          alpha500=None, return_diagnostics=False):
        """Evaluate only wavelength-dependent profiles and assemble K.

        Continuum has shape [*points,W]. Each line's velocity offsets may have
        any shape broadcastable to it: [W], [ray,1,W], or [points,1] for paired
        queries. Supplying alpha500 normalizes K to vertical continuum depth;
        otherwise the result is dimensional extinction in inverse metres.
        """
        if len(prepared.lines) != len(self.lines):
            raise ValueError("Prepared line state must match the configured spectral lines")
        if velocity_offsets.shape[0] != len(prepared.lines):
            raise ValueError("velocity_offsets must have one leading entry per line")
        expected = (*prepared.bulk_shift_km_s.shape, continuum_extinction.shape[-1])
        if continuum_extinction.shape != expected:
            raise ValueError("continuum_extinction must have shape [*points,wavelength]")
        if torch.broadcast_shapes(velocity_offsets.shape[1:], expected) != expected:
            raise ValueError("Line velocity offsets must broadcast to [*points,wavelength]")
        if alpha500 is not None and alpha500.shape != prepared.bulk_shift_km_s.shape:
            raise ValueError("alpha500 must share the prepared atmosphere point shape")
        eta_i = continuum_extinction if alpha500 is None else continuum_extinction / alpha500[..., None]
        continuum_ratio = eta_i if alpha500 is not None else None
        zero = torch.zeros_like(eta_i)
        eta_q = eta_u = eta_v = rho_q = rho_u = rho_v = zero
        line_absorption, damping_values, doppler_widths = {}, {}, {}
        for index, (line, pattern, local) in enumerate(zip(self.lines, self.patterns, prepared.lines)):
            offset = (velocity_offsets[index] + prepared.bulk_shift_km_s[..., None]) / local.width_km_s[..., None]
            groups = self._group_profiles(pattern, offset, local.splitting, local.damping, local.doppler_width_hz)
            phi_b, psi_b = groups["blue"]
            phi_p, psi_p = groups["pi"]
            phi_r, psi_r = groups["red"]
            strength = local.integrated_opacity if alpha500 is None else local.integrated_opacity / alpha500
            coefficient = .5 * strength[..., None]
            average_phi, average_psi = .5 * (phi_b + phi_r), .5 * (psi_b + psi_r)
            difference_phi, difference_psi = phi_p - average_phi, psi_p - average_psi
            eta_i = eta_i + coefficient * (phi_p * prepared.sin_inclination_squared[..., None]
                + average_phi * (1 + prepared.cos_inclination_squared[..., None]))
            eta_q = eta_q + coefficient * difference_phi * prepared.sin_squared_cos_two_azimuth[..., None]
            eta_u = eta_u + coefficient * difference_phi * prepared.sin_squared_sin_two_azimuth[..., None]
            eta_v = eta_v + coefficient * (phi_r - phi_b) * prepared.cos_inclination[..., None]
            rho_q = rho_q + coefficient * difference_psi * prepared.sin_squared_cos_two_azimuth[..., None]
            rho_u = rho_u + coefficient * difference_psi * prepared.sin_squared_sin_two_azimuth[..., None]
            rho_v = rho_v + coefficient * (psi_r - psi_b) * prepared.cos_inclination[..., None]
            if return_diagnostics:
                diagnostic_scale = local.integrated_opacity if alpha500 is None else local.integrated_opacity / alpha500
                line_absorption[line.id] = diagnostic_scale[..., None] * (phi_p + phi_b + phi_r) / 3.
                damping_values[line.id] = local.damping
                doppler_widths[line.id] = local.doppler_width_hz
        propagation = torch.stack((
            torch.stack((eta_i, eta_q, eta_u, eta_v), -1),
            torch.stack((eta_q, eta_i, rho_v, -rho_u), -1),
            torch.stack((eta_u, -rho_v, eta_i, rho_q), -1),
            torch.stack((eta_v, rho_u, -rho_q, eta_i), -1),
        ), -2)
        if not return_diagnostics:
            return propagation
        # Dimensional prepared evaluation reports dimensional line absorption;
        # the legacy forward wrapper supplies alpha500-normalized diagnostics.
        return propagation, PropagationDiagnostics(
            continuum_extinction if continuum_ratio is None else continuum_ratio,
            eta_i, line_absorption, damping_values, doppler_widths,
            prepared.electron_density, prepared.hydrogen_neutral, prepared.lower_level_populations)

    def forward(
        self, wavelength_angstrom, temperature, velocity_los, microturbulence,
        magnetic_field, continuum_extinction, alpha500, *, lower_level_populations,
        damping_electron_density, damping_hydrogen_neutral, normalize_to_alpha500,
        return_diagnostics=False, line_velocity_offsets_km_s=None, paired_wavelengths=False,
    ):
        """Checked public interface for standalone/reference synthesis."""
        wavelength = torch.as_tensor(wavelength_angstrom, dtype=temperature.dtype, device=temperature.device)
        alpha500 = torch.as_tensor(alpha500, dtype=temperature.dtype, device=temperature.device)
        if alpha500.shape == (*temperature.shape, 1):
            alpha500 = alpha500[..., 0]
        if paired_wavelengths:
            if wavelength.shape != temperature.shape or wavelength.ndim != 1:
                raise ValueError("Paired line queries require one-dimensional matching point/wavelength arrays")
            require_tensor(torch.isfinite(wavelength).all() & ((wavelength >= 2000.) & (wavelength <= 100000.)).all(),
                           "Paired air wavelengths must be finite and within 2000--100000 Angstrom")
        self._validate_spectral_inputs(wavelength[:1] if paired_wavelengths else wavelength,
                                      temperature, continuum_extinction, alpha500)
        prepared = self.prepare(temperature, velocity_los, microturbulence, magnetic_field,
                                lower_level_populations=lower_level_populations,
                                damping_electron_density=damping_electron_density,
                                damping_hydrogen_neutral=damping_hydrogen_neutral)
        if line_velocity_offsets_km_s is None:
            offsets = self.velocity_offsets_km_s(wavelength)
            if paired_wavelengths:
                offsets = offsets[..., None]
        else:
            offsets = torch.as_tensor(line_velocity_offsets_km_s, dtype=temperature.dtype, device=temperature.device)
            expected = ((len(self.lines), *temperature.shape, 1) if paired_wavelengths
                        else (len(self.lines), len(wavelength)))
            if offsets.shape != expected:
                raise ValueError("line_velocity_offsets_km_s must be finite [line,wavelength]")
            require_tensor(torch.isfinite(offsets).all(), "line_velocity_offsets_km_s must be finite [line,wavelength]")
        result = self.evaluate_prepared(prepared, offsets, continuum_extinction,
                                       alpha500=alpha500 if normalize_to_alpha500 else None,
                                       return_diagnostics=return_diagnostics)
        if not return_diagnostics or normalize_to_alpha500:
            return result
        # Historical diagnostics always use alpha500 units, even when K uses m^-1.
        propagation, diagnostic = result
        return propagation, PropagationDiagnostics(
            diagnostic.continuum_extinction_ratio / alpha500[..., None], diagnostic.eta_i,
            {name: value / alpha500[..., None] for name, value in diagnostic.line_absorption.items()},
            diagnostic.damping, diagnostic.doppler_width_hz, diagnostic.damping_electron_density,
            diagnostic.damping_hydrogen_neutral, diagnostic.lower_level_populations)
