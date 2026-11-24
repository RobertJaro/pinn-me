"""Pointwise LTE coefficients for stationary polarized radiation transport.

This module evaluates physical coefficients only. It has no atmospheric depth
grid, boundary condition or numerical transport solver. The continuum closure
is explicitly ``thermalized_continuum``: total absorption plus scattering is
paired with a thermal source, retaining the established LTE approximation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from prom3theus.core import SPEED_OF_LIGHT
from prom3theus.core.validation import require_tensor, tensor_validation_enabled

from .atomic import AtomicDatabase, SpectralLine
from .opacity import STICSpectralState, planck_lambda
from .plasma import SolarPlasmaState, SolarPlasmaTable
from .polarization import PolarizedLineOpacity, PreparedLineOpacity
from .wavelength import CenteredWavelength, air_to_vacuum_angstrom


@dataclass(frozen=True)
class LocalRadiationCoefficients:
    """Dimensional K [m^-1] and thermal Stokes source divided by fixed I0."""

    propagation_m1: torch.Tensor
    source: torch.Tensor


@dataclass(frozen=True)
class PreparedLTEAtmosphere:
    """Point physics for one forward; never copied across the wavelength axis."""

    plasma: SolarPlasmaState
    lines: PreparedLineOpacity
    continuum_logs: torch.Tensor


@dataclass(frozen=True)
class PreparedSpectrum:
    """Exact vacuum offsets and continuum interpolation on a broadcastable grid."""

    vacuum: torch.Tensor
    velocity_offsets: torch.Tensor
    continuum: STICSpectralState


class LTELocalCoefficients(nn.Module):
    """Evaluate local K and B_lambda for arbitrary atmospheric points.

    Scalars have an arbitrary common point shape; velocities (m/s) and magnetic
    fields (gauss) have one final Cartesian component axis. ``direction`` is the
    unit photon propagation vector. The right-handed ``stokes_basis`` contains
    rows (e1, e2, direction), in the same physical Cartesian frame as the fields.
    Wavelengths are vacuum Angstrom in the common inertial spectral frame.

    ``forward``/``evaluate`` uses a shared wavelength grid and returns point
    shape + [W,4,4] and [W,4]. ``evaluate_paired`` instead evaluates one wavelength
    per point, returning point shape + [4,4] and [4], without a Cartesian product.
    Material velocities must already include any rotating-to-inertial correction;
    detector motion and wavelength zero points never enter these coefficients.
    """

    def __init__(
        self,
        *,
        atomic_database: AtomicDatabase | None = None,
        lines: Sequence[SpectralLine] | None = None,
        line_ids: Sequence[str] | None = None,
        wavelength_window_angstrom: tuple[float, float] | None = None,
        plasma_provider: SolarPlasmaTable | None = None,
        faddeeva_coefficients: int = 32,
    ):
        super().__init__()
        if sum(value is not None for value in (lines, line_ids, wavelength_window_angstrom)) != 1:
            raise ValueError("Pass exactly one of lines, line_ids, or wavelength_window_angstrom.")
        self.atomic = atomic_database or AtomicDatabase()
        if lines is None:
            lines = (
                tuple(self.atomic.get_line(line_id) for line_id in line_ids)
                if line_ids is not None
                else self.atomic.select_lines(*wavelength_window_angstrom)
            )
        self.lines = tuple(lines)
        if not self.lines:
            raise ValueError("The selected radiation window contains no spectral lines")
        self.plasma_provider = plasma_provider or SolarPlasmaTable(self.atomic)
        self.line_opacity = PolarizedLineOpacity(
            self.lines, self.atomic, faddeeva_coefficients=faddeeva_coefficients
        )
        # Python constants retain the precise centres even after module.float().
        self._rest_vacuum_angstrom = tuple(
            air_to_vacuum_angstrom(torch.tensor(line.wavelength_air_angstrom, dtype=torch.float64)).item()
            for line in self.lines
        )
        for name in ("wavelength_vacuum", "velocity_offsets", "spectral_weight"):
            self.register_buffer("_prepared_" + name, torch.empty(0), persistent=False)
        for name in ("spectral_lower", "spectral_upper"):
            self.register_buffer("_prepared_" + name, torch.empty(0, dtype=torch.long), persistent=False)

    @staticmethod
    def _wavelength_tensor(wavelength, *, device) -> torch.Tensor:
        # Python lists and double detector grids must not lose sub-milliangstrom
        # offsets before line-centre subtraction. Casting happens afterwards.
        result = (
            wavelength.to(device=device)
            if isinstance(wavelength, torch.Tensor)
            else torch.as_tensor(wavelength, dtype=torch.float64, device=device)
        )
        if not result.is_floating_point():
            result = result.to(torch.float64)
        if result.dtype not in (torch.float32, torch.float64):
            raise TypeError("Radiation wavelengths require float32 or float64")
        if result.numel() == 0:
            raise ValueError("Radiation wavelengths must be nonempty")
        require_tensor(lambda: (torch.isfinite(result) & (result > 0)).all(),
                       "Radiation wavelengths must be finite and positive")
        return result

    def _velocity_offsets(self, vacuum: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            (SPEED_OF_LIGHT / 1000.0)
            * ((vacuum - int(rest)) - (rest - int(rest))) / vacuum
            for rest in self._rest_vacuum_angstrom
        ])

    @property
    def wavelength_grid_prepared(self) -> bool:
        return bool(self._prepared_wavelength_vacuum.numel())

    @torch.no_grad()
    def prepare_wavelength_grid(self, wavelength_vacuum_angstrom) -> None:
        """Cache immutable spectral coordinates, subtracting centres before casting."""
        vacuum = self._wavelength_tensor(
            wavelength_vacuum_angstrom, device=self.line_opacity.line_rest_frequency_hz.device
        )
        if vacuum.ndim == 0:
            vacuum = vacuum[None]
        if vacuum.ndim != 1:
            raise ValueError("A prepared radiation wavelength grid must be one-dimensional")
        spectral = self.plasma_provider.prepare_stic_spectral_state(vacuum)
        self._prepared_wavelength_vacuum = vacuum.detach().clone()
        self._prepared_velocity_offsets = self._velocity_offsets(vacuum).detach().clone()
        self._prepared_spectral_lower = spectral.lower_indices.detach().clone()
        self._prepared_spectral_upper = spectral.upper_indices.detach().clone()
        self._prepared_spectral_weight = spectral.weight.detach().clone()

    @staticmethod
    def _point_fields(atmosphere, direction, stokes_basis, *, thermodynamic_field="gas_pressure"):
        required = ("temperature", thermodynamic_field, "microturbulence", "velocity_field", "magnetic_field")
        missing = set(required).difference(atmosphere)
        if missing:
            raise KeyError(f"Missing local atmospheric fields: {sorted(missing)}")
        temperature = atmosphere["temperature"]
        if temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError("Local LTE coefficients require float32 or float64 atmospheric fields")
        for name in required:
            value = atmosphere[name]
            expected = (*temperature.shape, 3) if name in required[-2:] else temperature.shape
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if value.dtype != temperature.dtype or value.device != temperature.device:
                raise TypeError("Atmospheric fields must share dtype and device")
        direction = torch.as_tensor(direction, dtype=temperature.dtype, device=temperature.device)
        basis = torch.as_tensor(stokes_basis, dtype=temperature.dtype, device=temperature.device)
        direction = torch.broadcast_to(direction, (*temperature.shape, 3))
        basis = torch.broadcast_to(basis, (*temperature.shape, 3, 3))
        if tensor_validation_enabled():
            tolerance = 2.0e-5 if temperature.dtype == torch.float32 else 1.0e-9
            identity = torch.eye(3, dtype=temperature.dtype, device=temperature.device)
            # Keep geometry checks and vector projections at the field precision,
            # even when neural-network matmuls use TF32 (matmul_precision="high").
            gram = (basis[..., :, None, :] * basis[..., None, :, :]).sum(dim=-1)
            require_tensor(
                lambda: torch.isclose(gram, identity, atol=tolerance, rtol=tolerance).all()
                & torch.isclose(basis[..., 2, :], direction, atol=tolerance, rtol=tolerance).all()
                & torch.isclose(torch.linalg.cross(basis[..., 0, :], basis[..., 1, :]), direction,
                                atol=tolerance, rtol=tolerance).all(),
                "stokes_basis must be right-handed orthonormal rows (e1,e2,direction)")
        velocity_los = -(atmosphere["velocity_field"] * direction).sum(dim=-1)
        magnetic = (basis * atmosphere["magnetic_field"][..., None, :]).sum(dim=-1)
        return temperature, atmosphere[thermodynamic_field], atmosphere["microturbulence"], velocity_los, magnetic

    def prepare_atmosphere(self, atmosphere, *, direction, stokes_basis):
        """Prepare wavelength-independent physics once; retain all gradients."""
        point_fields = self._point_fields(
            atmosphere, direction, stokes_basis
        )
        temperature, pressure = point_fields[:2]
        state = self.plasma_provider.prepare_plasma_state(temperature, pressure)
        reservoirs = self.plasma_provider.reference_line_thermodynamics(
            temperature, pressure, state,
            include_electron_density=self.line_opacity.requires_stark_electron_density,
        )
        populations = self.plasma_provider.reference_lower_level_populations(
            self.lines, temperature, pressure, state,
            fe_i_population_over_partition=reservoirs["fe_i_population_over_partition"],
        )
        logs = self.plasma_provider._hot_continued_stic_logs(
            self.plasma_provider.stic_log_continuum_components, state)
        temperature, _, microturbulence, velocity_los, magnetic = point_fields
        lines = self.line_opacity.prepare(
            temperature, velocity_los, microturbulence, magnetic,
            lower_level_populations=populations,
            damping_electron_density=reservoirs.get("electron_density"),
            damping_hydrogen_neutral=reservoirs["hydrogen_neutral"])
        return PreparedLTEAtmosphere(state, lines, logs)

    def prepare_spectrum(self, wavelength) -> PreparedSpectrum:
        """Prepare a whole query once, before splitting its wavelength axis."""
        if wavelength is None:
            if not self.wavelength_grid_prepared:
                raise ValueError("A shared grid must be prepared before omitting wavelengths")
            return PreparedSpectrum(self._prepared_wavelength_vacuum,
                self._prepared_velocity_offsets, STICSpectralState(
                    self._prepared_spectral_lower, self._prepared_spectral_upper,
                    self._prepared_spectral_weight))
        device = self.line_opacity.line_rest_frequency_hz.device
        if isinstance(wavelength, CenteredWavelength):
            offset = wavelength.offset_angstrom.to(device=device)
            vacuum = self._wavelength_tensor(offset + wavelength.origin_angstrom, device=device)
            velocity_offsets = torch.stack([
                (SPEED_OF_LIGHT / 1000.) * (
                    offset + (wavelength.origin_angstrom - int(rest)) - (rest - int(rest))
                ) / vacuum
                for rest in self._rest_vacuum_angstrom
            ])
        else:
            vacuum = self._wavelength_tensor(wavelength, device=device)
            velocity_offsets = self._velocity_offsets(vacuum)
        if vacuum.ndim == 0:
            vacuum = vacuum[None]
            velocity_offsets = velocity_offsets[..., None]
        flat = self.plasma_provider.prepare_stic_spectral_state(vacuum.reshape(-1))
        spectral = STICSpectralState(flat.lower_indices.reshape(vacuum.shape),
            flat.upper_indices.reshape(vacuum.shape), flat.weight.reshape(vacuum.shape))
        return PreparedSpectrum(vacuum, velocity_offsets, spectral)

    def evaluate_spectrum(self, prepared, spectrum, *, radiance_scale=1.0):
        """Broadcast spectral queries over points; evaluate only spectral physics."""
        if isinstance(radiance_scale, torch.Tensor) and radiance_scale.requires_grad:
            raise ValueError("radiance_scale must be fixed, not a trainable tensor")
        temperature = prepared.plasma.temperature
        vacuum = spectrum.vacuum.to(temperature)
        shape = torch.broadcast_shapes((*temperature.shape, 1), vacuum.shape)
        if shape[:-1] != temperature.shape:
            raise ValueError("Spectral queries must broadcast over existing atmospheric points")
        absorption, scattering = self.plasma_provider.continuum_components(
            vacuum, prepared.plasma, spectral_state=spectrum.continuum,
            grid_wavelengths=True, radiative_logs=prepared.continuum_logs)
        propagation = self.line_opacity.evaluate_prepared(
            prepared.lines, spectrum.velocity_offsets.to(temperature), absorption + scattering)
        thermal = planck_lambda(temperature[..., None].expand(shape), vacuum.expand(shape),
            radiance_scale=radiance_scale, paired_wavelengths=True)
        zero = torch.zeros_like(thermal)
        source = torch.stack((thermal, zero, zero, zero), dim=-1)
        return LocalRadiationCoefficients(propagation, source)

    def evaluate_prepared(self, prepared, wavelength, *, radiance_scale=1.0, paired=False):
        """Checked convenience interface for shared or one-wavelength-per-point queries."""
        if isinstance(radiance_scale, torch.Tensor) and radiance_scale.requires_grad:
            raise ValueError("radiance_scale must be fixed, not a trainable tensor")
        if paired:
            wavelength = self._wavelength_tensor(wavelength, device=prepared.plasma.temperature.device)
            if wavelength.shape != prepared.plasma.temperature.shape:
                raise ValueError("Paired wavelengths must match the atmospheric point shape")
            spectrum = self.prepare_spectrum(wavelength[..., None])
        else:
            if wavelength is not None and torch.as_tensor(wavelength).ndim > 1:
                raise ValueError("Shared wavelengths must be a scalar or one-dimensional block")
            spectrum = self.prepare_spectrum(wavelength)
        result = self.evaluate_spectrum(prepared, spectrum, radiance_scale=radiance_scale)
        if paired:
            return LocalRadiationCoefficients(result.propagation_m1.squeeze(-3), result.source.squeeze(-2))
        return result

    def forward(self, atmosphere: Mapping[str, torch.Tensor], wavelength_vacuum_angstrom=None, *, direction, stokes_basis, radiance_scale=1.0):
        prepared = self.prepare_atmosphere(atmosphere, direction=direction, stokes_basis=stokes_basis)
        return self.evaluate_prepared(prepared, wavelength_vacuum_angstrom, radiance_scale=radiance_scale)

    evaluate = forward

    def evaluate_paired(self, atmosphere: Mapping[str, torch.Tensor], wavelength_vacuum_angstrom, *, direction, stokes_basis, radiance_scale=1.0):
        prepared = self.prepare_atmosphere(atmosphere, direction=direction, stokes_basis=stokes_basis)
        return self.evaluate_prepared(prepared, wavelength_vacuum_angstrom, radiance_scale=radiance_scale, paired=True)

    def metadata(self) -> dict:
        return {
            "line_ids": [line.id for line in self.lines],
            "wavelength_frame": "inertial_vacuum_angstrom",
            "coefficient_units": "m^-1",
            "continuum_closure": "thermalized_continuum",
            "source_closure": "unpolarized_planck; emissivity=K_source",
            "motion_approximation": "longitudinal_line_shift_only",
        }


__all__ = ["LTELocalCoefficients", "LocalRadiationCoefficients"]
