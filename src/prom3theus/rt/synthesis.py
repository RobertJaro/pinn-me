"""Standalone differentiable LTE polarized spectrum synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from torch import nn

from prom3theus.core import SPEED_OF_LIGHT
from .atomic import AtomicDatabase, SpectralLine
from .opacity import (
    ContinuumOpacity,
    STICSpectralState,
    planck_lambda,
)
from .polarization import PolarizedLineOpacity, PropagationDiagnostics
from .transfer import (
    GeometricHeightPath,
    OpticalDepthPath,
    PolarizedFormalSolver,
    RayDistancePath,
    TransferPath,
)
from .wavelength import air_to_vacuum_angstrom

if TYPE_CHECKING:
    from .atmosphere import StratifiedAtmosphere


@dataclass(frozen=True)
class SynthesisDiagnostics:
    """Depth-dependent state returned by an opt-in diagnostic synthesis."""

    gas_pressure: torch.Tensor
    reference_thermodynamics: dict[str, torch.Tensor]
    alpha500: torch.Tensor
    continuum_extinction: torch.Tensor
    propagation: PropagationDiagnostics
    cumulative_tau500_along_path: torch.Tensor | None = None


class LTESynthesizer(nn.Module):
    """Differentiable, depth-stratified LTE forward model.

    This class owns no inversion network and can therefore be used directly to
    synthesize arbitrary atmospheres.  Public wavelengths are absolute air
    wavelengths in Angstrom; all thermodynamic calculations use SI internally.
    """

    def __init__(
        self,
        log_tau500=None,
        *,
        atomic_database: AtomicDatabase | None = None,
        lines: Sequence[SpectralLine] | None = None,
        line_ids: Sequence[str] | None = None,
        wavelength_window_angstrom: tuple[float, float] | None = None,
        continuum_opacity: ContinuumOpacity | None = None,
        formal_solver: nn.Module | None = None,
        faddeeva_coefficients: int = 32,
    ):
        super().__init__()
        selections = sum(
            value is not None for value in (lines, line_ids, wavelength_window_angstrom)
        )
        if selections != 1:
            raise ValueError(
                "Pass exactly one of lines, line_ids, or wavelength_window_angstrom."
            )
        self.atomic = atomic_database or AtomicDatabase()
        self.continuum_opacity = continuum_opacity or ContinuumOpacity(self.atomic)
        self.formal_solver = formal_solver or PolarizedFormalSolver()

        if lines is None:
            if line_ids is not None:
                lines = tuple(self.atomic.get_line(line_id) for line_id in line_ids)
            else:
                assert wavelength_window_angstrom is not None
                lines = self.atomic.select_lines(*wavelength_window_angstrom)
        self.lines = tuple(lines)
        if not self.lines:
            raise ValueError("The selected synthesis window contains no spectral lines")
        self.requires_stark_electron_density = any(
            line.log_gamma_stark_s_cm3 is not None for line in self.lines
        )
        self.line_opacity = PolarizedLineOpacity(
            self.lines,
            atomic_database=self.atomic,
            faddeeva_coefficients=faddeeva_coefficients,
        )

        if log_tau500 is None:
            depth_grid = torch.empty(0, dtype=torch.float32)
        else:
            depth_grid = torch.as_tensor(log_tau500)
            if not depth_grid.is_floating_point():
                depth_grid = depth_grid.to(torch.get_default_dtype())
            depth_grid = depth_grid.detach().clone()
            self._validate_depth_grid(depth_grid)
        self.register_buffer("log_tau500", depth_grid)
        self.register_buffer(
            "_prepared_wavelength_air_angstrom",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_wavelength_vacuum_angstrom",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_continuum_wavelength_vacuum_angstrom",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_frequency_hz",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_stic_lower_indices",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_stic_upper_indices",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_stic_weight",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )

    @staticmethod
    def _validate_depth_grid(grid: torch.Tensor) -> None:
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError(
                "log_tau500 must be one-dimensional with at least two points"
            )
        if grid.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "LTE synthesis supports float32 and float64 depth grids only"
            )
        if not torch.isfinite(grid).all() or not torch.all(grid[1:] > grid[:-1]):
            raise ValueError(
                "log_tau500 must be finite and increase from top to bottom"
            )

    @staticmethod
    def _validate_wavelength_grid(wavelength: torch.Tensor) -> None:
        if wavelength.ndim != 1 or wavelength.numel() < 2:
            raise ValueError(
                "wavelength_angstrom must be a one-dimensional grid with at least two points"
            )
        if wavelength.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "LTE synthesis supports float32 and float64 wavelength grids only"
            )
        if not torch.isfinite(wavelength).all() or not torch.all(
            wavelength[1:] > wavelength[:-1]
        ):
            raise ValueError(
                "wavelength_angstrom must be finite and strictly increasing"
            )

    def _atmosphere_grid(self, atmosphere: "StratifiedAtmosphere") -> torch.Tensor:
        grid = atmosphere.log_tau500
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError(
                "Atmosphere depth grid must be one-dimensional with at least two points"
            )
        if self.log_tau500.numel():
            configured = self.log_tau500.to(grid)
            if configured.shape != grid.shape:
                raise ValueError(
                    "Atmosphere depth grid differs from the synthesizer grid"
                )
            self._validate_depth_grid(grid)
            if not torch.allclose(
                configured,
                grid,
                rtol=0.0,
                atol=2.0 * torch.finfo(grid.dtype).eps,
            ):
                raise ValueError(
                    "Atmosphere depth grid differs from the synthesizer grid"
                )
        return grid

    @property
    def wavelength_grid_prepared(self) -> bool:
        """Whether a fixed spectral grid has been compiled into module buffers."""

        return self._prepared_wavelength_air_angstrom.numel() >= 2

    @torch.no_grad()
    def prepare_wavelength_grid(self, wavelength_angstrom) -> None:
        """Compile one fixed air-wavelength grid for repeated synthesis.

        Air-to-vacuum conversion, frequency construction, and STiC spectral
        validation/index lookup depend only on the wavelength grid.  Preparing
        them once keeps those value-dependent operations and accelerator
        synchronizations out of every inversion batch. Pass ``None`` to
        :meth:`forward` to consume this prepared state.
        """

        wavelength = torch.as_tensor(wavelength_angstrom)
        if not wavelength.is_floating_point():
            wavelength = wavelength.to(torch.get_default_dtype())
        wavelength = wavelength.to(
            device=self.line_opacity.line_rest_frequency_hz.device
        )
        self._validate_wavelength_grid(wavelength)
        wavelength = wavelength.detach().clone()
        wavelength_vacuum = air_to_vacuum_angstrom(wavelength)
        continuum_wavelength = torch.cat(
            (wavelength_vacuum, wavelength_vacuum.new_tensor([5000.0]))
        )
        spectral_state = self.continuum_opacity.prepare_stic_spectral_state(
            continuum_wavelength
        )
        self._prepared_wavelength_air_angstrom = wavelength
        self._prepared_wavelength_vacuum_angstrom = wavelength_vacuum
        self._prepared_continuum_wavelength_vacuum_angstrom = continuum_wavelength
        self._prepared_frequency_hz = wavelength_vacuum.new_tensor(SPEED_OF_LIGHT) / (
            wavelength_vacuum * 1.0e-10
        )
        self._prepared_stic_lower_indices = (
            spectral_state.lower_indices.detach().clone()
        )
        self._prepared_stic_upper_indices = (
            spectral_state.upper_indices.detach().clone()
        )
        self._prepared_stic_weight = spectral_state.weight.detach().clone()

    def _gas_pressure(
        self,
        atmosphere: "StratifiedAtmosphere",
        grid: torch.Tensor,
    ) -> torch.Tensor:
        del grid
        if atmosphere.gas_pressure is None:
            raise ValueError(
                "LTE synthesis requires gas_pressure. In inversion it is a smooth "
                "coordinate-network output constrained by the configured force-balance residual; the "
                "forward solver deliberately performs no iterative pressure solve."
            )
        return atmosphere.gas_pressure

    @staticmethod
    def _source_vector(source_function: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(source_function)
        return torch.stack((source_function, zeros, zeros, zeros), dim=-1)

    def forward(
        self,
        atmosphere: "StratifiedAtmosphere",
        wavelength_angstrom=None,
        *,
        path: TransferPath,
        return_diagnostics: bool = False,
        radiance_scale: torch.Tensor | float | None = None,
    ):
        """Synthesize emergent Stokes profiles with shape ``[..., 4, wavelength]``.

        The atmosphere passed here already contains the line-formation-frame
        ``v_los`` chosen by the observation adapter. Instrument-specific frame
        handling (for example HMI observer motion or the Hinode Level-1
        wavelength gauge) is deliberately outside this standalone solver.
        """

        if not isinstance(
            path,
            (OpticalDepthPath, GeometricHeightPath, RayDistancePath),
        ):
            raise TypeError(
                "path must be OpticalDepthPath, GeometricHeightPath, or RayDistancePath."
            )
        grid = self._atmosphere_grid(atmosphere)
        temperature = atmosphere.temperature
        if temperature.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "LTE synthesis supports float32 and float64 atmospheres only"
            )
        if wavelength_angstrom is None:
            if not self.wavelength_grid_prepared:
                raise RuntimeError(
                    "prepare_wavelength_grid() must be called before synthesis without an explicit grid."
                )
            wavelength = self._prepared_wavelength_air_angstrom.to(temperature)
            wavelength_vacuum = self._prepared_wavelength_vacuum_angstrom.to(
                temperature
            )
            continuum_wavelength = (
                self._prepared_continuum_wavelength_vacuum_angstrom.to(temperature)
            )
            spectral_state = STICSpectralState(
                self._prepared_stic_lower_indices,
                self._prepared_stic_upper_indices,
                self._prepared_stic_weight.to(temperature),
            )
            frequency_hz = self._prepared_frequency_hz.to(temperature)
        else:
            wavelength = torch.as_tensor(
                wavelength_angstrom,
                dtype=temperature.dtype,
                device=temperature.device,
            )
            self._validate_wavelength_grid(wavelength)
            wavelength_vacuum = air_to_vacuum_angstrom(wavelength)
            continuum_wavelength = torch.cat(
                (wavelength_vacuum, wavelength.new_tensor([5000.0]))
            )
            spectral_state = None
            frequency_hz = None
        gas_pressure = self._gas_pressure(atmosphere, grid)
        if (
            not torch.isfinite(temperature).all()
            or not torch.isfinite(gas_pressure).all()
            or not torch.isfinite(atmosphere.microturbulence).all()
            or not torch.isfinite(atmosphere.velocity_field).all()
            or not torch.isfinite(atmosphere.magnetic_field).all()
            or torch.any(temperature <= 0)
            or torch.any(gas_pressure <= 0)
            or torch.any(atmosphere.microturbulence < 0)
            or torch.any(atmosphere.v_los.abs() >= SPEED_OF_LIGHT)
        ):
            raise ValueError(
                "LTE atmospheric fields must be finite, temperature and gas pressure "
                "must be positive, microturbulence must be non-negative, and |v_los| < c."
            )
        lookup_state = self.continuum_opacity.prepare_stic_lookup(
            temperature,
            gas_pressure,
        )
        continuum_and_reference = self.continuum_opacity(
            continuum_wavelength,
            temperature,
            gas_pressure,
            lookup_state,
            spectral_state=spectral_state,
        )
        continuum_extinction = continuum_and_reference[..., :-1]
        alpha500 = continuum_and_reference[..., -1]

        line_thermodynamics = self.continuum_opacity.reference_line_thermodynamics(
            temperature,
            gas_pressure,
            lookup_state,
            include_electron_density=self.requires_stark_electron_density,
        )
        damping_electron_density = line_thermodynamics.get("electron_density")
        damping_hydrogen_neutral = line_thermodynamics["hydrogen_neutral"]
        lower_level_populations = (
            self.continuum_opacity.reference_lower_level_populations(
                self.lines,
                temperature,
                gas_pressure,
                lookup_state,
                fe_i_population_over_partition=line_thermodynamics[
                    "fe_i_population_over_partition"
                ],
            )
        )
        propagation_result = self.line_opacity(
            wavelength,
            temperature,
            atmosphere.v_los,
            atmosphere.microturbulence,
            atmosphere.magnetic_field,
            continuum_extinction,
            alpha500,
            lower_level_populations=lower_level_populations,
            damping_electron_density=damping_electron_density,
            damping_hydrogen_neutral=damping_hydrogen_neutral,
            normalize_to_alpha500=not isinstance(path, RayDistancePath),
            return_diagnostics=return_diagnostics,
            frequency_hz=frequency_hz,
        )
        if return_diagnostics:
            propagation, propagation_diagnostics = propagation_result
        else:
            propagation = propagation_result
            propagation_diagnostics = None
        # When a scale is supplied, Planck radiance is normalized directly in
        # log space. The physical ~1e13 radiance is never materialized in the
        # training graph or propagated through the formal-solver backward.
        source_function = planck_lambda(
            temperature,
            wavelength_vacuum,
            radiance_scale=radiance_scale,
        )
        emergent = self.formal_solver(
            propagation,
            self._source_vector(source_function),
            grid,
            path=path,
            reference_extinction_m1=(
                alpha500 if isinstance(path, GeometricHeightPath) else None
            ),
        )

        stokes = emergent.movedim(-1, -2)
        if not return_diagnostics:
            return stokes
        diagnostics = SynthesisDiagnostics(
            gas_pressure=gas_pressure,
            reference_thermodynamics={
                **self.continuum_opacity.reference_thermodynamics(
                    temperature, gas_pressure, lookup_state
                ),
                "hydrogen_neutral": damping_hydrogen_neutral,
                "fe_i_population_over_partition": (
                    self.continuum_opacity.reference_fe_i_population_over_partition(
                        temperature, gas_pressure, lookup_state
                    )
                ),
            },
            alpha500=alpha500,
            continuum_extinction=continuum_extinction,
            propagation=propagation_diagnostics,
            cumulative_tau500_along_path=self._cumulative_tau500(
                alpha500,
                path=path,
            ),
        )
        return stokes, diagnostics

    @staticmethod
    def _cumulative_tau500(
        alpha500: torch.Tensor,
        *,
        path: TransferPath,
    ) -> torch.Tensor | None:
        """Integrate absolute 500-nm opacity along the realized path."""

        if isinstance(path, RayDistancePath):
            distance = torch.as_tensor(path.distance_m, device=alpha500.device)
            interval_m = (distance[..., 1:] - distance[..., :-1]).abs().to(alpha500)
        elif isinstance(path, GeometricHeightPath):
            height = torch.as_tensor(path.height_m).to(alpha500)
            ray_mu = torch.as_tensor(
                path.mu, dtype=alpha500.dtype, device=alpha500.device
            )
            while ray_mu.ndim < height.ndim:
                ray_mu = ray_mu.unsqueeze(-1)
            interval_m = (height[..., :-1] - height[..., 1:]).abs() / ray_mu
        elif isinstance(path, OpticalDepthPath):
            return None
        else:
            raise TypeError(
                "path must be OpticalDepthPath, GeometricHeightPath, or RayDistancePath."
            )
        increments = 0.5 * (alpha500[..., :-1] + alpha500[..., 1:]) * interval_m
        return torch.cat(
            (torch.zeros_like(alpha500[..., :1]), torch.cumsum(increments, dim=-1)),
            dim=-1,
        )

    def metadata(self) -> dict:
        return {
            "lines": [line.id for line in self.lines],
            "wavelength_medium": "air",
            "wavelength_unit": "angstrom",
            "internal_wavelength_medium": "vacuum",
            "air_to_vacuum_conversion": (
                "VALD/Piskunov inverse of the Morton (2000) standard-air "
                "refractive index"
            ),
            "air_to_vacuum_reference_url": (
                "https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion"
            ),
            "velocity_unit": "m/s (positive redshift)",
            "bulk_doppler_shift": (
                "exact special-relativistic frequency factor "
                "sqrt((1-v/c)/(1+v/c)); |v|<c"
            ),
            "dispersion_coordinate": (
                "red-positive: (nu_component - nu) / delta_nu_D, equivalent "
                "to (lambda - lambda_component) / delta_lambda_D locally"
            ),
            "magnetic_field_unit": (
                "gauss, observer Stokes frame [Bx(+Q reference), "
                "By(increasing azimuth), B_los(positive toward observer)]"
            ),
            "atomic_model_units": {
                "oscillator_strength": "dimensionless f; stored log(g_l*f) is divided by 2*J_l+1",
                "atomic_mass": "u in metadata, converted to kg for thermal Doppler widths",
                "lower_level_population": "m^-3",
                "integrated_line_extinction": "m^-1 Hz",
                "frequency_profile": "Hz^-1",
                "monochromatic_line_extinction": "m^-1",
                "doppler_width": "Hz",
                "damping_rate": "s^-1",
                "zeeman_splitting": "Hz from B in gauss",
                "propagation_matrix": (
                    "m^-1 for RayDistancePath; dimensionless per unit vertical "
                    "tau500 for OpticalDepthPath and GeometricHeightPath"
                ),
            },
            "formal_solver": type(self.formal_solver).__name__,
            "formal_solution_coordinates": (
                "caller-selected OpticalDepthPath, GeometricHeightPath, or "
                "RayDistancePath"
            ),
            "ray_geometry": {
                "mu": "cosine of the ray to the local vertical; 0 < mu <= 1",
                "transfer_equation": (
                    "RayDistancePath: dI/ds=-K_length(I-S); OpticalDepthPath: "
                    "mu*dI/dtau500=K_tau(I-S); GeometricHeightPath converts "
                    "K_tau with alpha500"
                ),
                "application": (
                    "mu is carried only by plane-parallel optical-depth and "
                    "geometric-height paths"
                ),
                "ray_distance_sampling": (
                    "ordered full-3D points along each observed ray; continuum "
                    "tau500 is the cumulative trapezoidal integral of alpha500 ds"
                ),
            },
            "pressure_source": "supplied atmosphere (no iterative force-balance solve)",
            "thermodynamic_source": "pinned offline-generated STiC/Wittmann lookup",
            "continuum_lookup": self.continuum_opacity.metadata(),
            "thermodynamic_roles": {
                "stic_wittmann": (
                    "total continuum extinction, mass density for the configured force balance, "
                    "electron density for Stark broadening, physical neutral atomic-H "
                    "density for ABO broadening, n(Fe I)/U(Fe I) for LTE lower-level "
                    "populations, and the FALC reference stratification and gravity"
                ),
            },
        }


__all__ = ["LTESynthesizer", "SynthesisDiagnostics"]
