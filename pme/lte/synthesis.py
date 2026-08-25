"""Standalone differentiable LTE polarized spectrum synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from torch import nn

from pme.lte.atomic import AtomicDatabase, SpectralLine
from pme.lte.opacity import ContinuumOpacity, planck_lambda
from pme.lte.polarization import PolarizedLineOpacity, PropagationDiagnostics
from pme.lte.transfer import PolarizedFormalSolver
from pme.lte.wavelength import air_to_vacuum_angstrom

if TYPE_CHECKING:
    from pme.lte.atmosphere import StratifiedAtmosphere


HINODE_SP_FE_LINE_IDS = ("FeI_6301.5008", "FeI_6302.4932")


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
        atomic_data_directory=None,
        atomic_database: AtomicDatabase | None = None,
        lines: Sequence[SpectralLine] | None = None,
        line_ids: Sequence[str] | None = None,
        wavelength_window_angstrom: tuple[float, float] = (6300.7, 6303.3),
        continuum_opacity: ContinuumOpacity | None = None,
        formal_solver: nn.Module | None = None,
        faddeeva_coefficients: int = 32,
    ):
        super().__init__()
        if atomic_database is not None and atomic_data_directory is not None:
            raise ValueError(
                "Pass either atomic_database or atomic_data_directory, not both."
            )
        self.atomic = atomic_database or AtomicDatabase(
            data_directory=atomic_data_directory
        )
        self.continuum_opacity = continuum_opacity or ContinuumOpacity(self.atomic)
        self.formal_solver = formal_solver or PolarizedFormalSolver()

        if lines is not None and line_ids is not None:
            raise ValueError("Pass either lines or line_ids, not both")
        if lines is None:
            if line_ids is not None:
                lines = tuple(self.atomic.get_line(line_id) for line_id in line_ids)
            else:
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

    @staticmethod
    def _validate_depth_grid(grid: torch.Tensor) -> None:
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("log_tau500 must be one-dimensional with at least two points")
        if not bool(torch.all(grid[1:] > grid[:-1])):
            raise ValueError("log_tau500 must increase from top to bottom")

    def _atmosphere_grid(self, atmosphere: "StratifiedAtmosphere") -> torch.Tensor:
        grid = atmosphere.log_tau500
        self._validate_depth_grid(grid)
        if self.log_tau500.numel():
            configured = self.log_tau500.to(grid)
            if configured.shape != grid.shape or not torch.allclose(
                configured,
                grid,
                rtol=0.0,
                atol=2.0 * torch.finfo(grid.dtype).eps,
            ):
                raise ValueError("Atmosphere depth grid differs from the synthesizer grid")
        return grid

    def _gas_pressure(self, atmosphere: "StratifiedAtmosphere", grid: torch.Tensor) -> torch.Tensor:
        del grid
        if atmosphere.gas_pressure is None:
            raise ValueError(
                "LTE synthesis requires gas_pressure. In inversion it is a smooth "
                "coordinate-network output constrained by the MHS residual; the "
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
        wavelength_angstrom,
        mu: torch.Tensor | float = 1.0,
        *,
        return_diagnostics: bool = False,
        radiance_scale: torch.Tensor | float | None = None,
        ray_distance_m: torch.Tensor | None = None,
    ):
        """Synthesize emergent Stokes profiles with shape ``[..., 4, wavelength]``.

        The only bulk velocity entering line formation is the physical
        atmosphere's solar-relative ``v_los``. Instrument-observer motion is
        an upstream data-calibration concern and is deliberately absent from
        this API; Hinode Level-1 ``DOP_RCV`` has already been removed by
        ``sp_prep``.
        """

        grid = self._atmosphere_grid(atmosphere)
        temperature = atmosphere.temperature
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        if wavelength.ndim != 1 or wavelength.numel() < 2:
            raise ValueError("wavelength_angstrom must be a one-dimensional grid")
        if not torch.isfinite(wavelength).all() or torch.any(wavelength <= 0):
            raise ValueError("wavelength_angstrom must be finite and positive")
        ray_mu = torch.as_tensor(mu, dtype=temperature.dtype, device=temperature.device)
        if ray_distance_m is None and (
            not torch.isfinite(ray_mu).all() or torch.any((ray_mu <= 0) | (ray_mu > 1))
        ):
            raise ValueError("mu must be finite and satisfy 0 < mu <= 1")

        gas_pressure = self._gas_pressure(atmosphere, grid)
        wavelength_vacuum = air_to_vacuum_angstrom(wavelength)
        continuum_extinction = self.continuum_opacity(
            wavelength_vacuum,
            temperature,
            gas_pressure,
        )
        reference_wavelength = wavelength.new_tensor([5000.0])
        alpha500 = self.continuum_opacity(
            reference_wavelength,
            temperature,
            gas_pressure,
        )[..., 0]

        damping_electron_density = (
            self.continuum_opacity.reference_electron_density(
                temperature, gas_pressure
            )
            if self.requires_stark_electron_density
            else None
        )
        damping_hydrogen_neutral = (
            self.continuum_opacity.reference_neutral_hydrogen_density(
                temperature, gas_pressure
            )
        )
        lower_level_populations = (
            self.continuum_opacity.reference_lower_level_populations(
                self.lines, temperature, gas_pressure
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
            return_diagnostics=return_diagnostics,
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
            mu=ray_mu,
            geometric_height_m=atmosphere.geometric_height_m,
            alpha500=(
                alpha500 if atmosphere.geometric_height_m is not None else None
            ),
            ray_distance_m=ray_distance_m,
        )

        stokes = emergent.movedim(-1, -2)
        if not return_diagnostics:
            return stokes
        diagnostics = SynthesisDiagnostics(
            gas_pressure=gas_pressure,
            reference_thermodynamics={
                **self.continuum_opacity.reference_thermodynamics(
                    temperature, gas_pressure
                ),
                "hydrogen_neutral": damping_hydrogen_neutral,
                "fe_i_population_over_partition": (
                    self.continuum_opacity.reference_fe_i_population_over_partition(
                        temperature, gas_pressure
                    )
                ),
            },
            alpha500=alpha500,
            continuum_extinction=continuum_extinction,
            propagation=propagation_diagnostics,
            cumulative_tau500_along_path=self._cumulative_tau500(
                alpha500,
                ray_distance_m=ray_distance_m,
                geometric_height_m=atmosphere.geometric_height_m,
                mu=ray_mu,
            ),
        )
        return stokes, diagnostics

    @staticmethod
    def _cumulative_tau500(
        alpha500: torch.Tensor,
        *,
        ray_distance_m: torch.Tensor | None,
        geometric_height_m: torch.Tensor | None,
        mu: torch.Tensor,
    ) -> torch.Tensor | None:
        """Integrate absolute 500-nm opacity along the realized path."""

        if ray_distance_m is not None:
            # These are solar-local offsets from the outer shell, not absolute
            # spacecraft distances. Difference before matching opacity dtype.
            distance = torch.as_tensor(ray_distance_m, device=alpha500.device)
            interval_m = (
                distance[..., 1:] - distance[..., :-1]
            ).abs().to(alpha500)
        elif geometric_height_m is not None:
            height = geometric_height_m.to(alpha500)
            ray_mu = mu
            while ray_mu.ndim < height.ndim:
                ray_mu = ray_mu.unsqueeze(-1)
            interval_m = (height[..., :-1] - height[..., 1:]).abs() / ray_mu
        else:
            return None
        if torch.any(interval_m <= 0):
            raise ValueError("The opacity-integration path must have nonzero layer lengths.")
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
                "https://www.astro.uu.se/valdwiki/"
                "Air-to-vacuum%20conversion"
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
                    "constructed dimensionless per unit vertical tau500; "
                    "multiplied by alpha500 to m^-1 when geometric heights are present"
                ),
            },
            "formal_solver": type(self.formal_solver).__name__,
            "formal_solution_coordinates": (
                "geometric height when supplied by the atmosphere; otherwise tau500"
            ),
            "ray_geometry": {
                "mu": "cosine of the ray to the local vertical; 0 < mu <= 1",
                "transfer_equation": (
                    "exact-ray mode: dI/ds=-K_length(I-S); legacy mode: "
                    "mu*dI/dtau500=K_tau(I-S)"
                ),
                "application": (
                    "not used by transfer when exact ray distances are supplied; "
                    "otherwise applied once as delta_tau500/mu"
                ),
                "exact_ray_sampling": (
                    "ordered full-3D points along each observed ray; continuum "
                    "tau500 is the cumulative trapezoidal integral of alpha500 ds"
                ),
            },
            "pressure_source": "supplied atmosphere (no iterative MHS solve)",
            "runtime_atomic_eos": (
                "none; production thermodynamics and line populations use the "
                "pinned offline-generated STiC lookup"
            ),
            "continuum_lookup": self.continuum_opacity.metadata(),
            "thermodynamic_roles": {
                "stic_wittmann": (
                    "total continuum extinction, mass density for the configured MHS, "
                    "electron density for Stark broadening, physical neutral atomic-H "
                    "density for ABO broadening, n(Fe I)/U(Fe I) for LTE lower-level "
                    "populations, and the FALC reference stratification and gravity"
                ),
                "barklem_saha": (
                    "reference-only atomic state and the inspectable H-minus-only "
                    "continuum decomposition; it is not used for production continuum, "
                    "Fe-I line populations, damping perturbers, or production MHS"
                ),
            },
        }


__all__ = ["HINODE_SP_FE_LINE_IDS", "LTESynthesizer", "SynthesisDiagnostics"]
