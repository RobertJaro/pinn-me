"""Typed configuration contract for the LTE inversion framework.

The schema deliberately contains data only.  Runtime modules may depend on this
package, while configuration loading never imports the solver, observations, or
instrument implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import math
from pathlib import Path
from typing import Any, Literal


_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def _plain_value(value: Any) -> Any:
    """Convert a configuration value to YAML/JSON-friendly built-in types."""

    if is_dataclass(value):
        return {
            field.name: _plain_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    return value


class ConfigNode:
    """Mixin shared by all typed configuration nodes."""

    def to_dict(self) -> dict[str, Any]:
        """Return a detached runtime mapping while preserving section names."""

        plain = _plain_value(self)
        if not isinstance(plain, dict):  # pragma: no cover - defensive invariant
            raise TypeError("A configuration node must serialize to a mapping.")
        return plain


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    _finite(float(value), name)
    if value < 0 if allow_zero else value <= 0:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")


def _ordered_pair(values: tuple[float, float], name: str) -> None:
    lower, upper = values
    _finite(lower, f"{name}[0]")
    _finite(upper, f"{name}[1]")
    if lower >= upper:
        raise ValueError(f"{name} must be strictly increasing")


def _quantiles(values: tuple[float, float], name: str) -> None:
    _ordered_pair(values, name)
    if values[0] < 0.0 or values[1] > 1.0:
        raise ValueError(f"{name} must lie within [0, 1]")


@dataclass(frozen=True, slots=True)
class SolverConfig(ConfigNode):
    kind: Literal["lte"]
    output_directory: Path
    work_directory: Path


@dataclass(frozen=True, slots=True)
class ResourceConfig(ConfigNode):
    bundle: Literal["packaged"]


@dataclass(frozen=True, slots=True)
class SlitSliceConfig(ConfigNode):
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("slit slice must satisfy 0 <= start < stop")


@dataclass(frozen=True, slots=True)
class DataLoaderConfig(ConfigNode):
    batch_size: int
    validation_batch_size: int | None = None
    validation_stride: int = 1
    preparation_workers: int = 1
    workers: int = 0
    pin_memory: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        _positive(self.batch_size, "batch_size")
        if self.validation_batch_size is not None:
            _positive(self.validation_batch_size, "validation_batch_size")
        _positive(self.validation_stride, "validation_stride")
        _positive(self.preparation_workers, "preparation_workers")
        _positive(self.workers, "workers", allow_zero=True)


@dataclass(frozen=True, slots=True)
class HinodeSelectionConfig(ConfigNode):
    slit_slice: SlitSliceConfig


@dataclass(frozen=True, slots=True)
class HinodeCalibrationConfig(ConfigNode):
    continuum_edge_samples: int
    quiet_sun_max_fractional_polarization: float
    quiet_sun_continuum_trim_quantiles: tuple[float, float]
    minimum_quiet_sun_pixels: int
    stokes_reference_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.continuum_edge_samples, "continuum_edge_samples")
        _positive(
            self.quiet_sun_max_fractional_polarization,
            "quiet_sun_max_fractional_polarization",
        )
        _quantiles(
            self.quiet_sun_continuum_trim_quantiles,
            "quiet_sun_continuum_trim_quantiles",
        )
        _positive(self.minimum_quiet_sun_pixels, "minimum_quiet_sun_pixels")
        _finite(self.stokes_reference_angle_deg, "stokes_reference_angle_deg")


@dataclass(frozen=True, slots=True)
class HinodeObservationConfig(ConfigNode):
    type: Literal["hinode_sp"]
    directory: Path
    selection: HinodeSelectionConfig
    loader: DataLoaderConfig
    calibration: HinodeCalibrationConfig


@dataclass(frozen=True, slots=True)
class HMISelectionConfig(ConfigNode):
    acquisition_indices: tuple[int, ...] | None = None
    validation_raster: int = 0

    def __post_init__(self) -> None:
        if self.validation_raster < 0:
            raise ValueError("validation_raster must be non-negative")
        if self.acquisition_indices is None:
            return
        if not self.acquisition_indices:
            raise ValueError("acquisition_indices must be null or non-empty")
        if any(index < 0 for index in self.acquisition_indices):
            raise ValueError("acquisition_indices must be non-negative")
        if tuple(sorted(set(self.acquisition_indices))) != self.acquisition_indices:
            raise ValueError("acquisition_indices must be unique and increasing")
        if self.validation_raster not in self.acquisition_indices:
            raise ValueError(
                "validation_raster must identify one of the selected acquisition_indices"
            )


@dataclass(frozen=True, slots=True)
class HMICalibrationConfig(ConfigNode):
    transmission_profile_directory: Path
    require_quality_zero: bool
    quiet_sun_max_fractional_polarization: float
    quiet_sun_trim_quantiles: tuple[float, float]
    minimum_quiet_sun_pixels: int
    calibration_sample_limit: int

    def __post_init__(self) -> None:
        _positive(
            self.quiet_sun_max_fractional_polarization,
            "quiet_sun_max_fractional_polarization",
        )
        _quantiles(self.quiet_sun_trim_quantiles, "quiet_sun_trim_quantiles")
        _positive(self.minimum_quiet_sun_pixels, "minimum_quiet_sun_pixels")
        _positive(self.calibration_sample_limit, "calibration_sample_limit")


@dataclass(frozen=True, slots=True)
class HMIObservationConfig(ConfigNode):
    type: Literal["hmi_stokes"]
    directory: Path
    selection: HMISelectionConfig
    loader: DataLoaderConfig
    calibration: HMICalibrationConfig


ObservationConfig = HinodeObservationConfig | HMIObservationConfig


@dataclass(frozen=True, slots=True)
class DepthGridConfig(ConfigNode):
    coordinate: Literal["log_tau500"]
    minimum: float
    maximum: float
    count: int

    def __post_init__(self) -> None:
        _ordered_pair((self.minimum, self.maximum), "depth grid bounds")
        if self.count < 2:
            raise ValueError("depth grid count must be at least two")


@dataclass(frozen=True, slots=True)
class AtmosphereGeometryConfig(ConfigNode):
    type: Literal["spherical_shell"]
    time_dependent: bool
    height_input_scale_m: float
    outer_height_megameter: float
    inner_height_megameter: float
    tangent_margin_m: float
    line_formation_outer_height_megameter: float | None = None

    def __post_init__(self) -> None:
        if self.line_formation_outer_height_megameter is None:
            object.__setattr__(
                self,
                "line_formation_outer_height_megameter",
                self.outer_height_megameter,
            )
        _positive(self.height_input_scale_m, "height_input_scale_m")
        _finite(self.outer_height_megameter, "outer_height_megameter")
        _finite(
            self.line_formation_outer_height_megameter,
            "line_formation_outer_height_megameter",
        )
        _finite(self.inner_height_megameter, "inner_height_megameter")
        if self.outer_height_megameter <= self.inner_height_megameter:
            raise ValueError("outer shell height must exceed inner shell height")
        if not (
            self.inner_height_megameter
            < self.line_formation_outer_height_megameter
            <= self.outer_height_megameter
        ):
            raise ValueError(
                "line-formation outer height must exceed the inner shell height "
                "and not exceed the full-domain outer height"
            )
        _positive(self.tangent_margin_m, "tangent_margin_m")


@dataclass(frozen=True, slots=True)
class UpperAtmosphereConfig(ConfigNode):
    """Reference thermodynamics above the LTE line-formation atmosphere."""

    type: Literal["hydrostatic_corona"]
    transition_region_top_megameter: float
    coronal_temperature_k: float
    reference_grid_points: int = 2048

    def __post_init__(self) -> None:
        _positive(
            self.transition_region_top_megameter,
            "transition_region_top_megameter",
        )
        _positive(self.coronal_temperature_k, "coronal_temperature_k")
        if self.coronal_temperature_k < 1.0e5:
            raise ValueError("coronal_temperature_k must be at least 1e5 K")
        if self.reference_grid_points < 64:
            raise ValueError("reference_grid_points must be at least 64")


@dataclass(frozen=True, slots=True)
class LogParameterConfig(ConfigNode):
    """Scale for an unbounded residual in natural-log parameter space."""

    log_scale: float

    def __post_init__(self) -> None:
        _positive(self.log_scale, "log_scale")


@dataclass(frozen=True, slots=True)
class VelocityParameterConfig(ConfigNode):
    scale_m_per_s: float
    maximum_m_per_s: float

    def __post_init__(self) -> None:
        _positive(self.scale_m_per_s, "scale_m_per_s")
        _positive(self.maximum_m_per_s, "maximum_m_per_s")
        if self.maximum_m_per_s >= _SPEED_OF_LIGHT_M_PER_S:
            raise ValueError("maximum_m_per_s must be subluminal")


@dataclass(frozen=True, slots=True)
class MagneticParameterConfig(ConfigNode):
    scale_gauss: float

    def __post_init__(self) -> None:
        _positive(self.scale_gauss, "scale_gauss")


@dataclass(frozen=True, slots=True)
class AtmosphereParameterConfig(ConfigNode):
    temperature: LogParameterConfig
    velocity: VelocityParameterConfig
    magnetic_field: MagneticParameterConfig
    microturbulence: LogParameterConfig
    gas_pressure: LogParameterConfig


@dataclass(frozen=True, slots=True)
class EncodingConfig(ConfigNode):
    type: Literal["fourier"]
    num_frequencies: tuple[int, ...]
    max_frequencies: tuple[int, ...]
    include_input: bool = True

    def __post_init__(self) -> None:
        if not self.num_frequencies:
            raise ValueError("num_frequencies must not be empty")
        if len(self.num_frequencies) != len(self.max_frequencies):
            raise ValueError("Fourier frequency lists must have equal length")
        if any(value <= 0 for value in self.num_frequencies):
            raise ValueError("num_frequencies entries must be positive")
        if any(value <= 0 for value in self.max_frequencies):
            raise ValueError("max_frequencies entries must be positive")


@dataclass(frozen=True, slots=True)
class NetworkConfig(ConfigNode):
    type: Literal["mlp"]
    hidden_dimension: int
    hidden_layers: int
    activation: Literal["silu", "gelu", "tanh"]
    encoding: EncodingConfig

    def __post_init__(self) -> None:
        _positive(self.hidden_dimension, "hidden_dimension")
        _positive(self.hidden_layers, "hidden_layers")


@dataclass(frozen=True, slots=True)
class AtmosphereConfig(ConfigNode):
    depth_grid: DepthGridConfig
    geometry: AtmosphereGeometryConfig
    reference_atmosphere: Literal["falc_82"]
    parameters: AtmosphereParameterConfig
    network: NetworkConfig
    upper_atmosphere: UpperAtmosphereConfig | None = None

    def __post_init__(self) -> None:
        expected_dimensions = 4 if self.geometry.time_dependent else 3
        frequency_dimensions = len(self.network.encoding.num_frequencies)
        if frequency_dimensions != expected_dimensions:
            raise ValueError(
                "Fourier frequency lists must contain one entry per atmosphere "
                f"coordinate ({expected_dimensions}); got {frequency_dimensions}"
            )
        if self.depth_grid.minimum < 0.0 < self.depth_grid.maximum and not (
            self.geometry.outer_height_megameter
            > 0.0
            > self.geometry.inner_height_megameter
        ):
            raise ValueError(
                "An atmosphere depth grid crossing log_tau500=0 requires a shell "
                "with positive outer height and negative inner height"
            )
        geometry = self.geometry
        extrapolation = (
            geometry.line_formation_outer_height_megameter
            < geometry.outer_height_megameter
        )
        if extrapolation != (self.upper_atmosphere is not None):
            raise ValueError(
                "upper_atmosphere must be configured exactly when the full domain "
                "extends above the LTE line-formation domain"
            )
        if self.upper_atmosphere is not None and not (
            geometry.line_formation_outer_height_megameter
            < self.upper_atmosphere.transition_region_top_megameter
            <= geometry.outer_height_megameter
        ):
            raise ValueError(
                "transition_region_top_megameter must lie above the line-formation "
                "top and at or below the full-domain top"
            )


@dataclass(frozen=True, slots=True)
class SynthesisConfig(ConfigNode):
    line_ids: tuple[str, ...]
    excluded_wavelength_windows_angstrom: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.line_ids or any(not value.strip() for value in self.line_ids):
            raise ValueError("line_ids must contain at least one non-empty identifier")
        if len(set(self.line_ids)) != len(self.line_ids):
            raise ValueError("line_ids must be unique")
        for index, interval in enumerate(self.excluded_wavelength_windows_angstrom):
            _ordered_pair(interval, f"excluded_wavelength_windows_angstrom[{index}]")


@dataclass(frozen=True, slots=True)
class SpectralPSFConfig(ConfigNode):
    type: Literal["gaussian"]
    fwhm_angstrom: float
    oversample: int
    truncate_sigma: float

    def __post_init__(self) -> None:
        _positive(self.fwhm_angstrom, "fwhm_angstrom")
        _positive(self.oversample, "oversample")
        _positive(self.truncate_sigma, "truncate_sigma")


@dataclass(frozen=True, slots=True)
class HinodeInstrumentConfig(ConfigNode):
    type: Literal["hinode_sp"]
    spectral_psf: SpectralPSFConfig
    line_of_sight_velocity_correction_m_per_s: float = 0.0
    optimize_line_of_sight_velocity_correction: bool = False

    def __post_init__(self) -> None:
        if type(self.optimize_line_of_sight_velocity_correction) is not bool:
            raise TypeError(
                "optimize_line_of_sight_velocity_correction must be boolean"
            )
        _finite(
            self.line_of_sight_velocity_correction_m_per_s,
            "line_of_sight_velocity_correction_m_per_s",
        )
        if (
            abs(self.line_of_sight_velocity_correction_m_per_s)
            >= _SPEED_OF_LIGHT_M_PER_S
        ):
            raise ValueError(
                "line_of_sight_velocity_correction_m_per_s must be subluminal"
            )


@dataclass(frozen=True, slots=True)
class HMIInstrumentConfig(ConfigNode):
    type: Literal["hmi_filter_profiles"]
    magnetic_azimuth_offset_deg: float
    line_of_sight_velocity_correction_m_per_s: float = 0.0
    optimize_line_of_sight_velocity_correction: bool = False

    def __post_init__(self) -> None:
        _finite(self.magnetic_azimuth_offset_deg, "magnetic_azimuth_offset_deg")
        if float(self.magnetic_azimuth_offset_deg) != 90.0:
            raise ValueError(
                "HMI magnetic_azimuth_offset_deg must be exactly 90 degrees"
            )
        if type(self.optimize_line_of_sight_velocity_correction) is not bool:
            raise TypeError(
                "optimize_line_of_sight_velocity_correction must be boolean"
            )
        _finite(
            self.line_of_sight_velocity_correction_m_per_s,
            "line_of_sight_velocity_correction_m_per_s",
        )
        if (
            abs(self.line_of_sight_velocity_correction_m_per_s)
            >= _SPEED_OF_LIGHT_M_PER_S
        ):
            raise ValueError(
                "line_of_sight_velocity_correction_m_per_s must be subluminal"
            )


InstrumentConfig = HinodeInstrumentConfig | HMIInstrumentConfig


@dataclass(frozen=True, slots=True)
class CoarseToFineConfig(ConfigNode):
    enabled: bool
    fine_sample_count: int | None = None
    uniform_weight_floor: float | None = None

    def __post_init__(self) -> None:
        if self.enabled and (
            self.fine_sample_count is None or self.uniform_weight_floor is None
        ):
            raise ValueError(
                "enabled coarse-to-fine sampling requires fine_sample_count and "
                "uniform_weight_floor"
            )
        if self.fine_sample_count is None:
            object.__setattr__(self, "fine_sample_count", 1)
        if self.uniform_weight_floor is None:
            object.__setattr__(self, "uniform_weight_floor", 0.0)
        _positive(self.fine_sample_count, "fine_sample_count")
        _positive(self.uniform_weight_floor, "uniform_weight_floor", allow_zero=True)
        if self.uniform_weight_floor > 1.0:
            raise ValueError("uniform_weight_floor must not exceed one")


@dataclass(frozen=True, slots=True)
class DepthSamplingConfig(ConfigNode):
    sample_count: int
    coarse_to_fine: CoarseToFineConfig

    def __post_init__(self) -> None:
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")


@dataclass(frozen=True, slots=True)
class VectorRegularizationConfig(ConfigNode):
    enabled: bool = False
    magnetic_weight: float = 0.0
    velocity_weight: float = 0.0
    decay_steps: int = 0

    def __post_init__(self) -> None:
        _positive(self.magnetic_weight, "magnetic_weight", allow_zero=True)
        _positive(self.velocity_weight, "velocity_weight", allow_zero=True)
        _positive(self.decay_steps, "decay_steps", allow_zero=not self.enabled)
        if self.enabled and self.decay_steps == 0:
            raise ValueError(
                "enabled vector regularization requires positive decay_steps"
            )
        if self.enabled and self.magnetic_weight == self.velocity_weight == 0.0:
            raise ValueError("enabled vector regularization requires a positive weight")
        if not self.enabled and any(
            value != 0
            for value in (
                self.magnetic_weight,
                self.velocity_weight,
                self.decay_steps,
            )
        ):
            raise ValueError(
                "disabled vector regularization requires zero weights and decay_steps"
            )


@dataclass(frozen=True, slots=True)
class CollocationConfig(ConfigNode):
    volume_points_per_step: int
    height_layers_per_step: int
    upper_boundary_points_per_step: int
    validation_height_layers: int
    validation_points_per_height: int
    upper_volume_points_per_step: int = 0
    upper_height_layers_per_step: int = 0
    side_boundary_points_per_step: int = 0
    side_height_layers_per_step: int = 0
    validation_upper_height_layers: int = 0

    def __post_init__(self) -> None:
        for name in (
            "volume_points_per_step",
            "height_layers_per_step",
            "validation_height_layers",
            "validation_points_per_height",
        ):
            _positive(getattr(self, name), name)
        for name in (
            "upper_volume_points_per_step",
            "upper_height_layers_per_step",
            "upper_boundary_points_per_step",
            "side_boundary_points_per_step",
            "side_height_layers_per_step",
            "validation_upper_height_layers",
        ):
            _positive(getattr(self, name), name, allow_zero=True)


@dataclass(frozen=True, slots=True)
class PhysicsNormalizationConfig(ConfigNode):
    length_m: float
    time_s: float
    magnetic_field_floor_gauss: float = 1.0
    velocity_scale_m_per_s: float = 1_000.0

    def __post_init__(self) -> None:
        _positive(self.length_m, "length_m")
        _positive(self.time_s, "time_s")
        _positive(self.magnetic_field_floor_gauss, "magnetic_field_floor_gauss")
        _positive(self.velocity_scale_m_per_s, "velocity_scale_m_per_s")


@dataclass(frozen=True, slots=True)
class EquationConfig(ConfigNode):
    enabled: bool
    weight: float

    def __post_init__(self) -> None:
        _positive(self.weight, "equation weight", allow_zero=True)
        if self.enabled != (self.weight > 0.0):
            raise ValueError(
                "equation enabled must be true exactly when its weight is positive"
            )


def _disabled_equation() -> EquationConfig:
    return EquationConfig(enabled=False, weight=0.0)


@dataclass(frozen=True, slots=True)
class PhysicsEquationConfig(ConfigNode):
    hydrostatic_equilibrium: EquationConfig
    magnetohydrostatic_equilibrium: EquationConfig
    momentum: EquationConfig
    magnetic_divergence: EquationConfig
    induction: EquationConfig
    continuity: EquationConfig
    upper_boundary_gas_pressure_prior: EquationConfig
    adiabatic_pressure: EquationConfig = field(default_factory=_disabled_equation)
    radial_magnetic_energy_gradient: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_boundary_open_velocity: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_domain_microturbulence_prior: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_domain_temperature_prior: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_boundary_current_free: EquationConfig = field(
        default_factory=_disabled_equation
    )
    side_boundary_open_velocity: EquationConfig = field(
        default_factory=_disabled_equation
    )
    side_boundary_current_free: EquationConfig = field(
        default_factory=_disabled_equation
    )


@dataclass(frozen=True, slots=True)
class PhysicsConfig(ConfigNode):
    vector_basis_matches_spatial_coordinates: bool
    collocation: CollocationConfig
    normalization: PhysicsNormalizationConfig
    equations: PhysicsEquationConfig
    adiabatic_index: float = 5.0 / 3.0
    upper_boundary_current_free_ramp_steps: int = 0

    def __post_init__(self) -> None:
        _finite(self.adiabatic_index, "adiabatic_index")
        if self.adiabatic_index <= 1.0:
            raise ValueError("adiabatic_index must be greater than one")
        if self.upper_boundary_current_free_ramp_steps < 0:
            raise ValueError(
                "upper_boundary_current_free_ramp_steps must be non-negative"
            )
        if (
            not (
                self.equations.upper_boundary_current_free.enabled
                or self.equations.side_boundary_current_free.enabled
            )
            and self.upper_boundary_current_free_ramp_steps != 0
        ):
            raise ValueError(
                "Disabled current-free boundaries require zero ramp steps"
            )
        force_balance_equations = (
            self.equations.hydrostatic_equilibrium,
            self.equations.magnetohydrostatic_equilibrium,
            self.equations.momentum,
        )
        if (
            sum(
                equation.enabled and equation.weight > 0.0
                for equation in force_balance_equations
            )
            > 1
        ):
            raise ValueError(
                "hydrostatic_equilibrium, magnetohydrostatic_equilibrium, and "
                "momentum are mutually exclusive"
            )
        volume_equations = (
            self.equations.hydrostatic_equilibrium,
            self.equations.magnetohydrostatic_equilibrium,
            self.equations.momentum,
            self.equations.magnetic_divergence,
            self.equations.radial_magnetic_energy_gradient,
            self.equations.induction,
            self.equations.continuity,
            self.equations.adiabatic_pressure,
        )
        volume_active = any(
            equation.enabled and equation.weight > 0.0 for equation in volume_equations
        )
        vector_derivatives_active = any(
            equation.enabled and equation.weight > 0.0
            for equation in (
                self.equations.magnetohydrostatic_equilibrium,
                self.equations.momentum,
                self.equations.magnetic_divergence,
                self.equations.radial_magnetic_energy_gradient,
                self.equations.induction,
                self.equations.continuity,
                self.equations.adiabatic_pressure,
                self.equations.upper_boundary_open_velocity,
                self.equations.upper_boundary_current_free,
                self.equations.side_boundary_open_velocity,
                self.equations.side_boundary_current_free,
            )
        )
        if (
            vector_derivatives_active
            and not self.vector_basis_matches_spatial_coordinates
        ):
            raise ValueError(
                "Vector differential equations require "
                "vector_basis_matches_spatial_coordinates=true"
            )
        if volume_active and (
            self.collocation.height_layers_per_step
            > self.collocation.volume_points_per_step
            or self.collocation.volume_points_per_step
            % self.collocation.height_layers_per_step
            != 0
        ):
            raise ValueError(
                "Active volume physics requires volume_points_per_step to be "
                "divisible by height_layers_per_step"
            )
        if volume_active and self.collocation.validation_height_layers < 2:
            raise ValueError(
                "Active volume physics requires at least two validation height layers"
            )
        upper_volume_active = any(
            equation.enabled and equation.weight > 0.0
            for equation in (
                self.equations.upper_domain_microturbulence_prior,
                self.equations.upper_domain_temperature_prior,
            )
        )
        if upper_volume_active and (
            self.collocation.upper_height_layers_per_step < 1
            or self.collocation.upper_volume_points_per_step
            % self.collocation.upper_height_layers_per_step
            != 0
            or self.collocation.validation_upper_height_layers < 2
        ):
            raise ValueError(
                "Active upper-domain physics requires positive, divisible upper "
                "collocation counts and at least two validation height layers"
            )
        side_boundary_active = any(
            equation.enabled and equation.weight > 0.0
            for equation in (
                self.equations.side_boundary_open_velocity,
                self.equations.side_boundary_current_free,
            )
        )
        side_points_per_height = (
            self.collocation.side_boundary_points_per_step
            // max(self.collocation.side_height_layers_per_step, 1)
        )
        if side_boundary_active and (
            self.collocation.side_height_layers_per_step < 1
            or self.collocation.side_boundary_points_per_step
            % self.collocation.side_height_layers_per_step
            != 0
            or side_points_per_height < 4
            or side_points_per_height % 4 != 0
            or self.collocation.validation_height_layers < 2
            or self.collocation.validation_points_per_height < 4
            or self.collocation.validation_points_per_height % 4 != 0
        ):
            raise ValueError(
                "Active side-boundary physics requires positive height layers, "
                "at least four points per height, and training/validation points "
                "per height divisible by four"
            )


@dataclass(frozen=True, slots=True)
class LearningRateConfig(ConfigNode):
    start: float
    end: float
    iterations: int | Literal["auto"]

    def __post_init__(self) -> None:
        _positive(self.start, "learning-rate start")
        _positive(self.end, "learning-rate end")
        if isinstance(self.iterations, int) and self.iterations <= 0:
            raise ValueError("learning-rate iterations must be positive or 'auto'")


@dataclass(frozen=True, slots=True)
class TrainingConfig(ConfigNode):
    depth_sampling: DepthSamplingConfig
    physics: PhysicsConfig
    learning_rate: LearningRateConfig
    vector_regularization: VectorRegularizationConfig = field(
        default_factory=VectorRegularizationConfig
    )


@dataclass(frozen=True, slots=True)
class StokesWeightConfig(ConfigNode):
    i: float
    q: float
    u: float
    v: float

    def __post_init__(self) -> None:
        for name in ("i", "q", "u", "v"):
            _positive(
                getattr(self, name), f"Stokes {name.upper()} weight", allow_zero=True
            )
        if self.i == self.q == self.u == self.v == 0.0:
            raise ValueError("at least one Stokes weight must be positive")


@dataclass(frozen=True, slots=True)
class StokesSigmaConfig(ConfigNode):
    """Effective one-sigma errors in fixed atlas-continuum intensity units."""

    i: float
    q: float
    u: float
    v: float

    def __post_init__(self) -> None:
        for name in ("i", "q", "u", "v"):
            _positive(getattr(self, name), f"Stokes {name.upper()} sigma")


@dataclass(frozen=True, slots=True)
class LossConfig(ConfigNode):
    type: Literal["huber"]
    stokes_weights: StokesWeightConfig
    stokes_sigmas: StokesSigmaConfig
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.huber_delta, "huber_delta")


@dataclass(frozen=True, slots=True)
class RaySamplingConfig(ConfigNode):
    batch_size: int = 8192
    max_pixels: int = 65_536
    max_profile_samples: int = 4096

    def __post_init__(self) -> None:
        _positive(self.batch_size, "ray batch_size")
        _positive(self.max_pixels, "max_pixels")
        _positive(self.max_profile_samples, "max_profile_samples")


@dataclass(frozen=True, slots=True)
class SliceSamplingConfig(ConfigNode):
    batch_size: int = 8192
    layer_count: int = 6
    longitude_points: int = 256
    latitude_points: int = 256
    radial_points: int = 192

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "layer_count",
            "longitude_points",
            "latitude_points",
            "radial_points",
        ):
            _positive(getattr(self, name), f"slice {name}")
        if (
            self.layer_count < 3
            or min(
                self.longitude_points,
                self.latitude_points,
                self.radial_points,
            )
            < 2
        ):
            raise ValueError(
                "Slice layer counts must be at least three and spatial counts at least two"
            )


@dataclass(frozen=True, slots=True)
class MeridionalSliceConfig(ConfigNode):
    enabled: bool = False
    longitude_deg: float | None = None

    def __post_init__(self) -> None:
        if self.enabled and self.longitude_deg is None:
            raise ValueError("enabled meridional slice requires longitude_deg")
        if not self.enabled and self.longitude_deg is not None:
            raise ValueError("disabled meridional slice requires longitude_deg=null")
        if self.longitude_deg is not None:
            _finite(self.longitude_deg, "longitude_deg")


@dataclass(frozen=True, slots=True)
class VisualizationConfig(ConfigNode):
    enabled: bool = False
    every_n_validations: int = 5
    ray_sampling: RaySamplingConfig = field(default_factory=RaySamplingConfig)
    slice_sampling: SliceSamplingConfig = field(default_factory=SliceSamplingConfig)
    dpi: int = 180
    meridional_slice: MeridionalSliceConfig = field(
        default_factory=MeridionalSliceConfig
    )
    include_initial: bool = True

    def __post_init__(self) -> None:
        _positive(self.every_n_validations, "every_n_validations")
        if self.dpi < 50:
            raise ValueError("dpi must be at least 50")


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig(ConfigNode):
    validation_every_n_epochs: int = 1
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        _positive(self.validation_every_n_epochs, "validation_every_n_epochs")


@dataclass(frozen=True, slots=True)
class RuntimeConfig(ConfigNode):
    max_epochs: int
    log_every_n_steps: int = 50
    gradient_clip_norm: float | None = 0.1
    validation_check_interval_steps: int | None = None

    def __post_init__(self) -> None:
        _positive(self.max_epochs, "max_epochs")
        _positive(self.log_every_n_steps, "log_every_n_steps")
        if self.validation_check_interval_steps is not None:
            _positive(
                self.validation_check_interval_steps,
                "validation_check_interval_steps",
            )
        if self.gradient_clip_norm is not None:
            _positive(self.gradient_clip_norm, "gradient_clip_norm")


@dataclass(frozen=True, slots=True)
class LoggingConfig(ConfigNode):
    type: Literal["wandb", "disabled"]
    project: str | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.type == "wandb" and (
            self.project is None
            or not self.project.strip()
            or self.run_name is None
            or not self.run_name.strip()
        ):
            raise ValueError(
                "W&B logging requires non-empty project and run_name values"
            )
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("logging tags must not contain empty values")


@dataclass(frozen=True, slots=True)
class InversionConfig(ConfigNode):
    """Complete, versioned configuration for one LTE inversion."""

    schema_version: Literal[2]
    solver: SolverConfig
    resources: ResourceConfig
    observation: ObservationConfig
    atmosphere: AtmosphereConfig
    synthesis: SynthesisConfig
    instrument: InstrumentConfig
    training: TrainingConfig
    loss: LossConfig
    diagnostics: DiagnosticsConfig
    runtime: RuntimeConfig
    logging: LoggingConfig

    def __post_init__(self) -> None:
        pair = (self.observation.type, self.instrument.type)
        supported_pairs = {
            ("hinode_sp", "hinode_sp"),
            ("hmi_stokes", "hmi_filter_profiles"),
        }
        if pair not in supported_pairs:
            raise ValueError(
                "incompatible observation/instrument types: "
                f"{self.observation.type!r} and {self.instrument.type!r}"
            )
        temporal_equations = (
            self.training.physics.equations.momentum,
            self.training.physics.equations.induction,
            self.training.physics.equations.continuity,
            self.training.physics.equations.adiabatic_pressure,
        )
        if (
            any(
                equation.enabled and equation.weight > 0.0
                for equation in temporal_equations
            )
            and not self.atmosphere.geometry.time_dependent
        ):
            raise ValueError(
                "Temporal LTE physics requires atmosphere.geometry.time_dependent=true"
            )
        if not self.atmosphere.geometry.time_dependent and (
            self.instrument.optimize_line_of_sight_velocity_correction
            or self.instrument.line_of_sight_velocity_correction_m_per_s != 0.0
        ):
            raise ValueError(
                "Static inversions must leave the LOS velocity correction disabled"
            )
        induction = self.training.physics.equations.induction
        if induction.enabled and induction.weight > 0.0:
            if not self.instrument.optimize_line_of_sight_velocity_correction:
                raise ValueError(
                    "Dynamic induction inversions require "
                    "instrument.optimize_line_of_sight_velocity_correction=true"
                )
        geometry = self.atmosphere.geometry
        extrapolation_enabled = (
            geometry.line_formation_outer_height_megameter
            < geometry.outer_height_megameter
        )
        upper_equations = (
            self.training.physics.equations.upper_domain_microturbulence_prior,
            self.training.physics.equations.upper_domain_temperature_prior,
            self.training.physics.equations.upper_boundary_open_velocity,
            self.training.physics.equations.upper_boundary_current_free,
        )
        if not extrapolation_enabled and any(
            equation.enabled for equation in upper_equations
        ):
            raise ValueError(
                "Upper-domain and outer-boundary equations require an "
                "atmospheric extension above the line-formation domain"
            )


__all__ = [
    "AtmosphereConfig",
    "ConfigNode",
    "HMIInstrumentConfig",
    "HMIObservationConfig",
    "HinodeInstrumentConfig",
    "HinodeObservationConfig",
    "InstrumentConfig",
    "InversionConfig",
    "ObservationConfig",
]
