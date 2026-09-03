"""Typed configuration contract for the LTE inversion framework.

The schema deliberately contains data only.  Runtime modules may depend on this
package, while configuration loading never imports the solver, observations, or
instrument implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
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
    validation_batch_size: int
    validation_stride: int
    preparation_workers: int
    workers: int
    pin_memory: bool
    progress: bool

    def __post_init__(self) -> None:
        _positive(self.batch_size, "batch_size")
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
    stokes_reference_angle_deg: float

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
    validation_raster: int

    def __post_init__(self) -> None:
        if self.validation_raster < 0:
            raise ValueError("validation_raster must be non-negative")


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

    def __post_init__(self) -> None:
        _positive(self.height_input_scale_m, "height_input_scale_m")
        _finite(self.outer_height_megameter, "outer_height_megameter")
        _finite(self.inner_height_megameter, "inner_height_megameter")
        if self.outer_height_megameter <= self.inner_height_megameter:
            raise ValueError("outer shell height must exceed inner shell height")
        _positive(self.tangent_margin_m, "tangent_margin_m")


@dataclass(frozen=True, slots=True)
class LogBoundedParameterConfig(ConfigNode):
    log10_bounds: tuple[float, float]
    log_scale: float

    def __post_init__(self) -> None:
        _ordered_pair(self.log10_bounds, "log10_bounds")
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
    temperature: LogBoundedParameterConfig
    velocity: VelocityParameterConfig
    magnetic_field: MagneticParameterConfig
    microturbulence: LogBoundedParameterConfig
    gas_pressure: LogBoundedParameterConfig


@dataclass(frozen=True, slots=True)
class EncodingConfig(ConfigNode):
    type: Literal["fourier"]
    num_frequencies: tuple[int, ...]
    max_frequencies: tuple[int, ...]
    include_input: bool

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


@dataclass(frozen=True, slots=True)
class SynthesisConfig(ConfigNode):
    line_ids: tuple[str, ...]
    excluded_wavelength_windows_angstrom: tuple[tuple[float, float], ...]

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
    radial_velocity_correction_m_per_s: float

    def __post_init__(self) -> None:
        _finite(
            self.radial_velocity_correction_m_per_s,
            "radial_velocity_correction_m_per_s",
        )
        if abs(self.radial_velocity_correction_m_per_s) >= _SPEED_OF_LIGHT_M_PER_S:
            raise ValueError("radial_velocity_correction_m_per_s must be subluminal")


@dataclass(frozen=True, slots=True)
class HMIInstrumentConfig(ConfigNode):
    type: Literal["hmi_filter_profiles"]
    radial_velocity_correction_m_per_s: float

    def __post_init__(self) -> None:
        _finite(
            self.radial_velocity_correction_m_per_s,
            "radial_velocity_correction_m_per_s",
        )
        if abs(self.radial_velocity_correction_m_per_s) >= _SPEED_OF_LIGHT_M_PER_S:
            raise ValueError("radial_velocity_correction_m_per_s must be subluminal")


InstrumentConfig = HinodeInstrumentConfig | HMIInstrumentConfig


@dataclass(frozen=True, slots=True)
class CoarseToFineConfig(ConfigNode):
    enabled: bool
    fine_sample_count: int
    uniform_weight_floor: float

    def __post_init__(self) -> None:
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
    enabled: bool
    magnetic_weight: float
    velocity_weight: float
    decay_steps: int

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

    def __post_init__(self) -> None:
        for name in (
            "volume_points_per_step",
            "height_layers_per_step",
            "upper_boundary_points_per_step",
            "validation_height_layers",
            "validation_points_per_height",
        ):
            _positive(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class PhysicsNormalizationConfig(ConfigNode):
    length_m: float
    time_s: float

    def __post_init__(self) -> None:
        _positive(self.length_m, "length_m")
        _positive(self.time_s, "time_s")


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


@dataclass(frozen=True, slots=True)
class PhysicsEquationConfig(ConfigNode):
    hydrostatic_equilibrium: EquationConfig
    magnetohydrostatic_equilibrium: EquationConfig
    momentum: EquationConfig
    magnetic_divergence: EquationConfig
    induction: EquationConfig
    continuity: EquationConfig
    upper_boundary_gas_pressure_prior: EquationConfig


@dataclass(frozen=True, slots=True)
class PhysicsConfig(ConfigNode):
    vector_basis_matches_spatial_coordinates: bool
    collocation: CollocationConfig
    normalization: PhysicsNormalizationConfig
    equations: PhysicsEquationConfig

    def __post_init__(self) -> None:
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
            self.equations.induction,
            self.equations.continuity,
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
                self.equations.induction,
                self.equations.continuity,
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
    vector_regularization: VectorRegularizationConfig
    physics: PhysicsConfig
    learning_rate: LearningRateConfig


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
class PolarizationTransformConfig(ConfigNode):
    type: Literal["asinh"]
    q_alpha: float
    u_alpha: float
    v_alpha: float

    def __post_init__(self) -> None:
        _positive(self.q_alpha, "q_alpha")
        _positive(self.u_alpha, "u_alpha")
        _positive(self.v_alpha, "v_alpha")


@dataclass(frozen=True, slots=True)
class LossConfig(ConfigNode):
    type: Literal["mse"]
    stokes_weights: StokesWeightConfig
    polarization_transform: PolarizationTransformConfig


@dataclass(frozen=True, slots=True)
class RaySamplingConfig(ConfigNode):
    batch_size: int
    max_pixels: int
    max_profile_samples: int

    def __post_init__(self) -> None:
        _positive(self.batch_size, "ray batch_size")
        _positive(self.max_pixels, "max_pixels")
        _positive(self.max_profile_samples, "max_profile_samples")


@dataclass(frozen=True, slots=True)
class SliceSamplingConfig(ConfigNode):
    batch_size: int
    layer_count: int
    longitude_points: int
    latitude_points: int
    radial_points: int

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
            min(
                self.layer_count,
                self.longitude_points,
                self.latitude_points,
                self.radial_points,
            )
            < 2
        ):
            raise ValueError(
                "Slice layer, longitude, latitude, and radial counts must be at least two"
            )


@dataclass(frozen=True, slots=True)
class MeridionalSliceConfig(ConfigNode):
    enabled: bool
    longitude_deg: float | None

    def __post_init__(self) -> None:
        if self.enabled and self.longitude_deg is None:
            raise ValueError("enabled meridional slice requires longitude_deg")
        if not self.enabled and self.longitude_deg is not None:
            raise ValueError("disabled meridional slice requires longitude_deg=null")
        if self.longitude_deg is not None:
            _finite(self.longitude_deg, "longitude_deg")


@dataclass(frozen=True, slots=True)
class VisualizationConfig(ConfigNode):
    enabled: bool
    every_n_validations: int
    ray_sampling: RaySamplingConfig
    slice_sampling: SliceSamplingConfig
    dpi: int
    meridional_slice: MeridionalSliceConfig
    include_initial: bool

    def __post_init__(self) -> None:
        _positive(self.every_n_validations, "every_n_validations")
        if self.dpi < 50:
            raise ValueError("dpi must be at least 50")


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig(ConfigNode):
    validation_every_n_epochs: int
    visualization: VisualizationConfig

    def __post_init__(self) -> None:
        _positive(self.validation_every_n_epochs, "validation_every_n_epochs")


@dataclass(frozen=True, slots=True)
class RuntimeConfig(ConfigNode):
    max_epochs: int
    log_every_n_steps: int
    validation_check_interval_steps: int
    gradient_clip_norm: float | None

    def __post_init__(self) -> None:
        _positive(self.max_epochs, "max_epochs")
        _positive(self.log_every_n_steps, "log_every_n_steps")
        _positive(
            self.validation_check_interval_steps, "validation_check_interval_steps"
        )
        if self.gradient_clip_norm is not None:
            _positive(self.gradient_clip_norm, "gradient_clip_norm")


@dataclass(frozen=True, slots=True)
class LoggingConfig(ConfigNode):
    type: Literal["wandb", "disabled"]
    project: str
    tags: tuple[str, ...]
    run_name: str

    def __post_init__(self) -> None:
        if not self.project.strip():
            raise ValueError("logging project must not be empty")
        if not self.run_name.strip():
            raise ValueError("logging run_name must not be empty")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("logging tags must not contain empty values")


@dataclass(frozen=True, slots=True)
class InversionConfig(ConfigNode):
    """Complete, versioned configuration for one LTE inversion."""

    schema_version: Literal[1]
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
