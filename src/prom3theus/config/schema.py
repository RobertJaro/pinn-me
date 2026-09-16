"""Typed configuration contract for the LTE inversion framework.

The schema deliberately contains data only.  Runtime modules may depend on this
package, while configuration loading never imports the solver, observations, or
instrument implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import math
from numbers import Real
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
    workers: int = 2
    pin_memory: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        _positive(self.batch_size, "batch_size")
        if self.validation_batch_size is not None:
            _positive(self.validation_batch_size, "validation_batch_size")
        _positive(self.validation_stride, "validation_stride")
        _positive(self.preparation_workers, "preparation_workers")
        _positive(self.workers, "workers", allow_zero=True)
        if type(self.pin_memory) is not bool:
            raise TypeError("pin_memory must be boolean")


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
class AtmosphereGeometryConfig(ConfigNode):
    type: Literal["spherical_shell"]
    time_dependent: bool
    height_input_scale_m: float
    outer_height_megameter: float
    inner_height_megameter: float
    tangent_margin_m: float
    line_formation_outer_height_megameter: float | None = None
    uniform_spatial_scaling: bool = False

    def __post_init__(self) -> None:
        if self.line_formation_outer_height_megameter is None:
            object.__setattr__(
                self,
                "line_formation_outer_height_megameter",
                self.outer_height_megameter,
            )
        if type(self.uniform_spatial_scaling) is not bool:
            raise TypeError("uniform_spatial_scaling must be boolean")
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
    representation: Literal["direct", "vector_potential", "potential_delta"] = "direct"
    reference_height_megameter: float | None = None
    # Only meaningful for representation="potential_delta": the delta-field
    # contribution's weight alpha stays 0 for potential_delta_cool_steps, then
    # ramps linearly to 1 over the following potential_delta_ramp_steps. At
    # alpha=0 the field is exactly grad(psi), i.e. curl-free by construction.
    potential_delta_cool_steps: int = 0
    potential_delta_ramp_steps: int = 0

    def __post_init__(self) -> None:
        _positive(self.scale_gauss, "scale_gauss")
        if self.representation not in {
            "direct",
            "vector_potential",
            "potential_delta",
        }:
            raise ValueError(
                "magnetic representation must be 'direct', 'vector_potential', "
                "or 'potential_delta'"
            )
        if self.reference_height_megameter is not None:
            value = self.reference_height_megameter
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("magnetic reference_height_megameter must be numeric or null")
            _finite(value, "magnetic reference_height_megameter")
            if self.representation in ("vector_potential", "potential_delta"):
                raise ValueError(
                    "magnetic reference_height_megameter is incompatible with "
                    f"{self.representation} representation"
                )
        for name in ("potential_delta_cool_steps", "potential_delta_ramp_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"magnetic {name} must be a non-negative integer")


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
class RadialWeightingConfig(ConfigNode):
    """Initial SIREN bandwidth envelope from the solar surface outward."""

    near_sun_weight: float = 1.0
    outer_weight: float = 0.1
    exponent: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.near_sun_weight, "near_sun_weight")
        _positive(self.outer_weight, "outer_weight")
        _positive(self.exponent, "exponent")
        if self.outer_weight > self.near_sun_weight:
            raise ValueError("outer_weight must not exceed near_sun_weight")


@dataclass(frozen=True, slots=True)
class NetworkConfig(ConfigNode):
    type: Literal["mlp", "siren"]
    hidden_dimension: int
    hidden_layers: int
    activation: Literal["silu", "gelu", "tanh"] | None = None
    encoding: EncodingConfig | None = None
    first_omega_0: float = 30.0
    hidden_omega_0: float = 1.0
    radial_weighting: RadialWeightingConfig | None = field(
        default_factory=RadialWeightingConfig
    )

    def __post_init__(self) -> None:
        _positive(self.hidden_dimension, "hidden_dimension")
        _positive(self.hidden_layers, "hidden_layers")
        if self.type == "mlp":
            if self.activation is None or self.encoding is None:
                raise ValueError("MLP networks require activation and encoding")
        elif self.activation is not None or self.encoding is not None:
            raise ValueError("SIREN networks do not use activation or encoding")
        _positive(self.first_omega_0, "first_omega_0")
        _positive(self.hidden_omega_0, "hidden_omega_0")


@dataclass(frozen=True, slots=True)
class AtmosphereConfig(ConfigNode):
    geometry: AtmosphereGeometryConfig
    reference_atmosphere: Literal["falc_82"]
    parameters: AtmosphereParameterConfig
    network: NetworkConfig
    upper_atmosphere: UpperAtmosphereConfig | None = None

    def __post_init__(self) -> None:
        expected_dimensions = 4 if self.geometry.time_dependent else 3
        frequency_dimensions = (
            len(self.network.encoding.num_frequencies)
            if self.network.encoding is not None
            else expected_dimensions
        )
        if self.network.type == "mlp" and frequency_dimensions != expected_dimensions:
            raise ValueError(
                "Fourier frequency lists must contain one entry per atmosphere "
                f"coordinate ({expected_dimensions}); got {frequency_dimensions}"
            )
        geometry = self.geometry
        magnetic_height = self.parameters.magnetic_field.reference_height_megameter
        if magnetic_height is not None and not (
            geometry.inner_height_megameter <= magnetic_height <= geometry.outer_height_megameter
        ):
            raise ValueError("magnetic reference_height_megameter must lie within the atmosphere shell")
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
    reference_log_tau500_bounds: tuple[float, float] = (-5.0, 1.0)

    def __post_init__(self) -> None:
        _ordered_pair(self.reference_log_tau500_bounds, "reference_log_tau500_bounds")
        if (
            not self.reference_log_tau500_bounds[0]
            < 0
            < self.reference_log_tau500_bounds[1]
        ):
            raise ValueError("reference_log_tau500_bounds must straddle zero")
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")


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
    validation_side_boundary_points: int = 0
    validation_upper_height_layers: int = 0
    height_sampling_power: float = 1.0
    # "height_grouped" draws height_layers_per_step discrete heights, each
    # shared by volume_points_per_step / height_layers_per_step points (the
    # original behavior; height_layers_per_step must divide evenly and is
    # otherwise unused). "fully_random" draws volume_points_per_step fully
    # independent points across the whole volume instead -- no discrete
    # height layers at all -- and every per-height-group normalization scale
    # in magnetofluid.py becomes a per-point local scale as a result (a group
    # of one point). height_layers_per_step is still validated but ignored
    # in this mode. Required by hard_example_enabled below.
    volume_sampling: Literal["height_grouped", "fully_random"] = "height_grouped"
    residual_adaptive_enabled: bool = False
    residual_adaptive_candidate_multiplier: int = 2
    residual_adaptive_fraction: float = 0.5
    residual_adaptive_start_step: int = 500
    residual_adaptive_update_every_n_steps: int = 250
    # Persistent hard-example carry-forward: each step keeps the
    # highest-residual points from the previous step (jittered) instead of
    # redrawing every collocation point from scratch, so a thin high-error
    # layer/region cannot simply be missed by the sampler. Independent of
    # residual_adaptive_* above, which only re-ranks candidates within one
    # already-selected discrete height layer.
    hard_example_enabled: bool = False
    hard_example_count: int = 0
    hard_example_start_step: int = 0
    hard_example_jitter_length_m: float = 50_000.0
    hard_example_jitter_time_hours: float = 0.0

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
            "validation_side_boundary_points",
            "validation_upper_height_layers",
        ):
            _positive(getattr(self, name), name, allow_zero=True)
        _positive(self.height_sampling_power, "height_sampling_power")
        if type(self.residual_adaptive_enabled) is not bool:
            raise TypeError("residual_adaptive_enabled must be boolean")
        _positive(
            self.residual_adaptive_candidate_multiplier,
            "residual_adaptive_candidate_multiplier",
        )
        _positive(
            self.residual_adaptive_fraction,
            "residual_adaptive_fraction",
            allow_zero=True,
        )
        if self.residual_adaptive_fraction > 1.0:
            raise ValueError("residual_adaptive_fraction must not exceed one")
        _positive(self.residual_adaptive_start_step, "residual_adaptive_start_step", allow_zero=True)
        _positive(
            self.residual_adaptive_update_every_n_steps,
            "residual_adaptive_update_every_n_steps",
        )
        if type(self.hard_example_enabled) is not bool:
            raise TypeError("hard_example_enabled must be boolean")
        _positive(self.hard_example_count, "hard_example_count", allow_zero=True)
        if self.hard_example_enabled and self.hard_example_count < 1:
            raise ValueError("Enabled hard-example sampling requires hard_example_count > 0")
        if self.hard_example_count > self.volume_points_per_step:
            raise ValueError(
                "hard_example_count cannot exceed volume_points_per_step"
            )
        _positive(self.hard_example_start_step, "hard_example_start_step", allow_zero=True)
        _positive(
            self.hard_example_jitter_length_m, "hard_example_jitter_length_m", allow_zero=True
        )
        _positive(
            self.hard_example_jitter_time_hours,
            "hard_example_jitter_time_hours",
            allow_zero=True,
        )
        if self.hard_example_enabled and self.volume_sampling != "fully_random":
            raise ValueError(
                "hard_example_enabled requires volume_sampling: fully_random "
                "(carrying forward the globally hardest points only makes "
                "sense without fixed discrete height groups)"
            )
        if self.volume_sampling == "fully_random" and self.residual_adaptive_enabled:
            raise ValueError(
                "residual_adaptive_enabled assumes more than one point per "
                "height group to rank within, which volume_sampling: "
                "fully_random does not have; use hard_example_enabled instead"
            )


@dataclass(frozen=True, slots=True)
class PhysicsNormalizationConfig(ConfigNode):
    length_m: float
    time_s: float
    magnetic_field_floor_gauss: float = 1.0
    velocity_scale_m_per_s: float = 1_000.0
    magnetic_field_scale_gauss: float | None = None
    force_balance_pressure_scale_pa: float | None = None
    force_balance_pressure_floor_pa: float = 0.0
    detach_normalization_scale: bool = True

    def __post_init__(self) -> None:
        _positive(self.length_m, "length_m")
        _positive(self.time_s, "time_s")
        _positive(self.magnetic_field_floor_gauss, "magnetic_field_floor_gauss")
        _positive(self.velocity_scale_m_per_s, "velocity_scale_m_per_s")
        if self.magnetic_field_scale_gauss is not None:
            _positive(self.magnetic_field_scale_gauss, "magnetic_field_scale_gauss")
        if self.force_balance_pressure_scale_pa is not None:
            _positive(self.force_balance_pressure_scale_pa, "force_balance_pressure_scale_pa")
        _positive(
            self.force_balance_pressure_floor_pa,
            "force_balance_pressure_floor_pa",
            allow_zero=True,
        )
        if type(self.detach_normalization_scale) is not bool:
            raise TypeError("detach_normalization_scale must be boolean")


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
class CoronalEnergyConfig(EquationConfig):
    cooling_table: Path | None = None
    minimum_height_megameter: float = 3.0
    conductivity_w_m_k72: float = 1.0e-11
    magnetic_floor_gauss: float = 0.1
    heating_w_m3: float = 1.0e-5
    heating_scale_height_megameter: float = 30.0

    def __post_init__(self) -> None:
        EquationConfig.__post_init__(self)
        for name in (
            "minimum_height_megameter",
            "conductivity_w_m_k72",
            "magnetic_floor_gauss",
            "heating_scale_height_megameter",
        ):
            _positive(getattr(self, name), name)
        _positive(self.heating_w_m3, "heating_w_m3", allow_zero=True)
        if self.enabled and self.cooling_table is None:
            raise ValueError("coronal_energy requires a prepared cooling_table")

    def validate_domain(self, atmosphere) -> None:
        if not self.enabled:
            return
        if atmosphere.upper_atmosphere is None:
            raise ValueError("coronal_energy requires a coronal atmosphere extension")
        lower = max(
            atmosphere.geometry.line_formation_outer_height_megameter,
            atmosphere.upper_atmosphere.transition_region_top_megameter,
        )
        if (
            not lower
            < self.minimum_height_megameter
            < atmosphere.geometry.outer_height_megameter
        ):
            raise ValueError(
                "coronal_energy minimum height must lie above the transition region and below the outer shell"
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
    adiabatic_pressure: EquationConfig = field(default_factory=_disabled_equation)
    coronal_energy: CoronalEnergyConfig = field(
        default_factory=lambda: CoronalEnergyConfig(enabled=False, weight=0.0)
    )
    magnetic_force_free: EquationConfig = field(default_factory=_disabled_equation)
    magnetic_current_free: EquationConfig = field(default_factory=_disabled_equation)
    radial_magnetic_energy_gradient: EquationConfig = field(
        default_factory=_disabled_equation
    )
    radial_magnetic_field: EquationConfig = field(default_factory=_disabled_equation)
    upper_boundary_open_velocity: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_boundary_current_free: EquationConfig = field(
        default_factory=_disabled_equation
    )
    side_boundary_current_free: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_boundary_no_inflow: EquationConfig = field(default_factory=_disabled_equation)
    side_boundary_no_inflow: EquationConfig = field(default_factory=_disabled_equation)
    upper_domain_microturbulence_prior: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_domain_temperature_prior: EquationConfig = field(
        default_factory=_disabled_equation
    )
    upper_boundary_tangential_magnetic_neumann: EquationConfig = field(
        default_factory=_disabled_equation
    )
    side_boundary_open_velocity: EquationConfig = field(
        default_factory=_disabled_equation
    )
    side_boundary_tangential_magnetic_neumann: EquationConfig = field(
        default_factory=_disabled_equation
    )


@dataclass(frozen=True, slots=True)
class PotentialPhotosphereConfig(ConfigNode):
    """Temporary horizontal disambiguation prior on photospheric source cells."""

    enabled: bool = False
    batch_size: int = 256
    weight: float = 0.1
    minimum_horizontal_field_gauss: float = 50.0
    start_step: int = 500
    ramp_steps: int = 500
    end_step: int = 8000

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("potential photosphere enabled must be boolean")
        _positive(self.weight, "photosphere weight")
        if isinstance(self.minimum_horizontal_field_gauss, bool):
            raise TypeError("photosphere minimum horizontal field must be numeric")
        _positive(self.minimum_horizontal_field_gauss, "photosphere minimum horizontal field")
        for name in ("batch_size", "start_step", "ramp_steps", "end_step"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"potential photosphere {name} must be an integer")
            _positive(value, name, allow_zero=name in ("start_step", "ramp_steps"))
        if not self.start_step + self.ramp_steps < self.end_step:
            raise ValueError("Potential photosphere requires ramp end < end step")


@dataclass(frozen=True, slots=True)
class PotentialBoundaryConfig(ConfigNode):
    """Progressive full-vector targets from a spherical photospheric potential field."""

    enabled: bool = False
    geometry: Literal["spherical_neumann"] = "spherical_neumann"
    start_step: int = 1000
    ramp_steps: int = 2000
    update_every_n_steps: int = 500
    freeze_step: int = 10000
    blend: float = 0.5
    weight: float = 1.0
    source_height_megameter: float = 0.0
    grid_size: int = 64
    source_supersampling: int = 2
    top_grid_size: int = 16
    side_horizontal_points: int = 32
    side_height_points: int = 16
    jitter_fraction: float = 1.0
    seed: int = 0
    batch_size: int = 256
    field_floor_gauss: float = 1.0
    photosphere: PotentialPhotosphereConfig = field(default_factory=PotentialPhotosphereConfig)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("potential_boundary.enabled must be boolean")
        if self.photosphere.enabled and (not self.enabled or self.photosphere.start_step < self.start_step):
            raise ValueError("Potential photosphere requires enabled boundary references before its start")
        if self.geometry != "spherical_neumann":
            raise ValueError("potential_boundary.geometry must be spherical_neumann")
        for name in ("start_step", "ramp_steps", "freeze_step"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"potential_boundary.{name} must be an integer")
            _positive(getattr(self, name), name, allow_zero=True)
        for name in ("update_every_n_steps", "grid_size", "source_supersampling",
                     "top_grid_size", "side_horizontal_points", "side_height_points", "batch_size"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"potential_boundary.{name} must be an integer")
            _positive(getattr(self, name), name)
        if self.grid_size < 4:
            raise ValueError("Potential grid_size >= 4 is required")
        _finite(self.jitter_fraction, "jitter_fraction")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("Potential jitter_fraction must be between zero and one")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Potential seed must be a non-negative integer")
        if self.freeze_step < self.start_step:
            raise ValueError("Potential freeze_step must not precede source capture")
        _positive(self.blend, "blend")
        if self.blend > 1:
            raise ValueError("Potential blend must be <= 1")
        _positive(self.weight, "weight")
        _positive(self.field_floor_gauss, "field_floor_gauss")
        _finite(self.source_height_megameter, "source_height_megameter")
        if self.source_height_megameter != 0:
            raise ValueError("Potential source_height_megameter must be 0 (photosphere)")


@dataclass(frozen=True, slots=True)
class PhysicsConfig(ConfigNode):
    vector_basis_matches_spatial_coordinates: bool
    collocation: CollocationConfig
    normalization: PhysicsNormalizationConfig
    equations: PhysicsEquationConfig
    adiabatic_index: float = 5.0 / 3.0
    magnetic_current_free_steps: int = 0
    magnetic_current_free_final_factor: float = 0.0
    potential_boundary: PotentialBoundaryConfig = field(default_factory=PotentialBoundaryConfig)
    loss_start_step: int = 0
    loss_ramp_steps: int = 0
    robust_loss_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.potential_boundary.enabled:
            for name in ("upper_boundary_current_free", "side_boundary_current_free",
                         "upper_boundary_tangential_magnetic_neumann", "side_boundary_tangential_magnetic_neumann"):
                equation = getattr(self.equations, name)
                if equation.enabled and equation.weight > 0:
                    raise ValueError(f"potential_boundary replaces {name}")
        _finite(self.adiabatic_index, "adiabatic_index")
        if (
            self.equations.coronal_energy.enabled
            and self.equations.adiabatic_pressure.enabled
        ):
            raise ValueError(
                "coronal_energy and global adiabatic_pressure are mutually exclusive"
            )
        if self.adiabatic_index <= 1.0:
            raise ValueError("adiabatic_index must be greater than one")
        if self.magnetic_current_free_steps < 0:
            raise ValueError("magnetic_current_free_steps must be non-negative")
        _finite(self.magnetic_current_free_final_factor, "magnetic_current_free_final_factor")
        if not 0.0 <= self.magnetic_current_free_final_factor <= 1.0:
            raise ValueError("magnetic_current_free_final_factor must be between zero and one")
        if self.magnetic_current_free_final_factor > 0 and self.magnetic_current_free_steps == 0:
            raise ValueError("magnetic_current_free_final_factor requires positive magnetic_current_free_steps")
        _positive(self.loss_start_step, "physics.loss_start_step", allow_zero=True)
        _positive(self.loss_ramp_steps, "physics.loss_ramp_steps", allow_zero=True)
        _positive(
            self.robust_loss_delta,
            "physics.robust_loss_delta",
            allow_zero=True,
        )
        if self.equations.magnetic_current_free.enabled != (
            self.magnetic_current_free_steps > 0
        ):
            raise ValueError(
                "magnetic_current_free must be enabled exactly when "
                "magnetic_current_free_steps is positive"
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
            self.equations.magnetic_force_free,
            self.equations.magnetic_current_free,
            self.equations.radial_magnetic_energy_gradient,
            self.equations.radial_magnetic_field,
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
                self.equations.magnetic_force_free,
                self.equations.magnetic_current_free,
                self.equations.radial_magnetic_energy_gradient,
                self.equations.induction,
                self.equations.continuity,
                self.equations.adiabatic_pressure,
                self.equations.coronal_energy,
                self.equations.upper_boundary_current_free,
                self.equations.side_boundary_current_free,
                self.equations.upper_boundary_no_inflow,
                self.equations.side_boundary_no_inflow,
                self.equations.upper_boundary_open_velocity,
                self.equations.upper_boundary_tangential_magnetic_neumann,
                self.equations.side_boundary_open_velocity,
                self.equations.side_boundary_tangential_magnetic_neumann,
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
                self.equations.coronal_energy,
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
                self.equations.side_boundary_current_free,
                self.equations.side_boundary_no_inflow,
                self.equations.side_boundary_open_velocity,
                self.equations.side_boundary_tangential_magnetic_neumann,
            )
        )
        if side_boundary_active and (
            self.collocation.side_boundary_points_per_step < 4
            or self.collocation.side_boundary_points_per_step % 4 != 0
            or self.collocation.validation_side_boundary_points < 4
            or self.collocation.validation_side_boundary_points % 4 != 0
        ):
            raise ValueError(
                "Active side-boundary physics requires at least four flat training "
                "and validation samples divisible by four"
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
class LossConfig(ConfigNode):
    type: Literal["mse", "asinh_mse"]
    stokes_weights: StokesWeightConfig
    asinh_scale: float = 1.0e-3
    qu_warmup_steps: int = 0

    def __post_init__(self):
        _positive(self.asinh_scale, "Stokes asinh scale")
        if type(self.qu_warmup_steps) is not int or self.qu_warmup_steps < 0:
            raise ValueError("qu_warmup_steps must be a non-negative integer")


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


__all__ = [
    "PotentialBoundaryConfig",
    "AtmosphereConfig",
    "ConfigNode",
    "HMIInstrumentConfig",
    "HMIObservationConfig",
    "HinodeInstrumentConfig",
    "HinodeObservationConfig",
    "InstrumentConfig",
    "ObservationConfig",
    "RadialWeightingConfig",
]
