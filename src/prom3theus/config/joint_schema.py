"""Typed schema-v5 contract for multimodal inversions.

This module contains configuration data and validation only.  It deliberately
does not import observation adapters, resource loaders, forward operators, or
training code. Common scientific building blocks live in ``config.schema``.
This is the only supported run schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
import math
from pathlib import Path
import re
from typing import Literal

from .schema import (
    AtmosphereConfig,
    ConfigNode,
    CoarseToFineConfig,
    HMIObservationConfig,
    HinodeObservationConfig,
    InstrumentConfig,
    LossConfig,
    LoggingConfig,
    MeridionalSliceConfig,
    PhysicsConfig,
    RaySamplingConfig,
    SliceSamplingConfig,
)

from .synthesis_schema import LTESynthesisConfig, LTEWindowConfig


AIA_TEMPERATURE_RESPONSE_RESOURCE = "aia_euv_v1:aia_temperature_response"
AIA_CHANNELS_ANGSTROM = (171, 193, 211)
_SAFE_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


def _require_bool(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_safe_id(value: str, name: str) -> None:
    _require_nonempty(value, name)
    if _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must match [a-z][a-z0-9_]*")


def _require_finite_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_finite_positive(value: float, name: str) -> None:
    _require_finite_nonnegative(value, name)
    if value == 0.0:
        raise ValueError(f"{name} must be strictly positive")


def _require_positive_integer(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class JointSolverConfig(ConfigNode):
    """Filesystem owner for a joint run."""

    kind: Literal["joint"]
    output_directory: Path
    work_directory: Path


@dataclass(frozen=True, slots=True)
class TimeWindowConfig(ConfigNode):
    """Half-open UTC observation interval; unzoned dates are interpreted as UTC."""

    start: str
    end: str

    def __post_init__(self):
        from dateutil.parser import parse

        times = []
        for name in ("start", "end"):
            value = parse(getattr(self, name))
            value = (
                value.replace(tzinfo=timezone.utc)
                if value.tzinfo is None
                else value.astimezone(timezone.utc)
            )
            object.__setattr__(self, name, value.isoformat())
            times.append(value)
        if times[1] <= times[0]:
            raise ValueError("time_window.end must be later than time_window.start")


@dataclass(frozen=True, slots=True)
class SceneConfig(ConfigNode):
    """Select the observation stream defining the shared scene affine."""

    reference_stream: str
    time_window: TimeWindowConfig | None = None

    def __post_init__(self) -> None:
        _require_safe_id(self.reference_stream, "scene.reference_stream")


@dataclass(frozen=True, slots=True)
class AIASelectionConfig(ConfigNode):
    """Select one stable, complete exposure group for validation."""

    validation_exposure_group: str

    def __post_init__(self) -> None:
        _require_nonempty(
            self.validation_exposure_group,
            "validation_exposure_group",
        )


@dataclass(frozen=True, slots=True)
class ImageDataLoaderConfig(ConfigNode):
    """Loader-only options excluded from immutable image-store signatures."""

    batch_size: int
    validation_batch_size: int
    workers: int = 4
    pin_memory: bool = False

    def __post_init__(self) -> None:
        _require_positive_integer(self.batch_size, "batch_size")
        _require_positive_integer(
            self.validation_batch_size,
            "validation_batch_size",
        )
        if type(self.workers) is not int:
            raise TypeError("workers must be an integer")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        _require_bool(self.pin_memory, "pin_memory")


@dataclass(frozen=True, slots=True)
class AIAObservationConfig(ConfigNode):
    """Prepared, native-grid AIA image sequence used by schema version 5."""

    type: Literal["aia_euv"]
    directory: Path
    channels_angstrom: tuple[int, ...]
    selection: AIASelectionConfig
    loader: ImageDataLoaderConfig
    binning: int = 1

    def __post_init__(self) -> None:
        _require_positive_integer(self.binning, "binning")
        channels = self.channels_angstrom
        if any(type(channel) is not int for channel in channels):
            raise TypeError("channels_angstrom must contain integers")
        if channels != AIA_CHANNELS_ANGSTROM:
            raise ValueError(
                "The initial AIA integration requires channels exactly in the "
                f"order {list(AIA_CHANNELS_ANGSTROM)}"
            )


JointObservationConfig = (
    HinodeObservationConfig | HMIObservationConfig | AIAObservationConfig
)


@dataclass(frozen=True, slots=True)
class DataWeightScheduleConfig(ConfigNode):
    """Optional early optimization weight for one observation stream.

    ``initial_weight`` is held for ``initial_steps`` and then linearly changed
    to the parent data-term ``weight`` over ``ramp_steps``.  The schedule is an
    optimizer policy, not part of the scientific/data contract.
    """

    initial_weight: float
    initial_steps: int = 0
    ramp_steps: int = 0

    def __post_init__(self) -> None:
        _require_finite_nonnegative(self.initial_weight, "initial_weight")
        _require_positive_integer(
            self.initial_steps,
            "initial_steps",
            minimum=0,
        )
        _require_positive_integer(
            self.ramp_steps,
            "ramp_steps",
            minimum=0,
        )
        if self.initial_steps == 0 and self.ramp_steps == 0:
            raise ValueError(
                "a data weight schedule requires initial_steps or ramp_steps"
            )


@dataclass(frozen=True, slots=True)
class LTEStokesDataTermConfig(ConfigNode):
    """Instrument likelihood of differentiable formal LTE Stokes synthesis."""

    type: Literal["lte_stokes"]
    synthesis_window: str
    weight: float
    instrument: InstrumentConfig
    objective: LossConfig
    excluded_wavelength_windows_angstrom: tuple[tuple[float, float], ...] = ()
    weight_schedule: DataWeightScheduleConfig | None = None

    def __post_init__(self) -> None:
        _require_safe_id(self.synthesis_window, "synthesis_window")
        _require_finite_nonnegative(self.weight, "Stokes stream weight")
        for interval in self.excluded_wavelength_windows_angstrom:
            if len(interval) != 2 or not all(math.isfinite(v) for v in interval) or interval[0] >= interval[1]:
                raise ValueError("excluded_wavelength_windows_angstrom must contain finite increasing pairs")


@dataclass(frozen=True, slots=True)
class AIASynthesisConfig(ConfigNode):
    """Optically thin full-shell synthesis settings for an AIA stream."""

    response_resource: Literal["aia_euv_v1:aia_temperature_response"]
    ray_end: Literal["atmosphere_outer_shell"] = "atmosphere_outer_shell"
    ray_samples: int = 192
    height_sampling_power: float = 2.0
    coarse_to_fine: CoarseToFineConfig = field(
        default_factory=lambda: CoarseToFineConfig(enabled=False)
    )
    training_jitter: bool = True

    def __post_init__(self) -> None:
        _require_positive_integer(self.ray_samples, "ray_samples", minimum=2)
        _require_finite_positive(
            self.height_sampling_power,
            "height_sampling_power",
        )
        if self.height_sampling_power < 1.0:
            raise ValueError(
                "height_sampling_power must be at least one so the full-shell "
                "grid is not depleted near the lower atmosphere"
            )


@dataclass(frozen=True, slots=True)
class AIAChannelWeightConfig(ConfigNode):
    """One explicit scientific weight for one AIA passband."""

    channel_angstrom: int
    weight: float

    def __post_init__(self) -> None:
        _require_positive_integer(self.channel_angstrom, "channel_angstrom")
        _require_finite_nonnegative(self.weight, "AIA channel weight")


@dataclass(frozen=True, slots=True)
class AIACalibrationConfig(ConfigNode):
    """Identifiable common/relative log-gain prior for AIA channels."""

    enabled: bool = True
    absolute_prior_fraction: float = 0.25
    relative_prior_fraction: float = 0.15

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "calibration.enabled")
        _require_finite_positive(
            self.absolute_prior_fraction,
            "absolute_prior_fraction",
        )
        _require_finite_positive(
            self.relative_prior_fraction,
            "relative_prior_fraction",
        )


@dataclass(frozen=True, slots=True)
class AIAObjectiveConfig(ConfigNode):
    """Mean-squared image matching retaining finite negative observations."""

    type: Literal["asinh_mse"]
    channel_weights: tuple[AIAChannelWeightConfig, ...]
    calibration: AIACalibrationConfig = field(default_factory=AIACalibrationConfig)

    def __post_init__(self) -> None:
        channels = tuple(item.channel_angstrom for item in self.channel_weights)
        if not channels:
            raise ValueError("channel_weights must not be empty")
        if len(set(channels)) != len(channels):
            raise ValueError("channel_weights must contain each channel once")
        if not any(item.weight > 0.0 for item in self.channel_weights):
            raise ValueError("at least one AIA channel weight must be positive")


@dataclass(frozen=True, slots=True)
class AIAOpticallyThinDataTermConfig(ConfigNode):
    """AIA forward model, likelihood, and nuisance-prior configuration."""

    type: Literal["aia_optically_thin"]
    weight: float
    synthesis: AIASynthesisConfig
    objective: AIAObjectiveConfig
    weight_schedule: DataWeightScheduleConfig | None = None

    def __post_init__(self) -> None:
        _require_finite_nonnegative(self.weight, "AIA stream weight")


JointDataTermConfig = LTEStokesDataTermConfig | AIAOpticallyThinDataTermConfig


@dataclass(frozen=True, slots=True)
class ObservationStreamConfig(ConfigNode):
    """Keep a prepared observation and its matching data term inseparable."""

    id: str
    observation: JointObservationConfig
    data_term: JointDataTermConfig

    def __post_init__(self) -> None:
        _require_safe_id(self.id, "stream id")
        pair = (self.observation.type, self.data_term.type)
        from .registry import COMPATIBLE_PAIRS

        supported = COMPATIBLE_PAIRS
        if pair not in supported:
            raise ValueError(
                f"incompatible observation/data-term types: {pair[0]!r} and {pair[1]!r}"
            )

        if isinstance(self.data_term, LTEStokesDataTermConfig):
            instrument_pair = (
                self.observation.type,
                self.data_term.instrument.type,
            )
            supported_instruments = {
                ("hinode_sp", "hinode_sp"),
                ("hmi_stokes", "hmi_filter_profiles"),
                ("muram_stored", "hmi_filter_profiles"),
                ("muram_stored", "hinode_sp"),
            }
            if instrument_pair not in supported_instruments:
                raise ValueError(
                    "incompatible observation/instrument types: "
                    f"{instrument_pair[0]!r} and {instrument_pair[1]!r}"
                )

        if isinstance(self.observation, AIAObservationConfig):
            objective_channels = tuple(
                item.channel_angstrom
                for item in self.data_term.objective.channel_weights
            )
            if set(objective_channels) != set(self.observation.channels_angstrom):
                raise ValueError(
                    "AIA channel_weights must contain exactly every configured "
                    "observation channel"
                )


@dataclass(frozen=True, slots=True)
class STICTableSupportRegularizationConfig(ConfigNode):
    """Softly penalize unsaturated LTE states outside finite STiC support."""

    id: str
    type: Literal["stic_table_support"]
    temperature_weight: float
    gas_pressure_weight: float
    sample_count: int
    height_layers: int
    domain: Literal["line_formation"] = "line_formation"

    def __post_init__(self) -> None:
        _require_safe_id(self.id, "atmosphere regularization id")
        _require_finite_nonnegative(
            self.temperature_weight,
            "temperature_weight",
        )
        _require_finite_nonnegative(
            self.gas_pressure_weight,
            "gas_pressure_weight",
        )
        if self.temperature_weight == self.gas_pressure_weight == 0.0:
            raise ValueError(
                "STiC support regularization requires a positive component weight"
            )
        _validate_grouped_samples(self.sample_count, self.height_layers)


@dataclass(frozen=True, slots=True)
class VectorMagnitudeRegularizationConfig(ConfigNode):
    """Optional shared-atmosphere vector prior on explicit scene samples."""

    id: str
    type: Literal["vector_magnitude"]
    magnetic_weight: float
    velocity_weight: float
    sample_count: int
    height_layers: int
    domain: Literal["full_shell", "line_formation"] = "full_shell"

    def __post_init__(self) -> None:
        _require_safe_id(self.id, "atmosphere regularization id")
        _require_finite_nonnegative(self.magnetic_weight, "magnetic_weight")
        _require_finite_nonnegative(self.velocity_weight, "velocity_weight")
        if self.magnetic_weight == self.velocity_weight == 0.0:
            raise ValueError(
                "Vector magnitude regularization requires a positive component weight"
            )
        _validate_grouped_samples(self.sample_count, self.height_layers)


def _validate_grouped_samples(sample_count: int, height_layers: int) -> None:
    _require_positive_integer(sample_count, "sample_count")
    _require_positive_integer(height_layers, "height_layers")
    if height_layers > sample_count or sample_count % height_layers != 0:
        raise ValueError("sample_count must be exactly divisible by height_layers")


AtmosphereRegularizationConfig = (
    STICTableSupportRegularizationConfig | VectorMagnitudeRegularizationConfig
)


@dataclass(frozen=True, slots=True)
class JointVisualizationConfig(ConfigNode):
    """Bounded lifecycle-independent diagnostic rendering settings."""

    enabled: bool = True
    render_atmosphere: bool = True
    render_streams: bool = True
    ray_sampling: RaySamplingConfig = field(default_factory=RaySamplingConfig)
    slice_sampling: SliceSamplingConfig = field(default_factory=SliceSamplingConfig)
    meridional_slice: MeridionalSliceConfig = field(
        default_factory=MeridionalSliceConfig
    )
    contribution_ray_count: int = 16
    dpi: int = 180

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "visualization.enabled")
        _require_bool(self.render_atmosphere, "render_atmosphere")
        _require_bool(self.render_streams, "render_streams")
        _require_positive_integer(
            self.contribution_ray_count,
            "contribution_ray_count",
            minimum=0,
        )
        _require_positive_integer(self.dpi, "dpi", minimum=50)


@dataclass(frozen=True, slots=True)
class JointDiagnosticsConfig(ConfigNode):
    """Diagnostic owner for the joint runtime."""

    visualization: JointVisualizationConfig = field(
        default_factory=JointVisualizationConfig
    )


@dataclass(frozen=True, slots=True)
class JointTrainingConfig(ConfigNode):
    """Joint optimization controls. A limit of -1 disables that stopping limit.

    With both max_steps and max_epochs set to -1, training runs until stopped.
    """

    max_steps: int
    learning_rate: float
    final_learning_rate: float
    optimizer: Literal["adam", "muon", "soap"] = "adam"
    device: str = "auto"
    devices: int | Literal["auto"] = "auto"
    max_epochs: int = -1
    # Fixed-length epochs use fresh random samples and validate at each epoch end.
    steps_per_epoch: int | None = None
    learning_rate_decay_steps: int | None = None
    checkpoint_every_n_steps: int = 1000
    validation_every_n_steps: int = 1000
    validation_samples_per_stream: int = 256
    log_every_n_steps: int = 10
    gradient_clip_norm: float | None = 0.1
    reference_stream: str | None = None
    resume_from_checkpoint: Path | None = None
    migrate_data_order: bool = False
    gpu_prefetch: bool = True
    reader_workers: int = 4
    loader_block_bytes: int = 33554432
    loader_cache_bytes: int = 134217728
    loader_prefetch_batches: int = 2
    loader_max_batch_bytes: int = 67108864
    loader_seed: int = 0
    loss_balance: "LossBalanceConfig" = field(default_factory=lambda: LossBalanceConfig())


    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "muon", "soap"):
            raise ValueError("training.optimizer must be 'adam', 'muon', or 'soap'")
        for name in ("migrate_data_order", "gpu_prefetch"):
            _require_bool(getattr(self, name), name)
        if self.devices != "auto":
            _require_positive_integer(self.devices, "devices")
        for name in ("reader_workers", "loader_block_bytes", "loader_cache_bytes",
                     "loader_prefetch_batches", "loader_max_batch_bytes"):
            _require_positive_integer(getattr(self, name), name)
        if type(self.loader_seed) is not int or not 0 <= self.loader_seed < 2**63:
            raise ValueError("loader_seed must be an integer in [0, 2**63)")
        if self.loader_block_bytes > self.loader_cache_bytes:
            raise ValueError("loader_block_bytes must not exceed loader_cache_bytes")
        for name in ("max_steps", "max_epochs"):
            value = getattr(self, name)
            if value != -1:
                _require_positive_integer(value, name)
        if self.learning_rate_decay_steps is not None:
            _require_positive_integer(self.learning_rate_decay_steps, "learning_rate_decay_steps")
        if self.steps_per_epoch is not None:
            _require_positive_integer(self.steps_per_epoch, "steps_per_epoch")
        for name in (
            "checkpoint_every_n_steps",
            "validation_every_n_steps",
            "validation_samples_per_stream",
            "log_every_n_steps",
        ):
            _require_positive_integer(getattr(self, name), name)
        for name in ("learning_rate", "final_learning_rate"):
            _require_finite_positive(getattr(self, name), name)
        if self.gradient_clip_norm is not None:
            _require_finite_positive(self.gradient_clip_norm, "gradient_clip_norm")
        _require_nonempty(self.device, "training.device")


@dataclass(frozen=True, slots=True)
class LossBalanceConfig(ConfigNode):
    """Gradient-based multipliers for balancing data and shared objectives."""

    enabled: bool = False
    start_step: int = 250
    update_every_n_steps: int = 100
    ema_decay: float = 0.9
    adaptation_rate: float = 0.1
    target_gradient_fraction: float = 0.25
    min_multiplier: float = 0.01
    max_multiplier: float = 100.0
    probe_parameter_count: int = 4
    # Only these objectives adapt; an empty list keeps every weight fixed.
    included_objectives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "loss_balance.enabled")
        _require_positive_integer(self.start_step, "loss_balance.start_step", minimum=0)
        _require_positive_integer(
            self.update_every_n_steps,
            "loss_balance.update_every_n_steps",
        )
        _require_finite_nonnegative(self.ema_decay, "loss_balance.ema_decay")
        if self.ema_decay >= 1.0:
            raise ValueError("loss_balance.ema_decay must be smaller than one")
        _require_finite_positive(
            self.adaptation_rate,
            "loss_balance.adaptation_rate",
        )
        if self.adaptation_rate > 1.0:
            raise ValueError("loss_balance.adaptation_rate must not exceed one")
        _require_finite_positive(
            self.target_gradient_fraction,
            "loss_balance.target_gradient_fraction",
        )
        if self.target_gradient_fraction > 1.0:
            raise ValueError(
                "loss_balance.target_gradient_fraction must not exceed one"
            )
        _require_finite_positive(
            self.min_multiplier,
            "loss_balance.min_multiplier",
        )
        _require_finite_positive(
            self.max_multiplier,
            "loss_balance.max_multiplier",
        )
        if self.max_multiplier < self.min_multiplier:
            raise ValueError(
                "loss_balance.max_multiplier must be at least min_multiplier"
            )
        _require_positive_integer(
            self.probe_parameter_count,
            "loss_balance.probe_parameter_count",
        )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.included_objectives
        ):
            raise ValueError(
                "loss_balance.included_objectives must contain non-empty strings"
            )
        if len(set(self.included_objectives)) != len(self.included_objectives):
            raise ValueError("loss_balance.included_objectives must be unique")


@dataclass(frozen=True, slots=True)
class JointInversionConfig(ConfigNode):
    """Complete HMI/Hinode plus AIA configuration contract."""

    schema_version: Literal[5]
    solver: JointSolverConfig
    scene: SceneConfig
    atmosphere: AtmosphereConfig
    synthesis: LTESynthesisConfig
    streams: tuple[ObservationStreamConfig, ...]
    diagnostics: JointDiagnosticsConfig
    physics: PhysicsConfig | None = None
    atmosphere_regularization: tuple[AtmosphereRegularizationConfig, ...] = ()
    training: JointTrainingConfig | None = None
    logging: LoggingConfig = field(
        default_factory=lambda: LoggingConfig(type="disabled")
    )

    def __post_init__(self) -> None:
        synthesis_windows = {window.id: window for window in self.synthesis.windows}
        requested_fields = {
            stream.data_term.synthesis_window for stream in self.streams
            if isinstance(stream.data_term, LTEStokesDataTermConfig)
        }
        if requested_fields != set(synthesis_windows):
            raise ValueError(
                "synthesis.windows must match the synthesis_window references: "
                f"missing={sorted(requested_fields - set(synthesis_windows))}, "
                f"unused={sorted(set(synthesis_windows) - requested_fields)}"
            )
        for window in synthesis_windows.values():
            if window.shell_height_bounds_megameter is not None:
                lower, upper = window.shell_height_bounds_megameter
                geometry = self.atmosphere.geometry
                if lower < geometry.inner_height_megameter or upper > geometry.line_formation_outer_height_megameter:
                    raise ValueError("synthesis shell_height_bounds_megameter must lie inside the photospheric line-formation domain")
        if not self.streams:
            raise ValueError("schema version 5 requires at least one stream")
        stream_ids = tuple(stream.id for stream in self.streams)
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream ids must be unique")
        if self.scene.reference_stream not in stream_ids:
            raise ValueError(
                "scene.reference_stream must identify one configured stream"
            )

        if (
            self.training is not None
            and self.training.reference_stream is not None
            and self.training.reference_stream not in stream_ids
        ):
            raise ValueError(
                "training.reference_stream must identify a configured stream"
            )

        regularization_ids = tuple(
            regularization.id for regularization in self.atmosphere_regularization
        )
        if len(set(regularization_ids)) != len(regularization_ids):
            raise ValueError("atmosphere regularization ids must be unique")
        if self.physics is None:
            return
        if self.physics.potential_boundary.enabled:
            if "potential_boundary" in (*stream_ids, *regularization_ids):
                raise ValueError("potential_boundary is reserved for the progressive boundary objective")
            geometry = self.atmosphere.geometry
            potential = self.physics.potential_boundary
            if not geometry.inner_height_megameter <= potential.source_height_megameter <= geometry.line_formation_outer_height_megameter:
                raise ValueError("Potential source height must lie in the line-formation domain")


        gauge = self.physics.equations.vector_potential_gauge
        if (
            gauge.enabled
            and gauge.weight > 0.0
            and self.atmosphere.parameters.magnetic_field.representation
            != "vector_potential"
        ):
            raise ValueError(
                "vector_potential_gauge requires the vector_potential magnetic "
                "representation"
            )

        temporal_equations = (
            self.physics.equations.momentum,
            self.physics.equations.induction,
            self.physics.equations.continuity,
            self.physics.equations.adiabatic_pressure,
            self.physics.equations.coronal_energy,
            self.physics.equations.upper_boundary_no_inflow,
        )
        self.physics.equations.coronal_energy.validate_domain(self.atmosphere)
        if (
            any(
                equation.enabled and equation.weight > 0.0
                for equation in temporal_equations
            )
            and not self.atmosphere.geometry.time_dependent
        ):
            raise ValueError(
                "Temporal joint physics requires atmosphere.geometry.time_dependent=true"
            )

        geometry = self.atmosphere.geometry
        extrapolation_enabled = (
            geometry.line_formation_outer_height_megameter
            < geometry.outer_height_megameter
        )
        upper_equations = (
            self.physics.equations.upper_domain_microturbulence_prior,
            self.physics.equations.upper_domain_temperature_prior,
            self.physics.equations.coronal_energy,
            self.physics.equations.upper_boundary_open_velocity,
            self.physics.equations.upper_boundary_tangential_magnetic_neumann,
        )
        if not extrapolation_enabled and any(
            equation.enabled for equation in upper_equations
        ):
            raise ValueError(
                "Upper-domain and outer-boundary equations require an atmospheric "
                "extension above the line-formation domain"
            )


Configuration = JointInversionConfig


__all__ = [
    "JointTrainingConfig",
    "LossBalanceConfig",
    "AIA_CHANNELS_ANGSTROM",
    "AIA_TEMPERATURE_RESPONSE_RESOURCE",
    "AIACalibrationConfig",
    "AIAChannelWeightConfig",
    "AIAObjectiveConfig",
    "AIAObservationConfig",
    "AIAOpticallyThinDataTermConfig",
    "AIASelectionConfig",
    "AIASynthesisConfig",
    "AtmosphereRegularizationConfig",
    "Configuration",
    "DataWeightScheduleConfig",
    "ImageDataLoaderConfig",
    "JointDataTermConfig",
    "JointDiagnosticsConfig",
    "JointInversionConfig",
    "JointObservationConfig",
    "JointSolverConfig",
    "JointVisualizationConfig",
    "LTEStokesDataTermConfig",
    "LTESynthesisConfig",
    "LTEWindowConfig",
    "ObservationStreamConfig",
    "STICTableSupportRegularizationConfig",
    "SceneConfig",
    "TimeWindowConfig",
    "VectorMagnitudeRegularizationConfig",
]
