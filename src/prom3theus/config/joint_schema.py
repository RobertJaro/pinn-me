"""Typed schema-v3 contract for multimodal inversions and optional setup checks.

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
    DepthSamplingConfig,
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
    SynthesisConfig,
)


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
    workers: int = 2
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
    """Prepared, native-grid AIA image sequence used by schema version 3."""

    type: Literal["aia_euv"]
    directory: Path
    channels_angstrom: tuple[int, ...]
    selection: AIASelectionConfig
    loader: ImageDataLoaderConfig

    def __post_init__(self) -> None:
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
class DisambiguationConfig(ConfigNode):
    """Temporary smooth 180-degree phase continuation for raw Stokes data.

    A SIREN predicts the phase in units of pi, so the synthesis angle is
    ``pi * phase_network_output``.  During training, the phase rotation is
    active from step zero, while its binary ``sin(phi)^2`` preference is held
    at zero for ``cold_steps`` and warmed up afterward.  At the fixed handoff,
    and in evaluation mode, the rotation and its loss are removed from the
    forward graph.
    """

    enabled: bool = False
    cold_steps: int = 2_000
    warmup_steps: int = 5_000
    handoff_step: int = 15_000
    binary_weight: float = 0.01
    hidden_dimension: int = 64
    hidden_layers: int = 3
    first_omega_0: float = 1.0
    hidden_omega_0: float = 1.0

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "disambiguation.enabled")
        _require_positive_integer(
            self.cold_steps,
            "disambiguation.cold_steps",
            minimum=0,
        )
        _require_positive_integer(
            self.warmup_steps,
            "disambiguation.warmup_steps",
            minimum=0,
        )
        _require_positive_integer(
            self.handoff_step,
            "disambiguation.handoff_step",
            minimum=0,
        )
        _require_finite_nonnegative(
            self.binary_weight,
            "disambiguation.binary_weight",
        )
        _require_positive_integer(
            self.hidden_dimension,
            "disambiguation.hidden_dimension",
        )
        _require_positive_integer(
            self.hidden_layers,
            "disambiguation.hidden_layers",
        )
        _require_finite_positive(
            self.first_omega_0,
            "disambiguation.first_omega_0",
        )
        _require_finite_positive(
            self.hidden_omega_0,
            "disambiguation.hidden_omega_0",
        )
        if self.enabled:
            if self.warmup_steps < 1:
                raise ValueError(
                    "Enabled disambiguation requires warmup_steps > 0 so the "
                    "binary penalty starts at zero."
                )
            if self.binary_weight <= 0.0:
                raise ValueError(
                    "Enabled disambiguation requires a positive binary_weight."
                )
            if self.handoff_step <= self.cold_steps + self.warmup_steps:
                raise ValueError(
                    "disambiguation.handoff_step must be greater than cold_steps "
                    "plus warmup_steps."
                )


@dataclass(frozen=True, slots=True)
class LTEStokesDataTermConfig(ConfigNode):
    """Complete LTE Stokes forward/objective contract for a v3 stream."""

    type: Literal["lte_stokes"]
    weight: float
    synthesis: SynthesisConfig
    instrument: InstrumentConfig
    objective: LossConfig
    depth_sampling: DepthSamplingConfig
    weight_schedule: DataWeightScheduleConfig | None = None
    disambiguation: DisambiguationConfig = field(default_factory=DisambiguationConfig)

    def __post_init__(self) -> None:
        _require_finite_nonnegative(self.weight, "LTE Stokes stream weight")


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


@dataclass(frozen=True, slots=True)
class VectorPotentialRegularizationConfig(ConfigNode):
    """Gauge and smoothness regularization for a vector-potential atmosphere."""

    id: str
    type: Literal["vector_potential"]
    gauge_weight: float
    smoothness_weight: float
    sample_count: int
    height_layers: int
    domain: Literal["full_shell", "line_formation"] = "full_shell"
    smoothness_step_megameter: float = 0.25

    def __post_init__(self) -> None:
        _require_safe_id(self.id, "atmosphere regularization id")
        _require_finite_nonnegative(self.gauge_weight, "gauge_weight")
        _require_finite_nonnegative(self.smoothness_weight, "smoothness_weight")
        if self.gauge_weight == self.smoothness_weight == 0.0:
            raise ValueError(
                "Vector-potential regularization requires a positive component weight"
            )
        _validate_grouped_samples(self.sample_count, self.height_layers)
        _require_finite_positive(
            self.smoothness_step_megameter,
            "smoothness_step_megameter",
        )


def _validate_grouped_samples(sample_count: int, height_layers: int) -> None:
    _require_positive_integer(sample_count, "sample_count")
    _require_positive_integer(height_layers, "height_layers")
    if height_layers > sample_count or sample_count % height_layers != 0:
        raise ValueError("sample_count must be exactly divisible by height_layers")


AtmosphereRegularizationConfig = (
    STICTableSupportRegularizationConfig
    | VectorMagnitudeRegularizationConfig
    | VectorPotentialRegularizationConfig
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
class DryRunConfig(ConfigNode):
    """Evaluation controls which cannot start optimization."""

    device: str = "auto"
    evaluate_gradients: bool = True
    gradient_sample_count: int = 32
    max_samples_per_stream: int = 4096
    quadrature_samples: tuple[int, ...] = (96, 192, 384)
    report_filename: str = "dry_run.json"

    def __post_init__(self) -> None:
        _require_nonempty(self.device, "dry_run.device")
        _require_bool(self.evaluate_gradients, "evaluate_gradients")
        _require_positive_integer(
            self.gradient_sample_count,
            "gradient_sample_count",
        )
        _require_positive_integer(
            self.max_samples_per_stream,
            "max_samples_per_stream",
        )
        if self.gradient_sample_count > self.max_samples_per_stream:
            raise ValueError(
                "gradient_sample_count must not exceed max_samples_per_stream"
            )
        samples = self.quadrature_samples
        if (
            not samples
            or any(type(value) is not int or value < 2 for value in samples)
            or tuple(sorted(set(samples))) != samples
        ):
            raise ValueError(
                "quadrature_samples must be unique increasing integers of at least two"
            )
        _require_nonempty(self.report_filename, "report_filename")
        report = Path(self.report_filename)
        if (
            report.name != self.report_filename
            or report.suffix.lower() != ".json"
            or self.report_filename in {".", ".."}
        ):
            raise ValueError("report_filename must be a JSON basename")


@dataclass(frozen=True, slots=True)
class JointTrainingConfig(ConfigNode):
    """Joint optimization and checkpoint controls, including externally launched ranks."""

    max_steps: int
    learning_rate: float
    final_learning_rate: float
    device: str = "auto"
    max_epochs: int = -1
    checkpoint_every_n_steps: int = 1000
    validation_every_n_steps: int = 1000
    validation_samples_per_stream: int = 256
    log_every_n_steps: int = 10
    warmup_steps: int = 0
    gradient_clip_norm: float | None = 0.1
    reference_stream: str | None = None
    resume_from_checkpoint: Path | None = None
    migrate_data_order: bool = False
    gpu_prefetch: bool = True
    reader_workers: int = 2
    loader_block_bytes: int = 33554432
    loader_cache_bytes: int = 134217728
    loader_prefetch_batches: int = 2
    loader_max_batch_bytes: int = 67108864
    loader_seed: int = 0
    loss_balance: "LossBalanceConfig" = field(default_factory=lambda: LossBalanceConfig())


    def __post_init__(self) -> None:
        for name in ("migrate_data_order", "gpu_prefetch"):
            _require_bool(getattr(self, name), name)
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
        if self.max_steps == self.max_epochs == -1:
            raise ValueError("Set a positive max_steps or max_epochs.")
        for name in (
            "checkpoint_every_n_steps",
            "validation_every_n_steps",
            "validation_samples_per_stream",
            "log_every_n_steps",
        ):
            _require_positive_integer(getattr(self, name), name)
        _require_positive_integer(self.warmup_steps, "warmup_steps", minimum=0)
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
    excluded_objectives: tuple[str, ...] = ()

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
            for value in self.excluded_objectives
        ):
            raise ValueError(
                "loss_balance.excluded_objectives must contain non-empty strings"
            )
        if len(set(self.excluded_objectives)) != len(self.excluded_objectives):
            raise ValueError("loss_balance.excluded_objectives must be unique")


@dataclass(frozen=True, slots=True)
class JointInversionConfig(ConfigNode):
    """Complete HMI/Hinode plus AIA configuration contract."""

    schema_version: Literal[3]
    solver: JointSolverConfig
    scene: SceneConfig
    atmosphere: AtmosphereConfig
    physics: PhysicsConfig
    atmosphere_regularization: tuple[AtmosphereRegularizationConfig, ...]
    streams: tuple[ObservationStreamConfig, ...]
    diagnostics: JointDiagnosticsConfig
    dry_run: DryRunConfig = field(default_factory=DryRunConfig)
    training: JointTrainingConfig | None = None
    logging: LoggingConfig = field(
        default_factory=lambda: LoggingConfig(type="disabled")
    )

    def __post_init__(self) -> None:
        if not self.streams:
            raise ValueError("schema version 3 requires at least one stream")
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
        if self.physics.potential_boundary.enabled:
            if "potential_boundary" in (*stream_ids, *regularization_ids):
                raise ValueError("potential_boundary is reserved for the progressive boundary objective")
            geometry = self.atmosphere.geometry
            potential = self.physics.potential_boundary
            if not geometry.inner_height_megameter <= potential.source_height_megameter <= geometry.line_formation_outer_height_megameter:
                raise ValueError("Potential source height must lie in the line-formation domain")


        temporal_equations = (
            self.physics.equations.momentum,
            self.physics.equations.induction,
            self.physics.equations.continuity,
            self.physics.equations.adiabatic_pressure,
            self.physics.equations.coronal_energy,
            self.physics.equations.upper_boundary_current_free,
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

        stokes_terms = tuple(
            stream.data_term
            for stream in self.streams
            if isinstance(stream.data_term, LTEStokesDataTermConfig)
        )
        if not self.atmosphere.geometry.time_dependent and any(
            term.instrument.optimize_line_of_sight_velocity_correction
            or term.instrument.line_of_sight_velocity_correction_m_per_s != 0.0
            for term in stokes_terms
        ):
            raise ValueError(
                "Static joint inversions must leave every LOS velocity correction "
                "disabled"
            )
        induction = self.physics.equations.induction
        if (
            induction.enabled
            and induction.weight > 0.0
            and not any(
                term.instrument.optimize_line_of_sight_velocity_correction
                for term in stokes_terms
            )
        ):
            raise ValueError(
                "Dynamic induction inversions require an optimized Stokes "
                "instrument LOS velocity correction"
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

        configured_quadrature = set(self.dry_run.quadrature_samples)
        for stream in self.streams:
            if isinstance(stream.data_term, AIAOpticallyThinDataTermConfig):
                ray_samples = stream.data_term.synthesis.ray_samples
                if ray_samples not in configured_quadrature:
                    raise ValueError(
                        "dry_run.quadrature_samples must include every configured "
                        "AIA ray_samples value"
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
    "DisambiguationConfig",
    "DryRunConfig",
    "ImageDataLoaderConfig",
    "JointDataTermConfig",
    "JointDiagnosticsConfig",
    "JointInversionConfig",
    "JointObservationConfig",
    "JointSolverConfig",
    "JointVisualizationConfig",
    "LTEStokesDataTermConfig",
    "ObservationStreamConfig",
    "STICTableSupportRegularizationConfig",
    "SceneConfig",
    "TimeWindowConfig",
    "VectorMagnitudeRegularizationConfig",
    "VectorPotentialRegularizationConfig",
]
