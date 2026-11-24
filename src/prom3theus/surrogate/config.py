"""Inputs for a single surrogate training run."""

import math
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Sequence

from .spectrum import read_window, resolve_window


@dataclass(frozen=True)
class Perturbations:
    """Atmospheric jitter for each fresh training point."""

    log_temperature: float = 0.0
    log_pressure: float = 0.0
    microturbulence_m_s: float = 0.0
    velocity_m_s: float = 0.0
    magnetic_field_fraction: float = 0.0  # Half-width of the uniform multiplicative field jitter.

    def __post_init__(self):
        if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0
               for x in vars(self).values()):
            raise ValueError("Jitter settings must be finite nonnegative numbers")

        if self.magnetic_field_fraction > 1:
            raise ValueError("magnetic_field_fraction must be between 0 and 1")


@dataclass(frozen=True)
class TrainingConfig:
    source_path: Path | None = None
    iteration: int | None = None
    output_dir: Path | None = None
    window_config: Path | None = None
    window: str | None = None
    instrument_config: Path | None = None
    vertical_axis: int | None = None
    component_order: Sequence[int] = (0, 1, 2)
    device: str = "auto"
    workers: int | None = None
    batch_size: int | None = None
    replay_fraction: float | None = None
    replay_buffer_size: int | None = None
    steps: int | None = None
    knots: int | None = None
    width: int | None = None
    hidden_layers: int | None = None
    architecture: str | None = None
    first_omega_0: float | None = None
    hidden_omega_0: float | None = None
    wavelength_count: int | None = None
    wavelength_jitter_fraction: float | None = None
    optimizer: str | None = None
    learning_rate: float | None = None
    final_learning_rate: float | None = None
    learning_rate_decay_steps: int | None = None
    seed: int = 42
    log_every: int = 50
    validate_every: int | None = None
    checkpoint_every: int = 1000
    ray_depths: int = 32
    log_tau: Sequence[float] = (-6, 2)
    perturbations: Perturbations | dict | None = None
    resume: bool = False

    @cached_property
    def _window_settings(self):
        return read_window(self.window_config, self.window) if self.window_config else None

    @cached_property
    def window_and_detector(self):
        return resolve_window(self._window_settings, self.window, self.instrument_config)

    def __post_init__(self):
        settings = self._window_settings or {}
        for name in ("source_path", "output_dir", "instrument_config"):
            if getattr(self, name) is None and name in settings:
                path = settings[name]
                if not isinstance(path, str) or not path.strip():
                    raise ValueError(f"{name} must be a nonempty path")
                object.__setattr__(self, name, self.window_config.parent / Path(path).expanduser())
        if self.iteration is None:
            object.__setattr__(self, "iteration", settings.get("iteration"))
        for name in ("source_path", "iteration", "output_dir"):
            if getattr(self, name) is None:
                raise ValueError(f"Supply {name} in the surrogate YAML or via --{name.replace('_', '-')}")
        if type(self.iteration) is not int:
            raise ValueError("iteration must be an integer")
        window, _ = self.window_and_detector
        if not isinstance(self.perturbations, Perturbations):
            jitter = window.get("jitter", {})
            if not isinstance(jitter, dict):
                raise ValueError("jitter must be a mapping of atmospheric jitter settings")
            # Older YAMLs included a probability gate. Only the widths control jitter now.
            jitter = {name: value for name, value in jitter.items() if name != "fraction"}
            widths = jitter | (self.perturbations or {})
            # Legacy zero floors are equivalent to omitting additive field jitter.
            for key in ("bx_gauss", "by_gauss", "bz_gauss", "magnetic_gauss"):
                if key in widths and widths.pop(key) != 0:
                    raise ValueError(f"{key} was replaced by multiplicative magnetic_field_fraction")
            try:
                settings = Perturbations(**widths)
            except TypeError as error:
                raise ValueError(f"Invalid jitter settings: {error}") from error
            object.__setattr__(self, "perturbations", settings)
        architecture = self.architecture
        if architecture is None:
            architecture = window.get("architecture", "silu")
        if architecture not in ("silu", "tanh", "siren"):
            raise ValueError("architecture must be silu, tanh or siren")
        object.__setattr__(self, "architecture", architecture)
        replay = self.replay_fraction
        if replay is None:
            replay = window.get("replay_fraction", 0.0)
        if type(replay) not in (float, int) or not 0 <= replay <= 1:
            raise ValueError("replay_fraction must be a number between 0 and 1")
        object.__setattr__(self, "replay_fraction", replay)

        wavelength_width = self.wavelength_jitter_fraction
        if wavelength_width is None:
            wavelength_width = window.get("wavelength_jitter_fraction", 0.0)
        if type(wavelength_width) not in (float, int) or not 0 <= wavelength_width < math.inf:
            raise ValueError("wavelength_jitter_fraction must be a finite nonnegative width")
        object.__setattr__(self, "wavelength_jitter_fraction", wavelength_width)
        for name, default, minimum in (
            ("width", 128, 1),
            ("hidden_layers", 3, 1),
            ("knots", 513, 2),
            ("wavelength_count", 513, 2),
            ("workers", 4, 0),
            ("batch_size", 1024, 1),
            ("replay_buffer_size", 4096, 0),
            ("steps", 20000, 1),
            ("validate_every", 5000, 1),
            ("learning_rate_decay_steps", 10000, 1),
        ):
            value = getattr(self, name)
            if value is None:
                value = window.get(name, default)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
            object.__setattr__(self, name, value)
        if self.replay_buffer_size < math.ceil(self.batch_size * self.replay_fraction):
            raise ValueError("replay_buffer_size must hold at least one step's replay points")
        for name in (
            "log_every",
            "checkpoint_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.ray_depths < 2:
            raise ValueError("ray_depths must be at least two")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.optimizer is None:
            object.__setattr__(self, "optimizer", window.get("optimizer", "soap"))
        for name, default in (
            ("learning_rate", 0.001),
            ("final_learning_rate", 0.0001),
            ("first_omega_0", 30.0),
            ("hidden_omega_0", 1.0),
        ):
            value = getattr(self, name)
            if value is None:
                value = window.get(name, default)
            if type(value) not in (float, int) or not 0 < value < math.inf:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if len(self.log_tau) != 2 or not self.log_tau[0] < self.log_tau[1]:
            raise ValueError("log_tau must specify an increasing interval")

    def arguments(self):
        """Keep the flat CLI argument representation in the run manifest."""
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values["config"] = values.pop("window_config")
        values.update(vars(values.pop("perturbations")))
        return values
