"""Scene-owned LTE spectral windows and numerical formal-transfer controls."""

from dataclasses import dataclass, field
import math
import re
from typing import Literal

from .schema import CoarseToFineConfig, ConfigNode


@dataclass(frozen=True, slots=True)
class LTEWindowConfig(ConfigNode):
    id: str
    line_ids: tuple[str, ...]
    wavelength_bounds_vacuum_angstrom: tuple[float, float]
    radiance_scale: float
    depth_count: int = 257
    coarse_to_fine: CoarseToFineConfig = field(default_factory=lambda: CoarseToFineConfig(enabled=False))
    jitter_fraction: float = 0.0
    wavelength_chunk_size: int = 16
    transfer_precision: Literal["float32", "float64"] = "float32"
    shell_height_bounds_megameter: tuple[float, float] | None = None
    lateral_halo_megameter: float = 0.0
    incoming_temperature_k: float | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or re.fullmatch(r"[a-z][a-z0-9_]*", self.id) is None:
            raise ValueError("LTE window id must match [a-z][a-z0-9_]*")
        if not self.line_ids or any(not isinstance(v, str) or not v.strip() for v in self.line_ids):
            raise ValueError("line_ids must contain nonempty identifiers")
        if len(set(self.line_ids)) != len(self.line_ids):
            raise ValueError("line_ids must be unique")
        bounds = self.wavelength_bounds_vacuum_angstrom
        if len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("Wavelength bounds must be finite, positive and increasing")
        if not math.isfinite(self.radiance_scale) or self.radiance_scale <= 0:
            raise ValueError("radiance_scale must be finite and positive")
        for name, minimum in (("depth_count", 2), ("wavelength_chunk_size", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.transfer_precision not in ("float32", "float64"):
            raise ValueError("transfer_precision must be float32 or float64")
        if (isinstance(self.jitter_fraction, bool)
                or not math.isfinite(self.jitter_fraction)
                or not 0 <= self.jitter_fraction <= 1):
            raise ValueError("jitter_fraction must be finite and between 0 and 1")
        if not math.isfinite(self.lateral_halo_megameter) or self.lateral_halo_megameter < 0:
            raise ValueError("lateral_halo_megameter must be finite and nonnegative")
        if self.incoming_temperature_k is not None and (
            not math.isfinite(self.incoming_temperature_k) or self.incoming_temperature_k <= 0
        ):
            raise ValueError("incoming_temperature_k must be finite and positive")
        if self.shell_height_bounds_megameter is not None:
            bounds = self.shell_height_bounds_megameter
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                raise ValueError("Shell bounds must be finite [inner, outer] heights")


@dataclass(frozen=True, slots=True)
class LTESynthesisConfig(ConfigNode):
    windows: tuple[LTEWindowConfig, ...]
    source_closure: Literal["thermalized_continuum"] = "thermalized_continuum"

    def __post_init__(self):
        ids = [window.id for window in self.windows]
        if len(set(ids)) != len(ids):
            raise ValueError("LTE window ids must be unique")


__all__ = ["LTEWindowConfig", "LTESynthesisConfig"]
