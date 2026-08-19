"""Deterministic global collocation sampling for spherical PINNs.

The sampler in this module is deliberately independent of a data-loader batch.
It samples a fixed global Sobol point set from an explicit space-time domain and
then partitions that set across distributed ranks.  All angular quantities are
in radians and all time/radius quantities are already in model-normalized units.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

import torch
import torch.distributed as dist


__all__ = [
    "LongitudeInterval",
    "SphericalCollocationDomain",
    "SphericalCollocationConfig",
    "SphericalCollocationBatch",
    "SmoothSphericalCollocationSampler",
]


_TWO_PI = 2.0 * math.pi


def _finite_float(name: str, value) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, not bool.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number.") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _interval(name: str, values, *, allow_zero_width: bool) -> tuple[float, float]:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise TypeError(f"{name} must contain exactly (minimum, maximum).")
    lower = _finite_float(f"{name}[0]", values[0])
    upper = _finite_float(f"{name}[1]", values[1])
    if upper < lower or (upper == lower and not allow_zero_width):
        relation = (
            "greater than or equal to" if allow_zero_width else "strictly greater than"
        )
        raise ValueError(f"{name}[1] must be {relation} {name}[0].")
    return lower, upper


@dataclass(frozen=True)
class LongitudeInterval:
    """A continuous, unwrapped longitude interval represented by center/width.

    Center/width avoids the usual ambiguity at the ``-pi``/``+pi`` seam.  For
    example, a two-degree interval across that seam can be represented by
    ``center=pi, width=radians(2)`` and has unwrapped bounds 179--181 degrees.
    """

    center: float
    width: float

    def __post_init__(self):
        center = _finite_float("longitude center", self.center)
        width = _finite_float("longitude width", self.width)
        if width <= 0 or width > _TWO_PI:
            raise ValueError("longitude width must be in the interval (0, 2*pi].")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "width", width)

    @property
    def bounds(self) -> tuple[float, float]:
        """Return continuous unwrapped ``(minimum, maximum)`` bounds."""
        half_width = self.width / 2.0
        return self.center - half_width, self.center + half_width

    @property
    def wrapped_center(self) -> float:
        """Return the center in the conventional ``[-pi, pi)`` interval."""
        return (self.center + math.pi) % _TWO_PI - math.pi

    @classmethod
    def from_unwrapped_bounds(cls, lower: float, upper: float) -> "LongitudeInterval":
        """Construct from explicit continuous bounds, which may cross ``pi``."""
        lower, upper = _interval(
            "unwrapped longitude bounds",
            (lower, upper),
            allow_zero_width=False,
        )
        return cls(center=(lower + upper) / 2.0, width=upper - lower)

    @classmethod
    def shortest_arc(cls, first: float, second: float) -> "LongitudeInterval":
        """Construct the shorter seam-safe arc connecting two wrapped angles.

        Antipodal endpoints are ambiguous and must instead be supplied as
        explicit unwrapped bounds or center/width.
        """
        first = _finite_float("first longitude", first)
        second = _finite_float("second longitude", second)
        delta = (second - first + math.pi) % _TWO_PI - math.pi
        if math.isclose(abs(delta), math.pi, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "Antipodal longitude endpoints do not define a unique shortest arc."
            )
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Longitude endpoints must define a nonzero arc.")
        return cls(center=first + delta / 2.0, width=abs(delta))


@dataclass(frozen=True)
class SphericalCollocationDomain:
    """Explicit global domain in normalized time and spherical coordinates."""

    normalized_time_range: tuple[float, float]
    latitude_range: tuple[float, float]
    longitude: LongitudeInterval
    normalized_radius_range: tuple[float, float]
    surface_radius: float = 1.0

    def __post_init__(self):
        time_range = _interval(
            "normalized_time_range",
            self.normalized_time_range,
            allow_zero_width=True,
        )
        latitude_range = _interval(
            "latitude_range", self.latitude_range, allow_zero_width=False
        )
        radius_range = _interval(
            "normalized_radius_range",
            self.normalized_radius_range,
            allow_zero_width=True,
        )
        if not isinstance(self.longitude, LongitudeInterval):
            raise TypeError("longitude must be a LongitudeInterval.")
        if latitude_range[0] < -math.pi / 2 or latitude_range[1] > math.pi / 2:
            raise ValueError("latitude_range must lie within [-pi/2, pi/2].")
        surface_radius = _finite_float("surface_radius", self.surface_radius)
        if surface_radius <= 0:
            raise ValueError("surface_radius must be positive.")
        if radius_range[0] <= 0:
            raise ValueError("normalized radii must be positive.")
        if radius_range[1] < surface_radius:
            raise ValueError(
                "normalized_radius_range must intersect or lie above surface_radius."
            )

        object.__setattr__(self, "normalized_time_range", time_range)
        object.__setattr__(self, "latitude_range", latitude_range)
        object.__setattr__(self, "normalized_radius_range", radius_range)
        object.__setattr__(self, "surface_radius", surface_radius)

    @classmethod
    def surface_shell(
        cls,
        *,
        normalized_time_range,
        latitude_range,
        longitude,
        surface_radius: float = 1.0,
        shell_height: float = 0.01,
    ) -> "SphericalCollocationDomain":
        """Construct a shell whose inner boundary is the model's solar surface."""
        surface_radius = _finite_float("surface_radius", surface_radius)
        shell_height = _finite_float("shell_height", shell_height)
        if shell_height < 0:
            raise ValueError("shell_height must be non-negative.")
        return cls(
            normalized_time_range=normalized_time_range,
            latitude_range=latitude_range,
            longitude=longitude,
            normalized_radius_range=(surface_radius, surface_radius + shell_height),
            surface_radius=surface_radius,
        )

    def canonical_metadata(self) -> dict:
        """Return a serialization-friendly description for logs/checkpoints."""
        return {
            "normalized_time_range": list(self.normalized_time_range),
            "latitude_range": list(self.latitude_range),
            "longitude_center": self.longitude.center,
            "longitude_width": self.longitude.width,
            "normalized_radius_range": list(self.normalized_radius_range),
            "surface_radius": self.surface_radius,
        }


@dataclass(frozen=True)
class SphericalCollocationConfig:
    """Sampling controls that do not alter the physical domain."""

    global_size: int = 4096
    seed: int = 0
    radial_measure: Literal["volume", "radius"] = "volume"

    def __post_init__(self):
        if isinstance(self.global_size, bool) or not isinstance(self.global_size, int):
            raise TypeError("global_size must be an integer.")
        if self.global_size <= 0:
            raise ValueError("global_size must be positive.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.radial_measure not in ("volume", "radius"):
            raise ValueError("radial_measure must be either 'volume' or 'radius'.")

    def canonical_metadata(self) -> dict:
        return {
            "global_size": self.global_size,
            "seed": self.seed,
            "radial_measure": self.radial_measure,
            "sequence": "sobol_scrambled_with_stateless_step_shift",
        }


@dataclass(frozen=True)
class SphericalCollocationBatch:
    """One rank's partition of a fixed global collocation point set."""

    coords: torch.Tensor
    normalized_time: torch.Tensor
    latitude: torch.Tensor
    longitude_unwrapped: torch.Tensor
    normalized_radius: torch.Tensor
    global_indices: torch.Tensor
    global_size: int
    rank: int
    world_size: int

    @property
    def local_size(self) -> int:
        return self.coords.shape[0]

    def ddp_scaled_mean(self, per_point: torch.Tensor) -> torch.Tensor:
        """Return a local loss whose DDP gradient average is the global mean.

        DDP averages gradients across ranks.  Multiplying each local sum by
        ``world_size / global_size`` therefore gives exactly the mean over the
        fixed global point set, including when partitions have unequal sizes.
        """
        if per_point.ndim == 0 or per_point.shape[0] != self.local_size:
            raise ValueError(
                f"per_point must have leading dimension {self.local_size}; "
                f"got shape {tuple(per_point.shape)}."
            )
        return per_point.sum(dim=0) * (self.world_size / self.global_size)


class SmoothSphericalCollocationSampler:
    """Stateless Sobol sampler over an explicit global spherical shell."""

    SOBOL_DIMENSION = 4  # normalized time, equal-area latitude, longitude, radius

    def __init__(
        self,
        domain: SphericalCollocationDomain,
        config: SphericalCollocationConfig | None = None,
    ):
        if not isinstance(domain, SphericalCollocationDomain):
            raise TypeError("domain must be a SphericalCollocationDomain.")
        if config is None:
            config = SphericalCollocationConfig()
        if not isinstance(config, SphericalCollocationConfig):
            raise TypeError("config must be a SphericalCollocationConfig.")
        self.domain = domain
        self.config = config

    def canonical_metadata(self) -> dict:
        return {
            "domain": self.domain.canonical_metadata(),
            "sampling": self.config.canonical_metadata(),
        }

    def _sobol_seed(self) -> int:
        payload = str(self.config.seed).encode("ascii")
        return (
            int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")
            & 0x7FFFFFFF
        )

    def _step_shift(self, global_step: int) -> torch.Tensor:
        """Return a deterministic Cranley-Patterson shift in the unit cube."""
        payload = f"{self.config.seed}:{global_step}".encode("ascii")
        digest = hashlib.blake2b(payload, digest_size=8 * self.SOBOL_DIMENSION).digest()
        values = [
            int.from_bytes(digest[index * 8 : (index + 1) * 8], "little") / 2**64
            for index in range(self.SOBOL_DIMENSION)
        ]
        return torch.tensor(values, dtype=torch.float64)

    @staticmethod
    def _distributed_context(rank, world_size) -> tuple[int, int]:
        if (rank is None) != (world_size is None):
            raise ValueError(
                "rank and world_size must either both be provided or both be omitted."
            )
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank, world_size = dist.get_rank(), dist.get_world_size()
            else:
                rank, world_size = 0, 1
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer.")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("world_size must be an integer.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}); got {rank}.")
        return rank, world_size

    def sample(
        self,
        global_step: int,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = True,
    ) -> SphericalCollocationBatch:
        """Sample and return this rank's deterministic global-point partition."""
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise TypeError("global_step must be an integer.")
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        if (
            not isinstance(dtype, torch.dtype)
            or not torch.empty((), dtype=dtype).is_floating_point()
        ):
            raise TypeError("dtype must be a floating-point torch dtype.")
        rank, world_size = self._distributed_context(rank, world_size)
        if self.config.global_size < world_size:
            raise ValueError(
                f"global_size ({self.config.global_size}) must be at least world_size ({world_size})."
            )

        engine = torch.quasirandom.SobolEngine(
            dimension=self.SOBOL_DIMENSION,
            scramble=True,
            seed=self._sobol_seed(),
        )
        # Generate the full CPU sequence on every rank, then slice it.  For the
        # intended O(10^3) point sets this is cheap and makes the global set
        # invariant to rank count and device-specific random implementations.
        unit = engine.draw(self.config.global_size, dtype=torch.float64)
        unit = torch.remainder(unit + self._step_shift(global_step), 1.0)
        start = self.config.global_size * rank // world_size
        stop = self.config.global_size * (rank + 1) // world_size
        unit = unit[start:stop]

        t_min, t_max = self.domain.normalized_time_range
        normalized_time = t_min + unit[:, 0] * (t_max - t_min)

        lat_min, lat_max = self.domain.latitude_range
        sin_lat_min, sin_lat_max = math.sin(lat_min), math.sin(lat_max)
        sin_latitude = sin_lat_min + unit[:, 1] * (sin_lat_max - sin_lat_min)
        latitude = torch.asin(sin_latitude.clamp(-1.0, 1.0))

        lon_min, lon_max = self.domain.longitude.bounds
        longitude = lon_min + unit[:, 2] * (lon_max - lon_min)

        r_min, r_max = self.domain.normalized_radius_range
        if self.config.radial_measure == "volume":
            radius = (r_min**3 + unit[:, 3] * (r_max**3 - r_min**3)).pow(1.0 / 3.0)
        else:
            radius = r_min + unit[:, 3] * (r_max - r_min)

        cos_latitude = torch.sqrt((1.0 - sin_latitude.square()).clamp_min(0.0))
        x = radius * cos_latitude * torch.cos(longitude)
        y = radius * cos_latitude * torch.sin(longitude)
        z = radius * sin_latitude
        coords = torch.stack([normalized_time, x, y, z], dim=-1)

        target = {"device": device, "dtype": dtype}
        coords = coords.to(**target).detach()
        coords.requires_grad_(requires_grad)
        normalized_time = normalized_time.to(**target)
        latitude = latitude.to(**target)
        longitude = longitude.to(**target)
        radius = radius.to(**target)
        global_indices = torch.arange(start, stop, dtype=torch.long, device=device)

        return SphericalCollocationBatch(
            coords=coords,
            normalized_time=normalized_time,
            latitude=latitude,
            longitude_unwrapped=longitude,
            normalized_radius=radius,
            global_indices=global_indices,
            global_size=self.config.global_size,
            rank=rank,
            world_size=world_size,
        )
