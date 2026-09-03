"""Compact spherical-shell sampling shared by LTE training and validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


@dataclass(frozen=True)
class SphericalShellDomain:
    """Immutable physical bounds for random and deterministic collocation."""

    longitude_center_rad: float
    longitude_offset_bounds_rad: tuple[float, float]
    latitude_bounds_rad: tuple[float, float]
    time_bounds_hours: tuple[float, float]
    height_bounds_Mm: tuple[float, float]
    solar_radius_m: float

    def __post_init__(self):
        for name in (
            "longitude_offset_bounds_rad",
            "latitude_bounds_rad",
            "time_bounds_hours",
            "height_bounds_Mm",
        ):
            bounds = tuple(map(float, getattr(self, name)))
            if len(bounds) != 2 or not all(map(math.isfinite, bounds)):
                raise ValueError(f"{name} must contain two finite bounds.")
            if bounds[1] < bounds[0]:
                raise ValueError(f"{name} must be increasing.")
            object.__setattr__(self, name, bounds)
        if self.height_bounds_Mm[1] <= self.height_bounds_Mm[0]:
            raise ValueError("The spherical shell must have positive radial extent.")
        if not (
            -0.5 * math.pi
            <= self.latitude_bounds_rad[0]
            <= self.latitude_bounds_rad[1]
            <= 0.5 * math.pi
        ):
            raise ValueError("latitude_bounds_rad must lie within [-pi/2, pi/2].")
        if (
            self.longitude_offset_bounds_rad[1] - self.longitude_offset_bounds_rad[0]
            > 2.0 * math.pi
        ):
            raise ValueError("Longitude support cannot span more than one full turn.")
        if not math.isfinite(self.longitude_center_rad):
            raise ValueError("longitude_center_rad must be finite.")
        if not math.isfinite(self.solar_radius_m) or self.solar_radius_m <= 0:
            raise ValueError("solar_radius_m must be finite and positive.")
        if self.solar_radius_m + self.height_bounds_Mm[0] * 1.0e6 <= 0:
            raise ValueError("The inner shell radius must be positive.")

    @classmethod
    def from_mapping(cls, value: Mapping) -> "SphericalShellDomain":
        options = dict(value)
        expected = {
            "surface_longitude_center_rad",
            "surface_longitude_offset_rad",
            "surface_latitude_rad",
            "time_hours",
            "height_Mm",
            "solar_radius_m",
        }
        missing = expected - set(options)
        unknown = set(options) - expected
        if missing or unknown:
            raise TypeError(
                "Physics sampling_domain must contain exactly the current fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        return cls(
            longitude_center_rad=options["surface_longitude_center_rad"],
            longitude_offset_bounds_rad=options["surface_longitude_offset_rad"],
            latitude_bounds_rad=options["surface_latitude_rad"],
            time_bounds_hours=options["time_hours"],
            height_bounds_Mm=options["height_Mm"],
            solar_radius_m=options["solar_radius_m"],
        )

    @classmethod
    def from_observation_bounds(
        cls,
        value: Mapping,
        shell_height_bounds_Mm,
    ) -> "SphericalShellDomain":
        """Combine cached angular/time support with one atmosphere shell."""

        if shell_height_bounds_Mm is None:
            raise ValueError("Physics sampling requires shell_height_bounds_Mm.")
        shell_bounds = tuple(map(float, shell_height_bounds_Mm))
        if len(shell_bounds) != 2:
            raise ValueError(
                "shell_height_bounds_Mm must contain [outer_height, inner_height]."
            )
        outer, inner = shell_bounds
        if not outer > inner:
            raise ValueError("Physical shell outer height must exceed inner height.")
        return cls(
            longitude_center_rad=value["surface_longitude_center_rad"],
            longitude_offset_bounds_rad=value["surface_longitude_offset_rad"],
            latitude_bounds_rad=value["surface_latitude_rad"],
            time_bounds_hours=value["time_hours"],
            height_bounds_Mm=(inner, outer),
            solar_radius_m=value["solar_radius_m"],
        )

    def configuration(self) -> dict:
        """Return the minimal serializable state consumed by ``LTEModule``."""

        return {
            "surface_longitude_center_rad": self.longitude_center_rad,
            "surface_longitude_offset_rad": list(self.longitude_offset_bounds_rad),
            "surface_latitude_rad": list(self.latitude_bounds_rad),
            "time_hours": list(self.time_bounds_hours),
            "height_Mm": list(self.height_bounds_Mm),
            "solar_radius_m": self.solar_radius_m,
        }

    def metadata(self) -> dict:
        """Describe the grouped shell sampler for the inversion artifact."""

        angle_scale = 180.0 / math.pi
        domain = self.configuration()
        domain.update(
            {
                "surface_longitude_center_deg": (
                    self.longitude_center_rad * angle_scale
                ),
                "surface_longitude_offset_deg": [
                    value * angle_scale for value in self.longitude_offset_bounds_rad
                ],
                "surface_latitude_deg": [
                    value * angle_scale for value in self.latitude_bounds_rad
                ],
                "radius_Rsun": [
                    (self.solar_radius_m + height * 1.0e6) / self.solar_radius_m
                    for height in self.height_bounds_Mm
                ],
                "support": (
                    "initialized spherical shell with random Carrington longitude, "
                    "surface-area latitude, radius, and time samples"
                ),
            }
        )
        return {
            "collocation_distribution": (
                "uniform random geometric-height layers; uniform longitude and "
                "surface-area latitude samples within the initialized domain"
            ),
            "sampling_domain": domain,
            "spatial_sampling": {
                "type": "grouped_random_spherical_shell",
                "support": "initialized Carrington angular, time, and radial bounds",
                "position_formula": (
                    "Cartesian position from sampled radius/latitude/longitude"
                ),
                "coordinate_frame": "heliocentric Carrington Cartesian",
                "selection_rule": (
                    "each radial group shares one height and contains independent "
                    "random longitude, surface-area latitude, and time points"
                ),
            },
        }

    def _positions(
        self,
        heights_Mm: torch.Tensor,
        longitude_fraction: torch.Tensor,
        latitude_fraction: torch.Tensor,
        time_fraction: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        spatial_dtype = torch.float64 if dtype is None else dtype
        time_dtype = torch.float32 if dtype is None else dtype
        heights = torch.as_tensor(heights_Mm, dtype=spatial_dtype, device=device)
        longitude_fraction = torch.as_tensor(
            longitude_fraction, dtype=spatial_dtype, device=device
        )
        latitude_fraction = torch.as_tensor(
            latitude_fraction, dtype=spatial_dtype, device=device
        )
        time_fraction = torch.as_tensor(time_fraction, dtype=time_dtype, device=device)
        longitude = (
            self.longitude_center_rad
            + self.longitude_offset_bounds_rad[0]
            + (
                self.longitude_offset_bounds_rad[1]
                - self.longitude_offset_bounds_rad[0]
            )
            * longitude_fraction
        )
        sine_latitude_min = math.sin(self.latitude_bounds_rad[0])
        sine_latitude_max = math.sin(self.latitude_bounds_rad[1])
        sine_latitude = (
            sine_latitude_min
            + (sine_latitude_max - sine_latitude_min) * latitude_fraction
        )
        radius_m = self.solar_radius_m + heights[:, None] * 1.0e6
        cosine_latitude = torch.sqrt((1.0 - sine_latitude.square()).clamp_min(0.0))
        sine_longitude = torch.sin(longitude)
        cosine_longitude = torch.cos(longitude)
        position_m = radius_m[..., None] * torch.stack(
            (
                cosine_latitude * cosine_longitude,
                cosine_latitude * sine_longitude,
                sine_latitude,
            ),
            dim=-1,
        )
        time = (
            self.time_bounds_hours[0]
            + (self.time_bounds_hours[1] - self.time_bounds_hours[0]) * time_fraction
        )
        return {"position_m": position_m, "time_hours": time[..., None]}

    def random_grouped(
        self,
        height_count: int,
        points_per_height: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw grouped points with one shared radius per group."""

        height_count = int(height_count)
        points_per_height = int(points_per_height)
        if height_count < 1 or points_per_height < 1:
            raise ValueError("Grouped sample counts must be positive.")
        spatial_dtype = torch.float64 if dtype is None else dtype
        shape = (height_count, points_per_height)
        random_fraction_count = (2 if dtype is None else 3) * points_per_height
        spatial_random = torch.rand(
            (height_count, 1 + random_fraction_count),
            dtype=spatial_dtype,
            device=device,
        )
        heights = self.height_bounds_Mm[0] + spatial_random[:, 0] * (
            self.height_bounds_Mm[1] - self.height_bounds_Mm[0]
        )
        longitude_fraction = spatial_random[:, 1 : 1 + points_per_height]
        latitude_fraction = spatial_random[
            :, 1 + points_per_height : 1 + 2 * points_per_height
        ]
        if dtype is None:
            time_fraction = torch.rand(shape, dtype=torch.float32, device=device)
        else:
            time_fraction = spatial_random[:, 1 + 2 * points_per_height :]
        return self._positions(
            heights,
            longitude_fraction,
            latitude_fraction,
            time_fraction,
            device=device,
            dtype=dtype,
        )

    def random_top(
        self,
        point_count: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw independent angular/time points at the upper shell boundary."""

        point_count = int(point_count)
        if point_count < 1:
            raise ValueError("Boundary sample count must be positive.")
        spatial_dtype = torch.float64 if dtype is None else dtype
        shape = (1, point_count)
        random_fraction_count = (2 if dtype is None else 3) * point_count
        spatial_random = torch.rand(
            (1, random_fraction_count), dtype=spatial_dtype, device=device
        )
        time_fraction = (
            torch.rand(shape, dtype=torch.float32, device=device)
            if dtype is None
            else spatial_random[:, 2 * point_count :]
        )
        return self._positions(
            torch.full(
                (1,),
                self.height_bounds_Mm[1],
                dtype=spatial_dtype,
                device=device,
            ),
            spatial_random[:, :point_count],
            spatial_random[:, point_count : 2 * point_count],
            time_fraction,
            device=device,
            dtype=dtype,
        )

    def _deterministic_at_heights(
        self,
        heights_Mm: torch.Tensor,
        point_count: int,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> dict[str, torch.Tensor]:
        spatial_dtype = torch.float64 if dtype is None else dtype
        height_count = int(heights_Mm.numel())
        index = torch.arange(point_count, dtype=spatial_dtype, device=device) + 0.5
        longitude = (index / point_count).expand(height_count, -1)
        latitude = torch.frac(index * ((math.sqrt(5.0) - 1.0) / 2.0)).expand(
            height_count, -1
        )
        time = torch.frac(index * (math.sqrt(2.0) - 1.0))
        if dtype is None:
            time = time.to(torch.float32)
        return self._positions(
            heights_Mm,
            longitude,
            latitude,
            time.expand(height_count, -1),
            device=device,
            dtype=dtype,
        )

    def deterministic_grouped(
        self,
        height_count: int,
        points_per_height: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build a fixed low-discrepancy shell sample for comparable validation."""

        height_count = int(height_count)
        points_per_height = int(points_per_height)
        if height_count < 2 or points_per_height < 1:
            raise ValueError(
                "Validation requires at least two heights and one point per height."
            )
        spatial_dtype = torch.float64 if dtype is None else dtype
        heights = torch.linspace(
            self.height_bounds_Mm[0],
            self.height_bounds_Mm[1],
            height_count,
            dtype=spatial_dtype,
            device=device,
        )
        return self._deterministic_at_heights(
            heights,
            points_per_height,
            device=device,
            dtype=dtype,
        )

    def deterministic_top(
        self,
        point_count: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build a fixed upper-boundary sample for comparable validation."""

        point_count = int(point_count)
        if point_count < 1:
            raise ValueError("Boundary validation sample count must be positive.")
        spatial_dtype = torch.float64 if dtype is None else dtype
        heights = torch.full(
            (1,),
            self.height_bounds_Mm[1],
            dtype=spatial_dtype,
            device=device,
        )
        return self._deterministic_at_heights(
            heights,
            point_count,
            device=device,
            dtype=dtype,
        )


__all__ = ["SphericalShellDomain"]
