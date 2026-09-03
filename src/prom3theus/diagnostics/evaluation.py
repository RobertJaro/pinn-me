"""Evaluate LTE atmosphere diagnostics on explicit physical samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import torch

from prom3theus.core import (
    cartesian_to_spherical,
    project_cartesian_to_spherical,
    spherical_to_cartesian,
)

from .sampling import subsample_grid


THERMODYNAMIC_FIELDS = (
    "temperature",
    "density",
    "pressure",
    "microturbulence",
)
MAGNETIC_FIELDS = ("b_r", "b_theta", "b_phi")
VELOCITY_FIELDS = ("v_r", "v_theta", "v_phi")


def _option_mapping(value: Mapping | None, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _integer_option(options: dict, name: str, default: int) -> int:
    value = options.pop(name, default)
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


@dataclass(frozen=True, slots=True)
class AtmosphereSampling:
    """Validated sample limits for atmosphere diagnostics."""

    ray_evaluation_batch_size: int
    max_ray_pixels: int
    max_profile_samples: int
    slice_longitude_points: int
    slice_latitude_points: int
    slice_radial_points: int
    slice_layer_count: int
    slice_evaluation_batch_size: int
    meridional_slice_enabled: bool
    meridional_slice_longitude_deg: float | None

    @classmethod
    def from_options(
        cls,
        *,
        ray_sampling: Mapping | None = None,
        slice_sampling: Mapping | None = None,
        meridional_slice: Mapping | None = None,
    ) -> "AtmosphereSampling":
        """Parse strict callback options into one immutable sampling contract."""

        ray_config = _option_mapping(ray_sampling, "ray_sampling")
        ray_evaluation_batch_size = _integer_option(ray_config, "batch_size", 8192)
        max_ray_pixels = _integer_option(ray_config, "max_pixels", 65_536)
        max_profile_samples = _integer_option(ray_config, "max_profile_samples", 4_096)
        if ray_config:
            raise TypeError(f"Unknown ray-sampling options: {sorted(ray_config)}")

        slice_config = _option_mapping(slice_sampling, "slice_sampling")
        slice_longitude_points = _integer_option(slice_config, "longitude_points", 256)
        slice_latitude_points = _integer_option(slice_config, "latitude_points", 256)
        slice_radial_points = _integer_option(slice_config, "radial_points", 192)
        slice_layer_count = _integer_option(slice_config, "layer_count", 4)
        slice_evaluation_batch_size = _integer_option(slice_config, "batch_size", 8192)
        if slice_config:
            raise TypeError(f"Unknown slice-sampling options: {sorted(slice_config)}")

        meridional_config = _option_mapping(meridional_slice, "meridional_slice")
        meridional_slice_enabled = meridional_config.pop("enabled", False)
        if type(meridional_slice_enabled) is not bool:
            raise TypeError("meridional_slice.enabled must be a boolean.")
        raw_longitude_deg = meridional_config.pop("longitude_deg", None)
        if meridional_config:
            raise TypeError(
                f"Unknown meridional-slice options: {sorted(meridional_config)}"
            )
        if meridional_slice_enabled and raw_longitude_deg is None:
            raise ValueError("An enabled meridional slice requires longitude_deg.")
        if raw_longitude_deg is not None and (
            isinstance(raw_longitude_deg, bool)
            or not isinstance(raw_longitude_deg, (int, float))
        ):
            raise TypeError("meridional_slice.longitude_deg must be a number or null.")
        meridional_slice_longitude_deg = (
            None if raw_longitude_deg is None else float(raw_longitude_deg)
        )
        if meridional_slice_longitude_deg is not None and not math.isfinite(
            meridional_slice_longitude_deg
        ):
            raise ValueError("Meridional-slice longitude_deg must be finite.")

        positive_counts = {
            "ray_sampling.batch_size": ray_evaluation_batch_size,
            "ray_sampling.max_pixels": max_ray_pixels,
            "ray_sampling.max_profile_samples": max_profile_samples,
            "slice_sampling.longitude_points": slice_longitude_points,
            "slice_sampling.latitude_points": slice_latitude_points,
            "slice_sampling.radial_points": slice_radial_points,
            "slice_sampling.layer_count": slice_layer_count,
            "slice_sampling.batch_size": slice_evaluation_batch_size,
        }
        invalid = [name for name, value in positive_counts.items() if value < 1]
        if invalid:
            raise ValueError(
                f"Visualization sample counts must be positive: {invalid}."
            )
        if min(slice_longitude_points, slice_latitude_points) < 2:
            raise ValueError(
                "Physical shell slices require at least two angular points."
            )
        if slice_radial_points < 2:
            raise ValueError(
                "A physical radial slice requires at least two radial points."
            )
        if slice_layer_count < 2:
            raise ValueError("Physical shell plots require at least two radial layers.")
        return cls(
            ray_evaluation_batch_size=ray_evaluation_batch_size,
            max_ray_pixels=max_ray_pixels,
            max_profile_samples=max_profile_samples,
            slice_longitude_points=slice_longitude_points,
            slice_latitude_points=slice_latitude_points,
            slice_radial_points=slice_radial_points,
            slice_layer_count=slice_layer_count,
            slice_evaluation_batch_size=slice_evaluation_batch_size,
            meridional_slice_enabled=meridional_slice_enabled,
            meridional_slice_longitude_deg=meridional_slice_longitude_deg,
        )


class AtmosphereEvaluator:
    """Evaluate atmosphere fields independently of plotting and trainer hooks."""

    def __init__(self, sampling: AtmosphereSampling):
        self.sampling = sampling
        for name in AtmosphereSampling.__dataclass_fields__:
            setattr(self, name, getattr(sampling, name))

    def evaluate_ray_optical_depth(
        self,
        pl_module,
        raster,
        *,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
    ) -> dict:
        if (rows is None) != (columns is None):
            raise ValueError("rows and columns must be supplied together.")
        model = pl_module.atmosphere_model
        parameter = next(model.parameters())
        height, width = raster.spatial_shape
        if rows is None or columns is None:
            rows, columns = subsample_grid(height, width, self.max_ray_pixels)
        coords = raster.coordinates[rows][:, columns]
        ray_direction = raster.ray_direction[rows][:, columns]
        surface_position_m = raster.surface_position_m[rows][:, columns]
        valid = raster.valid_mask[rows][:, columns]
        flat_coords = coords[valid].to(device=parameter.device, dtype=parameter.dtype)
        flat_ray_direction = ray_direction[valid].to(device=parameter.device)
        if flat_coords.numel() == 0:
            raise ValueError(
                "No valid observation pixels remain for atmosphere visualization."
            )

        was_training = model.training
        model.eval()
        chunks = {"tau500_ray": [], "geometric_height": []}
        traced_spherical = []
        sampled_depth_grid = None
        try:
            with torch.no_grad():
                for start in range(
                    0, flat_coords.shape[0], self.ray_evaluation_batch_size
                ):
                    batch_coords = flat_coords[
                        start : start + self.ray_evaluation_batch_size
                    ]
                    atmosphere, ray_trace = (
                        pl_module.forward_composition.trace_atmosphere(
                            batch_coords,
                            flat_ray_direction[
                                start : start + self.ray_evaluation_batch_size
                            ],
                            pl_module.sample_depth_grid(randomize=False),
                        )
                    )
                    sampled_depth_grid = atmosphere.log_tau500.detach().float().cpu()
                    gas_pressure = atmosphere.gas_pressure
                    if gas_pressure is None:
                        raise RuntimeError(
                            "LTE atmosphere visualization requires predicted gas pressure."
                        )
                    alpha500 = pl_module.synthesizer.continuum_opacity.volume_extinction_at_5000(
                        atmosphere.temperature, gas_pressure
                    )
                    distance_interval_m = (
                        ray_trace.distance_m[..., 1:] - ray_trace.distance_m[..., :-1]
                    )
                    tau_increment = (
                        0.5
                        * (alpha500[..., :-1] + alpha500[..., 1:])
                        * distance_interval_m
                    )
                    tau500_ray = torch.cat(
                        (
                            torch.zeros_like(alpha500[..., :1]),
                            torch.cumsum(tau_increment, dim=-1),
                        ),
                        dim=-1,
                    )
                    spherical = cartesian_to_spherical(ray_trace.position_m, torch)
                    scene_center_spherical = cartesian_to_spherical(
                        model.scene_basis[2].to(spherical), torch
                    )
                    longitude_reference = scene_center_spherical[2]
                    longitude = longitude_reference + torch.atan2(
                        torch.sin(spherical[..., 2] - longitude_reference),
                        torch.cos(spherical[..., 2] - longitude_reference),
                    )
                    traced_spherical.append(
                        torch.stack(
                            (spherical[..., 0], spherical[..., 1], longitude), dim=-1
                        )
                        .detach()
                        .float()
                        .cpu()
                    )
                    chunks["tau500_ray"].append(tau500_ray.detach().float().cpu())
                    chunks["geometric_height"].append(
                        (ray_trace.geometric_height_m / 1.0e6).detach().float().cpu()
                    )
        finally:
            model.train(was_training)

        fields = {name: torch.cat(values).numpy() for name, values in chunks.items()}
        depth = next(iter(fields.values())).shape[-1]
        map_fields = {}
        for name, values in fields.items():
            image = np.full((*valid.shape, depth), np.nan, dtype=np.float32)
            image[valid.numpy()] = values
            map_fields[name] = image
        base_spherical = cartesian_to_spherical(surface_position_m, torch)
        scene_center_spherical = cartesian_to_spherical(
            model.scene_basis[2].to(base_spherical), torch
        )
        longitude_reference = scene_center_spherical[2]
        base_longitude = longitude_reference + torch.atan2(
            torch.sin(base_spherical[..., 2] - longitude_reference),
            torch.cos(base_spherical[..., 2] - longitude_reference),
        )
        map_longitude_deg = np.broadcast_to(
            torch.rad2deg(base_longitude).numpy()[..., None], (*valid.shape, depth)
        ).copy()
        map_latitude_deg = np.broadcast_to(
            (90.0 - torch.rad2deg(base_spherical[..., 1])).numpy()[..., None],
            (*valid.shape, depth),
        ).copy()
        if traced_spherical:
            spherical = torch.cat(traced_spherical).numpy()
            map_longitude_deg[valid.numpy()] = np.rad2deg(spherical[..., 2])
            map_latitude_deg[valid.numpy()] = 90.0 - np.rad2deg(spherical[..., 1])
        shell_height_levels_m = np.nanmedian(fields["geometric_height"], axis=0) * 1.0e6
        if sampled_depth_grid is None:
            raise RuntimeError("Ray optical-depth sampling produced no depth grid.")
        return {
            "log_tau500": sampled_depth_grid.numpy(),
            "solar_radius_m": float(model.solar_radius_m.detach().cpu()),
            "shell_height_levels_m": shell_height_levels_m,
            "profile_fields": fields,
            "map_fields": map_fields,
            "map_longitude_deg": map_longitude_deg,
            "map_latitude_deg": map_latitude_deg,
        }

    @staticmethod
    def _observed_spherical_bounds(model, raster) -> tuple[float, float, float, float]:
        """Return unwrapped longitude and latitude bounds of valid surface points."""

        surface = raster.surface_position_m[raster.valid_mask].detach().float().cpu()
        if surface.shape[0] < 2:
            raise ValueError(
                "Physical slices require at least two valid surface points."
            )
        spherical = cartesian_to_spherical(surface, torch)
        center = cartesian_to_spherical(
            model.scene_basis[2].detach().float().cpu(), torch
        )
        longitude_center = center[2]
        longitude = longitude_center + torch.atan2(
            torch.sin(spherical[:, 2] - longitude_center),
            torch.cos(spherical[:, 2] - longitude_center),
        )
        latitude = 0.5 * math.pi - spherical[:, 1]
        bounds = (
            float(longitude.min()),
            float(longitude.max()),
            float(latitude.min()),
            float(latitude.max()),
        )
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("Observed spherical bounds must be finite.")
        if not bounds[1] > bounds[0] or not bounds[3] > bounds[2]:
            raise ValueError(
                "Observed spherical bounds must span longitude and latitude."
            )
        return bounds

    def _evaluate_physical_positions(
        self, pl_module, position_m: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """Evaluate plot fields at explicitly supplied Carrington positions."""

        model = pl_module.atmosphere_model
        parameter = next(model.parameters())
        flat_position = position_m.reshape(-1, 3).to(
            device=parameter.device, dtype=parameter.dtype
        )
        chunks = {
            name: []
            for name in (
                *THERMODYNAMIC_FIELDS,
                *MAGNETIC_FIELDS,
                *VELOCITY_FIELDS,
            )
        }
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(
                    0, flat_position.shape[0], self.slice_evaluation_batch_size
                ):
                    position = flat_position[
                        start : start + self.slice_evaluation_batch_size
                    ]
                    atmosphere = model.evaluate_position_points(position)
                    pressure = atmosphere["gas_pressure"]
                    density = (
                        pl_module.synthesizer.continuum_opacity.reference_mass_density(
                            atmosphere["temperature"], pressure
                        )
                    )
                    spherical = cartesian_to_spherical(position, torch)
                    magnetic_spherical = project_cartesian_to_spherical(
                        atmosphere["magnetic_field"], spherical, torch
                    )
                    velocity_spherical = (
                        project_cartesian_to_spherical(
                            atmosphere["velocity_field"], spherical, torch
                        )
                        / 1_000.0
                    )
                    values = {
                        "temperature": atmosphere["temperature"],
                        "density": density,
                        "pressure": pressure,
                        "microturbulence": atmosphere["microturbulence"] / 1_000.0,
                        "b_r": magnetic_spherical[..., 0],
                        "b_theta": magnetic_spherical[..., 1],
                        "b_phi": magnetic_spherical[..., 2],
                        "v_r": velocity_spherical[..., 0],
                        "v_theta": velocity_spherical[..., 1],
                        "v_phi": velocity_spherical[..., 2],
                    }
                    for name, value in values.items():
                        chunks[name].append(value.detach().float().cpu())
        finally:
            model.train(was_training)
        leading_shape = position_m.shape[:-1]
        return {
            name: torch.cat(values).numpy().reshape(leading_shape)
            for name, values in chunks.items()
        }

    def evaluate_shell_layers(
        self, pl_module, raster, shell_height_levels_m: np.ndarray
    ) -> dict:
        """Evaluate only requested longitude-latitude layers in physical space."""

        shell_height_levels_m = np.asarray(shell_height_levels_m)
        if (
            shell_height_levels_m.ndim != 1
            or shell_height_levels_m.size < 1
            or not np.issubdtype(shell_height_levels_m.dtype, np.number)
            or not np.isfinite(shell_height_levels_m).all()
        ):
            raise ValueError("shell_height_levels_m must be a non-empty finite vector.")
        model = pl_module.atmosphere_model
        lon_min, lon_max, lat_min, lat_max = self._observed_spherical_bounds(
            model, raster
        )
        longitude = torch.linspace(lon_min, lon_max, self.slice_longitude_points)
        latitude = torch.linspace(lat_min, lat_max, self.slice_latitude_points)
        latitude_grid, longitude_grid = torch.meshgrid(
            latitude, longitude, indexing="ij"
        )
        heights = torch.as_tensor(shell_height_levels_m, dtype=torch.float32)
        spherical = torch.stack(
            (
                (
                    model.solar_radius_m.detach().float().cpu() + heights[:, None, None]
                ).expand(-1, self.slice_latitude_points, self.slice_longitude_points),
                (0.5 * math.pi - latitude_grid)[None].expand(heights.numel(), -1, -1),
                longitude_grid[None].expand(heights.numel(), -1, -1),
            ),
            dim=-1,
        )
        position = spherical_to_cartesian(spherical, torch).permute(1, 2, 0, 3)
        fields = self._evaluate_physical_positions(pl_module, position)
        map_longitude = np.broadcast_to(
            np.rad2deg(longitude_grid.numpy())[..., None], position.shape[:-1]
        ).copy()
        map_latitude = np.broadcast_to(
            np.rad2deg(latitude_grid.numpy())[..., None], position.shape[:-1]
        ).copy()
        result = {
            "solar_radius_m": float(model.solar_radius_m.detach().cpu()),
            "shell_height_levels_m": np.asarray(shell_height_levels_m),
            "map_fields": fields,
            "map_longitude_deg": map_longitude,
            "map_latitude_deg": map_latitude,
        }
        if self.meridional_slice_enabled:
            result["slice_longitude_deg"] = math.degrees(
                self._meridional_slice_longitude(model)
            )
        return result

    def _meridional_slice_longitude(self, model) -> float:
        """Return the configured longitude on the scene-centred continuous branch."""

        requested = math.radians(self.meridional_slice_longitude_deg)
        center = float(
            cartesian_to_spherical(model.scene_basis[2].detach().float().cpu(), torch)[
                2
            ]
        )
        return center + math.atan2(
            math.sin(requested - center), math.cos(requested - center)
        )

    def evaluate_meridional_slice(self, pl_module, raster) -> dict:
        """Evaluate a constant-longitude radial plane in physical space."""

        model = pl_module.atmosphere_model
        longitude = self._meridional_slice_longitude(model)
        _, _, latitude_min, latitude_max = self._observed_spherical_bounds(
            model, raster
        )
        latitude = torch.linspace(
            latitude_min, latitude_max, self.slice_latitude_points
        )
        outer_height_m, inner_height_m = (
            float(value) * 1.0e6 for value in model.shell_height_bounds_Mm
        )
        height = torch.linspace(
            outer_height_m, inner_height_m, self.slice_radial_points
        )
        latitude_grid, height_grid = torch.meshgrid(latitude, height, indexing="ij")
        spherical = torch.stack(
            (
                model.solar_radius_m.detach().float().cpu() + height_grid,
                0.5 * math.pi - latitude_grid,
                torch.full_like(latitude_grid, longitude),
            ),
            dim=-1,
        )
        position = spherical_to_cartesian(spherical, torch)
        fields = self._evaluate_physical_positions(pl_module, position)
        shape = (self.slice_latitude_points, 1, self.slice_radial_points)
        return {
            "solar_radius_m": float(model.solar_radius_m.detach().cpu()),
            "map_fields": {
                **{name: values[:, None, :] for name, values in fields.items()},
                "geometric_height": (height_grid / 1.0e6).numpy()[:, None, :],
            },
            "map_latitude_deg": np.rad2deg(latitude_grid.numpy())[:, None, :],
            "map_longitude_deg": np.full(
                shape, math.degrees(longitude), dtype=np.float32
            ),
            "slice_longitude_deg": math.degrees(longitude),
        }


__all__ = [
    "AtmosphereEvaluator",
    "AtmosphereSampling",
    "MAGNETIC_FIELDS",
    "THERMODYNAMIC_FIELDS",
    "VELOCITY_FIELDS",
]
