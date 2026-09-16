"""Differentiable optically thin synthesis along native observation rays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch

from prom3theus.observations.scene import SceneContract
from .depth_sampling import (
    DepthRefinement,
    jitter_depth_grid,
    contribution_fine_distances,
    merge_depth_samples,
)
from prom3theus.rt.optically_thin import SI_EMISSION_MEASURE_TO_CGS, trapezoid_weights


@dataclass(frozen=True, slots=True)
class RayIntegralResult:
    """Raw image prediction and the physical samples that produced it."""

    raw_prediction: torch.Tensor
    all_channel_prediction: torch.Tensor
    channel_index: torch.Tensor
    position_m: torch.Tensor
    distance_m: torch.Tensor
    line_element_m: torch.Tensor
    chart_time_coordinates: torch.Tensor
    geometric_height_m: torch.Tensor
    temperature_k: torch.Tensor
    gas_pressure: torch.Tensor
    electron_density_m3: torch.Tensor
    emission_measure_contribution_cm5: torch.Tensor
    channel_emission_contribution: torch.Tensor | None

    @property
    def prediction(self) -> torch.Tensor:
        """Alias used by observation objectives before nuisance calibration."""

        return self.raw_prediction

    def diagnostics(self) -> dict[str, torch.Tensor]:
        values = {
            "position_m": self.position_m,
            "distance_m": self.distance_m,
            "line_element_m": self.line_element_m,
            "chart_time_coordinates": self.chart_time_coordinates,
            "geometric_height_m": self.geometric_height_m,
            "temperature_k": self.temperature_k,
            "gas_pressure": self.gas_pressure,
            "electron_density_m3": self.electron_density_m3,
            "emission_measure_contribution_cm5": (
                self.emission_measure_contribution_cm5
            ),
            "all_channel_prediction": self.all_channel_prediction,
            "channel_index": self.channel_index,
        }
        if self.channel_emission_contribution is not None:
            values["channel_emission_contribution"] = self.channel_emission_contribution
        return values


def _reference_tensor(atmosphere_model: Any) -> torch.Tensor:
    reference = getattr(atmosphere_model, "solar_radius_m", None)
    if isinstance(reference, torch.Tensor) and reference.is_floating_point():
        return reference
    parameters = getattr(atmosphere_model, "parameters", None)
    if callable(parameters):
        try:
            parameter = next(parameters())
        except StopIteration:
            parameter = None
        if isinstance(parameter, torch.Tensor) and parameter.is_floating_point():
            return parameter
    raise TypeError(
        "atmosphere_model must expose a floating solar_radius_m tensor or parameter."
    )


def _channel_vocabulary(emission_operator: Any) -> tuple[str, ...]:
    channels: Sequence[int | str] | None = getattr(
        emission_operator, "channels_angstrom", None
    )
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
        raise TypeError("emission_operator must expose channels_angstrom.")
    values = tuple(str(channel).strip() for channel in channels)
    if (
        not values
        or any(not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "emission_operator channels_angstrom must be non-empty and unique."
        )
    return values


class RayIntegralForwardComposition:
    """Compose scene geometry, one shared atmosphere, EOS, and EUV emission.

    This class is intentionally not an ``nn.Module``.  It holds references to
    the atmosphere and emission operator registered by their owning data term,
    avoiding duplicate module ownership or state-dict prefixes.
    """

    def __init__(
        self,
        *,
        atmosphere_model: Any,
        scene: SceneContract,
        emission_operator: Any,
        sample_count: int = 192,
        height_power: float = 2.0,
        depth_refinement: DepthRefinement | None = None,
    ) -> None:
        if not isinstance(scene, SceneContract):
            raise TypeError("scene must be a SceneContract.")
        if not callable(emission_operator):
            raise TypeError("emission_operator must be callable.")
        if type(sample_count) is not int or sample_count < 2:
            raise ValueError("sample_count must be an integer of at least two.")
        height_power = float(height_power)
        if not math.isfinite(height_power) or height_power <= 0:
            raise ValueError("height_power must be finite and strictly positive.")
        _reference_tensor(atmosphere_model)
        eos = getattr(atmosphere_model, "thermodynamic_eos", None)
        if eos is None or not callable(getattr(eos, "electron_density", None)):
            raise TypeError(
                "atmosphere_model.thermodynamic_eos must provide electron_density(T, P)."
            )
        self.atmosphere_model = atmosphere_model
        self.scene = scene
        self.emission_operator = emission_operator
        self.sample_count = sample_count
        self.height_power = height_power
        self.depth_refinement = depth_refinement or DepthRefinement(
            enabled=False, sample_count=1, uniform_weight_floor=0.0
        )
        if self.depth_refinement.enabled and not callable(
            getattr(emission_operator, "local_emission_measure_integrand", None)
        ):
            raise TypeError("Refinement requires a local emission integrand.")
        self.channels_angstrom = _channel_vocabulary(emission_operator)

    def _outer_height_m(self) -> float:
        try:
            bounds = tuple(
                float(value) for value in self.atmosphere_model.shell_height_bounds_Mm
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "atmosphere_model must declare shell_height_bounds_Mm=[outer, inner]."
            ) from error
        if (
            len(bounds) != 2
            or not all(math.isfinite(value) for value in bounds)
            or not bounds[0] > 0 > bounds[1]
        ):
            raise ValueError(
                "shell_height_bounds_Mm must contain finite [outer, inner] bounds "
                "that straddle the photosphere."
            )
        outer_height_m = bounds[0] * 1.0e6
        scene_inner, scene_outer = self.scene.height_bounds_m
        if scene_inner > 0 or not math.isclose(
            scene_outer,
            outer_height_m,
            rel_tol=1.0e-10,
            abs_tol=1.0,
        ):
            raise ValueError(
                "SceneContract height bounds must include the photosphere and share "
                "the atmosphere outer shell."
            )
        model_radius = getattr(
            self.atmosphere_model,
            "solar_radius_value_m",
            getattr(self.atmosphere_model, "solar_radius_m", None),
        )
        if model_radius is not None:
            radius = float(torch.as_tensor(model_radius).detach().cpu())
            if not math.isclose(
                radius, self.scene.solar_radius_m, rel_tol=2.0e-7, abs_tol=100.0
            ):
                raise ValueError(
                    "Atmosphere and SceneContract must share one solar radius."
                )
        return outer_height_m

    def _validate_batch(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("An image synthesis batch must be a mapping.")
        required = {
            "surface_position_m",
            "ray_direction",
            "absolute_tai_seconds",
            "channel_index",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"Image synthesis batch is missing fields: {missing}.")
        reference = _reference_tensor(self.atmosphere_model)
        surface = torch.as_tensor(
            batch["surface_position_m"],
            dtype=torch.float64,
            device=reference.device,
        )
        ray = torch.as_tensor(
            batch["ray_direction"], dtype=torch.float64, device=reference.device
        )
        absolute_time = torch.as_tensor(
            batch["absolute_tai_seconds"],
            dtype=torch.float64,
            device=reference.device,
        )
        raw_channel_index = torch.as_tensor(
            batch["channel_index"], device=reference.device
        )
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if raw_channel_index.dtype not in integer_dtypes:
            raise TypeError("channel_index must use an integer dtype.")
        channel_index = raw_channel_index.to(torch.long)
        if surface.ndim != 2 or surface.shape[-1] != 3 or not surface.shape[0]:
            raise ValueError("surface_position_m must have non-empty shape [ray, 3].")
        if ray.shape != surface.shape:
            raise ValueError("ray_direction must match surface_position_m shape.")
        ray_count = surface.shape[0]
        if absolute_time.shape != (ray_count,):
            raise ValueError("absolute_tai_seconds must have shape [ray].")
        if channel_index.shape != (ray_count,):
            raise ValueError("channel_index must have shape [ray].")
        if (
            not torch.isfinite(surface).all()
            or not torch.isfinite(ray).all()
            or not torch.isfinite(absolute_time).all()
        ):
            raise ValueError("Image ray geometry and times must be finite.")
        norm = torch.linalg.vector_norm(ray, dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), rtol=0.0, atol=2.0e-5):
            raise ValueError("ray_direction must contain observer-to-Sun unit vectors.")
        ray = ray / norm[:, None]
        radius = torch.linalg.vector_norm(surface, dim=-1)
        expected_radius = torch.full_like(radius, self.scene.solar_radius_m)
        if not torch.allclose(radius, expected_radius, rtol=2.0e-5, atol=1.0):
            raise ValueError(
                "surface_position_m must lie on the SceneContract photosphere."
            )
        surface = surface * (self.scene.solar_radius_m / radius)[..., None]
        outward = -ray
        if torch.any((surface * outward).sum(dim=-1) <= 0):
            raise ValueError(
                "ray_direction must point from the observer toward an on-disk surface."
            )
        if torch.any(channel_index < 0) or torch.any(
            channel_index >= len(self.channels_angstrom)
        ):
            raise IndexError("channel_index lies outside emission-operator channels.")
        return surface, ray, absolute_time, channel_index

    def _sample_rays(
        self,
        surface_position_m: torch.Tensor,
        ray_direction: torch.Tensor,
        outer_height_m: float,
        *,
        sample_count: int,
        height_power: float,
        randomize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Geometry is evaluated in float64 so the zero-distance photospheric
        # root is not lost to cancellation at solar-radius scales.
        surface = surface_position_m.to(torch.float64)
        outward = -ray_direction.to(torch.float64)
        fraction = torch.linspace(
            0.0,
            1.0,
            sample_count,
            dtype=torch.float64,
            device=surface.device,
        )
        fraction = jitter_depth_grid(
            fraction.expand(surface.shape[0], -1), randomize=randomize
        )
        target_height = outer_height_m * fraction.pow(height_power)
        projection = (surface * outward).sum(dim=-1, keepdim=True)
        # Rationalized near-side root avoids subtracting solar-radius-scale
        # numbers when a perturbed shell lies very close to the photosphere.
        radial_increment = target_height * (
            2.0 * self.scene.solar_radius_m + target_height
        )
        distance = radial_increment / (
            torch.sqrt(projection.square() + radial_increment) + projection
        )
        if not torch.isfinite(distance).all() or torch.any(
            torch.diff(distance, dim=1) <= 0
        ):
            raise ValueError(
                "Ray samples must increase strictly from the photosphere to the outer shell."
            )
        position = surface[:, None, :] + distance[..., None] * outward[:, None, :]
        return position, distance

    def _atmosphere_samples(self, chart_time, geometric_height_m, distance_m):
        fields = self.atmosphere_model.evaluate_chart_height_points(
            chart_time,
            geometric_height_m,
        )
        if not isinstance(fields, Mapping):
            raise TypeError("Atmosphere evaluation must return a field mapping.")
        missing = {"temperature", "gas_pressure"} - set(fields)
        if missing:
            raise KeyError(
                f"Atmosphere evaluation is missing fields: {sorted(missing)}."
            )
        temperature = fields["temperature"]
        gas_pressure = fields["gas_pressure"]
        if (
            not isinstance(temperature, torch.Tensor)
            or not isinstance(gas_pressure, torch.Tensor)
            or temperature.shape != distance_m.shape
            or gas_pressure.shape != distance_m.shape
        ):
            raise ValueError(
                "Atmosphere temperature and gas_pressure must have shape [ray, sample]."
            )
        if (
            not torch.isfinite(temperature).all()
            or not torch.isfinite(gas_pressure).all()
            or torch.any(temperature <= 0)
            or torch.any(gas_pressure <= 0)
        ):
            raise FloatingPointError(
                "Atmosphere temperature and gas pressure must be finite and positive."
            )
        electron_density = self.atmosphere_model.thermodynamic_eos.electron_density(
            temperature,
            gas_pressure,
        )
        if (
            not isinstance(electron_density, torch.Tensor)
            or electron_density.shape != temperature.shape
            or not torch.isfinite(electron_density).all()
            or torch.any(electron_density < 0)
        ):
            raise FloatingPointError(
                "The shared thermodynamic EOS returned invalid electron density."
            )
        return temperature, gas_pressure, electron_density

    def synthesize(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        surface_position_m: torch.Tensor | None = None,
        ray_direction: torch.Tensor | None = None,
        absolute_tai_seconds: torch.Tensor | None = None,
        channel_index: torch.Tensor | None = None,
        sample_count: int | None = None,
        height_sampling_power: float | None = None,
        randomize: bool = False,
        refine: bool = True,
    ) -> RayIntegralResult:
        """Forward render one mixed-channel batch without applying calibration."""

        explicit = {
            "surface_position_m": surface_position_m,
            "ray_direction": ray_direction,
            "absolute_tai_seconds": absolute_tai_seconds,
            "channel_index": channel_index,
        }
        if batch is None:
            if any(value is None for value in explicit.values()):
                missing = sorted(
                    name for name, value in explicit.items() if value is None
                )
                raise KeyError(f"Image synthesis inputs are missing fields: {missing}.")
            batch = explicit
        elif any(value is not None for value in explicit.values()):
            raise ValueError(
                "Pass either one batch mapping or explicit image-ray fields, not both."
            )
        selected_count = self.sample_count if sample_count is None else sample_count
        if type(selected_count) is not int or selected_count < 2:
            raise ValueError("sample_count must be an integer of at least two.")
        selected_power = (
            self.height_power
            if height_sampling_power is None
            else float(height_sampling_power)
        )
        if not math.isfinite(selected_power) or selected_power <= 0:
            raise ValueError(
                "height_sampling_power must be finite and strictly positive."
            )
        surface, ray, absolute_time, channel_index = self._validate_batch(batch)
        outer_height_m = self._outer_height_m()
        position_m, distance_m = self._sample_rays(
            surface,
            ray,
            outer_height_m,
            sample_count=selected_count,
            height_power=selected_power,
            randomize=randomize,
        )
        ray_time = absolute_time[:, None].expand(distance_m.shape)
        chart_time, geometric_height_m = self.scene.transform(position_m, ray_time)
        temperature, gas_pressure, electron_density = self._atmosphere_samples(
            chart_time, geometric_height_m, distance_m
        )
        if refine and self.depth_refinement.enabled:
            with torch.no_grad():
                local = self.emission_operator.local_emission_measure_integrand(
                    temperature, electron_density
                )
                selected = local.gather(
                    -1, channel_index[:, None, None].expand(-1, distance_m.shape[1], 1)
                ).squeeze(-1)
                masses = (
                    0.5
                    * (selected[:, 1:] + selected[:, :-1])
                    * torch.diff(distance_m, dim=-1)
                )
                fine_distance = contribution_fine_distances(
                    masses,
                    distance_m,
                    self.depth_refinement.sample_count,
                    self.depth_refinement.uniform_weight_floor,
                    randomize=randomize,
                )
                fine_position = (
                    surface[:, None, :] - fine_distance[..., None] * ray[:, None, :]
                )
                fine_chart, fine_height = self.scene.transform(
                    fine_position, absolute_time[:, None].expand(fine_distance.shape)
                )
                merged_distance, order = torch.sort(
                    torch.cat((distance_m, fine_distance), dim=-1), dim=-1
                )
            fine_temperature, fine_pressure, fine_density = self._atmosphere_samples(
                fine_chart, fine_height, fine_distance
            )
            temperature = merge_depth_samples(temperature, fine_temperature, order)
            gas_pressure = merge_depth_samples(gas_pressure, fine_pressure, order)
            electron_density = merge_depth_samples(
                electron_density, fine_density, order
            )
            position_m = merge_depth_samples(position_m, fine_position, order)
            chart_time = merge_depth_samples(chart_time, fine_chart, order)
            geometric_height_m = merge_depth_samples(
                geometric_height_m, fine_height, order
            )
            distance_m = merged_distance
        all_channels = self.emission_operator(
            temperature,
            electron_density,
            distance_m,
            sample_dim=1,
        )
        expected_shape = (surface.shape[0], len(self.channels_angstrom))
        if (
            not isinstance(all_channels, torch.Tensor)
            or all_channels.shape != expected_shape
        ):
            raise ValueError(
                "emission_operator must return [ray, channel] with shape "
                f"{expected_shape}."
            )
        if not torch.isfinite(all_channels).all() or torch.any(all_channels < 0):
            raise FloatingPointError(
                "The emission operator returned invalid raw image intensities."
            )
        raw_prediction = torch.gather(all_channels, 1, channel_index[:, None]).squeeze(
            1
        )
        weights = trapezoid_weights(distance_m)
        contribution = (
            SI_EMISSION_MEASURE_TO_CGS * electron_density.to(weights).square() * weights
        )
        channel_contribution = None
        local_integrand = getattr(
            self.emission_operator, "local_emission_measure_integrand", None
        )
        if callable(local_integrand):
            local = local_integrand(temperature, electron_density)
            expected_local_shape = (*temperature.shape, len(self.channels_angstrom))
            if (
                not isinstance(local, torch.Tensor)
                or local.shape != expected_local_shape
                or not torch.isfinite(local).all()
                or torch.any(local < 0)
            ):
                raise ValueError(
                    "emission_operator.local_emission_measure_integrand must return "
                    "finite non-negative [ray, sample, channel] values."
                )
            channel_contribution = (
                SI_EMISSION_MEASURE_TO_CGS * local.to(weights) * weights[..., None]
            )
        return RayIntegralResult(
            raw_prediction=raw_prediction,
            all_channel_prediction=all_channels,
            channel_index=channel_index,
            position_m=position_m,
            distance_m=distance_m,
            line_element_m=weights,
            chart_time_coordinates=chart_time,
            geometric_height_m=geometric_height_m,
            temperature_k=temperature,
            gas_pressure=gas_pressure,
            electron_density_m3=electron_density,
            emission_measure_contribution_cm5=contribution,
            channel_emission_contribution=channel_contribution,
        )

    __call__ = synthesize


__all__ = ["RayIntegralForwardComposition", "RayIntegralResult"]
