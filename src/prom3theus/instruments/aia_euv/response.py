"""Differentiable access to the pinned AIA temperature response."""

from __future__ import annotations

from collections.abc import Sequence
import json
import math

import torch
from torch import nn
from prom3theus.rt.optically_thin import NE2_RESPONSE_SCALE_CM5

from prom3theus.resources import (
    AIA_TEMPERATURE_RESPONSE_RESOURCE,
    resolve_resource_reference,
)


def _open_text(resource):
    return (
        resource.open("r", encoding="utf-8")
        if hasattr(resource, "open")
        else open(resource, encoding="utf-8")
    )


def canonical_aia_channel(channel: int | str) -> str:
    """Return the numeric canonical identifier for one AIA EUV channel."""

    if isinstance(channel, bool):
        raise ValueError("An AIA channel must be an integer or string identifier.")
    value = str(channel).strip().upper()
    if value.startswith("A"):
        value = value[1:]
    if not value.isdigit():
        raise ValueError(f"Invalid AIA channel identifier {channel!r}.")
    return str(int(value))


class AIAResponseTable(nn.Module):
    """Piecewise-linear AIA response with compact support in temperature.

    The returned channel-last values have units
    ``DN s^-1 pixel^-1 cm^5``.  The atmosphere temperature is never clipped or
    otherwise changed: only this observation operator has exact-zero response
    outside its audited CHIANTI interval.  The first resource version is the
    total response at fixed reference ``log10(ne/cm^-3)=9``; runtime density
    dependence enters through the separate ``ne^2`` emission measure.
    """

    response_unit = "DN s^-1 pixel^-1 cm^5"
    emission_measure_convention = "electron_density_squared"

    def __init__(
        self,
        channels: Sequence[int | str] | None = None,
        *,
        resource_reference: str = AIA_TEMPERATURE_RESPONSE_RESOURCE,
    ):
        super().__init__()
        resource = resolve_resource_reference(resource_reference)
        with _open_text(resource) as handle:
            document = json.load(handle)
        available = tuple(str(channel) for channel in document["channels"])
        selected = (
            available
            if channels is None
            else tuple(canonical_aia_channel(channel) for channel in channels)
        )
        if not selected:
            raise ValueError("At least one AIA channel must be selected.")
        if len(set(selected)) != len(selected):
            raise ValueError("Selected AIA channels must be unique.")
        missing = [channel for channel in selected if channel not in available]
        if missing:
            raise KeyError(
                f"AIA response {resource_reference!r} has no channels {missing}."
            )
        indices = [available.index(channel) for channel in selected]
        log_temperature = torch.tensor(
            document["axes"]["log10_temperature_k"], dtype=torch.float64
        )
        response = torch.tensor(document["response"], dtype=torch.float64)[indices]
        if (
            log_temperature.ndim != 1
            or log_temperature.numel() < 2
            or response.shape != (len(selected), log_temperature.numel())
            or not torch.isfinite(log_temperature).all()
            or not torch.isfinite(response).all()
            or torch.any(response < 0.0)
            or not torch.all(log_temperature[1:] > log_temperature[:-1])
            or document.get("response_unit") != self.response_unit
            or document.get("emission_measure_convention")
            != self.emission_measure_convention
            or document.get("outside_domain", {}).get("behavior") != "exact_zero"
        ):
            raise ValueError("Invalid verified AIA response table.")
        self.channels = selected
        self.available_channels = available
        self.resource_reference = resource_reference
        self.calibration_convention_id = document["calibration_convention_id"]
        self.reference_log10_electron_density_cm3 = float(
            document["reference_log10_electron_density_cm3"]
        )
        edge_taper = document.get("outside_domain", {}).get("edge_taper", {})
        if (
            edge_taper.get("method") != "quintic_smootherstep"
            or edge_taper.get("endpoint_value") != "exact_zero"
        ):
            raise ValueError("Invalid AIA response edge-taper contract.")
        self.edge_taper_width_log10_temperature_k = float(
            edge_taper.get("width_log10_temperature_k", math.nan)
        )
        if (
            not math.isfinite(self.edge_taper_width_log10_temperature_k)
            or self.edge_taper_width_log10_temperature_k <= 0.0
            or self.edge_taper_width_log10_temperature_k
            > float(log_temperature[-1] - log_temperature[0]) / 2.0
        ):
            raise ValueError("Invalid AIA response edge-taper width.")
        self.components_available = bool(document["components"]["available"])
        self.register_buffer("log_temperature", log_temperature, persistent=False)
        # Normalize the resource before interpolation enters the autograd graph.
        # Rescaling physical K(T) afterward leaves huge intermediate dL/dK values.
        self.register_buffer(
            "normalized_values", response / NE2_RESPONSE_SCALE_CM5, persistent=False
        )

    @property
    def temperature_bounds_k(self) -> tuple[float, float]:
        return (
            10.0 ** float(self.log_temperature[0]),
            10.0 ** float(self.log_temperature[-1]),
        )

    def forward(self, temperature_k: torch.Tensor) -> torch.Tensor:
        """Evaluate ``K_c(T)`` with channels on the final tensor axis."""

        return self.normalized_response(temperature_k) * NE2_RESPONSE_SCALE_CM5

    def normalized_response(self, temperature_k: torch.Tensor) -> torch.Tensor:
        """Evaluate K(T)/NE2_RESPONSE_SCALE_CM5 for stable emission synthesis."""

        if not isinstance(temperature_k, torch.Tensor):
            raise TypeError("temperature_k must be a torch.Tensor.")
        if not temperature_k.is_floating_point():
            raise TypeError("temperature_k must have a floating-point dtype.")
        if not torch.isfinite(temperature_k).all() or torch.any(temperature_k <= 0.0):
            raise ValueError("temperature_k must be finite and strictly positive.")
        if temperature_k.numel() == 0:
            return temperature_k.new_empty((*temperature_k.shape, len(self.channels)))

        axis = self.log_temperature.to(temperature_k)
        values = self.normalized_values.to(temperature_k)
        query = torch.log10(temperature_k)
        in_domain = (query >= axis[0]) & (query <= axis[-1])

        # Bounding only the lookup coordinate prevents huge extrapolation
        # intermediates.  ``temperature_k`` itself remains unchanged and the
        # response branch outside the atomic support is exactly zero.
        lookup = query.clamp(min=axis[0], max=axis[-1])
        flat = lookup.reshape(-1).contiguous()
        upper = torch.searchsorted(axis, flat, right=True).clamp(1, axis.numel() - 1)
        lower = upper - 1
        width = axis[upper] - axis[lower]
        fraction = (flat - axis[lower]) / width
        lower_values = values[:, lower].transpose(0, 1)
        upper_values = values[:, upper].transpose(0, 1)
        interpolated = lower_values + fraction[:, None] * (upper_values - lower_values)
        interpolated = interpolated.reshape(*temperature_k.shape, len(self.channels))
        width = query.new_tensor(self.edge_taper_width_log10_temperature_k)

        def smootherstep(value: torch.Tensor) -> torch.Tensor:
            phase = value.clamp(0.0, 1.0)
            return phase.pow(3) * (phase * (phase * 6.0 - 15.0) + 10.0)

        window = smootherstep((query - axis[0]) / width) * smootherstep(
            (axis[-1] - query) / width
        )
        tapered = interpolated * window[..., None]
        return torch.where(in_domain[..., None], tapered, torch.zeros_like(tapered))

    def extra_repr(self) -> str:
        lower, upper = self.temperature_bounds_k
        return (
            f"channels={self.channels}, temperature_k=[{lower:g}, {upper:g}], "
            f"reference_log10_ne_cm3={self.reference_log10_electron_density_cm3:g}"
        )


__all__ = ["AIAResponseTable", "canonical_aia_channel"]
