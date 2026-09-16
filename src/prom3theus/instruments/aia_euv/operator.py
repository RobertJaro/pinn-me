"""Differentiable AIA emissivity and line-of-sight observation operator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from prom3theus.rt.optically_thin import (
    NE2_RESPONSE_SCALE_CM5,
    SI_EMISSION_MEASURE_TO_CGS,
    integrate_ne2_response,
    ne2_response_emissivity,
)

from .response import AIAResponseTable


class AIAEmissionOperator(nn.Module):
    """Map local ``T`` and physical ``n_e`` to AIA count rates.

    The response table contains the instrument/atomic factor ``K_c(T)``.  The
    density is supplied by the shared atmosphere EOS and is never inferred by
    a separate AIA-only field.  Neither input is clipped: the compact response
    alone is exactly zero outside its audited temperature interval.
    """

    def __init__(
        self,
        channels_angstrom: Sequence[int],
        *,
        response_resource: str,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in channels_angstrom)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("AIA channels must be non-empty and unique.")
        self.channels_angstrom = channels
        self.response_table = AIAResponseTable(
            channels,
            resource_reference=response_resource,
        )

    def local_response(self, temperature_k: torch.Tensor) -> torch.Tensor:
        """Return channel-last ``K_c(T)`` in DN s^-1 pixel^-1 cm^5."""

        return self.response_table(temperature_k)

    def local_emission_measure_integrand(
        self,
        temperature_k: torch.Tensor,
        electron_density_m3: torch.Tensor,
    ) -> torch.Tensor:
        """Return channel-last ``n_e^2 K_c(T)`` before unit conversion."""

        response = self.response_table.normalized_response(temperature_k)
        if electron_density_m3.shape != temperature_k.shape:
            raise ValueError("electron_density_m3 must match temperature_k.")
        if (
            not electron_density_m3.is_floating_point()
            or not torch.isfinite(electron_density_m3).all()
            or torch.any(electron_density_m3 < 0.0)
        ):
            raise ValueError(
                "electron_density_m3 must be finite, floating point, and non-negative."
            )
        return (
            ne2_response_emissivity(
                response,
                electron_density_m3,
                response_scale_cm5=NE2_RESPONSE_SCALE_CM5,
            )
            / SI_EMISSION_MEASURE_TO_CGS
        )

    def forward(
        self,
        temperature_k: torch.Tensor,
        electron_density_m3: torch.Tensor,
        distance_m: torch.Tensor,
        *,
        sample_dim: int = -1,
    ) -> torch.Tensor:
        """Integrate every configured channel along the sampled rays."""

        response = self.response_table.normalized_response(temperature_k)
        return integrate_ne2_response(
            response,
            electron_density_m3,
            distance_m,
            sample_dim=sample_dim,
            response_scale_cm5=NE2_RESPONSE_SCALE_CM5,
        )

    def metadata(self) -> dict:
        lower, upper = self.response_table.temperature_bounds_k
        return {
            "type": "aia_optically_thin_ne2_response",
            "channels_angstrom": list(self.channels_angstrom),
            "response_resource": self.response_table.resource_reference,
            "response_unit": self.response_table.response_unit,
            "emission_measure_convention": (
                self.response_table.emission_measure_convention
            ),
            "temperature_support_k": [lower, upper],
            "outside_temperature_support": "response_exact_zero; atmosphere_unchanged",
            "edge_taper": {
                "method": "quintic_smootherstep",
                "width_log10_temperature_k": (
                    self.response_table.edge_taper_width_log10_temperature_k
                ),
            },
            "reference_log10_electron_density_cm3": (
                self.response_table.reference_log10_electron_density_cm3
            ),
            "components_available": self.response_table.components_available,
        }


__all__ = ["AIAEmissionOperator"]
