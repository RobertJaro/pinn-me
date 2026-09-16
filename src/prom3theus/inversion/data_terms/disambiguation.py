"""Smooth observer-frame phase continuation for spectropolarimetric fitting."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from prom3theus.core import SIRENModel


class ObserverPhaseRotation(nn.Module):
    """Predict a smooth observer-pixel phase in units of pi.

    The phase network is deliberately independent of the shared atmosphere.
    It receives normalized surface chart coordinates (and time for a dynamic
    atmosphere), so its output is constant along the depth samples of one
    observer ray.  The whole SIREN, including its output head, uses the
    canonical random initialization: a fresh run starts at an arbitrary,
    spatially varying phase rather than the identity rotation.  The owning
    Stokes term removes the temporary phase from the synthesis path at its
    configured hard handoff step and in evaluation.
    """

    def __init__(
        self,
        *,
        spatial_coordinate_center_mm,
        spatial_coordinate_scale_mm,
        time_dependent: bool,
        time_coordinate_center_hours: float,
        time_coordinate_scale_hours: float,
        hidden_dimension: int,
        hidden_layers: int,
        first_omega_0: float,
        hidden_omega_0: float,
    ) -> None:
        super().__init__()
        if type(time_dependent) is not bool:
            raise TypeError("time_dependent must be boolean")
        if type(hidden_dimension) is not int or hidden_dimension < 1:
            raise ValueError("hidden_dimension must be a positive integer")
        if type(hidden_layers) is not int or hidden_layers < 1:
            raise ValueError("hidden_layers must be a positive integer")
        spatial_center = torch.as_tensor(
            spatial_coordinate_center_mm,
            dtype=torch.float32,
        )
        spatial_scale = torch.as_tensor(
            spatial_coordinate_scale_mm,
            dtype=torch.float32,
        )
        if spatial_center.shape != (2,) or spatial_scale.shape != (2,):
            raise ValueError(
                "Phase spatial coordinate center and scale must each contain two values"
            )
        if not torch.isfinite(spatial_center).all() or not torch.isfinite(
            spatial_scale
        ).all() or torch.any(spatial_scale <= 0.0):
            raise ValueError("Phase spatial coordinate normalization must be finite")
        time_center = float(time_coordinate_center_hours)
        time_scale = float(time_coordinate_scale_hours)
        if not math.isfinite(time_center) or not math.isfinite(time_scale) or time_scale <= 0.0:
            raise ValueError("Phase time coordinate normalization must be finite and positive")

        self.time_dependent = time_dependent
        self.first_omega_0 = float(first_omega_0)
        self.hidden_omega_0 = float(hidden_omega_0)
        self.register_buffer("spatial_coordinate_center_mm", spatial_center)
        self.register_buffer("spatial_coordinate_scale_mm", spatial_scale)
        self.register_buffer(
            "time_coordinate_center_hours",
            torch.tensor(time_center, dtype=torch.float32),
        )
        self.register_buffer(
            "time_coordinate_scale_hours",
            torch.tensor(time_scale, dtype=torch.float32),
        )
        # SIRENModel applies its canonical uniform random initialization to
        # every layer, including the output head: a fresh run starts at a
        # random, spatially varying phase rather than the identity rotation.
        self.network = SIRENModel(
            in_dim=3 if time_dependent else 2,
            out_dim=1,
            dim=hidden_dimension,
            n_layers=hidden_layers,
            first_omega_0=self.first_omega_0,
            hidden_omega_0=self.hidden_omega_0,
            radial_weighting_config=None,
        )

    def _inputs(self, coordinates: torch.Tensor) -> torch.Tensor:
        if not isinstance(coordinates, torch.Tensor):
            raise TypeError("Phase coordinates must be a tensor")
        if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
            raise ValueError("Phase coordinates must end in [x, y, time]")
        coordinates = coordinates.to(
            device=self.spatial_coordinate_center_mm.device,
            dtype=self.spatial_coordinate_center_mm.dtype,
        )
        spatial = (
            coordinates[..., :2] - self.spatial_coordinate_center_mm
        ) / self.spatial_coordinate_scale_mm
        if not self.time_dependent:
            return spatial
        time = (
            coordinates[..., 2:3] - self.time_coordinate_center_hours
        ) / self.time_coordinate_scale_hours
        return torch.cat((spatial, time), dim=-1)

    def raw_output(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return the unconstrained phase-network output before pi scaling."""

        return self.network(self._inputs(coordinates)).squeeze(-1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return the synthesis phase in radians as ``network_output * pi``."""

        return self.raw_output(coordinates) * math.pi

    def metadata(self) -> Mapping[str, object]:
        return {
            "coordinate_frame": "normalized observer surface chart coordinates",
            "time_dependent": self.time_dependent,
            "phase_scale_rad": math.pi,
            "network": {
                "type": "siren",
                "output_initialization": "random",
                "dim": self.network.in_layer.out_features,
                "n_layers": len(self.network.activations),
                "first_omega_0": self.first_omega_0,
                "hidden_omega_0": self.hidden_omega_0,
            },
        }


__all__ = ["ObserverPhaseRotation"]
