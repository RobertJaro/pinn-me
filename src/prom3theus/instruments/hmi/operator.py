"""Differentiable SDO/HMI filter-profile integration."""

from __future__ import annotations

import json
import math

import torch
from torch import nn

from prom3theus.resources import resource_path, verify_manifest_resource


def _instrument_provenance() -> dict:
    resource_name = "hmi_stokes/instrument_hmi.json"
    path = resource_path(resource_name)
    manifest_path = resource_path("sources.json")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_resource(
        path, manifest, resource_name=resource_name, kind="instrument"
    )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_wavelength(wavelength: torch.Tensor, name: str) -> None:
    if wavelength.ndim != 1 or wavelength.numel() < 2:
        raise ValueError(
            f"{name} must be a one-dimensional grid with at least two points."
        )
    if not torch.isfinite(wavelength).all() or not torch.all(
        wavelength[1:] > wavelength[:-1]
    ):
        raise ValueError(f"{name} must contain finite, strictly increasing values.")


class HMIFilterProfiles(nn.Module):
    """Integrate LTE quadrature spectra through batch-local HMI profiles."""

    def __init__(
        self,
        quadrature_wavelength_angstrom,
        inner_half_width_angstrom: float,
    ):
        super().__init__()
        provenance = _instrument_provenance()
        quadrature = torch.as_tensor(
            quadrature_wavelength_angstrom, dtype=torch.float64
        )
        _validate_wavelength(quadrature, "quadrature_wavelength_angstrom")
        half_width = float(inner_half_width_angstrom)
        if not math.isfinite(half_width) or half_width <= 0:
            raise ValueError("inner_half_width_angstrom must be finite and positive.")
        self.inner_half_width_angstrom = half_width
        self.tuning_reference_angstrom = float(
            provenance["spectral_line"][
                "instrument_tuning_reference_air_wavelength_angstrom"
            ]
        )
        self.provenance = provenance
        lower = self.tuning_reference_angstrom - half_width
        upper = self.tuning_reference_angstrom + half_width
        if float(quadrature[0]) <= lower or float(quadrature[-1]) >= upper:
            raise ValueError(
                "HMI quadrature nodes must lie inside the continuum endpoints."
            )
        self.register_buffer("quadrature_wavelength_angstrom", quadrature)

    def synthesis_grid(self, observed_wavelength_angstrom) -> torch.Tensor:
        observed = torch.as_tensor(observed_wavelength_angstrom)
        if not observed.is_floating_point():
            observed = observed.float()
        _validate_wavelength(observed, "observed_wavelength_angstrom")
        quadrature = self.quadrature_wavelength_angstrom.to(observed)
        _validate_wavelength(quadrature, "quadrature_wavelength_angstrom")
        endpoints = observed.new_tensor(
            [
                self.tuning_reference_angstrom - self.inner_half_width_angstrom,
                self.tuning_reference_angstrom + self.inner_half_width_angstrom,
            ]
        )
        # At visible wavelengths one float32 ULP is about 5e-4 Angstrom.  The
        # outermost Gauss--Legendre node of the production HMI grid lies less
        # than one ULP inside each nominal continuum endpoint, so a direct cast
        # can collapse both pairs onto identical values.  Keep the fast float32
        # forward model and move only a collided auxiliary endpoint by one
        # representable value away from the quadrature interval.
        if endpoints[0] >= quadrature[0]:
            endpoints[0] = torch.nextafter(
                quadrature[0], quadrature.new_tensor(-torch.inf)
            )
        if endpoints[1] <= quadrature[-1]:
            endpoints[1] = torch.nextafter(
                quadrature[-1], quadrature.new_tensor(torch.inf)
            )
        return torch.cat(
            (
                endpoints[:1],
                quadrature,
                endpoints[1:],
            )
        )

    def forward(
        self,
        stokes: torch.Tensor,
        synthesis_wavelength_angstrom,
        observed_wavelength_angstrom,
        *,
        spectral_weights,
        continuum_weights,
    ) -> torch.Tensor:
        synthesis_wavelength = torch.as_tensor(
            synthesis_wavelength_angstrom, dtype=stokes.dtype, device=stokes.device
        )
        observed_wavelength = torch.as_tensor(
            observed_wavelength_angstrom, dtype=stokes.dtype, device=stokes.device
        )
        if stokes.ndim < 3 or stokes.shape[-2:] != (4, synthesis_wavelength.numel()):
            raise ValueError("stokes must end in [4, synthesis_wavelength].")
        if not stokes.is_floating_point() or not torch.isfinite(stokes).all():
            raise ValueError("stokes must contain finite real floating-point values.")
        _validate_wavelength(synthesis_wavelength, "synthesis_wavelength_angstrom")
        _validate_wavelength(observed_wavelength, "observed_wavelength_angstrom")
        expected_synthesis = self.synthesis_grid(observed_wavelength).to(stokes)
        tolerance = 8 * torch.finfo(stokes.dtype).eps * expected_synthesis.abs().max()
        if synthesis_wavelength.shape != expected_synthesis.shape or torch.any(
            torch.abs(synthesis_wavelength - expected_synthesis) > tolerance
        ):
            raise ValueError(
                "synthesis_wavelength_angstrom does not match the configured HMI response."
            )
        weights = torch.as_tensor(
            spectral_weights, dtype=stokes.dtype, device=stokes.device
        )
        continuum = torch.as_tensor(
            continuum_weights, dtype=stokes.dtype, device=stokes.device
        )
        expected_prefix = stokes.shape[:-2]
        if weights.shape[:-2] != expected_prefix:
            raise ValueError("HMI weights must have shape [..., filter, quadrature].")
        if weights.shape[-2] != observed_wavelength.numel():
            raise ValueError(
                "HMI response filter count does not match the observation grid."
            )
        if weights.shape[-1] != self.quadrature_wavelength_angstrom.numel():
            raise ValueError("HMI response weights do not match the quadrature grid.")
        if continuum.shape != weights.shape[:-1]:
            raise ValueError("HMI continuum_weights must have shape [..., filter].")
        if (
            not torch.isfinite(weights).all()
            or not torch.isfinite(continuum).all()
            or torch.any(weights < 0)
            or torch.any(continuum < 0)
            or not torch.allclose(
                weights.sum(dim=-1) + continuum,
                torch.ones_like(continuum),
                rtol=1e-4,
                atol=1e-5,
            )
        ):
            raise ValueError(
                "HMI spectral and continuum weights must be finite, non-negative, "
                "and sum to one."
            )
        quadrature_stokes = stokes[..., 1:-1]
        integrated = torch.matmul(
            quadrature_stokes,
            weights.transpose(-1, -2),
        )
        # The phase-map archive has already integrated the far blocking-filter
        # tails into one scalar. There is no polarized continuum in this model.
        edge_continuum = 0.5 * (stokes[..., 0, 0] + stokes[..., 0, -1])
        integrated[..., 0, :] += continuum * edge_continuum.unsqueeze(-1)
        return integrated

    def metadata(self) -> dict:
        return {
            "type": "hmi_batch_local_filter_profiles",
            "quadrature_wavelength_count": int(
                self.quadrature_wavelength_angstrom.numel()
            ),
            "synthesis_wavelength_count": int(
                self.quadrature_wavelength_angstrom.numel() + 2
            ),
            "inner_half_width_angstrom": self.inner_half_width_angstrom,
            "tuning_reference_air_wavelength_angstrom": self.tuning_reference_angstrom,
            "provenance": self.provenance,
        }


__all__ = ["HMIFilterProfiles"]
