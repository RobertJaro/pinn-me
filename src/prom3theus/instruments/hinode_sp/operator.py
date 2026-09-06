"""Differentiable Gaussian spectral response for Hinode/SOT-SP."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math

import torch
import torch.nn.functional as F
from torch import nn

from prom3theus.instruments.base import MagneticAzimuthConvention
from prom3theus.resources import resource_path, verify_manifest_resource


def _instrument_provenance() -> dict:
    resource_name = "hinode_sp/instrument_hinode_sp.json"
    path = resource_path(resource_name)
    manifest_path = resource_path("sources.json")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_resource(
        path, manifest, resource_name=resource_name, kind="instrument"
    )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class HinodeSpectralPSF(nn.Module):
    """Normalized Gaussian approximation to the SOT-SP spectral response."""

    def __init__(
        self,
        fwhm_angstrom: float,
        oversample: int = 4,
        truncate_sigma: float = 4.0,
    ) -> None:
        super().__init__()
        provenance = _instrument_provenance()
        if not math.isfinite(float(fwhm_angstrom)) or fwhm_angstrom <= 0:
            raise ValueError("fwhm_angstrom must be finite and positive.")
        if int(oversample) != oversample or oversample < 1:
            raise ValueError("oversample must be a positive integer.")
        if not math.isfinite(float(truncate_sigma)) or truncate_sigma <= 0:
            raise ValueError("truncate_sigma must be finite and positive.")
        self.fwhm_angstrom = float(fwhm_angstrom)
        self.oversample = int(oversample)
        self.truncate_sigma = float(truncate_sigma)
        self.polarization_convention = MagneticAzimuthConvention(
            name="identity",
            offset_deg=0.0,
        )
        self.provenance = provenance
        self.register_buffer(
            "_prepared_kernel", torch.empty(0, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "_prepared_sample_left",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_sample_right",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_sample_fraction",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_synthesis_wavelength",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_prepared_observed_wavelength",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self._prepared_radius = 0
        self._prepared_synthesis_count = 0

    @property
    def sigma_angstrom(self) -> float:
        return self.fwhm_angstrom / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    @property
    def response_half_width_angstrom(self) -> float:
        return self.truncate_sigma * self.sigma_angstrom

    @staticmethod
    def _validate_wavelength(wavelength: torch.Tensor, name: str) -> None:
        if wavelength.ndim != 1 or wavelength.numel() < 2:
            raise ValueError(
                f"{name} must be a one-dimensional grid with at least two points."
            )
        if not torch.isfinite(wavelength).all():
            raise ValueError(f"{name} must contain only finite values.")
        if not torch.all(wavelength[1:] > wavelength[:-1]):
            raise ValueError(f"{name} must be strictly increasing.")

    def synthesis_grid(self, observed_wavelength_angstrom) -> torch.Tensor:
        """Return a uniform oversampled grid enclosing the observations."""

        observed = torch.as_tensor(observed_wavelength_angstrom)
        if not observed.is_floating_point():
            observed = observed.to(dtype=torch.float32)
        self._validate_wavelength(observed, "observed_wavelength_angstrom")
        spacing = torch.min(observed[1:] - observed[:-1]) / self.oversample
        margin = self.response_half_width_angstrom
        intervals = int(
            torch.ceil(
                (observed[-1] + margin - (observed[0] - margin)) / spacing
            ).item()
        )
        return torch.linspace(
            observed[0] - margin,
            observed[-1] + margin,
            intervals + 1,
            device=observed.device,
            dtype=observed.dtype,
        )

    def _kernel(self, wavelength: torch.Tensor) -> tuple[torch.Tensor, int]:
        spacing = wavelength[1:] - wavelength[:-1]
        reference_spacing = spacing.mean()
        tolerance = torch.maximum(
            1e-4 * reference_spacing.abs(),
            8 * torch.finfo(wavelength.dtype).eps * wavelength.abs().max(),
        )
        if torch.any(torch.abs(spacing - reference_spacing) > tolerance):
            raise ValueError("The synthesis wavelength grid must be uniform.")
        radius = max(
            1,
            int(
                math.ceil(
                    self.response_half_width_angstrom
                    / float(reference_spacing.detach().cpu())
                )
            ),
        )
        offsets = (
            torch.arange(
                -radius, radius + 1, device=wavelength.device, dtype=wavelength.dtype
            )
            * reference_spacing
        )
        kernel = torch.exp(-0.5 * (offsets / self.sigma_angstrom).square())
        return kernel / kernel.sum(), radius

    @property
    def prepared(self) -> bool:
        return self._prepared_kernel.numel() > 0

    def prepare(
        self, synthesis_wavelength_angstrom, observed_wavelength_angstrom
    ) -> None:
        """Compile the fixed convolution and interpolation plan."""

        synthesis = torch.as_tensor(synthesis_wavelength_angstrom)
        observed = torch.as_tensor(
            observed_wavelength_angstrom,
            dtype=synthesis.dtype,
            device=synthesis.device,
        )
        if not synthesis.is_floating_point():
            synthesis = synthesis.float()
            observed = observed.float()
        self._validate_wavelength(synthesis, "synthesis_wavelength_angstrom")
        self._validate_wavelength(observed, "observed_wavelength_angstrom")
        epsilon = 8 * torch.finfo(synthesis.dtype).eps
        if (
            observed[0] < synthesis[0] - epsilon
            or observed[-1] > synthesis[-1] + epsilon
        ):
            raise ValueError("Observed wavelengths must lie inside the synthesis grid.")
        kernel, radius = self._kernel(synthesis)
        right = torch.searchsorted(synthesis, observed).clamp(1, synthesis.numel() - 1)
        left = right - 1
        fraction = (observed - synthesis[left]) / (synthesis[right] - synthesis[left])
        self._prepared_kernel = kernel.detach().clone()
        self._prepared_sample_left = left.detach().clone()
        self._prepared_sample_right = right.detach().clone()
        self._prepared_sample_fraction = fraction.detach().clone()
        self._prepared_synthesis_wavelength = synthesis.detach().clone()
        self._prepared_observed_wavelength = observed.detach().clone()
        self._prepared_radius = int(radius)
        self._prepared_synthesis_count = int(synthesis.numel())

    def forward(
        self,
        stokes: torch.Tensor,
        synthesis_wavelength_angstrom,
        observed_wavelength_angstrom,
    ) -> torch.Tensor:
        """Convolve profiles and sample them at observed wavelengths."""

        if not self.prepared:
            raise RuntimeError(
                "HinodeSpectralPSF.prepare() must be called before forward()."
            )
        if stokes.ndim < 2 or stokes.shape[-2] != 4:
            raise ValueError(
                f"stokes must end in [4, wavelength]; got {tuple(stokes.shape)}."
            )
        if not stokes.is_floating_point() or not torch.isfinite(stokes).all():
            raise ValueError("stokes must contain finite real floating-point values.")
        synthesis = torch.as_tensor(
            synthesis_wavelength_angstrom,
            dtype=stokes.dtype,
            device=stokes.device,
        )
        observed = torch.as_tensor(
            observed_wavelength_angstrom,
            dtype=stokes.dtype,
            device=stokes.device,
        )
        self._validate_wavelength(synthesis, "synthesis_wavelength_angstrom")
        self._validate_wavelength(observed, "observed_wavelength_angstrom")
        expected_synthesis = self._prepared_synthesis_wavelength.to(stokes)
        expected_observed = self._prepared_observed_wavelength.to(stokes)
        tolerance = 8 * torch.finfo(stokes.dtype).eps * expected_synthesis.abs().max()
        if (
            synthesis.shape != expected_synthesis.shape
            or observed.shape != expected_observed.shape
            or torch.any(torch.abs(synthesis - expected_synthesis) > tolerance)
            or torch.any(torch.abs(observed - expected_observed) > tolerance)
        ):
            raise ValueError(
                "Forward wavelength grids do not match the prepared Hinode response."
            )
        if stokes.shape[-1] != self._prepared_synthesis_count:
            raise ValueError(
                "stokes wavelength dimension does not match the synthesis grid."
            )
        original_shape = stokes.shape
        flat = stokes.reshape(-1, 1, original_shape[-1])
        padded = F.pad(
            flat, (self._prepared_radius, self._prepared_radius), mode="replicate"
        )
        convolved = F.conv1d(
            padded, self._prepared_kernel.to(stokes).reshape(1, 1, -1)
        ).reshape(original_shape)
        left = convolved.index_select(-1, self._prepared_sample_left.to(stokes.device))
        right = convolved.index_select(
            -1, self._prepared_sample_right.to(stokes.device)
        )
        fraction_shape = *([1] * (convolved.ndim - 1)), -1
        fraction = self._prepared_sample_fraction.to(stokes).reshape(fraction_shape)
        return left + fraction * (right - left)

    def metadata(self) -> dict:
        return {
            "type": "normalized_gaussian",
            "fwhm_angstrom": self.fwhm_angstrom,
            "sigma_angstrom": self.sigma_angstrom,
            "oversample": self.oversample,
            "truncate_sigma": self.truncate_sigma,
            "polarization_convention": self.polarization_convention.metadata(),
            "constant_preserving_boundary": "replicate",
            "provenance": self.provenance,
        }


def _build_hinode_sp(
    spectral_psf: Mapping,
) -> HinodeSpectralPSF:
    """Build the only supported Hinode response from the strict config shape."""

    psf = dict(spectral_psf)
    response_type = psf.pop("type")
    if response_type != "gaussian":
        raise ValueError("instrument.spectral_psf.type must be 'gaussian'.")
    allowed = {"fwhm_angstrom", "oversample", "truncate_sigma"}
    unknown = set(psf) - allowed
    missing = allowed - set(psf)
    if unknown or missing:
        raise TypeError(
            "instrument.spectral_psf must contain exactly fwhm_angstrom, "
            f"oversample, and truncate_sigma; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}."
        )
    return HinodeSpectralPSF(**psf)


__all__ = ["HinodeSpectralPSF"]
