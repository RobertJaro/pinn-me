"""Differentiable Hinode/SOT-SP spectral degradation."""

from __future__ import annotations

import math
import json
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from pme.lte.atomic import verify_manifest_resource


def _instrument_provenance(data_directory=None) -> dict:
    data_root = (
        files("pme.lte").joinpath("data")
        if data_directory is None
        else Path(data_directory).expanduser().resolve()
    )
    path = data_root.joinpath("instrument_hinode_sp.json")
    manifest_path = data_root.joinpath("sources.json")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_resource(path, manifest, kind="instrument")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class HinodeSpectralPSF(nn.Module):
    """Normalized Gaussian approximation to the SOT-SP spectral response.

    The response is evaluated on a uniformly oversampled synthesis grid and the
    convolved spectra are linearly sampled at the exact observed wavelengths.
    Replicated boundary padding and a unit-sum discrete kernel guarantee that a
    constant input spectrum remains constant, including at both window edges.

    Parameters
    ----------
    fwhm_angstrom:
        Gaussian full width at half maximum.  If omitted, use 25 mA: the
        approximate FWHM of the pre-launch tunable-laser measurement reported
        by Lites et al. (2013).  This is explicitly a Gaussian approximation;
        the measured response has asymmetric and elevated inner wings.
    oversample:
        Number of synthesis intervals per smallest observed wavelength interval.
    truncate_sigma:
        Gaussian half-width retained in units of sigma.
    """

    def __init__(
        self,
        fwhm_angstrom: float | None = None,
        oversample: int = 4,
        truncate_sigma: float = 4.0,
        response_offsets_angstrom=None,
        response_weights=None,
        response_provenance: dict | None = None,
        data_directory=None,
    ):
        super().__init__()
        provenance = _instrument_provenance(data_directory)
        if fwhm_angstrom is None:
            fwhm_angstrom = provenance["modeled_spectral_response"]["fwhm_angstrom"]
        if not math.isfinite(float(fwhm_angstrom)) or fwhm_angstrom <= 0:
            raise ValueError("fwhm_angstrom must be finite and positive.")
        if int(oversample) != oversample or oversample < 1:
            raise ValueError("oversample must be a positive integer.")
        if not math.isfinite(float(truncate_sigma)) or truncate_sigma <= 0:
            raise ValueError("truncate_sigma must be finite and positive.")
        self.fwhm_angstrom = float(fwhm_angstrom)
        self.oversample = int(oversample)
        self.truncate_sigma = float(truncate_sigma)
        self.provenance = provenance
        if (response_offsets_angstrom is None) != (response_weights is None):
            raise ValueError(
                "response_offsets_angstrom and response_weights must be supplied together."
            )
        if response_offsets_angstrom is None:
            offsets = torch.empty(0, dtype=torch.float64)
            weights = torch.empty(0, dtype=torch.float64)
            self.response_provenance = None
            self._response_offsets_metadata = []
            self._response_weights_metadata = []
        else:
            offsets = torch.as_tensor(response_offsets_angstrom, dtype=torch.float64)
            weights = torch.as_tensor(response_weights, dtype=torch.float64)
            if offsets.ndim != 1 or offsets.numel() < 3 or weights.shape != offsets.shape:
                raise ValueError("Custom spectral response offsets/weights must be matching 1-D arrays.")
            if not torch.isfinite(offsets).all() or not torch.isfinite(weights).all():
                raise ValueError("Custom spectral response must be finite.")
            if not torch.all(offsets[1:] > offsets[:-1]):
                raise ValueError("Custom response offsets must increase strictly.")
            if offsets[0] >= 0 or offsets[-1] <= 0:
                raise ValueError("Custom response offsets must span zero wavelength displacement.")
            if torch.any(weights < 0) or not torch.any(weights > 0):
                raise ValueError("Custom response weights must be non-negative and nonzero.")
            if not response_provenance:
                raise ValueError(
                    "A custom response requires response_provenance; an unlabeled kernel "
                    "must not enter scientific synthesis."
                )
            weights = weights / weights.sum()
            self.response_provenance = dict(response_provenance)
            self._response_offsets_metadata = offsets.tolist()
            self._response_weights_metadata = weights.tolist()
        self.register_buffer("response_offsets_angstrom", offsets)
        self.register_buffer("response_weights", weights)

    @property
    def sigma_angstrom(self) -> float:
        return self.fwhm_angstrom / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    @property
    def response_half_width_angstrom(self) -> float:
        if self.response_offsets_angstrom.numel():
            return float(self.response_offsets_angstrom.abs().max())
        return self.truncate_sigma * self.sigma_angstrom

    @staticmethod
    def _validate_wavelength(wavelength: torch.Tensor, name: str) -> None:
        if wavelength.ndim != 1 or wavelength.numel() < 2:
            raise ValueError(f"{name} must be a one-dimensional grid with at least two points.")
        if not torch.isfinite(wavelength).all():
            raise ValueError(f"{name} must contain only finite values.")
        if not torch.all(wavelength[1:] > wavelength[:-1]):
            raise ValueError(f"{name} must be strictly increasing.")

    def synthesis_grid(
        self,
        observed_wavelength_angstrom,
    ) -> torch.Tensor:
        """Return a uniform oversampled grid enclosing the observations."""

        observed = torch.as_tensor(observed_wavelength_angstrom)
        if not observed.is_floating_point():
            observed = observed.to(dtype=torch.float32)
        self._validate_wavelength(observed, "observed_wavelength_angstrom")
        spacing = torch.min(observed[1:] - observed[:-1]) / self.oversample
        margin = self.response_half_width_angstrom
        start = observed[0] - margin
        stop = observed[-1] + margin
        intervals = int(torch.ceil((stop - start) / spacing).item())
        return torch.linspace(start, stop, intervals + 1, device=observed.device, dtype=observed.dtype)

    def _kernel(self, wavelength: torch.Tensor) -> tuple[torch.Tensor, int]:
        spacing = wavelength[1:] - wavelength[:-1]
        reference_spacing = spacing.mean()
        # Absolute wavelength values near 6300 A lose sub-mA precision in
        # float32.  Include that representation floor when checking a linspace.
        tolerance = torch.maximum(
            1e-4 * reference_spacing.abs(),
            8 * torch.finfo(wavelength.dtype).eps * wavelength.abs().max(),
        )
        if torch.any(torch.abs(spacing - reference_spacing) > tolerance):
            raise ValueError("The oversampled synthesis wavelength grid must be uniform.")
        radius = max(1, int(math.ceil(self.response_half_width_angstrom /
                                      float(reference_spacing.detach().cpu()))))
        offsets = torch.arange(-radius, radius + 1, device=wavelength.device, dtype=wavelength.dtype)
        offsets = offsets * reference_spacing
        if self.response_offsets_angstrom.numel():
            source_offsets = self.response_offsets_angstrom.to(wavelength)
            source_weights = self.response_weights.to(wavelength)
            indices = torch.searchsorted(source_offsets, offsets).clamp(1, source_offsets.numel() - 1)
            left = indices - 1
            right = indices
            fraction = (offsets - source_offsets[left]) / (
                source_offsets[right] - source_offsets[left]
            )
            kernel = source_weights[left] + fraction * (
                source_weights[right] - source_weights[left]
            )
            inside = (offsets >= source_offsets[0]) & (offsets <= source_offsets[-1])
            kernel = torch.where(inside, kernel, torch.zeros_like(kernel))
        else:
            kernel = torch.exp(-0.5 * (offsets / self.sigma_angstrom).square())
        kernel = kernel / kernel.sum()
        return kernel, radius

    @staticmethod
    def _linear_sample(
        values: torch.Tensor,
        source_wavelength: torch.Tensor,
        target_wavelength: torch.Tensor,
    ) -> torch.Tensor:
        indices = torch.searchsorted(source_wavelength, target_wavelength)
        indices = indices.clamp(1, source_wavelength.numel() - 1)
        left = indices - 1
        right = indices
        x0 = source_wavelength[left]
        x1 = source_wavelength[right]
        weight = (target_wavelength - x0) / (x1 - x0)
        y0 = values.index_select(-1, left)
        y1 = values.index_select(-1, right)
        shape = *([1] * (values.ndim - 1)), -1
        return y0 + weight.reshape(shape) * (y1 - y0)

    def forward(
        self,
        stokes: torch.Tensor,
        synthesis_wavelength_angstrom,
        observed_wavelength_angstrom,
    ) -> torch.Tensor:
        """Convolve and sample Stokes profiles.

        ``stokes`` must have shape ``[..., 4, synthesis_wavelength]``.  Returned
        profiles have shape ``[..., 4, observed_wavelength]``.
        """

        synthesis_wavelength = torch.as_tensor(
            synthesis_wavelength_angstrom, device=stokes.device, dtype=stokes.dtype
        )
        observed_wavelength = torch.as_tensor(
            observed_wavelength_angstrom, device=stokes.device, dtype=stokes.dtype
        )
        self._validate_wavelength(synthesis_wavelength, "synthesis_wavelength_angstrom")
        self._validate_wavelength(observed_wavelength, "observed_wavelength_angstrom")
        if stokes.ndim < 2 or stokes.shape[-2] != 4:
            raise ValueError(f"stokes must end in [4, wavelength]; got {tuple(stokes.shape)}.")
        if stokes.shape[-1] != synthesis_wavelength.numel():
            raise ValueError("stokes wavelength dimension does not match the synthesis grid.")
        epsilon = 8 * torch.finfo(stokes.dtype).eps
        if observed_wavelength[0] < synthesis_wavelength[0] - epsilon \
                or observed_wavelength[-1] > synthesis_wavelength[-1] + epsilon:
            raise ValueError("Observed wavelengths must lie inside the synthesis grid.")

        kernel, radius = self._kernel(synthesis_wavelength)
        original_shape = stokes.shape
        flat = stokes.reshape(-1, 1, original_shape[-1])
        padded = F.pad(flat, (radius, radius), mode="replicate")
        convolved = F.conv1d(padded, kernel.reshape(1, 1, -1))
        convolved = convolved.reshape(original_shape)
        return self._linear_sample(convolved, synthesis_wavelength, observed_wavelength)

    def metadata(self) -> dict:
        custom = bool(self.response_offsets_angstrom.numel())
        return {
            "type": "tabulated_spectral_response" if custom else "normalized_gaussian",
            "fwhm_angstrom": None if custom else self.fwhm_angstrom,
            "sigma_angstrom": None if custom else self.sigma_angstrom,
            "gaussian_fallback_fwhm_angstrom": self.fwhm_angstrom,
            "oversample": self.oversample,
            "truncate_sigma": self.truncate_sigma,
            "constant_preserving_boundary": "replicate",
            "provenance": self.provenance,
            "custom_response": (
                None
                if not custom
                else {
                    "offsets_angstrom": list(self._response_offsets_metadata),
                    "weights": list(self._response_weights_metadata),
                    "offset_convention": (
                        "kernel samples ordered from bluer to redder input displacement"
                    ),
                    "provenance": self.response_provenance,
                }
            ),
        }
