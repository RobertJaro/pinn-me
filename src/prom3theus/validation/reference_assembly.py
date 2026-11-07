"""Development-only construction of numerical synthesis reference fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from prom3theus.instruments import build_instrument
from prom3theus.inversion.depth_sampling import DepthRefinement
from prom3theus.validation.reference_configuration import DepthSamplingSettings
from prom3theus.rt import StratifiedAtmosphereModel
from .reference_forward import LTEForwardComposition, LTESynthesisBackend
from .reference_synthesis import LTESynthesizer

@dataclass(frozen=True, slots=True)
class ForwardAssembly:
    """Numerical reference chain and prepared wavelength grid for validation."""

    synthesizer: LTESynthesizer
    instrument: nn.Module
    composition: LTEForwardComposition
    synthesis_wavelength_angstrom: torch.Tensor
    continuum_opacity: nn.Module | None


def build_forward_assembly(
    *,
    atmosphere_model: StratifiedAtmosphereModel,
    synthesizer_config: Mapping[str, Any],
    instrument_config: Mapping[str, Any],
    velocity_synthesis_mode: str,
    depth_sampling: DepthSamplingSettings,
    wavelength_angstrom: torch.Tensor,
) -> ForwardAssembly:
    """Build an independent differentiable numerical comparison fixture."""

    synthesizer = LTESynthesizer(
        # The atmosphere is a continuous coordinate model. Training supplies
        # the grid carried by each stratified atmosphere realization.
        log_tau500=None,
        continuum_opacity=getattr(atmosphere_model, "thermodynamic_eos", None),
        **synthesizer_config,
    )
    instrument = build_instrument(instrument_config)
    composition = LTEForwardComposition(
        atmosphere_model=atmosphere_model,
        backend=LTESynthesisBackend(synthesizer),
        instrument=instrument,
        velocity_synthesis_mode=velocity_synthesis_mode,
        reference_log_tau500_bounds=depth_sampling.reference_log_tau500_bounds,
        height_sampling=depth_sampling.height_sampling,
        depth_refinement=DepthRefinement(
            enabled=depth_sampling.coarse_to_fine_enabled,
            sample_count=depth_sampling.fine_sample_count,
            uniform_weight_floor=depth_sampling.uniform_weight_floor,
        ),
    )
    # Prepare small spectral differences before casting runtime buffers. Casting
    # absolute wavelengths near 6000 A first loses Doppler shifts of tens of m/s.
    synthesis_wavelength = composition.prepare_wavelength_grid(
        wavelength_angstrom.to(dtype=torch.float64)
    )
    runtime_reference = getattr(
        atmosphere_model, "solar_radius_m", torch.empty((), dtype=torch.float32)
    )
    synthesizer.to(runtime_reference)
    instrument.to(runtime_reference)
    return ForwardAssembly(
        synthesizer=synthesizer,
        instrument=instrument,
        composition=composition,
        synthesis_wavelength_angstrom=synthesis_wavelength,
        continuum_opacity=getattr(synthesizer, "continuum_opacity", None),
    )
