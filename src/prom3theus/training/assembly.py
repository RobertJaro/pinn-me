"""Component assembly for the LTE training runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

import torch
from torch import nn

from prom3theus.instruments import build_instrument
from prom3theus.inversion.constraints.magnetofluid import (
    EQUATION_NAMES,
    MagnetofluidConstraints,
)
from prom3theus.inversion.forward import (
    DepthRefinement,
    LTEForwardComposition,
    LTESynthesisBackend,
)
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.rt import LTESynthesizer, StratifiedAtmosphereModel

from .configuration import DepthSamplingSettings


@dataclass(frozen=True, slots=True)
class ForwardAssembly:
    """The trainable LTE synthesis chain and its prepared wavelength grid."""

    synthesizer: LTESynthesizer
    instrument: nn.Module
    composition: LTEForwardComposition
    synthesis_wavelength_angstrom: torch.Tensor
    continuum_opacity: nn.Module | None


@dataclass(frozen=True, slots=True)
class PhysicsAssembly:
    """A physics objective plus its sampling and validation state."""

    constraints: MagnetofluidConstraints
    gravity_m_per_s2: float | None
    upper_boundary_current_free_ramp_steps: int
    volume_points_per_step: int
    height_layers_per_step: int
    upper_volume_points_per_step: int
    upper_height_layers_per_step: int
    upper_boundary_points_per_step: int
    side_boundary_points_per_step: int
    side_height_layers_per_step: int
    validation_height_layers: int
    validation_upper_height_layers: int
    validation_points_per_height: int
    sampling_domain_config: Mapping[str, Any] | None
    sampling_domain: SphericalShellDomain | None
    upper_sampling_domain_config: Mapping[str, Any] | None
    upper_sampling_domain: SphericalShellDomain | None
    validation_position_m: torch.Tensor
    validation_time_hours: torch.Tensor
    validation_upper_position_m: torch.Tensor
    validation_upper_time_hours: torch.Tensor
    validation_boundary_position_m: torch.Tensor
    validation_boundary_time_hours: torch.Tensor
    validation_side_position_m: torch.Tensor
    validation_side_time_hours: torch.Tensor
    validation_side_normal: torch.Tensor

    def model_configuration(self) -> dict[str, Any]:
        """Return the exact normalized physics constructor representation."""

        return {
            **self.constraints.configuration(),
            "gravity_m_per_s2": self.gravity_m_per_s2,
            "upper_boundary_current_free_ramp_steps": (
                self.upper_boundary_current_free_ramp_steps
            ),
            "volume_points_per_step": self.volume_points_per_step,
            "height_layers_per_step": self.height_layers_per_step,
            "upper_volume_points_per_step": self.upper_volume_points_per_step,
            "upper_height_layers_per_step": self.upper_height_layers_per_step,
            "upper_boundary_points_per_step": (self.upper_boundary_points_per_step),
            "side_boundary_points_per_step": self.side_boundary_points_per_step,
            "side_height_layers_per_step": self.side_height_layers_per_step,
            "validation_height_layers": self.validation_height_layers,
            "validation_upper_height_layers": self.validation_upper_height_layers,
            "validation_points_per_height": (self.validation_points_per_height),
            "sampling_domain": deepcopy(self.sampling_domain_config),
            "upper_sampling_domain": deepcopy(self.upper_sampling_domain_config),
        }


def build_forward_assembly(
    *,
    atmosphere_model: StratifiedAtmosphereModel,
    synthesizer_config: Mapping[str, Any],
    instrument_config: Mapping[str, Any],
    velocity_synthesis_mode: str,
    depth_sampling: DepthSamplingSettings,
    wavelength_angstrom: torch.Tensor,
) -> ForwardAssembly:
    """Build the differentiable atmosphere-to-observation composition."""

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
        depth_refinement=DepthRefinement(
            enabled=depth_sampling.coarse_to_fine_enabled,
            sample_count=depth_sampling.fine_sample_count,
            uniform_weight_floor=depth_sampling.uniform_weight_floor,
        ),
    )
    synthesis_wavelength = composition.prepare_wavelength_grid(wavelength_angstrom)
    return ForwardAssembly(
        synthesizer=synthesizer,
        instrument=instrument,
        composition=composition,
        synthesis_wavelength_angstrom=synthesis_wavelength,
        continuum_opacity=getattr(synthesizer, "continuum_opacity", None),
    )


def build_physics_assembly(
    config: Mapping[str, Any],
    *,
    reference_gravity_m_per_s2: float | None,
) -> PhysicsAssembly:
    """Build and validate the closed magnetofluid sampling configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("physics_config must be a mapping.")
    options = deepcopy(dict(config))
    expected = {
        "equations",
        "gravity_m_per_s2",
        "adiabatic_index",
        "upper_boundary_current_free_ramp_steps",
        "volume_points_per_step",
        "height_layers_per_step",
        "upper_volume_points_per_step",
        "upper_height_layers_per_step",
        "upper_boundary_points_per_step",
        "side_boundary_points_per_step",
        "side_height_layers_per_step",
        "validation_height_layers",
        "validation_upper_height_layers",
        "validation_points_per_height",
        "sampling_domain",
        "upper_sampling_domain",
        "vector_basis_matches_spatial_coordinates",
        "normalization",
    }
    if set(options) != expected:
        raise TypeError(
            "physics_config must match the exact current physics assembly contract."
        )
    equations = deepcopy(options.pop("equations"))
    if not isinstance(equations, Mapping) or set(equations) != set(EQUATION_NAMES):
        raise TypeError(
            "physics_config.equations must contain every current LTE equation."
        )
    configured_gravity = options.pop("gravity_m_per_s2")
    adiabatic_index = options.pop("adiabatic_index")
    if configured_gravity is not None and (
        not isinstance(configured_gravity, Real) or isinstance(configured_gravity, bool)
    ):
        raise TypeError("gravity_m_per_s2 must be numeric or null.")
    gravity_m_per_s2 = None if configured_gravity is None else float(configured_gravity)
    current_free_ramp_steps = options.pop("upper_boundary_current_free_ramp_steps")
    if not isinstance(current_free_ramp_steps, Integral) or isinstance(
        current_free_ramp_steps, bool
    ):
        raise TypeError("upper_boundary_current_free_ramp_steps must be an integer.")
    current_free_ramp_steps = int(current_free_ramp_steps)
    if current_free_ramp_steps < 0:
        raise ValueError("upper_boundary_current_free_ramp_steps cannot be negative.")

    def count(name: str) -> int:
        value = options.pop(name)
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
        return int(value)

    volume_points_per_step = count("volume_points_per_step")
    height_layers_per_step = count("height_layers_per_step")
    upper_volume_points_per_step = count("upper_volume_points_per_step")
    upper_height_layers_per_step = count("upper_height_layers_per_step")
    upper_boundary_points_per_step = count("upper_boundary_points_per_step")
    side_boundary_points_per_step = count("side_boundary_points_per_step")
    side_height_layers_per_step = count("side_height_layers_per_step")
    validation_height_layers = count("validation_height_layers")
    validation_upper_height_layers = count("validation_upper_height_layers")
    validation_points_per_height = count("validation_points_per_height")
    sampling_domain_config = options.pop("sampling_domain")
    if sampling_domain_config is not None and not isinstance(
        sampling_domain_config, Mapping
    ):
        raise TypeError("sampling_domain must be a mapping or null.")
    upper_sampling_domain_config = options.pop("upper_sampling_domain")
    if upper_sampling_domain_config is not None and not isinstance(
        upper_sampling_domain_config, Mapping
    ):
        raise TypeError("upper_sampling_domain must be a mapping or null.")
    vector_basis_matches = options.pop("vector_basis_matches_spatial_coordinates")
    normalization = options.pop("normalization")
    if options:  # pragma: no cover - guarded by the exact key contract above
        raise RuntimeError(f"Unconsumed physics options: {sorted(options)}")

    constraints = MagnetofluidConstraints(
        equations,
        gravity_m_per_s2=gravity_m_per_s2,
        adiabatic_index=adiabatic_index,
        vector_basis_matches_spatial_coordinates=vector_basis_matches,
        normalization=normalization,
    )
    if gravity_m_per_s2 is not None and (
        not math.isfinite(gravity_m_per_s2) or gravity_m_per_s2 <= 0
    ):
        raise ValueError("Force-balance gravity must be finite and positive.")
    if (
        reference_gravity_m_per_s2 is not None
        and gravity_m_per_s2 is not None
        and not math.isclose(
            gravity_m_per_s2,
            float(reference_gravity_m_per_s2),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "Configured physics gravity does not match the LTE opacity resource."
        )
    if min(volume_points_per_step, height_layers_per_step) < 0:
        raise ValueError("Physics volume and height-layer counts cannot be negative.")
    if constraints.volume_active:
        if (
            height_layers_per_step < 1
            or height_layers_per_step > volume_points_per_step
            or volume_points_per_step % height_layers_per_step != 0
        ):
            raise ValueError(
                "Active volume physics requires a positive volume sample count "
                "exactly divisible by the positive height-layer count."
            )
    else:
        volume_points_per_step = 0
        height_layers_per_step = 0
    if min(upper_volume_points_per_step, upper_height_layers_per_step) < 0:
        raise ValueError(
            "Upper-domain volume and height-layer counts cannot be negative."
        )
    if constraints.upper_volume_active:
        if (
            upper_height_layers_per_step < 1
            or upper_height_layers_per_step > upper_volume_points_per_step
            or upper_volume_points_per_step % upper_height_layers_per_step != 0
        ):
            raise ValueError(
                "Active upper-domain physics requires a positive upper-volume "
                "sample count divisible by its positive height-layer count."
            )
    else:
        upper_volume_points_per_step = 0
        upper_height_layers_per_step = 0
    if upper_boundary_points_per_step < 0:
        raise ValueError("Upper-boundary sample count cannot be negative.")
    if constraints.upper_boundary_active and upper_boundary_points_per_step < 1:
        raise ValueError(
            "Active boundary physics requires a positive upper-boundary sample count."
        )
    if not constraints.upper_boundary_active:
        upper_boundary_points_per_step = 0
    if min(side_boundary_points_per_step, side_height_layers_per_step) < 0:
        raise ValueError("Side-boundary sample counts cannot be negative.")
    if constraints.side_boundary_active:
        if (
            side_height_layers_per_step < 1
            or side_boundary_points_per_step % side_height_layers_per_step != 0
            or side_boundary_points_per_step // side_height_layers_per_step < 4
            or (side_boundary_points_per_step // side_height_layers_per_step) % 4
            != 0
        ):
            raise ValueError(
                "Active side-boundary physics requires a positive sample count "
                "divisible into height layers with equal points on four faces."
            )
    else:
        side_boundary_points_per_step = 0
        side_height_layers_per_step = 0
    if validation_height_layers < 0:
        raise ValueError("Physics validation height-layer count cannot be negative.")
    if constraints.volume_active and validation_height_layers < 2:
        raise ValueError(
            "Active volume physics validation requires at least two height layers."
        )
    if not constraints.volume_active and not constraints.side_boundary_active:
        validation_height_layers = 0
    if validation_upper_height_layers < 0:
        raise ValueError(
            "Upper-domain validation height-layer count cannot be negative."
        )
    if constraints.upper_volume_active and validation_upper_height_layers < 2:
        raise ValueError(
            "Active upper-domain validation requires at least two height layers."
        )
    if not constraints.upper_volume_active:
        validation_upper_height_layers = 0
    if validation_points_per_height < 0 or (
        constraints.any_active and validation_points_per_height < 1
    ):
        raise ValueError(
            "Physics validation points must be non-negative and positive when "
            "an equation is enabled."
        )
    if not constraints.any_active:
        validation_points_per_height = 0
    if constraints.side_boundary_active and (
        validation_points_per_height < 4
        or validation_points_per_height % 4 != 0
    ):
        raise ValueError(
            "Side-boundary validation points per height must be at least four and "
            "divisible by four."
        )

    sampling_domain = (
        None
        if sampling_domain_config is None
        else SphericalShellDomain.from_mapping(sampling_domain_config)
    )
    upper_sampling_domain = (
        None
        if upper_sampling_domain_config is None
        else SphericalShellDomain.from_mapping(upper_sampling_domain_config)
    )
    if constraints.upper_volume_active and upper_sampling_domain is None:
        raise ValueError(
            "Active upper-domain physics requires an initialized upper sampling domain."
        )
    if constraints.side_boundary_active and sampling_domain is None:
        raise ValueError(
            "Active side-boundary physics requires an initialized sampling domain."
        )
    validation_volume = {
        "position_m": torch.empty(0, 0, 3, dtype=torch.float32),
        "time_hours": torch.empty(0, 0, 1, dtype=torch.float32),
    }
    validation_boundary = {
        "position_m": torch.empty(0, 3, dtype=torch.float32),
        "time_hours": torch.empty(0, 1, dtype=torch.float32),
    }
    validation_upper = {
        "position_m": torch.empty(0, 0, 3, dtype=torch.float32),
        "time_hours": torch.empty(0, 0, 1, dtype=torch.float32),
    }
    validation_side = {
        "position_m": torch.empty(0, 0, 3, dtype=torch.float32),
        "time_hours": torch.empty(0, 0, 1, dtype=torch.float32),
        "normal": torch.empty(0, 0, 3, dtype=torch.float32),
    }
    if sampling_domain is not None and constraints.volume_active:
        validation_volume = sampling_domain.deterministic_grouped(
            validation_height_layers,
            validation_points_per_height,
        )
    if sampling_domain is not None and constraints.upper_boundary_active:
        grouped_boundary = sampling_domain.deterministic_top(
            validation_points_per_height
        )
        validation_boundary = {
            name: value.squeeze(0) for name, value in grouped_boundary.items()
        }
    if upper_sampling_domain is not None and constraints.upper_volume_active:
        validation_upper = upper_sampling_domain.deterministic_grouped(
            validation_upper_height_layers,
            validation_points_per_height,
        )
    if sampling_domain is not None and constraints.side_boundary_active:
        validation_side = sampling_domain.deterministic_sides(
            validation_height_layers,
            validation_points_per_height,
        )
    return PhysicsAssembly(
        constraints=constraints,
        gravity_m_per_s2=gravity_m_per_s2,
        upper_boundary_current_free_ramp_steps=current_free_ramp_steps,
        volume_points_per_step=volume_points_per_step,
        height_layers_per_step=height_layers_per_step,
        upper_volume_points_per_step=upper_volume_points_per_step,
        upper_height_layers_per_step=upper_height_layers_per_step,
        upper_boundary_points_per_step=upper_boundary_points_per_step,
        side_boundary_points_per_step=side_boundary_points_per_step,
        side_height_layers_per_step=side_height_layers_per_step,
        validation_height_layers=validation_height_layers,
        validation_upper_height_layers=validation_upper_height_layers,
        validation_points_per_height=validation_points_per_height,
        sampling_domain_config=sampling_domain_config,
        sampling_domain=sampling_domain,
        upper_sampling_domain_config=upper_sampling_domain_config,
        upper_sampling_domain=upper_sampling_domain,
        validation_position_m=validation_volume["position_m"],
        validation_time_hours=validation_volume["time_hours"].to(torch.float32),
        validation_upper_position_m=validation_upper["position_m"],
        validation_upper_time_hours=validation_upper["time_hours"].to(torch.float32),
        validation_boundary_position_m=validation_boundary["position_m"],
        validation_boundary_time_hours=validation_boundary["time_hours"].to(
            torch.float32
        ),
        validation_side_position_m=validation_side["position_m"],
        validation_side_time_hours=validation_side["time_hours"].to(torch.float32),
        validation_side_normal=validation_side["normal"].to(torch.float32),
    )


__all__ = [
    "ForwardAssembly",
    "PhysicsAssembly",
    "build_forward_assembly",
    "build_physics_assembly",
]
