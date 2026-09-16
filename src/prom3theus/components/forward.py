"""Built-in forward providers; reconstruction uses constructor contracts only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from prom3theus.config.joint_schema import ObservationStreamConfig
from prom3theus.instruments import resolve_instrument_config
from prom3theus.inversion.data_terms import AIAEUVObservationTerm, StokesObservationTerm


def _stokes_instrument_options(
    stream: ObservationStreamConfig, specification: Any
) -> tuple[dict[str, Any], float, bool]:
    instrument = stream.data_term.instrument.to_dict()
    velocity = float(instrument.pop("line_of_sight_velocity_correction_m_per_s"))
    optimize_velocity = bool(
        instrument.pop("optimize_line_of_sight_velocity_correction")
    )
    for name, value in specification.instrument_options.items():
        if name in instrument and instrument[name] != value:
            raise ValueError(
                f"Configured instrument option {name!r} conflicts with the observation."
            )
        instrument[name] = value
    return resolve_instrument_config(instrument), velocity, optimize_velocity


def _build_stokes(stream, loaded, atmosphere_model, scene):
    data_term = stream.data_term
    synthesis = data_term.synthesis.to_dict()
    excluded = synthesis.pop("excluded_wavelength_windows_angstrom")
    objective = data_term.objective.to_dict()
    stokes_weights = {
        component.upper(): value
        for component, value in objective.pop("stokes_weights").items()
    }
    instrument, correction, optimize_correction = _stokes_instrument_options(
        stream, loaded.specification
    )
    specification = loaded.specification
    options = dict(
        atmosphere_model=atmosphere_model,
        observation_id=specification.observation_id,
        wavelength_angstrom=specification.wavelength_angstrom,
        continuum_indices=specification.continuum_indices,
        atlas_continuum_radiance_w_m3_sr=(specification.radiance_scale_w_m3_sr),
        velocity_synthesis_mode=specification.velocity_synthesis_mode.value,
        synthesizer_config=synthesis,
        instrument_config=instrument,
        objective_config=objective,
        weight_config=stokes_weights,
        wavelength_exclude_windows_angstrom=excluded,
        depth_sampling_config=data_term.depth_sampling.to_dict(),
        instrument_line_of_sight_velocity_correction_m_per_s=correction,
        optimize_instrument_line_of_sight_velocity_correction=(optimize_correction),
        disambiguation_config=data_term.disambiguation.to_dict(),
    )
    term = StokesObservationTerm(**options)
    term.construction = {
        "type": "lte_stokes",
        "options": {k: v for k, v in options.items() if k != "atmosphere_model"},
    }
    return term


def _build_aia(stream, loaded, atmosphere_model, scene):
    data_term = stream.data_term
    objective = data_term.objective
    channel_weights_by_id = {
        item.channel_angstrom: item.weight for item in objective.channel_weights
    }
    calibration = objective.calibration
    options = dict(
        atmosphere_model=atmosphere_model,
        scene=scene,
        observation_id=loaded.specification.observation_id,
        channels_angstrom=stream.observation.channels_angstrom,
        response_resource=data_term.synthesis.response_resource,
        ray_samples=data_term.synthesis.ray_samples,
        coarse_to_fine=data_term.synthesis.coarse_to_fine.to_dict(),
        training_jitter=data_term.synthesis.training_jitter,
        height_sampling_power=data_term.synthesis.height_sampling_power,
        asinh_scales=loaded.setup_metadata["asinh_scales"],
        intensity_scales=loaded.setup_metadata["intensity_scales"],
        channel_weights=[
            channel_weights_by_id[channel]
            for channel in stream.observation.channels_angstrom
        ],
        calibration_enabled=calibration.enabled,
        calibration_absolute_prior_fraction=calibration.absolute_prior_fraction,
        calibration_relative_prior_fraction=calibration.relative_prior_fraction,
    )
    term = AIAEUVObservationTerm(**options)
    term.construction = {
        "type": "aia_optically_thin",
        "options": {
            k: v for k, v in options.items() if k not in ("atmosphere_model", "scene")
        },
    }
    return term


@dataclass(frozen=True)
class ForwardProvider:
    build: object
    reconstruct: object


def _restore_stokes(options, atmosphere, scene):
    return StokesObservationTerm(atmosphere_model=atmosphere, **options)


def _restore_aia(options, atmosphere, scene):
    return AIAEUVObservationTerm(atmosphere_model=atmosphere, scene=scene, **options)


_PROVIDERS = {
    "lte_stokes": ForwardProvider(_build_stokes, _restore_stokes),
    "aia_optically_thin": ForwardProvider(_build_aia, _restore_aia),
}


def register_forward_component(name, provider):
    if name in _PROVIDERS:
        raise ValueError(f"Forward component already registered: {name}")
    if not isinstance(provider, ForwardProvider):
        raise TypeError("Expected ForwardProvider")
    _PROVIDERS[name] = provider


def _default_term_builder(stream, loaded, atmosphere_model, scene):
    return _PROVIDERS[stream.data_term.type].build(
        stream, loaded, atmosphere_model, scene
    )


def reconstruct_term(contract, atmosphere, scene):
    term = _PROVIDERS[contract["type"]].reconstruct(
        dict(contract["options"]), atmosphere, scene
    )
    term.construction = contract
    return term


def predict_term(term, batch):
    parameter = next(term.parameters())

    def move(value):
        import torch

        if isinstance(value, torch.Tensor):
            return value.to(device=parameter.device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    return term.predict(move(batch))
