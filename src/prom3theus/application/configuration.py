"""Translate typed scientific configuration into shared model constructor options."""

from typing import Any

from prom3theus.config.schema import AtmosphereConfig


def atmosphere_options(atmosphere: AtmosphereConfig) -> dict[str, Any]:
    geometry = atmosphere.geometry
    parameters = atmosphere.parameters
    network = atmosphere.network
    return {
        "height_input_scale_m": geometry.height_input_scale_m,
        "uniform_spatial_scaling": geometry.uniform_spatial_scaling,
        "time_dependent": geometry.time_dependent,
        "shell_height_bounds_Mm": [
            geometry.outer_height_megameter,
            geometry.inner_height_megameter,
        ],
        "line_formation_height_bounds_Mm": [
            geometry.line_formation_outer_height_megameter,
            geometry.inner_height_megameter,
        ],
        "tangent_margin_m": geometry.tangent_margin_m,
        "reference_atmosphere_config": atmosphere.reference_atmosphere,
        "upper_atmosphere_config": (
            None
            if atmosphere.upper_atmosphere is None
            else atmosphere.upper_atmosphere.to_dict()
        ),
        "temperature_log_scale": parameters.temperature.log_scale,
        "velocity_scale_m_per_s": parameters.velocity.scale_m_per_s,
        "velocity_max_m_per_s": parameters.velocity.maximum_m_per_s,
        "magnetic_scale_gauss": parameters.magnetic_field.scale_gauss,
        "magnetic_representation": parameters.magnetic_field.representation,
        "magnetic_potential_delta_cool_steps": (
            parameters.magnetic_field.potential_delta_cool_steps
        ),
        "magnetic_potential_delta_ramp_steps": (
            parameters.magnetic_field.potential_delta_ramp_steps
        ),
        **({
            "magnetic_reference_height_megameter": parameters.magnetic_field.reference_height_megameter,
        } if parameters.magnetic_field.reference_height_megameter is not None else {}),
        "microturbulence_log_scale": parameters.microturbulence.log_scale,
        "gas_pressure_log_scale": parameters.gas_pressure.log_scale,
        "model_config": (
            {
                "type": "siren",
                "dim": network.hidden_dimension,
                "n_layers": network.hidden_layers,
                "first_omega_0": network.first_omega_0,
                "hidden_omega_0": network.hidden_omega_0,
                "radial_weighting_config": (network.radial_weighting.to_dict() if network.radial_weighting is not None else None),
            }
            if network.type == "siren"
            else {
                "type": "mlp",
                "dim": network.hidden_dimension,
                "n_layers": network.hidden_layers,
                "activation": network.activation,
                "encoding_config": network.encoding.to_dict(),
            }
        ),
    }
