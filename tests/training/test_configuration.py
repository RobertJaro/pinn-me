"""Unit tests for the strict LTE training configuration boundary."""

from __future__ import annotations

import pytest
import torch

from prom3theus.training.assembly import build_physics_assembly
from prom3theus.training.configuration import (
    resolve_coordinate_grids,
    resolve_depth_sampling,
    resolve_learning_rate,
    resolve_objective_weighting,
    resolve_vector_regularization,
)


def test_coordinate_grids_are_float32_and_strictly_increasing():
    grids = resolve_coordinate_grids([-5.0, -1.0, 1.0], [6301.0, 6302.0])

    assert grids.log_tau500.dtype == torch.float32
    assert grids.wavelength_angstrom.dtype == torch.float32
    with pytest.raises(ValueError, match="top to bottom"):
        resolve_coordinate_grids([-5.0, -5.0], [6301.0, 6302.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_coordinate_grids([-5.0, 1.0], [6302.0, 6301.0])


def test_depth_and_vector_option_sets_reject_unknown_fields():
    depth = {
        "sample_count": 8,
        "coarse_to_fine": {
            "enabled": True,
            "fine_sample_count": 4,
            "uniform_weight_floor": 0.05,
        },
    }
    vector = {
        "enabled": True,
        "magnetic_weight": 1.0,
        "velocity_weight": 0.0,
        "decay_steps": 10,
    }
    with pytest.raises(TypeError, match="coarse_to_fine"):
        resolve_depth_sampling(
            {
                **depth,
                "coarse_to_fine": {
                    **depth["coarse_to_fine"],
                    "unexpected": True,
                },
            },
        )
    with pytest.raises(TypeError, match="vector_regularization_config"):
        resolve_vector_regularization({**vector, "unexpected": 1.0})
    with pytest.raises(ValueError, match="positive magnetic or velocity"):
        resolve_vector_regularization(
            {**vector, "magnetic_weight": 0.0, "velocity_weight": 0.0}
        )
    with pytest.raises(TypeError, match="sample_count must be an integer"):
        resolve_depth_sampling({**depth, "sample_count": 8.5})
    with pytest.raises(TypeError, match="enabled must be boolean"):
        resolve_vector_regularization({**vector, "enabled": "true"})


def test_physics_collocation_counts_are_strict_integers():
    physics = {
        "equations": {
            name: {"enabled": False, "weight": 0.0}
            for name in (
                "hydrostatic_equilibrium",
                "magnetohydrostatic_equilibrium",
                "momentum",
                "magnetic_divergence",
                "induction",
                "continuity",
                "upper_boundary_gas_pressure_prior",
            )
        },
        "gravity_m_per_s2": None,
        "volume_points_per_step": 16.5,
        "height_layers_per_step": 4,
        "upper_boundary_points_per_step": 4,
        "validation_height_layers": 2,
        "validation_points_per_height": 4,
        "sampling_domain": None,
        "vector_basis_matches_spatial_coordinates": True,
        "normalization": {"length_m": 1.0e6, "time_s": 3600.0},
    }
    with pytest.raises(TypeError, match="volume_points_per_step must be an integer"):
        build_physics_assembly(
            physics,
            reference_gravity_m_per_s2=None,
        )


def test_objective_weighting_applies_exclusions_and_preserves_component_order():
    wavelength = torch.tensor([6301.0, 6301.5, 6302.0])
    weighting = resolve_objective_weighting(
        wavelength,
        weight_config={"V": 4.0, "I": 1.0, "Q": 2.0, "U": 3.0},
        wavelength_weights=[1.0, 2.0, 3.0],
        wavelength_exclude_windows_angstrom=[[6301.4, 6301.6]],
        continuum_indices=[0, 2],
        atlas_continuum_radiance_w_m3_sr=3.06e13,
    )

    torch.testing.assert_close(
        weighting.stokes_weights, torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    torch.testing.assert_close(
        weighting.wavelength_weights, torch.tensor([1.0, 0.0, 3.0])
    )
    assert weighting.wavelength_exclude_windows_angstrom == [[6301.4, 6301.6]]
    with pytest.raises(KeyError, match="Stokes weights must contain exactly"):
        resolve_objective_weighting(
            wavelength,
            weight_config={"I": 1.0, "Q": 1.0, "U": 1.0, "R": 1.0},
            wavelength_weights=None,
            wavelength_exclude_windows_angstrom=[],
            continuum_indices=[0, 2],
            atlas_continuum_radiance_w_m3_sr=3.06e13,
        )
    with pytest.raises(TypeError, match="continuum_indices must contain integers"):
        resolve_objective_weighting(
            wavelength,
            weight_config={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0},
            wavelength_weights=None,
            wavelength_exclude_windows_angstrom=None,
            continuum_indices=[0.5, 2.0],
            atlas_continuum_radiance_w_m3_sr=3.06e13,
        )


def test_learning_rate_is_an_exact_closed_schedule():
    scheduled = resolve_learning_rate(
        {"start": 1.0e-3, "end": 1.0e-4, "iterations": "auto"}
    )
    assert scheduled.schedule == {
        "start": 1.0e-3,
        "end": 1.0e-4,
        "iterations": "auto",
    }
    assert scheduled.model_configuration == scheduled.schedule
    with pytest.raises(TypeError, match="contain exactly"):
        resolve_learning_rate(
            {
                "start": 1.0e-3,
                "end": 1.0e-4,
                "iterations": 10,
                "warmup": 2,
            }
        )
    with pytest.raises(TypeError, match="schedule mapping"):
        resolve_learning_rate(1.0e-4)
    with pytest.raises(ValueError, match="positive or 'auto'"):
        resolve_learning_rate({"start": 1.0e-3, "end": 1.0e-4, "iterations": True})
