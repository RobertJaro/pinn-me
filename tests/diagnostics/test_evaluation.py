"""Strict option parsing for atmosphere diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prom3theus.diagnostics.evaluation import (
    AtmosphereEvaluator,
    AtmosphereSampling,
)


def test_sampling_counts_are_not_silently_truncated():
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        AtmosphereSampling.from_options(ray_sampling={"batch_size": 1.5})


def test_meridional_enabled_requires_an_actual_boolean():
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        AtmosphereSampling.from_options(
            meridional_slice={"enabled": "false", "longitude_deg": 0.0}
        )


def test_ray_grid_indices_must_be_supplied_as_a_pair():
    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options())

    with pytest.raises(ValueError, match="supplied together"):
        evaluator.evaluate_ray_optical_depth(object(), object(), rows=None, columns=[0])


def test_shell_height_levels_must_be_finite():
    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options())

    with pytest.raises(ValueError, match="finite vector"):
        evaluator.evaluate_shell_layers(object(), object(), [0.0, float("nan")])


def test_physical_slice_density_uses_the_hybrid_atmosphere_eos():
    class HybridEOS:
        @staticmethod
        def mass_density(temperature, pressure):
            assert torch.all(temperature == 1.0e6)
            return pressure * 2.0

    class Atmosphere(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.thermodynamic_eos = HybridEOS()

        def evaluate_position_points(self, position):
            count = position.shape[0]
            scalar = self.scale.expand(count)
            vector = position * 0.0
            return {
                "temperature": scalar * 1.0e6,
                "gas_pressure": scalar * 0.25,
                "microturbulence": scalar * 1.0e3,
                "magnetic_field": vector,
                "velocity_field": vector,
            }

    class StrictSTICOpacity:
        @staticmethod
        def reference_mass_density(*args, **kwargs):
            del args, kwargs
            raise AssertionError("diagnostics must not bypass the hybrid EoS")

    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options())
    module = SimpleNamespace(
        atmosphere_model=Atmosphere(),
        synthesizer=SimpleNamespace(continuum_opacity=StrictSTICOpacity()),
    )
    fields = evaluator._evaluate_physical_positions(
        module,
        torch.tensor([[695_700_000.0, 0.0, 0.0], [715_700_000.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(fields["density"], 0.5)
    np.testing.assert_allclose(fields["current_density"], 0.0)


def test_current_density_is_curl_b_over_mu0_in_si_units():
    class HybridEOS:
        @staticmethod
        def mass_density(temperature, pressure):
            return pressure / temperature

    class Atmosphere(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.thermodynamic_eos = HybridEOS()

        def evaluate_position_points(self, position):
            count = position.shape[0]
            scalar = self.scale.expand(count)
            # B_y = x [G] gives curl(B)_z = 1 G m^-1.
            magnetic = torch.stack(
                (torch.zeros_like(position[:, 0]), position[:, 0], torch.zeros_like(position[:, 0])),
                dim=-1,
            )
            return {
                "temperature": scalar * 6_000.0,
                "gas_pressure": scalar,
                "microturbulence": scalar * 1_000.0,
                "magnetic_field": magnetic,
                "velocity_field": position * 0.0,
            }

    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options())
    fields = evaluator._evaluate_physical_positions(
        SimpleNamespace(atmosphere_model=Atmosphere()),
        torch.tensor([[695_700_000.0, 0.0, 0.0], [695_700_100.0, 0.0, 0.0]]),
    )

    expected = 1.0e-4 / (4.0 * np.pi * 1.0e-7)
    np.testing.assert_allclose(fields["current_density"], expected, rtol=1.0e-6)


def test_physical_slices_include_observer_magnetic_angles_and_los_velocity():
    class HybridEOS:
        @staticmethod
        def mass_density(temperature, pressure):
            return pressure / temperature

    class Atmosphere(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.thermodynamic_eos = HybridEOS()

        def evaluate_position_points(self, position):
            count = position.shape[0]
            scalar = self.scale.expand(count)
            connected = position * 0.0
            magnetic = connected + position.new_tensor((1.0, 2.0, 3.0))
            velocity = connected + position.new_tensor((4_000.0, 5_000.0, 6_000.0))
            return {
                "temperature": scalar * 6_000.0,
                "gas_pressure": scalar,
                "microturbulence": scalar * 1_000.0,
                "magnetic_field": magnetic,
                "velocity_field": velocity,
            }

    observer_basis = torch.tensor(
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    evaluator = AtmosphereEvaluator(AtmosphereSampling.from_options())
    fields = evaluator._evaluate_physical_positions(
        SimpleNamespace(atmosphere_model=Atmosphere()),
        torch.tensor([[695_700_000.0, 0.0, 0.0]]),
        observer_basis=observer_basis,
    )

    np.testing.assert_allclose(
        fields["field_strength"][0],
        np.sqrt(14.0),
    )
    np.testing.assert_allclose(
        fields["inclination"][0],
        np.rad2deg(np.arccos(3.0 / np.sqrt(14.0))),
        rtol=2.0e-7,
    )
    np.testing.assert_allclose(
        fields["azimuth"][0],
        np.rad2deg(np.arctan2(-1.0, 2.0)),
    )
    np.testing.assert_allclose(fields["v_toward"][0], 6.0)
