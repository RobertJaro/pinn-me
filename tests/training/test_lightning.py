"""Focused tests for the Lightning orchestration boundary."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from prom3theus.inversion.objective import StokesObjective
from prom3theus.inversion.constraints import MagnetofluidConstraints
from prom3theus.inversion.forward import (
    DepthRefinement,
    LTEForwardComposition,
)
from prom3theus.rt import StratifiedAtmosphereModel
from prom3theus.training.lightning import LTEInversionModule


def test_constructor_requires_the_explicit_lte_runtime_contract():
    parameters = inspect.signature(LTEInversionModule).parameters
    assert {
        "learning_rate",
        "observation_id",
        "velocity_synthesis_mode",
        "instrument_line_of_sight_velocity_correction_m_per_s",
        "vector_regularization_config",
        "run_metadata",
    } <= set(parameters)


def test_current_free_boundary_ramps_without_delaying_other_boundaries():
    harness = SimpleNamespace(
        upper_boundary_current_free_ramp_steps=100,
        _trainer=SimpleNamespace(global_step=9),
    )
    assert LTEInversionModule._current_free_boundary_factor(harness) == 0.1
    harness._trainer.global_step = 1000
    assert LTEInversionModule._current_free_boundary_factor(harness) == 1.0
    harness.upper_boundary_current_free_ramp_steps = 0
    assert LTEInversionModule._current_free_boundary_factor(harness) == 1.0


def test_current_free_boundary_validation_computes_derivatives_under_no_grad():
    model = StratifiedAtmosphereModel(
        torch.linspace(-5.0, 1.0, 3),
        scene_geometry_config={
            "solar_radius_m": 695_700_000.0,
            "scene_basis": torch.eye(3),
        },
        model_config={
            "type": "mlp",
            "dim": 8,
            "n_layers": 2,
            "activation": "silu",
            "encoding_config": {"type": "identity"},
        },
    )
    constraints = MagnetofluidConstraints(
        {
            "upper_boundary_current_free": {"enabled": True, "weight": 1.0e-3},
            "side_boundary_current_free": {"enabled": True, "weight": 1.0e-3},
        },
        vector_basis_matches_spatial_coordinates=True,
    )
    harness = SimpleNamespace(
        atmosphere_model=model,
        physics=constraints,
        _current_free_boundary_factor=lambda: 1.0,
    )
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.05, -0.03, 0.0]])
    top_height = coordinates.new_full((2,), model.shell_height_bounds_Mm[0] * 1.0e6)
    boundary = {
        "position_m": model.position_from_coords_height(coordinates, top_height),
        "time_hours": coordinates[:, 2:3],
    }
    side = {
        "position_m": boundary["position_m"].reshape(1, 2, 3),
        "time_hours": boundary["time_hours"].reshape(1, 2, 1),
        "normal": torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]]),
    }

    with torch.no_grad():
        losses, weighted = LTEInversionModule._physics_objective(
            harness,
            None,
            None,
            boundary,
            side,
            create_graph=False,
        )

    assert torch.isfinite(losses["upper_boundary_current_free"])
    assert torch.isfinite(losses["side_boundary_current_free"])
    assert torch.isfinite(weighted)


def test_shared_step_consumes_the_canonical_observation_batch_fields():
    class Harness:
        stokes_names = ("I", "Q", "U", "V")
        observation_id = "hinode_sp"
        velocity_synthesis_mode = SimpleNamespace(
            value="carrington_registered_relative"
        )
        instrument_type = "hinode_sp"
        atmosphere_model = torch.nn.Linear(1, 1)
        physics = SimpleNamespace(any_active=False)
        vector_regularization_enabled = False

        def synthesize(self, coordinates, **options):
            assert coordinates is batch["coordinates"]
            assert options["ray_direction"] is batch["ray_direction"]
            assert options["stokes_basis"] is batch["stokes_basis"]
            assert (
                options["removed_solar_los_velocity_m_per_s"]
                is batch["removed_solar_los_velocity_m_per_s"]
            )
            return {"stokes": batch["stokes"]}

        @staticmethod
        def _stokes_objective(prediction, target):
            assert prediction is target
            return torch.zeros(4), torch.zeros(())

        @staticmethod
        def log_dict(*args, **kwargs):
            del args, kwargs

    batch = {
        "coordinates": torch.zeros(2, 3),
        "ray_direction": torch.tensor([[0.0, 0.0, -1.0]]).expand(2, 3),
        "stokes_basis": torch.eye(3).expand(2, 3, 3),
        "removed_solar_los_velocity_m_per_s": torch.zeros(2),
        "stokes": torch.zeros(2, 4, 2),
    }
    loss = LTEInversionModule._shared_step(Harness(), batch, "valid")
    torch.testing.assert_close(loss, torch.zeros(()))

    malformed = {**batch, "coords": batch["coordinates"]}
    malformed.pop("coordinates")
    with pytest.raises(KeyError, match="coordinates"):
        LTEInversionModule._shared_step(Harness(), malformed, "valid")


def test_artifact_metadata_keeps_the_exact_physical_radiance_scale():
    radiance_scale = 30_612_345_678_901.0
    module = LTEInversionModule(
        log_tau500=[-5.0, 1.0],
        wavelength_angstrom=[6301.4, 6301.6],
        atmosphere_config={
            "scene_geometry_config": {
                "solar_radius_m": 695_700_000.0,
                "scene_basis": torch.eye(3),
            },
            "model_config": {
                "type": "mlp",
                "dim": 8,
                "n_layers": 2,
                "activation": "silu",
                "encoding_config": {"type": "identity"},
            },
        },
        synthesizer_config={"line_ids": ["FeI_6301.5008"]},
        instrument_config={
            "type": "hinode_sp",
            "spectral_psf": {
                "type": "gaussian",
                "fwhm_angstrom": 0.025,
                "oversample": 4,
                "truncate_sigma": 4.0,
            },
        },
        stokes_loss_config={
            "type": "huber",
            "stokes_sigmas": {"I": 5.0e-3, "Q": 2.0e-3, "U": 2.0e-3, "V": 2.0e-3},
            "huber_delta": 1.0,
        },
        weight_config={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0},
        wavelength_weights=None,
        wavelength_exclude_windows_angstrom=[],
        continuum_indices=[0, 1],
        atlas_continuum_radiance_w_m3_sr=radiance_scale,
        depth_sampling_config={
            "sample_count": 2,
            "coarse_to_fine": {
                "enabled": False,
                "fine_sample_count": 1,
                "uniform_weight_floor": 0.0,
            },
        },
        physics_config={
            "equations": {
                name: {"enabled": False, "weight": 0.0}
                for name in (
                    "hydrostatic_equilibrium",
                    "magnetohydrostatic_equilibrium",
                    "momentum",
                    "magnetic_divergence",
                    "induction",
                    "continuity",
                    "adiabatic_pressure",
                    "upper_boundary_open_velocity",
                    "side_boundary_open_velocity",
                    "side_boundary_current_free",
                    "upper_domain_microturbulence_prior",
                    "upper_domain_temperature_prior",
                    "radial_magnetic_energy_gradient",
                    "upper_boundary_current_free",
                    "upper_boundary_gas_pressure_prior",
                )
            },
            "gravity_m_per_s2": None,
            "adiabatic_index": 5.0 / 3.0,
            "upper_boundary_current_free_ramp_steps": 0,
            "volume_points_per_step": 0,
            "height_layers_per_step": 0,
            "upper_volume_points_per_step": 0,
            "upper_height_layers_per_step": 0,
            "upper_boundary_points_per_step": 0,
            "side_boundary_points_per_step": 0,
            "side_height_layers_per_step": 0,
            "validation_height_layers": 0,
            "validation_upper_height_layers": 0,
            "validation_points_per_height": 0,
            "sampling_domain": None,
            "upper_sampling_domain": None,
            "vector_basis_matches_spatial_coordinates": True,
            "normalization": {"length_m": 1.0e6, "time_s": 3600.0},
        },
        learning_rate={"start": 1.0e-4, "end": 1.0e-5, "iterations": "auto"},
        run_metadata={},
        observation_id="hinode-test",
        velocity_synthesis_mode="carrington_registered_relative",
        instrument_line_of_sight_velocity_correction_m_per_s=0.0,
        optimize_instrument_line_of_sight_velocity_correction=True,
        vector_regularization_config={
            "enabled": False,
            "magnetic_weight": 0.0,
            "velocity_weight": 0.0,
            "decay_steps": 0,
        },
    )

    assert module.atlas_continuum_radiance_w_m3_sr.dtype is torch.float32
    assert module.hparams["atlas_continuum_radiance_w_m3_sr"] == radiance_scale
    assert (
        module.synthesizer.continuum_opacity
        is module.atmosphere_model.thermodynamic_eos
    )
    assert module.instrument_line_of_sight_velocity_correction_normalized.requires_grad
    correction = module.instrument_line_of_sight_velocity_correction_m_per_s
    correction.add(1.0).square().backward()
    assert (
        module.instrument_line_of_sight_velocity_correction_normalized.grad is not None
    )


def test_vector_regularization_shrinks_only_nonzero_components():
    magnetic = torch.tensor([[[300.0, 0.0, 0.0]]], requires_grad=True)
    velocity = torch.tensor([[[0.0, 2000.0, 0.0]]], requires_grad=True)
    harness = SimpleNamespace(
        vector_regularization_enabled=True,
        vector_regularization_magnetic_weight=1.0e-2,
        vector_regularization_velocity_weight=2.0e-2,
        vector_regularization_decay_steps=100,
        atmosphere_model=SimpleNamespace(
            magnetic_scale_gauss=100.0,
            velocity_scale_m_per_s=1000.0,
        ),
        _trainer=SimpleNamespace(global_step=25),
    )
    harness._vector_regularization_factor = lambda: (
        LTEInversionModule._vector_regularization_factor(harness)
    )
    atmosphere = SimpleNamespace(
        magnetic_field=magnetic,
        velocity_field=velocity,
    )

    loss, factor = LTEInversionModule._vector_regularization(harness, atmosphere)
    assert factor == pytest.approx(0.75)
    expected = 0.75 * (1.0e-2 * 3.0 + 2.0e-2 * (4.0 / 3.0))
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()

    assert magnetic.grad[0, 0, 0] > 0
    assert velocity.grad[0, 0, 1] > 0
    torch.testing.assert_close(magnetic.grad[0, 0, 1:], torch.zeros(2))
    torch.testing.assert_close(velocity.grad[0, 0, [0, 2]], torch.zeros(2))


def test_observation_operator_receives_batch_local_response_mapping():
    class Backend:
        uses_prepared_wavelength = False

        @staticmethod
        def prepare_wavelength_grid(wavelength_angstrom):
            del wavelength_angstrom

        @staticmethod
        def reference_extinction(atmosphere):
            del atmosphere
            raise AssertionError("unused")

        @staticmethod
        def synthesize(atmosphere, wavelength_angstrom, *, radiance_scale, path):
            del atmosphere, wavelength_angstrom, radiance_scale, path
            raise AssertionError("unused")

    class Operator(torch.nn.Module):
        def forward(self, stokes, synthesis_wavelength, observed_wavelength, *, gain):
            del synthesis_wavelength, observed_wavelength
            return stokes * gain[..., None, None]

    composition = LTEForwardComposition(
        atmosphere_model=SimpleNamespace(),
        backend=Backend(),
        instrument=Operator(),
        velocity_synthesis_mode="carrington_registered_relative",
        depth_refinement=DepthRefinement(False, 1, 0.0),
    )
    stokes = torch.ones(2, 4, 2)
    result = composition.apply_instrument(
        stokes,
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
        {"gain": torch.tensor([2.0, 3.0])},
    )
    torch.testing.assert_close(result[0], torch.full((4, 2), 2.0))
    torch.testing.assert_close(result[1], torch.full((4, 2), 3.0))
    with pytest.raises(TypeError, match="mapping"):
        composition.apply_instrument(
            stokes,
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
            [torch.ones(())],
        )


def test_auto_learning_rate_schedule_uses_estimated_optimizer_steps():
    class OptimizationHarness(torch.nn.Module):
        configure_optimizers = LTEInversionModule.configure_optimizers

        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(()))
            self.learning_rate = 1.0e-3
            self.learning_rate_schedule = {
                "start": 1.0e-3,
                "end": 1.0e-4,
                "iterations": "auto",
            }
            self.resolved_learning_rate_iterations = None
            self.trainer = SimpleNamespace(estimated_stepping_batches=40)

    module = OptimizationHarness()
    configured = module.configure_optimizers()
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert configured["lr_scheduler"]["interval"] == "step"
    assert module.resolved_learning_rate_iterations == 40
    assert scheduler.gamma**40 == pytest.approx(0.1)


def test_stokes_weight_reduction_follows_prediction_dtype():
    harness = SimpleNamespace(
        wavelength_weights=torch.tensor([1.0, 1.0], dtype=torch.float32),
        stokes_weights=torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        stokes_loss=StokesObjective(
            stokes_sigmas={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0}
        ),
    )
    prediction = torch.zeros(2, 4, 2, dtype=torch.float64)
    target = torch.ones_like(prediction)

    component_loss, total = LTEInversionModule._stokes_objective(
        harness, prediction, target
    )

    assert component_loss.dtype is torch.float64
    assert total.dtype is torch.float64
    torch.testing.assert_close(total, torch.tensor(0.5, dtype=torch.float64))


def test_stokes_objective_rejects_nonfinite_synthesis_and_observations():
    harness = SimpleNamespace(
        wavelength_weights=torch.ones(2),
        stokes_weights=torch.ones(4),
        stokes_loss=StokesObjective(
            stokes_sigmas={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0}
        ),
    )
    finite = torch.zeros(1, 4, 2)
    invalid_prediction = finite.clone()
    invalid_prediction[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="LTE synthesis produced 1"):
        LTEInversionModule._stokes_objective(harness, invalid_prediction, finite)

    invalid_target = finite.clone()
    invalid_target[0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="Observed batch contains 1"):
        LTEInversionModule._stokes_objective(harness, finite, invalid_target)


def test_optimizer_guards_reject_nonfinite_parameters_and_gradients():
    class GuardHarness(torch.nn.Module):
        on_train_batch_start = LTEInversionModule.on_train_batch_start
        on_after_backward = LTEInversionModule.on_after_backward

        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.ones(()))

    harness = GuardHarness()
    harness.value.grad = torch.tensor(float("inf"))
    with pytest.raises(FloatingPointError, match="gradients"):
        harness.on_after_backward()
    assert harness.value.grad is None

    with torch.no_grad():
        harness.value.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="parameters"):
        harness.on_train_batch_start({}, 0)


def test_unclipped_gradient_diagnostic_logs_the_global_rank_maximum():
    class GradientHarness(torch.nn.Module):
        on_before_optimizer_step = LTEInversionModule.on_before_optimizer_step

        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.ones(2))
            self.logged = None

        def log(self, name, value, **options):
            self.logged = (name, value, options)

    harness = GradientHarness()
    harness.value.grad = torch.tensor([-3.0, 4.0])

    harness.on_before_optimizer_step(None)

    name, value, options = harness.logged
    assert name == "train.gradient_max_abs_unclipped"
    torch.testing.assert_close(value, torch.tensor(4.0))
    assert options["sync_dist"] is True
    assert options["reduce_fx"] == "max"
