from pathlib import Path
import inspect
from types import SimpleNamespace

import pytest
import torch
import yaml

from pme.inversion_lte import _resolve_log_tau500, _resolve_physics_activation
from pme.train.lte_module import LTEModule


def test_log_tau_grid_resolution_is_ordered():
    grid = _resolve_log_tau500({"min": -5.0, "max": 1.0, "count": 13})
    assert len(grid) == 13 and grid[0] == -5.0 and grid[-1] == 1.0
    with pytest.raises(ValueError):
        _resolve_log_tau500([0.0, -1.0])


def test_physics_activation_only_recognizes_current_streams():
    volume, anchor = _resolve_physics_activation({
        "equations": {
            "magnetohydrostatic_equilibrium": {"enabled": True},
            "mean_radial_optical_depth_anchor": {"enabled": True},
        }
    })
    assert volume and anchor
    assert not _resolve_physics_activation({"equations": {}})[0]
    volume, anchor = _resolve_physics_activation({
        "equations": {
            "upper_boundary_gas_pressure_prior": {"enabled": True},
        }
    })
    assert not volume and anchor


def test_lte_module_api_has_learning_rate_and_no_legacy_mapping():
    parameters = inspect.signature(LTEModule).parameters
    assert "learning_rate" in parameters
    assert "lr_params" not in parameters
    assert "height_mapping_config" not in parameters


def test_coarse_to_fine_distances_retain_coarse_grid_and_concentrate_near_tau_one():
    distance = torch.linspace(0.0, 10.0, 6).expand(2, -1)
    alpha = torch.tensor(
        [[0.02, 0.03, 0.08, 0.4, 1.0, 1.0], [0.02, 0.03, 0.08, 0.4, 1.0, 1.0]]
    )
    refined = LTEModule._importance_refined_distances(alpha, distance, 8, 0.05)
    assert refined.shape == (2, 14)
    assert torch.all(refined[:, 1:] > refined[:, :-1])
    for coarse in distance[0]:
        assert torch.any(refined[0] == coarse)
    torch.testing.assert_close(refined[0], refined[1])


def test_auto_learning_rate_schedule_uses_estimated_optimizer_steps():
    class OptimizationHarness(torch.nn.Module):
        configure_optimizers = LTEModule.configure_optimizers

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


@pytest.mark.parametrize("name", ["lte_diagnostic_local.yaml", "lte_full_resolution.yaml"])
def test_supplied_configs_are_shell_only_and_fixed_weight(name):
    path = Path(__file__).parents[2] / "config" / "hinode" / name
    config = yaml.safe_load(path.read_text())
    learning_rate = config["training"]["learning_rate"]
    if name == "lte_full_resolution.yaml":
        assert learning_rate == {"start": 1.0e-3, "end": 1.0e-4, "iterations": "auto"}
        assert config["data"]["validation_batch_size"] == 64
    else:
        assert learning_rate == 1.0e-5
    assert "lr_params" not in config["training"]
    assert "coordinate_mode" not in config["atmosphere"]
    assert config["atmosphere"]["shell_height_bounds_Mm"] == [1.5, -0.1]
    equations = config["training"]["physics_config"]["equations"]
    refinement = config["training"]["depth_sampling_config"]["coarse_to_fine"]
    assert refinement["enabled"] is True
    assert refinement["fine_sample_count"] == 32
    physics = config["training"]["physics_config"]
    assert (
        physics["volume_points_per_step"]
        // physics["height_layers_per_step"]
        == 32
    )
    if name == "lte_full_resolution.yaml":
        assert physics["height_layers_per_step"] == 32
    assert set(equations) == {
        "magnetohydrostatic_equilibrium",
        "magnetic_divergence",
        "mean_radial_optical_depth_anchor",
        "upper_boundary_gas_pressure_prior",
    }
    assert equations["upper_boundary_gas_pressure_prior"]["weight"] == 1.0e-4
    assert all(
        item["weight"] == 1.0e-3
        for name, item in equations.items()
        if name != "upper_boundary_gas_pressure_prior"
    )
    assert all(isinstance(item["weight"], float) for item in equations.values())
