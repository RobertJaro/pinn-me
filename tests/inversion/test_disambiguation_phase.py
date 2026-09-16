import math

import pytest
import torch
from torch import nn

from prom3theus.config import DisambiguationConfig
from prom3theus.core import SIRENModel
from prom3theus.inversion.data_terms.disambiguation import ObserverPhaseRotation
from prom3theus.inversion.data_terms.stokes import StokesObservationTerm
from prom3theus.inversion.forward import rotate_transverse_stokes_field


def _phase(*, time_dependent=False):
    return ObserverPhaseRotation(
        spatial_coordinate_center_mm=(2.0, -3.0),
        spatial_coordinate_scale_mm=(10.0, 20.0),
        time_dependent=time_dependent,
        time_coordinate_center_hours=5.0,
        time_coordinate_scale_hours=2.0,
        hidden_dimension=16,
        hidden_layers=2,
        first_omega_0=1.0,
        hidden_omega_0=1.0,
    )


def test_phase_network_scales_raw_output_by_pi_and_is_randomly_initialized():
    torch.manual_seed(4)
    phase = _phase()
    coordinates = torch.tensor(
        [[2.0, -3.0, 5.0], [4.0, 7.0, 5.0]],
    )

    raw = phase.raw_output(coordinates)
    actual = phase(coordinates)

    torch.testing.assert_close(actual, raw * math.pi)
    assert torch.any(raw != 0.0)
    assert torch.any(phase.network.out_layer.weight != 0.0)
    assert torch.any(phase.network.in_layer.weight != 0.0)
    assert torch.isfinite(actual).all()
    assert isinstance(phase.network, SIRENModel)
    assert phase.metadata()["network"]["type"] == "siren"
    assert phase.metadata()["network"]["output_initialization"] == "random"


def test_phase_output_head_receives_a_gradient():
    torch.manual_seed(4)
    phase = _phase()
    coordinates = torch.tensor(
        [[2.0, -3.0, 5.0], [4.0, 7.0, 5.0]],
    )

    loss = phase(coordinates).sum()
    loss.backward()

    assert phase.network.out_layer.weight.grad is not None
    assert torch.any(phase.network.out_layer.weight.grad != 0.0)


def test_phase_models_are_independently_randomly_initialized():
    torch.manual_seed(11)
    first = _phase()
    torch.manual_seed(12)
    second = _phase()

    assert any(
        not torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        )
    )


def test_dynamic_phase_uses_surface_time_but_not_ray_depth():
    torch.manual_seed(5)
    phase = _phase(time_dependent=True)
    coordinates = torch.tensor(
        [[2.0, -3.0, 5.0], [2.0, -3.0, 7.0]],
    )

    values = phase(coordinates)

    assert values.shape == (2,)
    assert not torch.allclose(values[0], values[1])


def test_transverse_rotation_preserves_los_and_magnitude_and_broadcasts_depth():
    field = torch.tensor(
        [[[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]],
    )

    rotated = rotate_transverse_stokes_field(field, math.pi / 2.0)
    expected = torch.tensor(
        [[[-4.0, 3.0, 5.0], [-4.0, 3.0, 5.0]]],
    )

    torch.testing.assert_close(rotated, expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated[..., :2], dim=-1),
        torch.linalg.vector_norm(field[..., :2], dim=-1),
    )
    torch.testing.assert_close(rotated[..., 2], field[..., 2])


def test_pi_rotation_is_the_180_degree_branch():
    field = torch.tensor([[3.0, 4.0, 5.0]])

    rotated = rotate_transverse_stokes_field(field, math.pi)

    torch.testing.assert_close(rotated, torch.tensor([[-3.0, -4.0, 5.0]]))


def test_phase_schedule_has_cold_start_warmup_then_fixed_handoff():
    term = StokesObservationTerm.__new__(StokesObservationTerm)
    nn.Module.__init__(term)
    term.disambiguation_options = {
        "enabled": True,
        "cold_steps": 2_000,
        "warmup_steps": 5_000,
        "handoff_step": 15_000,
        "binary_weight": 0.01,
        "hidden_dimension": 16,
        "hidden_layers": 2,
        "first_omega_0": 1.0,
        "hidden_omega_0": 1.0,
    }
    term.phase_rotation = nn.Identity()

    for step, active, expected_weight in (
        (0, True, 0.0),
        (1_999, True, 0.0),
        (2_000, True, 0.0),
        (4_500, True, 0.005),
        (7_000, True, 0.01),
        (14_999, True, 0.01),
        (15_000, False, 0.0),
    ):
        term.set_step(step)
        assert term._phase_is_active() is active
        assert term._phase_binary_weight() == pytest.approx(expected_weight)

    term.eval()
    term.set_step(0)
    assert term._phase_is_active() is False
    assert term._phase_binary_weight() == 0.0


def test_phase_diagnostics_follows_the_network_until_the_hard_handoff():
    torch.manual_seed(13)
    term = StokesObservationTerm.__new__(StokesObservationTerm)
    nn.Module.__init__(term)
    term.disambiguation_options = {
        "enabled": True,
        "cold_steps": 2_000,
        "warmup_steps": 5_000,
        "handoff_step": 15_000,
        "binary_weight": 0.01,
        "hidden_dimension": 16,
        "hidden_layers": 2,
        "first_omega_0": 1.0,
        "hidden_omega_0": 1.0,
    }
    term.phase_rotation = _phase()
    coordinates = torch.tensor([[2.0, -3.0, 5.0], [4.0, 7.0, 5.0]])

    term.set_step(0)
    phase = term.phase_for_diagnostics(coordinates)
    assert phase is not None
    # A fresh run starts at the network's (now randomly initialized) output,
    # not a hardcoded identity rotation.
    torch.testing.assert_close(phase, term.phase_rotation(coordinates))
    assert torch.any(phase != 0.0)

    term.set_step(15_000)
    torch.testing.assert_close(
        term.phase_for_diagnostics(coordinates),
        torch.zeros(coordinates.shape[:-1]),
    )


def test_disambiguation_config_requires_handoff_after_warmup():
    with pytest.raises(ValueError, match="handoff_step"):
        DisambiguationConfig(
            enabled=True,
            warmup_steps=10,
            handoff_step=10,
            binary_weight=0.01,
        )


def test_hmi_potential_preset_enables_fixed_phase_handoff():
    from prom3theus.config import load_config

    config = load_config("configs/hmi_aia_potential.yaml")
    stokes = config.streams[0].data_term
    disambiguation = config.streams[0].data_term.disambiguation

    assert stokes.weight == 10.0
    assert stokes.weight_schedule is None
    assert disambiguation.enabled
    assert disambiguation.cold_steps == 8_000
    assert disambiguation.warmup_steps == 2_000
    assert disambiguation.handoff_step == 15_000
    assert disambiguation.binary_weight == 0.03
    assert disambiguation.first_omega_0 == 100.0
    assert disambiguation.hidden_omega_0 == 1.0
