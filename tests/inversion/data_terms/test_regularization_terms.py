from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prom3theus.inversion.constraints.magnetofluid import PhysicsResult
from prom3theus.inversion.data_terms import PhysicsConstraintTerm


class _DerivativeAtmosphere(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(2.0))


class _DerivativeConstraints(nn.Module):
    volume_active = True
    upper_volume_active = False
    upper_boundary_active = False
    side_boundary_active = False
    loss_weights = {"magnetic_divergence": 1.0}

    @staticmethod
    def is_active(name: str) -> bool:
        return name == "magnetic_divergence"

    def volume(
        self,
        atmosphere_model,
        position_m,
        time_hours,
        *,
        create_graph,
        return_state,
        height_group_shape,
    ):
        del time_hours, return_state, height_group_shape
        position = position_m.detach().requires_grad_(True)
        field = atmosphere_model.slope * position[:, :1]
        derivative = torch.autograd.grad(
            field.sum(), position, create_graph=create_graph
        )[0][:, 0]
        return PhysicsResult(
            losses={"magnetic_divergence": derivative.square().mean()},
            weights=self.loss_weights,
        )


class _DerivativeDomain:
    def __init__(self) -> None:
        self.calls = 0

    def random_grouped(self, height_count, points_per_height, *, device):
        self.calls += 1
        assert (height_count, points_per_height) == (2, 1)
        return {
            "position_m": torch.tensor(
                [[[3.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]], device=device
            ),
            "time_hours": torch.zeros(2, 1, 1, device=device),
        }


def _assembly():
    empty_volume = torch.empty(0, 0, 3)
    empty_time_volume = torch.empty(0, 0, 1)
    empty_points = torch.empty(0, 3)
    empty_time = torch.empty(0, 1)
    domain = _DerivativeDomain()
    return SimpleNamespace(
        constraints=_DerivativeConstraints(),
        sampling_domain=domain,
        upper_sampling_domain=None,
        volume_points_per_step=2,
        height_layers_per_step=2,
        upper_volume_points_per_step=0,
        upper_height_layers_per_step=0,
        upper_boundary_points_per_step=0,
        side_boundary_points_per_step=0,
        magnetic_current_free_steps=0,
        magnetic_current_free_final_factor=0.0,
        validation_position_m=torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]),
        validation_time_hours=torch.zeros(2, 1, 1),
        validation_upper_position_m=empty_volume,
        validation_upper_time_hours=empty_time_volume,
        validation_boundary_position_m=empty_points,
        validation_boundary_time_hours=empty_time,
        validation_side_position_m=empty_points,
        validation_side_time_hours=empty_time,
        validation_side_normal=empty_points,
    )


def test_physics_term_is_training_ready_by_default_and_backpropagates_pde():
    atmosphere = _DerivativeAtmosphere()
    assembly = _assembly()
    term = PhysicsConstraintTerm(assembly)

    result = term.evaluate(atmosphere)
    gradient = torch.autograd.grad(result.loss, atmosphere.slope)[0]

    assert term.create_graph is True
    assert assembly.sampling_domain.calls == 1
    torch.testing.assert_close(result.loss, torch.tensor(4.0))
    torch.testing.assert_close(gradient, torch.tensor(4.0))


class _SharedBatchConstraints(nn.Module):
    """Records the exact batch each equation group receives."""

    volume_active = True
    upper_volume_active = True
    upper_boundary_active = False
    side_boundary_active = False
    loss_weights = {"magnetic_divergence": 1.0, "coronal_energy": 1.0}

    def __init__(self) -> None:
        super().__init__()
        self.volume_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.upper_domain_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    @staticmethod
    def is_active(name: str) -> bool:
        return name in ("magnetic_divergence", "coronal_energy")

    def volume(
        self,
        atmosphere_model,
        position_m,
        time_hours,
        *,
        create_graph,
        return_state,
        height_group_shape,
    ):
        del atmosphere_model, create_graph, return_state, height_group_shape
        self.volume_calls.append((position_m, time_hours))
        zero = position_m.new_zeros(())
        return PhysicsResult(losses={"magnetic_divergence": zero}, weights=self.loss_weights)

    def upper_domain(
        self,
        atmosphere_model,
        position_m,
        time_hours,
        *,
        height_group_shape,
        create_graph,
    ):
        del atmosphere_model, height_group_shape, create_graph
        self.upper_domain_calls.append((position_m, time_hours))
        zero = position_m.new_zeros(())
        return PhysicsResult(losses={"coronal_energy": zero}, weights=self.loss_weights)


def test_upper_volume_equations_share_the_volume_sampling_batch():
    """coronal_energy must not draw its own, second collocation batch.

    It should reuse exactly the positions/times already sampled for the
    ordinary volume equations (momentum, induction, ...), not a separate
    upper_sampling_domain draw -- masking which points count toward its own
    loss happens at reduction time, not via a second sampling pass.
    """

    atmosphere = _DerivativeAtmosphere()
    assembly = _assembly()
    constraints = _SharedBatchConstraints()
    assembly.constraints = constraints
    term = PhysicsConstraintTerm(assembly)

    term.evaluate(atmosphere)

    assert len(constraints.volume_calls) == 1
    assert len(constraints.upper_domain_calls) == 1
    volume_position, volume_time = constraints.volume_calls[0]
    upper_position, upper_time = constraints.upper_domain_calls[0]
    torch.testing.assert_close(volume_position, upper_position)
    torch.testing.assert_close(volume_time, upper_time)
    # Exactly one collocation draw from the domain for both equation groups.
    assert assembly.sampling_domain.calls == 1


class _TaggedHardExampleDomain:
    """Returns points tagged by call number so tests can trace their origin."""

    # Chosen so the synthetic small-magnitude test coordinates below are not
    # pulled back onto the shell by the real radial-clamp safeguard.
    solar_radius_m = 0.0
    height_bounds_megameter = (-1.0e6, 1.0e6)
    time_bounds_hours = (0.0, 2.0)

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def random_grouped(self, height_count, points_per_height, *, height_power, device):
        del height_power
        self.calls.append((height_count, points_per_height))
        generation = len(self.calls)
        base = 1000.0 * generation
        x = base + torch.arange(height_count, dtype=torch.float32)
        position = torch.stack(
            (x, torch.zeros(height_count), torch.zeros(height_count)), dim=-1
        )
        position = position.unsqueeze(1).expand(height_count, points_per_height, 3).clone()
        return {
            "position_m": position.to(device),
            "time_hours": torch.zeros(height_count, points_per_height, 1, device=device),
        }


def test_hard_examples_carry_the_highest_residual_point_into_the_next_step():
    assembly = _assembly()
    assembly.sampling_domain = _TaggedHardExampleDomain()
    assembly.volume_points_per_step = 3
    assembly.height_layers_per_step = 3
    assembly.volume_sampling = "fully_random"
    assembly.hard_example_enabled = True
    assembly.hard_example_count = 1
    assembly.hard_example_start_step = 0
    assembly.hard_example_jitter_length_m = 0.0
    assembly.hard_example_jitter_time_hours = 0.0
    constraints = assembly.constraints
    constraints.loss_weights = {"magnetic_divergence": 1.0}
    constraints.is_active = lambda name: name == "magnetic_divergence"

    def volume(atmosphere_model, position_m, time_hours, *, return_diagnostics=False, **kwargs):
        del atmosphere_model, time_hours, kwargs
        loss = position_m[:, 0].square().mean()
        diagnostics = None
        if return_diagnostics:
            # The residual grows with x, so the point with the largest x is
            # unambiguously the hardest one to carry forward.
            diagnostics = {
                "magnetic_divergence": {"normalized_residual": position_m[:, :1]}
            }
        return PhysicsResult(
            losses={"magnetic_divergence": loss},
            weights=constraints.loss_weights,
            equation_diagnostics=diagnostics,
        )

    constraints.volume = volume
    term = PhysicsConstraintTerm(assembly)
    term.set_step(0)

    # First active step: no carried-forward set yet, bootstrap fully random.
    term.evaluate(_DerivativeAtmosphere())
    assert assembly.sampling_domain.calls == [(3, 1)]
    assert term._hard_position_m is not None
    assert term._hard_position_m.shape == (1, 1, 3)
    # Highest x from generation 1 is 1000 + 2 = 1002.
    torch.testing.assert_close(
        term._hard_position_m[0, 0, 0], torch.tensor(1002.0)
    )

    # Second step: only the remaining (3 - 1) points are freshly drawn, and
    # the previous hard point is reused (unperturbed since jitter is zero).
    samples = term._samples(next(_DerivativeAtmosphere().parameters()), _DerivativeAtmosphere())
    assert assembly.sampling_domain.calls == [(3, 1), (2, 1)]
    position = samples["volume_position_m"]
    assert position.shape == (3, 1, 3)
    torch.testing.assert_close(position[0, 0, 0], torch.tensor(1002.0))


def _residual_grows_with_x(constraints):
    """Install a volume stub whose residual increases with the x coordinate."""

    seen: list[tuple[int, int]] = []

    def volume(
        atmosphere_model,
        position_m,
        time_hours,
        *,
        return_diagnostics=False,
        height_group_shape=None,
        **kwargs,
    ):
        del atmosphere_model, time_hours, kwargs
        seen.append(height_group_shape)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "magnetic_divergence": {"normalized_residual": position_m[:, :1]}
            }
        return PhysicsResult(
            losses={"magnetic_divergence": position_m[:, 0].square().mean()},
            weights=constraints.loss_weights,
            equation_diagnostics=diagnostics,
        )

    constraints.volume = volume
    constraints.loss_weights = {"magnetic_divergence": 1.0}
    constraints.is_active = lambda name: name == "magnetic_divergence"
    return seen


def test_hard_examples_rank_random_points_globally_without_height_groups():
    assembly = _assembly()
    assembly.sampling_domain = _TaggedHardExampleDomain()
    assembly.volume_points_per_step = 4
    assembly.height_layers_per_step = 4
    assembly.volume_sampling = "fully_random"
    assembly.hard_example_enabled = True
    assembly.hard_example_count = 2
    assembly.hard_example_start_step = 0
    assembly.hard_example_jitter_length_m = 0.0
    assembly.hard_example_jitter_time_hours = 0.0
    _residual_grows_with_x(assembly.constraints)
    term = PhysicsConstraintTerm(assembly)
    term.set_step(0)

    # Bootstrap step: no carried-forward set yet, so the whole batch is fresh.
    term.evaluate(_DerivativeAtmosphere())
    assert assembly.sampling_domain.calls == [(4, 1)]
    # Generation 1 spans x = 1000..1003, so the two globally hardest points
    # carried into the next step are 1003 and 1002.
    torch.testing.assert_close(
        term._hard_position_m[:, 0, 0], torch.tensor([1003.0, 1002.0])
    )

    samples = term._samples(
        next(_DerivativeAtmosphere().parameters()), _DerivativeAtmosphere()
    )

    # Only the remaining (4 - 2) points are freshly drawn.
    assert assembly.sampling_domain.calls == [(4, 1), (2, 1)]
    torch.testing.assert_close(
        samples["volume_position_m"][:, 0, 0],
        torch.tensor([1003.0, 1002.0, 2000.0, 2001.0]),
    )


def test_regularization_reports_only_enabled_components_even_at_zero():
    from prom3theus.inversion.data_terms import AtmosphereRegularizationTerm

    atmosphere = SimpleNamespace(
        magnetic_scale_gauss=1.0,
        velocity_scale_m_per_s=1.0,
        evaluate_position_points=lambda *args, **kwargs: {
            "magnetic_field": torch.zeros(2, 3),
            "velocity_field": torch.ones(2, 3),
        },
    )
    term = AtmosphereRegularizationTerm(
        kind="vector_magnitude",
        position_m=torch.zeros(2, 3),
        time_hours=torch.zeros(2),
        component_weights={"magnetic": 1.0, "velocity": 0.0},
    )
    result = term.evaluate(atmosphere)
    assert result.active and set(result.component_losses) == {"magnetic"}
    assert result.component_losses["magnetic"] == 0
    term.component_weights["magnetic"] = 0.0
    result = term.evaluate(atmosphere)
    assert not result.active and result.component_losses == {}


@pytest.mark.parametrize("steps, final_factor, late_factor", [
    (0, 0.0, 1.0), (2, 0.0, 0.0), (2, 0.1, 0.1), (2, 1.0, 1.0),
])
@pytest.mark.parametrize("training", [False, True])
def test_current_penalty_with_and_without_schedule(steps, final_factor, late_factor, training):
    assembly = _assembly()
    assembly.magnetic_current_free_steps = steps
    assembly.magnetic_current_free_final_factor = final_factor
    constraints = assembly.constraints
    constraints.loss_weights = {"magnetic_current_free": 1.0}
    constraints.is_active = lambda name: name == "magnetic_current_free"
    constraints.volume = lambda *args, **kwargs: PhysicsResult(
        losses={"magnetic_current_free": torch.tensor(4.0)},
        weights=constraints.loss_weights,
    )
    term = PhysicsConstraintTerm(assembly)
    term.train(training)
    term.set_step(1)
    torch.testing.assert_close(term.evaluate(_DerivativeAtmosphere()).loss, torch.tensor(4.0))
    for step in (2, 10000):
        term.set_step(step)
        result = term.evaluate(_DerivativeAtmosphere())
        assert result.active == (late_factor > 0)
        if late_factor == 0:
            assert result.component_losses == {}
        torch.testing.assert_close(result.loss, torch.tensor(4.0 * late_factor))
