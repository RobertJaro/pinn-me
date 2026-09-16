"""Surface geometry, axis normalization, sampling and hard photospheric cutoff."""

import copy

import pytest
import torch

from prom3theus.config.schema import PotentialBoundaryConfig, PotentialPhotosphereConfig
from prom3theus.inversion.data_terms.potential_boundary import (
    ProgressivePotentialBoundary,
)
from prom3theus.inversion.sampling import SphericalShellDomain


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(10.0))
        self.positions = []

    def evaluate_position_rsun(self, positions, time):
        self.positions.append(positions.detach().clone())
        return {
            "magnetic_field": self.amplitude
            * (positions + positions.new_tensor([0.0, 0.2, 0.0]))
            * (1 + time).reshape(-1, 1)
        }


def make_term():
    options = PotentialBoundaryConfig(
        enabled=True,
        start_step=0,
        ramp_steps=0,
        update_every_n_steps=2,
        freeze_step=20,
        grid_size=4,
        top_grid_size=3,
        side_horizontal_points=3,
        side_height_points=2,
        batch_size=8,
        photosphere=PotentialPhotosphereConfig(
            enabled=True, batch_size=5, start_step=2, ramp_steps=2, end_step=9
        ),
    ).to_dict()
    domain = SphericalShellDomain(
        0.0, (-0.04, 0.04), (-0.04, 0.04), (0.0, 2.0), (-0.1, 10.0), 696e6
    )
    return ProgressivePotentialBoundary(
        options, domain, domain, torch.eye(3), observation_times_hours=[0.0, 0.3, 2.0]
    )


def test_only_surface_queries_with_exact_photosphere_and_no_volume_normalization():
    term = make_term()
    for name, (begin, end) in term.surface_ranges.items():
        positions = term.positions[begin:end]
        radius = positions.norm(dim=-1)
        height = (radius - term.radius) / 1e6
        lon = torch.atan2(positions[:, 1], positions[:, 0])
        lat = torch.asin(positions[:, 2] / radius)
        if name == "photosphere":
            torch.testing.assert_close(
                height, torch.zeros_like(height), atol=2e-13, rtol=0
            )
            assert len(height) == 16
        elif name == "top":
            torch.testing.assert_close(height, torch.full_like(height, 10.0))
        else:
            coordinate = lon if name in ("west", "east") else lat
            expected = 0.04 if name in ("west", "north") else -0.04
            torch.testing.assert_close(
                coordinate, torch.full_like(coordinate, expected)
            )
            assert len(torch.unique(height.round(decimals=6))) == 2
            assert (height > 0).all()
    assert not hasattr(term, "normalization_positions")
    assert not hasattr(term, "interior_positions")


def test_axis_means_are_per_surface_per_time_and_per_side_height():
    term = make_term()
    for face, (name, (begin, end)) in enumerate(term.surface_ranges.items()):
        nz, nx = term.surface_shapes[name]
        values = (
            1
            + face * 100
            + 1000 * torch.arange(3)[:, None, None]
            + 10 * torch.arange(nz)[None, :, None]
            + torch.arange(nx)[None, None, :]
        )
        # Alternating signs distinguish mean magnitude from magnitude of mean.
        sign = torch.where(torch.arange(nx) % 2 == 0, 1.0, -1.0)
        term.targets[:, begin:end, 0] = (values * sign).reshape(3, -1)
    term._update_normalization()
    side_strengths = []
    for side_name, (side_begin, side_end) in term.surface_ranges.items():
        if side_name in ("top", "photosphere"):
            continue
        side_shape = term.surface_shapes[side_name]
        side_strengths.append(
            term.targets[:, side_begin:side_end]
            .norm(dim=-1)
            .reshape(3, *side_shape)
        )
    side_mean = torch.cat(side_strengths, dim=-1).mean(-1, keepdim=True)
    for name, (begin, end) in term.surface_ranges.items():
        nz, nx = term.surface_shapes[name]
        actual = term.normalization[:, begin:end].reshape(3, nz, nx)
        strength = term.targets[:, begin:end].norm(dim=-1).reshape(3, nz, nx)
        if name in ("top", "photosphere"):
            expected = strength.mean((-2, -1), keepdim=True).expand_as(strength)
        else:
            expected = side_mean.expand_as(strength)
        torch.testing.assert_close(actual, expected)
    for side_name in ("west", "east", "north", "south"):
        begin, end = term.surface_ranges[side_name]
        side_nz, side_nx = term.surface_shapes[side_name]
        torch.testing.assert_close(
            term.normalization[:, begin:end].reshape(3, side_nz, side_nx),
            side_mean.expand(3, side_nz, side_nx),
        )


def test_targets_detached_blending_radial_boundary_and_cached_surface_operator():
    term, model = make_term(), Model()
    term.set_step(4)
    term.prepare(model)
    assert term.potential_operator.shape[0] == len(term.positions)
    first = term.targets.clone()
    operator = term.potential_operator
    begin, end = term.surface_ranges["photosphere"]
    units = term.positions[begin:end] / term.radius
    torch.testing.assert_close(
        (term.targets[:, begin:end].double() * units).sum(-1).float(),
        term.source_reference,
        atol=3e-6,
        rtol=2e-6,
    )
    assert not term.targets.requires_grad
    for positions in model.positions:
        torch.testing.assert_close(
            positions.norm(dim=-1), torch.ones_like(positions[:, 0])
        )
    with torch.no_grad():
        model.amplitude.mul_(2)
    term.set_step(6)
    term.prepare(model)
    assert term.potential_operator is operator
    torch.testing.assert_close(term.targets, 1.5 * first)
    torch.testing.assert_close(term.sample_targets, term.targets[:, term.sample_order])
    torch.testing.assert_close(
        term.sample_normalization, term.normalization[:, term.sample_order]
    )


def test_random_contiguous_chunks_gradients_restore_and_hard_cutoff(monkeypatch):
    term, model = make_term(), Model()
    term.set_step(4)
    term.prepare(model)
    original = term._photosphere_residual
    sampled = []

    def residual(model, positions, times, targets, normalization):
        assert positions.is_contiguous()
        sampled.append(positions.clone())
        return original(model, positions, times, targets, normalization)

    monkeypatch.setattr(term, "_photosphere_residual", residual)
    result = term.evaluate(model)
    result.component_losses["photosphere"].backward()
    assert model.amplitude.grad is not None
    assert model.amplitude.grad.abs() < 1e-6  # Orientation is amplitude-invariant.
    first = sampled[-1]
    for _ in range(10):
        term.evaluate(model)
    assert any(not torch.equal(x, first) for x in sampled[1:])
    restored = make_term()
    restored.load_state_dict(copy.deepcopy(term.state_dict()))
    rng = torch.get_rng_state()
    expected = term.evaluate(model).loss
    torch.set_rng_state(rng)
    torch.testing.assert_close(restored.evaluate(model).loss, expected)
    term.set_step(8)
    term.prepare(model)
    term.set_step(9)  # No reference refresh due at this step.
    model.positions.clear()
    result = term.evaluate(model)
    assert "photosphere" not in result.component_losses
    assert result.metrics["photosphere_weight"] == 0
    assert all((p.norm(dim=-1) > 1).all() for p in model.positions)
    assert set(result.component_losses) == {"top", "side"}


@pytest.mark.parametrize(
    "step,weight", [(0, 0), (2, 0), (3, 0.05), (4, 0.1), (8, 0.1), (9, 0), (100, 0)]
)
def test_photosphere_schedule(step, weight):
    term = make_term()
    term.set_step(step)
    assert term._photosphere_weight() == pytest.approx(weight)


def _chart_from_positions(term, name, positions):
    """Independent Cartesian inverse for checking the actual fitting positions."""
    import math

    domain = term.surface_domains["photosphere" if name == "photosphere" else "outer"]
    radius = positions.norm(dim=-1)
    angle = torch.atan2(positions[:, 1], positions[:, 0]) - domain.longitude_center_rad
    angle = torch.atan2(angle.sin(), angle.cos())
    lon = (angle - domain.longitude_offset_bounds_rad[0]) / (
        domain.longitude_offset_bounds_rad[1] - domain.longitude_offset_bounds_rad[0]
    )
    mu0, mu1 = map(math.sin, domain.latitude_bounds_rad)
    lat = (positions[:, 2] / radius - mu0) / (mu1 - mu0)
    height = (radius - term.radius) / (domain.height_bounds_Mm[1] * 1e6)
    if name in ("top", "photosphere"):
        uv = torch.stack((lon, lat), -1)
        torch.testing.assert_close(
            radius,
            torch.full_like(
                radius,
                term.radius
                if name == "photosphere"
                else term.radius + domain.height_bounds_Mm[1] * 1e6,
            ),
        )
    else:
        edge = lat if name in ("west", "east") else lon
        fixed = lon if name in ("west", "east") else lat
        torch.testing.assert_close(
            fixed,
            torch.full_like(fixed, float(name in ("west", "north"))),
            atol=1e-12,
            rtol=1e-12,
        )
        assert (height > 0).all() and (height < 1).all()
        uv = torch.stack((height, edge), -1)
    assert (uv > 0).all() and (uv < 1).all()
    return uv


@pytest.mark.parametrize("fraction", [0.25, 1.0])
def test_jitter_is_fresh_in_selected_cells_and_stays_on_every_surface(fraction):
    from dataclasses import replace

    template = make_term()
    # Distinct source footprint and a joint domain crossing the longitude seam.
    domain = replace(template.surface_domains["outer"], longitude_center_rad=3.13)
    source = replace(
        domain,
        longitude_offset_bounds_rad=(-0.02, 0.03),
        latitude_bounds_rad=(-0.01, 0.025),
    )
    term = ProgressivePotentialBoundary(
        template.options | {"jitter_fraction": fraction},
        domain,
        source,
        torch.eye(3),
        observation_times_hours=[0.0, 0.3, 2.0],
    )
    for name, (begin, end) in term.surface_ranges.items():
        times = torch.zeros(end - begin)
        saved = term.sample_positions.clone()
        positions, _, _ = term._training_samples(name, begin, end, times)
        next_positions, _, _ = term._training_samples(name, begin, end, times)
        assert not torch.equal(positions, next_positions)
        assert not torch.equal(positions, saved[begin:end])
        torch.testing.assert_close(term.sample_positions, saved)
        shape = term.surface_shapes[name]
        uv = _chart_from_positions(term, name, positions)
        cell_coordinates = uv * uv.new_tensor(shape)
        cells = term.sample_order[begin:end] - begin
        expected_cells = torch.stack((cells // shape[1], cells % shape[1]), -1)
        torch.testing.assert_close(cell_coordinates.floor().long(), expected_cells)
        assert (
            (cell_coordinates - expected_cells - 0.5).abs() <= fraction / 2 + 1e-12
        ).all()


def test_jitter_targets_and_side_scales_follow_actual_positions_and_time():
    term = make_term()
    for face, (name, (begin, end)) in enumerate(term.surface_ranges.items()):
        n0, n1 = term.surface_shapes[name]
        u, v = torch.meshgrid(
            torch.arange(n0) + 0.5, torch.arange(n1) + 0.5, indexing="ij"
        )
        uv = torch.stack((u / n0, v / n1), -1).reshape(-1, 2)

        def field(uv, time):
            u, v = uv.unbind(-1)
            return torch.stack(
                (
                    1 + 2 * u + 3 * v + u * v + time,
                    -u + 5 * v - time,
                    2 * u - v + 3 * time,
                ),
                -1,
            )

        for anchor, time in enumerate(term.times):
            term.targets[anchor, begin:end] = field(uv, time)
            term.normalization[anchor, begin:end] = (
                10
                + face
                + time
                + (2 * uv[:, 0] if name not in ("top", "photosphere") else 0)
            )
        # Distinct source Br per cell and time, independent of interpolated B.
        term.source_reference.copy_(
            100
            + torch.arange(term.source_reference.shape[1])[None]
            + term.times[:, None]
        )
        times = torch.linspace(0.1, 1.9, end - begin)
        positions, actual, scale = term._training_samples(name, begin, end, times)
        uv = _chart_from_positions(term, name, positions).float()
        expected = field(uv, times)
        if name == "photosphere":
            direction = (positions / term.radius).float()
            br = 100 + (term.sample_order[begin:end] - begin) + times
            expected += (br - (expected * direction).sum(-1))[:, None] * direction
            torch.testing.assert_close((actual * direction).sum(-1), br)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        expected_scale = 10 + face + times
        if name not in ("top", "photosphere"):
            expected_scale += 2 * uv[:, 0].clamp(0.5 / n0, 1 - 0.5 / n0)
        torch.testing.assert_close(scale, expected_scale)
        assert not actual.requires_grad and not scale.requires_grad


def test_training_model_receives_jittered_coordinates_and_validation_does_not(
    monkeypatch,
):
    term, model = make_term(), Model()
    term.set_step(4)
    term.prepare(model)
    cache = copy.deepcopy(term.state_dict())
    # Selecting jittered fitting points must never reconstruct the potential.
    import prom3theus.inversion.data_terms.potential_boundary as implementation

    def forbidden(*args, **kwargs):
        raise AssertionError("Jitter must not rebuild a potential operator")

    monkeypatch.setattr(implementation, "spherical_potential_matrix", forbidden)
    monkeypatch.setattr(implementation, "spherical_photosphere_matrix", forbidden)
    captured = []
    sample = term._training_samples

    def capture(name, start, stop, times):
        # Only the selected contiguous shuffled cell chunk is perturbed.
        assert stop - start <= (5 if name == "photosphere" else 8)
        result = sample(name, start, stop, times)
        captured.append(result[0])
        assert not torch.equal(result[0], term.sample_positions[start:stop])
        return result

    monkeypatch.setattr(term, "_training_samples", capture)
    model.positions.clear()
    loss = term.evaluate(model).loss
    loss.backward()
    assert model.amplitude.grad.abs() > 1e-6
    assert len(captured) == len(model.positions) == 6
    for expected, actual in zip(captured, model.positions, strict=True):
        torch.testing.assert_close(actual, expected / term.radius, atol=0, rtol=0)
    for key, value in cache.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(term.state_dict()[key], value)
    term.eval()
    monkeypatch.setattr(term, "_training_samples", forbidden)
    one, two = term.evaluate(model), term.evaluate(model)
    torch.testing.assert_close(one.loss, two.loss, atol=0, rtol=0)


def test_zero_jitter_preserves_contiguous_fixed_tensor_reads():
    term = make_term()
    term.options["jitter_fraction"] = 0
    for name, (begin, end) in term.surface_ranges.items():
        positions, _, _ = term._training_samples(
            name, begin, end, torch.zeros(end - begin)
        )
        assert (
            positions.untyped_storage().data_ptr()
            == term.sample_positions.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(
            positions, term.sample_positions[begin:end], atol=0, rtol=0
        )


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_jitter_fraction_validation(fraction):
    with pytest.raises(ValueError):
        PotentialBoundaryConfig(jitter_fraction=fraction)


class HorizontalProbe(torch.nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = torch.nn.Parameter(torch.tensor(field, dtype=torch.float64))

    def evaluate_position_rsun(self, positions, times):
        return {"magnetic_field": self.field.expand(len(positions), -1)}


@pytest.mark.parametrize(
    "field,expected",
    [
        ([300.0, 20.0, 0.0], 0.0),
        ([900.0, 200.0, 0.0], 0.0),
        # 90 degrees off the reference: zero under the old 90-degree
        # threshold, now (1/3)**2 past the 45-degree threshold.
        ([-800.0, 0.0, 20.0], 1.0 / 9.0),
        # Exactly at the 45-degree threshold: still zero.
        ([100.0, 20.0, 20.0], 0.0),
        # 135 degrees off the reference: ((3/4 - 1/4) / (1 - 1/4))**2 = (2/3)**2.
        ([100.0, -20.0, 20.0], 4.0 / 9.0),
        ([100.0, -20.0, -20.0], 4.0 / 9.0),
        ([100.0, -20.0, 0.0], 1.0),
    ],
)
def test_photosphere_matches_only_signed_horizontal_angle(field, expected):
    term = make_term()
    model = HorizontalProbe(field)
    positions = torch.tensor([[term.radius, 0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[-200.0, 50.0, 0.0]], dtype=torch.float64)
    loss = term._photosphere_residual(
        model, positions, torch.zeros(1), target, torch.ones(1)
    ).mean()
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=loss.dtype))
    loss.backward()
    assert model.field.grad[0] == 0  # Br receives no constraint.
    assert abs((model.field.grad[1:] * model.field.detach()[1:]).sum()) < 1e-12
    if expected > 0:
        assert model.field.grad.norm() > 0  # Includes exact polarity reversal.
    else:
        torch.testing.assert_close(model.field.grad, torch.zeros_like(model.field.grad))


def test_photosphere_weak_reference_skipped_and_zero_prediction_has_finite_gradient():
    term = make_term()
    positions = torch.tensor([[term.radius, 0.0, 0.0]], dtype=torch.float64)
    for target, expected in [([300.0, 0.0, 0.0], 0.0), ([300.0, 50.0, 0.0], 0.5)]:
        model = HorizontalProbe([200.0, 0.0, 0.0])
        loss = term._photosphere_residual(
            model,
            positions,
            torch.zeros(1),
            torch.tensor([target], dtype=torch.float64),
            torch.ones(1),
        ).mean()
        loss.backward()
        assert loss.item() == expected
        assert torch.isfinite(model.field.grad).all()
        assert model.field.grad[0] == 0
        if expected:
            assert model.field.grad[1] < 0


@pytest.mark.parametrize(
    "reference_strength,active",
    [(0.0, False), (49.9, False), (50.0, True), (100.0, True)],
)
@pytest.mark.parametrize("prediction_strength", [0.01, 10.0, 1000.0])
def test_photosphere_threshold_uses_reference_horizontal_strength(
    reference_strength, active, prediction_strength
):
    term = make_term()
    model = HorizontalProbe([5000.0, -prediction_strength, 0.0])
    positions = torch.tensor([[term.radius, 0.0, 0.0]], dtype=torch.float64)
    # Large radial reference must not qualify weak horizontal fields.
    target = torch.tensor([[5000.0, reference_strength, 0.0]], dtype=torch.float64)
    result = term._photosphere_residual(
        model, positions, torch.zeros(1), target, torch.ones(1)
    ).mean()
    assert result.item() == float(active)
    result.backward()
    assert torch.isfinite(model.field.grad).all()
    assert (model.field.grad.norm() > 0) == active


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf"), True])
def test_photosphere_threshold_requires_positive_finite_gauss(threshold):
    with pytest.raises((TypeError, ValueError)):
        PotentialPhotosphereConfig(minimum_horizontal_field_gauss=threshold)


def test_photosphere_loss_averages_only_considered_points_not_the_full_batch(
    monkeypatch,
):
    """Excluded (zero-residual) points must not dilute the photosphere loss."""

    term, model = make_term(), Model()
    # Large enough that every chunk (training) and every eval sub-batch
    # covers the whole photosphere pool in one call.
    term.options["photosphere"]["batch_size"] = 1000
    term.set_step(4)
    term.prepare(model)
    begin, end = term.surface_ranges["photosphere"]
    count = end - begin
    assert count > 1
    considered = max(1, count // 2)
    pattern = torch.cat(
        (torch.full((considered, 1), 2.0), torch.zeros(count - considered, 1))
    ).to(torch.float64)

    def half_considered(model_, positions, times, targets, normalization):
        assert len(positions) == count
        return pattern

    monkeypatch.setattr(term, "_photosphere_residual", half_considered)

    expected = term._photosphere_weight() * 2.0
    term.train()
    training_loss = term.evaluate(model).component_losses["photosphere"]
    assert training_loss.item() == pytest.approx(expected)

    term.eval()
    eval_loss = term.evaluate(model).component_losses["photosphere"]
    assert eval_loss.item() == pytest.approx(expected)


def test_photosphere_loss_is_zero_not_nan_when_everything_is_excluded(monkeypatch):
    term, model = make_term(), Model()
    term.options["photosphere"]["batch_size"] = 1000
    term.set_step(4)
    term.prepare(model)
    begin, end = term.surface_ranges["photosphere"]
    count = end - begin

    def none_considered(model_, positions, times, targets, normalization):
        return torch.zeros(len(positions), 1, dtype=torch.float64)

    monkeypatch.setattr(term, "_photosphere_residual", none_considered)

    term.train()
    training_loss = term.evaluate(model).component_losses["photosphere"]
    assert torch.isfinite(training_loss)
    assert training_loss.item() == 0.0

    term.eval()
    eval_loss = term.evaluate(model).component_losses["photosphere"]
    assert torch.isfinite(eval_loss)
    assert eval_loss.item() == 0.0
