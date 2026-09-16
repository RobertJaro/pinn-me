import copy
import torch
from prom3theus.config.schema import PotentialBoundaryConfig
from prom3theus.inversion.sampling import SphericalShellDomain
from prom3theus.inversion.data_terms.potential_boundary import ProgressivePotentialBoundary


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(10.))
        self.calls = 0

    def evaluate_position_rsun(self, position, time):
        self.calls += 1
        field = self.amplitude * torch.stack((position[:, 1] * 20, position[:, 0] * 0, position[:, 0] * 0), -1)
        return {'magnetic_field': field * (1 + time).reshape(-1, 1)}


def make_term(**kwargs):
    times = kwargs.pop("observation_times_hours", [0., 1., 2.])
    time_dependent = kwargs.pop("time_dependent", True)
    options = PotentialBoundaryConfig(start_step=2, ramp_steps=2, update_every_n_steps=2,
        freeze_step=7, grid_size=8, top_grid_size=2, side_horizontal_points=2, side_height_points=2, batch_size=8).to_dict()
    options.update(kwargs)
    domain = SphericalShellDomain(0., (-.04, .04), (-.04, .04), (0., 2.), (0., 20.), 696e6)
    return ProgressivePotentialBoundary(options, domain, domain, [[0, 1, 0], [0, 0, 1], [1, 0, 0]], observation_times_hours=times, time_dependent=time_dependent)


def test_schedule_blending_gradients_freeze_and_restore():
    model, term = Model(), make_term()
    assert not term.evaluate(model).active
    assert model.calls == 0
    term.set_step(2)
    assert not term.evaluate(model).active
    first = term.targets.clone()
    assert term.update_count == 1
    assert not first.requires_grad
    torch.testing.assert_close(first[2], first[0] * 3)
    term.set_step(3)
    result = term.evaluate(model)
    assert result.active and result.loss > 0
    result.loss.backward()
    assert model.amplitude.grad.abs() > 0
    assert term.update_count == 1
    with torch.no_grad():
        model.amplitude.mul_(2)
    term.set_step(4)
    term.evaluate(model)
    torch.testing.assert_close(term.targets, first * 1.5)
    restored = make_term()
    restored.load_state_dict(copy.deepcopy(term.state_dict()))
    assert restored.get_extra_state() == term.get_extra_state()
    torch.testing.assert_close(restored.targets, term.targets)
    rng = torch.get_rng_state()
    expected = term.evaluate(model).loss
    torch.set_rng_state(rng)
    torch.testing.assert_close(restored.evaluate(model).loss, expected)
    for current in (term, restored):
        current.set_step(7)
        current.evaluate(model)
        assert current.frozen and current.last_update == 7
        final = current.targets.clone()
        current.set_step(20)
        current.evaluate(model)
        torch.testing.assert_close(current.targets, final)
    torch.testing.assert_close(restored.targets, term.targets)


def test_validation_does_not_refresh_and_source_is_photospheric():
    term, model = make_term(), Model()
    term.eval()
    term.set_step(10)
    assert not term.evaluate(model).active
    assert model.calls == 0
    radii = term.source_positions.norm(dim=-1)
    torch.testing.assert_close(radii, radii.new_full(radii.shape, 696e6))
    assert (term.positions.norm(dim=-1) > term.radius).all()
    assert not hasattr(term, "normalization_positions")
    assert not hasattr(term, 'plane_z')
    term.train()
    term.evaluate(model)
    state = copy.deepcopy(term.state_dict())
    term.eval()
    term.evaluate(model)
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(term.state_dict()[name], value)
        else:
            assert term.state_dict()[name] == value


def test_startup_gradient_probe_preserves_live_state():
    from prom3theus.application.joint_evaluation import _bounded_shared_gradient_samples
    from prom3theus.application.runtime import state_dict_sha256
    term, model = make_term(), Model()
    before = state_dict_sha256(term)
    with _bounded_shared_gradient_samples(term, maximum_count=4):
        result = term.evaluate(model)
        assert result.active
    result.loss.backward()
    assert torch.isfinite(model.amplitude.grad)
    assert state_dict_sha256(term) == before
    assert term.update_count == 0


def test_lightning_checkpoint_restores_frozen_boundary(tmp_path):
    from pytorch_lightning import Trainer
    from prom3theus.config.joint_schema import JointTrainingConfig
    from prom3theus.inversion.joint import JointForwardModel
    from prom3theus.inversion.data_terms.base import ObservationDataTerm, DataTermBatchResult
    from prom3theus.training.joint import JointInversionModule

    class Observation(ObservationDataTerm):
        def __init__(self, model):
            super().__init__()
            object.__setattr__(self, 'atmosphere', model)

        def evaluate_batch(self, batch):
            return DataTermBatchResult(self.atmosphere.amplitude.square(), {}, {}, 1)

    class Batches:
        def __iter__(self):
            for _ in range(6):
                yield {'stokes': {}}

        def __len__(self):
            return 6

    def module():
        model = Model()
        term = make_term(start_step=0, ramp_steps=1, freeze_step=2, update_every_n_steps=1)
        forward = JointForwardModel(model, {'stokes': Observation(model)}, {'stokes': 1.}, {'potential_boundary': term})
        return JointInversionModule(forward, JointTrainingConfig(max_steps=6, device='cpu', learning_rate=.001, final_learning_rate=.001), {'contract': {}})

    def trainer(steps):
        return Trainer(max_steps=steps, accelerator='cpu', logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path)

    original = module()
    first = trainer(4)
    first.fit(original, train_dataloaders=Batches())
    path = tmp_path / 'potential.ckpt'
    first.save_checkpoint(path)
    saved = original.model.shared_objectives['potential_boundary']
    restored = module()
    trainer(6).fit(restored, train_dataloaders=Batches(), ckpt_path=path)
    actual = restored.model.shared_objectives['potential_boundary']
    assert actual.frozen and actual.last_update == 2 and actual.update_count == 3
    assert actual.step == 5
    torch.testing.assert_close(actual.targets, saved.targets)
    torch.testing.assert_close(actual.normalization, saved.normalization)


def test_reference_refresh_precedes_all_shared_losses():
    from prom3theus.inversion.joint import JointForwardModel
    from prom3theus.inversion.data_terms.base import ObservationDataTerm, DataTermBatchResult, SharedObjectiveTerm, SharedTermResult

    model, reference = Model(), make_term()
    reference.set_step(2)

    class Observation(ObservationDataTerm):
        def evaluate_batch(self, batch):
            return DataTermBatchResult(model.amplitude.square(), {}, {}, 1)

    class Consumer(SharedObjectiveTerm):
        def evaluate(self, atmosphere):
            assert reference.last_update == 2
            scale = reference.normalization[0]
            assert scale is not None and torch.isfinite(scale).all()
            return SharedTermResult(atmosphere.amplitude.square() / (scale.mean() + 1), {}, {})

    joint = JointForwardModel(model, {'stokes': Observation()}, {'stokes': 1.},
                              {'physics': Consumer(), 'potential_boundary': reference})
    joint.evaluate_batches({'stokes': {}}).loss.backward()
    assert reference.update_count == 1


def test_refresh_reuses_spherical_operator_for_all_times_and_products(monkeypatch):
    import prom3theus.inversion.data_terms.potential_boundary as implementation
    builder = implementation.spherical_potential_matrix
    queries = []

    def build(cells, points):
        queries.append(points.copy())
        return builder(cells, points)

    monkeypatch.setattr(implementation, 'spherical_potential_matrix', build)
    term = make_term()
    term.set_step(2)
    term.prepare(Model())
    term.set_step(4)
    term.prepare(Model())
    assert len(queries) == 1
    assert len(queries[0]) == len(term.positions)
    assert (torch.from_numpy(queries[0]).norm(dim=-1) > 1).all()
    assert 'potential_operator' not in term.state_dict()


def test_negative_shell_base_does_not_create_subphotospheric_targets():
    options = PotentialBoundaryConfig(grid_size=4, top_grid_size=2, side_horizontal_points=2, side_height_points=2).to_dict()
    def build(inner):
        domain = SphericalShellDomain(0., (-.04, .04), (-.04, .04), (0., 2.), (inner, 20.), 696e6)
        return ProgressivePotentialBoundary(options, domain, domain, torch.eye(3), observation_times_hours=[0., 1., 2.])
    clipped, photospheric = build(-.1), build(0.)
    torch.testing.assert_close(clipped.positions, photospheric.positions)
    torch.testing.assert_close(clipped.source_positions, photospheric.source_positions)
    assert (clipped.positions.norm(dim=-1) > clipped.radius).all()


def test_rejects_buried_reference_checkpoint():
    import pytest
    term = make_term()
    for version in (3, 4, 5, 6):
        with pytest.raises(ValueError, match='Unsupported progressive potential checkpoint'):
            term.set_extra_state({**term.get_extra_state(), 'version': version})


def test_source_geometry_does_not_depend_on_exterior_query_height_or_count():
    options = PotentialBoundaryConfig(grid_size=4, top_grid_size=2, side_horizontal_points=2, side_height_points=2).to_dict()
    domain = SphericalShellDomain(0., (-.04, .04), (-.04, .04), (0., 2.), (-.1, 20.), 696e6)
    source = ProgressivePotentialBoundary(options, domain, domain, torch.eye(3), observation_times_hours=[0., 1., 2.])
    from dataclasses import replace
    taller = ProgressivePotentialBoundary(
        options | {'top_grid_size': 3, 'side_horizontal_points': 4},
        replace(domain, height_bounds_Mm=(-.1, 80.)), domain, torch.eye(3), observation_times_hours=[0., 1., 2.])
    torch.testing.assert_close(source.source_positions, taller.source_positions)
    torch.testing.assert_close(source.source_cells, taller.source_cells)


def test_all_irregular_observation_times_get_independent_reference_fields():
    class QuadraticTimeModel(Model):
        def evaluate_position_rsun(self, position, time):
            fields = super().evaluate_position_rsun(position, time)
            fields['magnetic_field'] = fields['magnetic_field'] * ((1 + time.square()) / (1 + time)).reshape(-1, 1)
            return fields
    term = make_term(observation_times_hours=[1.73, .61, .17, .61])
    torch.testing.assert_close(term.times, torch.tensor([0., .17, .61, 1.73, 2.]))
    term.set_step(2)
    term.prepare(QuadraticTimeModel())
    for index, time in enumerate(term.times):
        torch.testing.assert_close(term.targets[index], term.targets[0] * (1 + time.square()))
    assert term.targets.shape[0] == 5
    assert term.normalization.shape[0] == 5
    restored = make_term(observation_times_hours=[.17, .61, 1.73])
    restored.load_state_dict(copy.deepcopy(term.state_dict()))
    torch.testing.assert_close(restored.times, term.times)
    torch.testing.assert_close(restored.targets, term.targets)


def test_validation_covers_every_anchor_in_bounded_batches_with_mean_weight():
    term = make_term(observation_times_hours=[.2, .7, 1.8], top_grid_size=2, side_horizontal_points=2, side_height_points=2, batch_size=3)
    term.set_step(4)
    term.last_update = 4
    term.normalization.fill_(1.)
    values = torch.arange(len(term.times), dtype=torch.float32)
    term.targets[:, :term.top_count] = values[:, None, None]
    term.targets[:, term.top_count:] = 2 * values[:, None, None]

    class ZeroModel(Model):
        def __init__(self):
            super().__init__()
            self.seen = []
        def evaluate_position_rsun(self, position, time):
            assert len(position) <= 3
            self.seen.extend(time.tolist())
            return {'magnetic_field': position * self.amplitude * 0}
    model = ZeroModel()
    term.eval()
    result = term.evaluate(model)
    assert set(model.seen) == set(term.times.tolist())
    assert len(model.seen) == len(term.times) * len(term.positions)
    # Equal top/all-sides budgets: normalized errors are v²/2 and (2v)²/2.
    torch.testing.assert_close(result.loss.float(), 2.5 * values.square().mean())


def test_static_reference_uses_one_time_without_observation_anchors():
    term = make_term(time_dependent=False, observation_times_hours=None)
    torch.testing.assert_close(term.times, torch.tensor([1.]))
    term.set_step(2)
    term.prepare(Model())
    assert term.targets.shape[0] == 1


def test_dynamic_reference_requires_valid_observation_times():
    import pytest
    for times in (None, [], [float('nan')], [[0., 1.]], [-.1], [2.1]):
        with pytest.raises(ValueError):
            make_term(observation_times_hours=times)


def test_training_interpolates_at_irregular_physical_times():
    term = make_term(observation_times_hours=[.2, .7, 1.8])
    query = torch.tensor([0., .1, .2, .45, .7, 1., 1.8, 2.])
    values = term.times.square()[:, None].expand(-1, len(query))
    import numpy as np
    expected = torch.tensor(np.interp(query.numpy(), term.times.numpy(), term.times.square().numpy())).float()
    torch.testing.assert_close(term._time_values(values, query), expected)
    vectors = values[:, :, None].expand(-1, -1, 3)
    torch.testing.assert_close(term._time_values(vectors, query), expected[:, None].expand(-1, 3))


def test_exterior_loss_uses_detached_surface_scale_and_floor():
    term, model = make_term(), Model()
    positions = term.positions[:3]
    times = torch.zeros(3)
    targets = torch.ones(3, 3)
    prediction = model.evaluate_position_rsun(positions / term.radius, times)['magnetic_field']
    unnormalized = (prediction - targets).square()
    for scale in (0., 1., 1000.):
        denominator = (scale**2 + term.options['field_floor_gauss']**2) ** 0.5
        expected = unnormalized / denominator**2
        actual = term._residual(
            model, positions, times, targets, torch.full((3,), scale)
        )
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual.mean(), model.amplitude, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected.mean(), model.amplitude, retain_graph=True)[0]
        torch.testing.assert_close(gradient, expected_gradient)
    term.options['field_floor_gauss'] = 1e6
    actual = term._residual(model, positions, times, targets, torch.zeros(3))
    torch.testing.assert_close(actual, unnormalized / 1e12)


def test_frozen_source_capture_can_precede_completion_of_weight_ramp():
    term, model = make_term(start_step=2, freeze_step=2, ramp_steps=4), Model()
    term.set_step(2)
    assert not term.evaluate(model).active
    assert term.frozen and term.update_count == 1
    targets = term.targets.clone()
    with torch.no_grad():
        model.amplitude.mul_(2)
    term.set_step(6)
    assert term.evaluate(model).active
    assert term.update_count == 1
    torch.testing.assert_close(term.targets, targets)
