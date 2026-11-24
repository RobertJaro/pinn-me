"""Hard-example selection, cached wavelength alignment and online teacher counts."""

import torch

from prom3theus.rt.coefficient_mapper import CoefficientMapper
from prom3theus.rt.local_coefficients import LTELocalCoefficients
from prom3theus.surrogate import training
from prom3theus.surrogate.replay import HardProfileBuffer
from prom3theus.surrogate.sampling import FreshPointBatches
from prom3theus.surrogate.teacher import training_pair


def test_replayed_targets_match_each_points_original_shift(monkeypatch):
    wavelengths = torch.linspace(6300., 6307., 16, dtype=torch.float64)
    buffer = HardProfileBuffer(12, wavelengths, seed=12)
    model = CoefficientMapper(line_ids=["FeI_6302.4932"], bounds=[6300., 6307.], knots=12, width=8)
    optimizer = torch.optim.AdamW(model.parameters())
    teacher_counts, forward_counts = [], []
    recorded = {}

    originals = {}
    teacher = LTELocalCoefficients(line_ids=model.spec["line_ids"]).requires_grad_(False)

    def teacher_pair(teacher, points, query):
        teacher_counts.append(len(points))
        assert query.ndim == 1
        assert not torch.equal(query, wavelengths)
        for point in points:
            originals[float(point[0])] = (point.clone(), query.clone())
        return training_pair(teacher, points, query)

    prepare, coefficients, profile_loss = model.prepare_spectrum, model.coefficients, model.profile_loss_channels

    def record_grid(query):
        recorded["grid"] = query
        return prepare(query)

    def record_points(points, spectrum):
        forward_counts.append(len(points))
        recorded["points"] = points
        return coefficients(points, spectrum)

    def verify_labels(prediction, targets):
        grid = recorded["grid"].expand(len(targets), -1)
        for point, wavelengths_used, target in zip(recorded["points"], grid, targets):
            pressure_point, original_grid = originals[float(point[0])]
            assert torch.equal(wavelengths_used, original_grid)
            expected_point, expected_target = training_pair(teacher, pressure_point[None], original_grid)
            assert torch.equal(point, expected_point[0])
            torch.testing.assert_close(target, expected_target[0], rtol=2e-6, atol=0)
        return profile_loss(prediction, targets)

    monkeypatch.setattr(training, "training_pair", teacher_pair)
    monkeypatch.setattr(model, "prepare_spectrum", record_grid)
    monkeypatch.setattr(model, "coefficients", record_points)
    monkeypatch.setattr(model, "profile_loss_channels", verify_labels)
    generator = torch.Generator().manual_seed(11)
    for step, fresh in enumerate((8, 4, 4, 0), start=1):
        points = torch.randn(fresh, 7, generator=generator) * 300
        # Unique temperatures identify the original draws independently of buffer slots.
        points[:, 0] = 5000 + step * 10 + torch.arange(fresh)
        points[:, 1] = points[:, 1].abs() + 10  # gas pressure before EOS conversion
        points[:, 2] = points[:, 2].abs()
        training._train_step(model, teacher, optimizer, points, wavelengths, step,
                             replay=buffer, replay_count=8-fresh, wavelength_jitter_fraction=0.25)
        if step == 3:
            state = buffer.state_dict()
            buffer = HardProfileBuffer(12, wavelengths, seed=12)
            buffer.load_state_dict(state)
    assert teacher_counts == [8, 4, 4]
    assert forward_counts == [8, 8, 8, 8]
    assert buffer.size == buffer.capacity


def test_replay_refreshes_scores_and_evicts_solved_profiles():
    wave = torch.zeros(2, dtype=torch.float64)
    buffer = HardProfileBuffer(3, wave, seed=4)
    points = torch.arange(3.).reshape(3, 1).expand(3, 7)
    targets = points[:, None].expand(3, 2, 7)
    buffer.update(points, targets, wave[0], torch.tensor([1e6, 1e-6, 1e-6]), None)
    chosen = buffer.sample_indices(1, step=2)
    assert chosen.tolist() == [0]
    # The former hardest profile is now solved. The fresh example gets its slot.
    buffer.update(points[:1]+10, targets[:1]+10, wave[0]+1, torch.tensor([0.5, 0.]), chosen)
    assert buffer.points[:, 0].tolist() == [10., 1., 2.]
    assert buffer.targets[0, 0, 0] == 10 and buffer.offsets[0] == 1
    assert buffer.scores[0] == 0.5


def test_fresh_point_allocation():
    class Sizes:
        batch_size = 1024

        def sample_batch(self, index, *, size):
            return size

    batches = FreshPointBatches(Sizes(), 0.5)
    assert [batches[i] for i in range(3)] == [1024, 512, 512]
    # Odd batch sizes carry fractional replay counts across steps.
    small = Sizes()
    small.batch_size = 5
    batches = FreshPointBatches(small, 0.5)
    assert [batches[i] for i in range(4)] == [5, 3, 2, 3]
