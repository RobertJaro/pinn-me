import pytest
import torch

from prom3theus.inversion.depth_sampling import (
    importance_fine_distances,
    jitter_depth_grid,
    merge_depth_samples,
)


def test_jitter_preserves_endpoints_count_and_order(monkeypatch):
    base = torch.linspace(-5.0, 1.0, 9)
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.ones_like(value))
    sampled = jitter_depth_grid(base, randomize=True)
    assert sampled.shape == base.shape
    assert sampled[0] == base[0]
    assert sampled[-1] == base[-1]
    assert torch.all(sampled[1:] > sampled[:-1])


def test_jitter_uses_local_spacing_for_nonuniform_grids(monkeypatch):
    base = torch.tensor([-5.0, -1.0, -0.9, 1.0])
    draws = iter((torch.tensor([1.0, 0.0]),))
    monkeypatch.setattr(torch, "rand_like", lambda _: next(draws))

    sampled = jitter_depth_grid(base, randomize=True)

    assert torch.all(sampled[1:] > sampled[:-1])
    torch.testing.assert_close(sampled, torch.tensor([-5.0, -0.96, -0.94, 1.0]))


def test_importance_refinement_orders_a_folded_detached_proposal():
    opacity = torch.tensor([[1.0, 3.0, 2.0]])
    fine = importance_fine_distances(
        opacity,
        torch.tensor([[0.0, 2.0, 1.0]]),
        2,
        0.1,
    )

    assert torch.isfinite(fine).all()
    assert torch.all(fine >= 0.0)
    assert torch.all(fine <= 2.0)


def test_importance_refinement_rejects_nonphysical_extinction():
    distance = torch.tensor([[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="finite and non-negative"):
        importance_fine_distances(torch.tensor([[1.0, -1.0, 1.0]]), distance, 2, 0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        importance_fine_distances(
            torch.tensor([[1.0, float("nan"), 1.0]]), distance, 2, 0.1
        )


def test_importance_refinement_is_finite_for_an_opaque_interval():
    distance = torch.tensor([[0.0, 1.0, 2.0]])
    opacity = torch.full_like(distance, torch.finfo(distance.dtype).max)

    refined = importance_fine_distances(opacity, distance, 3, 0.05)

    assert torch.isfinite(refined).all()
    assert torch.all(refined > distance[..., :1])
    assert torch.all(refined < distance[..., -1:])


def test_importance_refinement_never_duplicates_float32_coarse_edges():
    distance = torch.linspace(0.0, 1.6e6, 25).unsqueeze(0)
    opacity = torch.ones_like(distance)

    fine = importance_fine_distances(opacity, distance, 16, 0.05)
    merged = torch.sort(torch.cat((distance, fine), dim=-1), dim=-1).values

    assert torch.all(merged[..., 1:] > merged[..., :-1])


def test_merge_depth_samples_supports_scalars_and_vectors():
    order = torch.tensor([[0, 2, 1]])
    scalar = merge_depth_samples(
        torch.tensor([[1.0, 3.0]]), torch.tensor([[2.0]]), order
    )
    assert torch.equal(scalar, torch.tensor([[1.0, 2.0, 3.0]]))

    coarse = torch.tensor([[[1.0, 10.0], [3.0, 30.0]]])
    fine = torch.tensor([[[2.0, 20.0]]])
    vector = merge_depth_samples(coarse, fine, order)
    assert torch.equal(vector, torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]))
