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


def test_fine_quantiles_are_stratified_and_eval_preserves_rng():
    from prom3theus.inversion.depth_sampling import contribution_fine_distances

    distances = torch.tensor([[0.0, 1.0, 2.0]] * 8, dtype=torch.float64)
    mass = torch.ones(8, 2, dtype=torch.float64, requires_grad=True)
    state = torch.random.get_rng_state()
    fixed = contribution_fine_distances(mass, distances, 16, 0.05)
    assert torch.equal(state, torch.random.get_rng_state())
    expected = ((torch.arange(16, dtype=torch.float64) + 0.5) / 8).expand(8, -1)
    torch.testing.assert_close(fixed, expected)
    first = contribution_fine_distances(mass, distances, 16, 0.05, randomize=True)
    assert not first.requires_grad
    assert not torch.equal(first[0], first[1])
    assert torch.all(torch.diff(first, dim=-1) > 0)
    assert torch.equal((first * 8).floor().long(), torch.arange(16).expand(8, -1))
    torch.random.set_rng_state(state)
    replay = contribution_fine_distances(mass, distances, 16, 0.05, randomize=True)
    torch.testing.assert_close(first, replay, rtol=0, atol=0)


@pytest.mark.parametrize("randomize", [False, True])
def test_stokes_reference_sampling_preserves_physical_distribution(randomize):
    from types import SimpleNamespace
    from prom3theus.rt import RadialReferenceAtmosphere
    from prom3theus.inversion.depth_sampling import reference_height_grid

    reference = RadialReferenceAtmosphere("falc_82")
    model = SimpleNamespace(
        solar_radius_m=torch.tensor(695700000.0),
        reference_atmosphere=reference,
        line_formation_height_bounds_Mm=(1.5, -0.1),
    )
    q = jitter_depth_grid(torch.linspace(-5, 1, 51), randomize=randomize)
    height = reference_height_grid(model, q)
    # Original piecewise FALC mapping, with physical endpoints pinned at shell bounds.
    reference_height = reference.height_from_log_tau(q)
    expected = torch.where(
        q <= 0,
        reference_height * (1.5e6 / float(reference_height[0])),
        reference_height * (-1e5 / float(reference_height[-1])),
    )
    torch.testing.assert_close(height, expected)
    torch.testing.assert_close(height[[0, -1]], torch.tensor([1.5e6, -1e5]))
    assert torch.all(height[1:] < height[:-1])
    torch.testing.assert_close(
        reference_height_grid(model, torch.tensor([0.0])), torch.tensor([0.0])
    )
    with pytest.raises(ValueError, match="sampling bounds"):
        reference_height_grid(model, torch.tensor([-6.0, 1.0]))


def test_depth_randomness_is_allocated_for_all_rays_together(monkeypatch):
    import ast
    import inspect
    import textwrap
    from prom3theus.inversion.depth_sampling import jitter_depth_grid, contribution_fine_distances

    calls = []
    original_rand, original_like = torch.rand, torch.rand_like

    def rand(*args, **kwargs):
        value = original_rand(*args, **kwargs)
        calls.append(value.shape)
        return value

    def rand_like(*args, **kwargs):
        value = original_like(*args, **kwargs)
        calls.append(value.shape)
        return value

    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "rand_like", rand_like)
    base = torch.linspace(0., 1., 32).expand(128, -1)
    jitter_depth_grid(base, randomize=True)
    contribution_fine_distances(torch.ones(128, 31), base, 16, .1, randomize=True)
    assert calls == [torch.Size((128, 30)), torch.Size((128, 16))]
    for method in (jitter_depth_grid, contribution_fine_distances):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        assert not any(isinstance(node, (ast.For, ast.While, ast.ListComp,
                                        ast.GeneratorExp)) for node in ast.walk(tree))


def _refinement_problem(depth=21):
    distance = torch.arange(depth, dtype=torch.float64).unsqueeze(0)
    # A continuum reaching unit optical depth only near the deep end.
    alpha500 = 1.0e-6 * torch.exp(distance)
    # A line core saturating far above it, where alpha500 is still negligible.
    line = 3.0 * torch.exp(-0.5 * ((distance - 4.0) / 0.8) ** 2)
    return alpha500, line, distance


def test_line_extinction_pulls_refinement_toward_the_core_forming_layers():
    alpha500, line, distance = _refinement_problem()

    continuum_only = importance_fine_distances(alpha500, distance, 32, 0.0)
    with_line = importance_fine_distances(
        alpha500, distance, 32, 0.0, line_extinction=line
    )

    core = distance[0, 6]
    # Guided by the continuum alone, refinement never reaches the core layers.
    assert torch.all(continuum_only > core)
    # The two channels carry equal mass, so half the samples move there and
    # half still resolve the continuum.
    assert (with_line < core).double().mean() == 0.5
    assert with_line.max() > core
    assert with_line.mean() < continuum_only.mean()
    assert torch.isfinite(with_line).all()
    assert torch.all(with_line > distance[..., 0])
    assert torch.all(with_line < distance[..., -1])


def test_omitting_line_extinction_preserves_the_continuum_only_proposal():
    alpha500, _, distance = _refinement_problem()

    torch.testing.assert_close(
        importance_fine_distances(alpha500, distance, 16, 0.05, line_extinction=None),
        importance_fine_distances(alpha500, distance, 16, 0.05),
        rtol=0.0,
        atol=0.0,
    )


def test_line_extinction_is_validated_against_the_reference_extinction():
    alpha500, line, distance = _refinement_problem()

    with pytest.raises(ValueError, match="share the alpha500 shape"):
        importance_fine_distances(
            alpha500, distance, 4, 0.05, line_extinction=line[..., :-1]
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        importance_fine_distances(
            alpha500, distance, 4, 0.05, line_extinction=-line
        )
