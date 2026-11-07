import torch

from prom3theus.validation.depth_convergence import measure_depth_convergence


def test_convergence_separates_pixel_normalization_bias_and_jitter_preserving_rng():
    torch.manual_seed(31)
    state = torch.random.get_rng_state()
    reference = torch.zeros(2, 4, 3)
    reference[:, 0] = torch.tensor([1.0, 0.2])[:, None]
    reference[:, 3] = torch.tensor([0.1, 0.02])[:, None]
    calls = []

    def evaluate(coarse, fine, randomize):
        calls.append((coarse, fine, randomize))
        actual = reference.clone()
        if coarse == 25:
            actual[:, 0] *= 0.9
            actual[:, 3] *= 0.5
        if randomize:
            actual += torch.rand_like(actual) * 0.001
        return actual

    report = measure_depth_convergence(
        evaluate, levels=((25, 16), (64, 32)), jitter_repeats=3
    )

    assert torch.equal(state, torch.random.get_rng_state())
    coarse, fine = report["rows"]
    torch.testing.assert_close(
        torch.tensor(coarse["rms_error_per_continuum"]),
        torch.tensor([0.1, 0.0, 0.0, 0.05]),
    )
    assert coarse["relative_l2_per_component"][1:3] == [None, None]
    assert coarse["relative_l2_per_component"][3] == 0.5
    assert fine["rms_error_per_continuum"] == [0.0] * 4
    assert all(value > 0 for value in coarse["jitter_rms_per_continuum"])
    assert calls[0] == (1537, 0, False)
    assert sum(randomize for _, _, randomize in calls) == 6

    replay = measure_depth_convergence(
        evaluate, levels=((25, 16), (64, 32)), jitter_repeats=3
    )
    assert report == replay
