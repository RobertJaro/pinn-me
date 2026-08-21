import pytest
import torch

from pme.lte.zeeman import ZeemanPattern, zeeman_components


@pytest.mark.parametrize(
    ("j_lower", "j_upper", "lande_lower", "lande_upper", "expected_lande"),
    [
        (2.0, 2.0, 1.84, 1.50, 1.67),
        (1.0, 0.0, 2.50, 0.00, 2.50),
    ],
)
def test_hinode_line_patterns_obey_selection_rules_and_group_normalization(
    j_lower, j_upper, lande_lower, lande_upper, expected_lande
):
    pattern = ZeemanPattern(j_lower, j_upper, lande_lower, lande_upper)
    assert pattern.effective_lande == pytest.approx(expected_lande, abs=1e-12)
    assert pattern.components

    for component in pattern.components:
        assert component.delta_m in (-1, 0, 1)
        assert component.m_upper - component.m_lower == pytest.approx(component.delta_m)
        assert component.strength > 0

    for delta_m in (-1, 0, 1):
        group = [component for component in pattern.components if component.delta_m == delta_m]
        if group:
            assert sum(component.strength for component in group) == pytest.approx(1.0, abs=1e-12)

    for group_name in pattern.groups:
        shifts, strengths = pattern.group_tensors(
            group_name, like=torch.zeros((), dtype=torch.float32)
        )
        assert shifts.dtype == torch.float32
        assert strengths.dtype == torch.float32
        assert torch.all(strengths > 0)
        torch.testing.assert_close(strengths.sum(), torch.tensor(1.0))


def test_normal_triplet_has_expected_shifts_and_equal_strengths():
    components = zeeman_components(
        j_lower=0.0,
        j_upper=1.0,
        lande_lower=0.0,
        lande_upper=1.0,
    )
    assert len(components) == 3
    by_delta_m = {component.delta_m: component for component in components}
    assert set(by_delta_m) == {-1, 0, 1}
    assert by_delta_m[-1].wavelength_shift_factor == pytest.approx(1.0)
    assert by_delta_m[0].wavelength_shift_factor == pytest.approx(0.0)
    assert by_delta_m[1].wavelength_shift_factor == pytest.approx(-1.0)
    assert all(component.strength == pytest.approx(1.0) for component in components)


def test_group_centers_reproduce_effective_lande_factor():
    pattern = ZeemanPattern(2.0, 2.0, 1.84, 1.50)
    centers = {}
    for name in pattern.groups:
        shifts, strengths = pattern.group_tensors(name, like=torch.zeros((), dtype=torch.float64))
        centers[name] = (shifts * strengths).sum().item()

    assert centers["pi"] == pytest.approx(0.0, abs=1e-12)
    assert centers["blue"] == pytest.approx(-pattern.effective_lande, abs=1e-12)
    assert centers["red"] == pytest.approx(pattern.effective_lande, abs=1e-12)
