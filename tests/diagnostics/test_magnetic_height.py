import numpy as np
import pytest

from prom3theus.diagnostics.magnetic_height import _height_metrics


def test_height_metrics_distinguish_constant_magnetism_from_isothermal_or_zero_fields():
    magnetic = np.tile([[[100., -20., 40.]]], (2, 3, 1))
    temperature = np.tile([6000., 5500., 5000.], (2, 1))
    assert _height_metrics(magnetic, temperature, [0, .15, .3])["passed"]
    for field, thermal, failed in (
        (np.zeros_like(magnetic), temperature, "nonzero_magnetic_field"),
        (magnetic, np.full_like(temperature, 5000), "temperature_height_variation_present"),
        (magnetic * np.array([1, 2, 1])[None, :, None], temperature, "magnetic_height_independence"),
    ):
        report = _height_metrics(field, thermal, [0, .15, .3])
        assert not report["passed"]
        assert not report["checks"][failed]


def test_nonfinite_height_arrays_cannot_pass():
    with pytest.raises(ValueError, match="non-finite"):
        _height_metrics(np.full((2, 3, 3), np.nan), np.ones((2, 3)), [0, .15, .3])
