"""Validation tests for differentiable special-function modules."""

from __future__ import annotations

import pytest

from prom3theus.core import Faddeeva


@pytest.mark.parametrize("n_coefs", [True, 4.5])
def test_faddeeva_requires_an_integer_coefficient_count(n_coefs):
    with pytest.raises(TypeError, match="must be an integer"):
        Faddeeva(n_coefs)
