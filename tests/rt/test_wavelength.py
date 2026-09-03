import pytest
import torch

from prom3theus.rt import air_to_vacuum_angstrom


def test_air_to_vacuum_uses_physical_optical_medium_and_preserves_precision():
    wavelength_air = torch.tensor(
        [6173.3352, 6301.5008, 6302.4932], dtype=torch.float64
    )
    wavelength_vacuum = air_to_vacuum_angstrom(wavelength_air)
    torch.testing.assert_close(
        wavelength_vacuum,
        torch.tensor(
            [6175.043573238136, 6303.2435744457125, 6304.236240909655],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=5.0e-10,
    )
    assert torch.all(wavelength_vacuum > wavelength_air)


def test_air_to_vacuum_rejects_low_precision_and_invalid_domain():
    with pytest.raises(TypeError, match="float32 and float64"):
        air_to_vacuum_angstrom(torch.tensor([6302.0], dtype=torch.float16))
    with pytest.raises(ValueError, match="2000 to 100000"):
        air_to_vacuum_angstrom(torch.tensor([1500.0]))
