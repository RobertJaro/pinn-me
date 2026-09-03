import numpy as np
import pytest

from prom3theus.rt import (
    disk_center_continuum_radiance,
    disk_center_intensity_radiance,
    load_solar_reference,
)


def test_absolute_references_declare_standard_air_and_si_radiance():
    for resource_name in (
        "hmi_stokes/solar_reference_617nm.json",
        "hinode_sp/solar_reference_630nm.json",
    ):
        reference = load_solar_reference(resource_name)
        assert reference["units"]["wavelength"] == "standard-air angstrom"
        assert reference["units"]["continuum_radiance"] == "W m^-3 sr^-1"
        lower, upper = reference["wavelength_range_air_angstrom"]
        assert lower <= reference["wavelength_air_angstrom"][0]
        assert upper >= reference["wavelength_air_angstrom"][-1]


@pytest.mark.parametrize(
    "interpolate",
    [disk_center_continuum_radiance, disk_center_intensity_radiance],
)
def test_absolute_reference_interpolation_rejects_nonfinite_wavelengths(interpolate):
    reference = load_solar_reference("hinode_sp/solar_reference_630nm.json")
    with pytest.raises(ValueError, match="outside"):
        interpolate(reference, np.array([6301.0, np.nan]))
