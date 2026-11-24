"""Ray-distance transfer agrees with columns and handles vacuum inflow."""

import numpy as np
import pytest
import torch

from prom3theus.muram.atmosphere_builder import build_native_atmosphere
from prom3theus.muram.ray_transfer import RayStokesRenderer
from prom3theus.rt.opacity import planck_lambda
from prom3theus.validation.reference_synthesis import LTESynthesizer
from prom3theus.validation.reference_transfer import GeometricHeightPath
from tests.muram.test_muram_synthesis import snapshot


WAVELENGTH = torch.tensor([6173.15, 6173.3352, 6173.52], dtype=torch.float64)
LINES = ("FeI_6173.3352",)


def test_vertical_ray_matches_native_column_reference():
    q = snapshot().synthesis_ready_quantities()
    atmosphere = build_native_atmosphere(q, dtype=torch.float64)
    renderer = RayStokesRenderer(LINES, WAVELENGTH, dtype=torch.float64)
    reference = LTESynthesizer(line_ids=LINES).double()
    reference.prepare_wavelength_grid(WAVELENGTH)
    expected = reference(atmosphere, path=GeometricHeightPath(atmosphere.geometric_height_m),
                         radiance_scale=1e13)
    fields = {name: np.ascontiguousarray(q[name][..., ::-1].reshape(4, -1))
              for name in ("temperature", "pressure", "bx", "by", "bz", "vx", "vy", "vz")}
    distance = (atmosphere.geometric_height_m[:, :1] - atmosphere.geometric_height_m)
    actual = renderer(fields, distance, torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)


def _constant_fields(depth=2):
    fields = {name: torch.zeros((2, depth), dtype=torch.float64)
              for name in ("bx", "by", "bz", "vx", "vy", "vz")}
    fields["temperature"] = torch.full((2, depth), 6000., dtype=torch.float64)
    fields["pressure"] = torch.full((2, depth), 10., dtype=torch.float64)
    return fields


def test_side_boundary_does_not_inject_a_planck_background():
    renderer = RayStokesRenderer(LINES, WAVELENGTH, dtype=torch.float64)
    fields = _constant_fields()
    distance = torch.tensor([[0., 0.01], [0., 0.01]], dtype=torch.float64)
    stokes = renderer(fields, distance, torch.tensor([True, False]))
    blackbody = planck_lambda(fields["temperature"][0, -1], renderer.wavelength_vacuum,
                             radiance_scale=1e13)
    torch.testing.assert_close(stokes[0, 0], blackbody, rtol=1e-12, atol=1e-12)
    assert torch.all(stokes[1, 0] >= 0)
    assert torch.all(stokes[1, 0] < 0.001 * blackbody)
    assert torch.all(stokes[:, 1:] == 0)


def test_duplicate_endpoint_padding_does_not_add_emission():
    renderer = RayStokesRenderer(LINES, WAVELENGTH, dtype=torch.float64)
    fields = _constant_fields(5)
    padded = torch.tensor([[0., 0., 2e4, 4e4, 4e4]] * 2, dtype=torch.float64)
    compact = torch.tensor([[0., 2e4, 4e4]] * 2, dtype=torch.float64)
    expected = renderer({name: value[:, [0, 2, 4]] for name, value in fields.items()},
                        compact, torch.tensor([True, False]))
    actual = renderer(fields, padded, torch.tensor([True, False]))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("distance", [[[1., 0.], [1., 0.]], [[0., 0.], [0., 0.]],
                                      [[0., float("nan")], [0., 1.]]])
def test_bad_ray_distance_rejected(distance):
    renderer = RayStokesRenderer(LINES, WAVELENGTH, dtype=torch.float64)
    with pytest.raises(ValueError, match="Ray distances"):
        renderer(_constant_fields(), distance, torch.ones(2, dtype=torch.bool))


def test_uniform_depth_sampling_converges_on_a_smooth_atmosphere():
    renderer = RayStokesRenderer(LINES, WAVELENGTH, dtype=torch.float64)
    spectra = []
    for count in (128, 256, 512):
        distance = torch.linspace(0., 2e5, count, dtype=torch.float64).expand(2, -1)
        fields = _constant_fields(count)
        fraction = distance / distance[:, -1:]
        fields["temperature"] = 4500. + 2000. * fraction
        fields["pressure"] = 10. * torch.exp(np.log(2e5) * fraction)
        fields["bz"][:] = 200.
        spectra.append(renderer(fields, distance, torch.ones(2, dtype=torch.bool)))
    coarse = (spectra[0] - spectra[2]).abs().max()
    finer = (spectra[1] - spectra[2]).abs().max()
    assert finer < coarse
    assert coarse < 0.005 * spectra[2][:, 0].abs().max()
