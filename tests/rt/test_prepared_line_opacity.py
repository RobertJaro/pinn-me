"""Reuse local line physics across spectral blocks without changing gradients."""
from unittest.mock import patch

import pytest
import torch

from prom3theus.rt.atomic import AtomicDatabase
from prom3theus.rt.polarization import PolarizedLineOpacity


def problem():
    atomic = AtomicDatabase()
    lines = tuple(atomic.get_line(name) for name in ('FeI_6301.5008', 'FeI_6302.4932'))
    opacity = PolarizedLineOpacity(lines, atomic).double()
    temperature = torch.tensor([[4800., 5700., 6500.], [5000., 5900., 6800.]], requires_grad=True)
    temperature = temperature.double().detach().requires_grad_()
    field = torch.tensor([350., -250., 900.], dtype=torch.float64).expand(2, 3, 3).clone().requires_grad_()
    arguments = dict(temperature=temperature, velocity_los=torch.full_like(temperature, 1200.),
                     microturbulence=torch.full_like(temperature, 800.), magnetic_field=field,
                     lower_level_populations={line.id: torch.full_like(temperature, 1e14) for line in lines},
                     damping_electron_density=None, damping_hydrogen_neutral=torch.full_like(temperature, 1e22))
    wavelength = torch.linspace(6301.2, 6302.8, 17, dtype=torch.float64)
    return opacity, arguments, wavelength


def derivatives(value, inputs):
    first = torch.autograd.grad(value.square().sum(), inputs, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(gradient.square().sum() for gradient in first), inputs, retain_graph=True)
    return first, second


def test_preparation_is_reused_across_chunks_with_first_and_second_derivatives():
    opacity, arguments, wavelength = problem()
    offsets = opacity.velocity_offsets_km_s(wavelength)
    continuum = torch.full((2, 3, len(wavelength)), 1e-7, dtype=torch.float64)
    with patch.object(opacity, '_orientation', wraps=opacity._orientation) as orientation:
        prepared = opacity.prepare(**arguments)
        chunks = [opacity.evaluate_prepared(prepared, block, continuum[..., start:start+5])
                  for start, block in zip(range(0, len(wavelength), 5), offsets.split(5, dim=-1))]
        assert orientation.call_count == 1
    actual = torch.cat(chunks, dim=-3)
    expected = opacity(wavelength, **arguments, continuum_extinction=continuum,
                       alpha500=torch.ones_like(arguments['temperature']), normalize_to_alpha500=False)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-15)
    inputs = (arguments['temperature'], arguments['magnetic_field'])
    for actual_order, expected_order in zip(derivatives(actual * 1e5, inputs), derivatives(expected * 1e5, inputs)):
        for a, b in zip(actual_order, expected_order):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)
            assert torch.isfinite(a).all()


def test_per_ray_wavelengths_broadcast_without_expanding_prepared_atmosphere():
    opacity, arguments, wavelength = problem()
    prepared = opacity.prepare(**arguments)
    waves = torch.stack((wavelength, wavelength + .012)).requires_grad_()
    offsets = opacity.velocity_offsets_km_s(waves)[:, :, None, :]
    continuum = torch.full((2, 3, len(wavelength)), 1e-7, dtype=torch.float64)
    actual = opacity.evaluate_prepared(prepared, offsets, continuum)
    expected = []
    for ray in range(2):
        point_arguments = {name: ({key: value[ray] for key, value in tensor.items()}
                                  if isinstance(tensor, dict) else tensor[ray] if tensor is not None else None)
                           for name, tensor in arguments.items()}
        expected.append(opacity(waves[ray], **point_arguments, continuum_extinction=continuum[ray],
                                alpha500=torch.ones(3, dtype=torch.float64), normalize_to_alpha500=False))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-15)
    actual_gradient = torch.autograd.grad(actual.square().sum(), waves, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), waves)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-20)
    assert prepared.lines[0].damping.shape == (2, 3)


@pytest.mark.parametrize('normalized', [True, False])
def test_diagnostics_keep_reference_tau500_normalization(normalized):
    opacity, arguments, wavelength = problem()
    continuum = torch.full((2, 3, len(wavelength)), 3e-7, dtype=torch.float64)
    alpha500 = torch.full((2, 3), 2e-7, dtype=torch.float64)
    matrix, diagnostics = opacity(wavelength, **arguments, continuum_extinction=continuum,
                                  alpha500=alpha500, normalize_to_alpha500=normalized, return_diagnostics=True)
    _, reference = opacity(wavelength, **arguments, continuum_extinction=continuum,
                           alpha500=alpha500, normalize_to_alpha500=True, return_diagnostics=True)
    torch.testing.assert_close(diagnostics.continuum_extinction_ratio, continuum / alpha500[..., None])
    for name, value in diagnostics.line_absorption.items():
        torch.testing.assert_close(value, reference.line_absorption[name])
    torch.testing.assert_close(diagnostics.eta_i, matrix[..., 0, 0])


def test_line_center_proxy_matches_independent_frequency_unit_formula_and_gradients():
    """Keep the legacy SI formula as an independent check of shared preparation."""
    from prom3theus.core import K_BOLTZMANN, SPEED_OF_LIGHT
    from prom3theus.rt.opacity import damping_rate, integrated_line_opacity

    opacity, arguments, _ = problem()
    temperature = arguments['temperature']
    turbulence = arguments['microturbulence'].requires_grad_()
    actual = opacity.line_center_extinction(
        temperature, turbulence, lower_level_populations=arguments['lower_level_populations'],
        damping_electron_density=None, damping_hydrogen_neutral=arguments['damping_hydrogen_neutral'])
    expected = torch.zeros_like(temperature)
    for index, line in enumerate(opacity.lines):
        frequency = temperature.new_tensor(opacity._rest_frequencies_hz[index])
        # Reference uses metres/second and Hz directly; prepared lines use km/s.
        velocity_width = torch.sqrt(2 * K_BOLTZMANN / opacity.line_atomic_mass_kg[index]
                                    * temperature + turbulence.square())
        frequency_width = frequency * velocity_width / SPEED_OF_LIGHT
        damping = damping_rate(line, temperature, torch.zeros_like(temperature),
                               arguments['damping_hydrogen_neutral']) / (4 * torch.pi * frequency_width)
        profile, _ = opacity.profile.dimensionless(torch.zeros_like(temperature), damping,
                                                   assume_nonnegative_damping=True)
        integrated = integrated_line_opacity(line, arguments['lower_level_populations'][line.id],
                                            temperature, rest_frequency_hz=frequency)
        expected = expected + integrated * profile / frequency_width
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)
    for actual_order, expected_order in zip(derivatives(actual * 1e5, (temperature, turbulence)),
                                            derivatives(expected * 1e5, (temperature, turbulence))):
        for a, b in zip(actual_order, expected_order):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
