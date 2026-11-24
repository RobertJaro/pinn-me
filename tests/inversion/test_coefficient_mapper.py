"""Local invariants, end-to-end gradients and portable reconstruction for a surrogate."""

import pytest
import torch

from prom3theus.rt.coefficient_mapper import (
    CoefficientMapper,
    FORMAT,
    independent_coefficients,
    propagation_matrix,
)
from prom3theus.rt.wavelength import CenteredWavelength
from prom3theus.config.synthesis_schema import LTEWindowConfig
from test_lte_integration import make_system


def mapper(**options):
    return CoefficientMapper(
        line_ids=["FeI_6302.4932"], bounds=[6300.0, 6307.0], width=12, knots=21, **options
    )


def test_teacher_and_inversion_use_identical_standardized_log_density_inputs():
    from prom3theus.rt.local_coefficients import LTELocalCoefficients
    from prom3theus.surrogate.teacher import training_pair

    model = mapper()
    teacher = LTELocalCoefficients(line_ids=model.spec["line_ids"])
    points = torch.tensor(
        [[6000.0, 12000.0, 120.0, 450.0, 60.0, -30.0, 200.0]], requires_grad=True
    )
    wave = torch.tensor([6302.0])
    inputs, _ = training_pair(teacher, points, wave)
    temperature, pressure, xi, velocity = points[:, :4].unbind(-1)
    density = teacher.plasma_provider.mass_density(temperature, pressure)
    basis = torch.eye(3)
    state = dict(
        temperature=temperature,
        mass_density=density,
        microturbulence=xi,
        velocity_field=-velocity[:, None] * basis[2],
        magnetic_field=points[:, 4:],
    )
    prepared = model.prepare_atmosphere(state, direction=basis[2], stokes_basis=basis)
    torch.testing.assert_close(prepared, inputs)
    model.input_center.copy_(torch.arange(model.input_center.numel()))
    model.input_scale.fill_(2.0)
    network_inputs = []
    hook = model.network.register_forward_pre_hook(
        lambda _, args: network_inputs.append(args[0])
    )
    model.coefficients(prepared, model.prepare_spectrum(wave))
    hook.remove()
    encoded = network_inputs[0][:, :2]
    expected = (
        torch.stack((temperature.log(), density.log()), -1) - model.input_center[:2]
    ) / 2
    torch.testing.assert_close(encoded, expected)
    (gradient,) = torch.autograd.grad(encoded.sum(), points)
    assert torch.isfinite(gradient).all() and gradient[0, 1] != 0


def test_unversioned_input_spec_preserves_legacy_pressure_checkpoint(tmp_path):
    model = mapper(spectral_frame="observer", linear_residual=False)
    spec = dict(model.spec)
    del spec["thermodynamic_input"]
    path = tmp_path / "pressure.pt"
    torch.save(
        dict(format="prom3theus.lte_mapper.v1", spec=spec, mapper=model.state_dict()),
        path,
    )
    assert CoefficientMapper.load(path).thermodynamic_input == "gas_pressure"
    assert CoefficientMapper.from_saved_spec(spec).thermodynamic_input == "gas_pressure"


def test_legacy_density_checkpoint_preserves_velocity_encoding_and_predictions(tmp_path):
    model = mapper(spectral_frame="observer", linear_residual=False)
    path = tmp_path / "observer.pt"
    torch.save(dict(format=FORMAT, spec=model.spec, mapper=model.state_dict()), path)
    restored = CoefficientMapper.load(path)
    points = torch.tensor([[6000., 2e-4, 0., 1200., 100., 80., 400.]])
    wave = torch.tensor([6303.2, 6304.1])
    assert restored.features(points)[0, 3] == points[0, 3] / 1000
    assert restored.linear is None
    torch.testing.assert_close(restored.coefficients(points, restored.prepare_spectrum(wave)),
                               model.coefficients(points, model.prepare_spectrum(wave)), rtol=0, atol=0)


def test_linear_and_residual_branches_add_in_knot_space_and_learn_together():
    model = mapper()
    points = torch.tensor([[6000., 2e-4, 0., 0., 100., 80., 400.]])
    features = model.features(points)
    assert torch.count_nonzero(model.network(features)) == 0
    with torch.no_grad():
        model.network[-1].bias.fill_(0.1)
    expected = model.linear(features) + model.network(features)
    grid = torch.linspace(*model.spec["bounds"], model.spec["knots"], dtype=torch.float64)
    result = model.coefficients(points, model.prepare_spectrum(grid))
    expected_i = (model.log_extinction_center + model.log_extinction_scale *
                  expected.unflatten(-1, (len(grid), 5))[..., 0]).exp()
    torch.testing.assert_close(result[..., 0], expected_i, atol=0, rtol=1e-5)
    result[..., 0].log().mean().backward()
    for weight in (model.linear.weight, model.network[-1].weight):
        assert weight.grad is not None and weight.grad.abs().sum() > 0


def test_doppler_lookup_agrees_with_rest_profiles_and_preserves_velocity_gradient():
    from prom3theus.core import SPEED_OF_LIGHT

    model = mapper().double()
    points = torch.tensor([[6000., 2e-4, 0., 0., 100., 80., 400.]], dtype=torch.float64).repeat(3, 1)
    rest = points.clone()
    points[:, 3] = torch.tensor([-2000., 0., 2000.])
    points.requires_grad_()
    torch.testing.assert_close(model.features(points), model.features(rest))
    wave = torch.tensor([6300., 6303.12, 6303.46, 6307.], dtype=torch.float64)
    spectrum = model.prepare_spectrum(CenteredWavelength(6300., wave - 6300.))
    moving = model.evaluate_spectrum(points, spectrum, radiance_scale=1e13)
    beta = points[:, 3:4] / SPEED_OF_LIGHT
    rest_wave = (wave * ((1-beta)/(1+beta)).sqrt()).clamp(*model.spec["bounds"])
    static = model.coefficients(rest, model.prepare_spectrum(rest_wave))
    actual = independent_coefficients(moving.propagation_m1)
    torch.testing.assert_close(actual, static, atol=0, rtol=1e-9)
    torch.testing.assert_close(moving.source, moving.source[:1].expand_as(moving.source))
    slope = torch.autograd.grad(actual[:, 1, 0].log().sum(), points)[0][:, 3]
    expected_slope = torch.autograd.grad(static[:, 1, 0].log().sum(), points)[0][:, 3]
    assert slope.abs().min() > 0
    torch.testing.assert_close(slope, expected_slope, atol=0, rtol=1e-9)


def test_basis_batched_shift_and_polarity():
    model = mapper()
    points = torch.tensor(
        [
            [6000.0, 2e-4, 300.0, 600.0, 10.0, 30.0, 300.0],
            [5800.0, 1e-4, 0.0, -500.0, 100.0, 60.0, 150.0],
        ]
    ).unsqueeze(1)
    shift = torch.tensor([0.01, -0.02], requires_grad=True)
    offset = torch.linspace(1.0, 4.0, 15) + shift[:, None, None]
    spectrum = model.prepare_spectrum(CenteredWavelength(6300.0, offset))
    prediction = model.coefficients(points, spectrum)
    assert prediction.shape == (2, 1, 15, 7)
    torch.testing.assert_close(
        independent_coefficients(propagation_matrix(prediction)), prediction
    )
    for i in range(2):
        single = model.coefficients(
            points[i], model.prepare_spectrum(CenteredWavelength(6300.0, offset[i]))
        )
        torch.testing.assert_close(single, prediction[i])
    prediction.sum().backward()
    assert shift.grad.isfinite().all() and shift.grad.abs().sum() > 0

    flipped = points.clone()
    flipped[..., 4:] *= -1
    mirrored = model.coefficients(flipped, spectrum)
    torch.testing.assert_close(
        mirrored, prediction * points.new_tensor([1, 1, 1, -1, 1, 1, -1])
    )
    rotated = points.clone()
    rotated[..., 4], rotated[..., 5] = -points[..., 5], points[..., 4]
    torch.testing.assert_close(
        model.coefficients(rotated, spectrum),
        prediction * points.new_tensor([1, -1, -1, 1, -1, -1, 1]),
    )
    flipped[..., 4:] = 0
    assert torch.count_nonzero(model.coefficients(flipped, spectrum)[..., 1:]) == 0
    assert torch.all(prediction[..., 1:4].norm(dim=-1) <= prediction[..., 0])


@pytest.mark.parametrize("options", [
    {},
    {"architecture": "tanh", "hidden_layers": 4},
    {"architecture": "siren", "hidden_layers": 6, "first_omega_0": 12., "hidden_omega_0": 2.},
])
def test_mapper_window_retains_atmosphere_gradient_and_restores_without_external_checkpoint(
    tmp_path, options,
):
    model = mapper(**options)
    path = tmp_path / "mapper.pt"
    torch.save(dict(format=FORMAT, spec=model.spec, mapper=model.state_dict()), path)
    atmosphere, window, term, batch = make_system(
        coefficient_backend="nn_surrogate",
        mapper_checkpoint=path,
        coarse_to_fine=dict(
            enabled=True, fine_sample_count=4, uniform_weight_floor=0.05
        ),
    )
    window.eval()
    term.eval()
    profile = term.predict(batch)
    profile.square().sum().backward()
    assert all(
        not p.requires_grad and p.grad is None for p in window.mapper.parameters()
    )
    assert any(
        p.grad is not None and p.grad.isfinite().all() and p.grad.abs().max() > 0
        for p in atmosphere.parameters()
    )
    path.unlink()
    options = window.construction
    spec = options.pop("mapper_spec")
    restored = type(window)(atmosphere, configuration=options, mapper_spec=spec)
    restored.load_state_dict(window.state_dict())
    object.__setattr__(term, "synthesis_window", restored.eval())
    torch.testing.assert_close(profile, term.predict(batch))


def test_backend_contracts(tmp_path):
    base = dict(
        id="fe",
        line_ids=("FeI_6302.4932",),
        wavelength_bounds_vacuum_angstrom=(6300, 6307),
        radiance_scale=1e13,
    )
    assert LTEWindowConfig(**base).coefficient_backend == "exact_lte"
    with pytest.raises(ValueError, match="mapper_checkpoint"):
        LTEWindowConfig(**base, coefficient_backend="nn_surrogate")
    with pytest.raises(ValueError, match="mapper_checkpoint"):
        LTEWindowConfig(**base, mapper_checkpoint=tmp_path / "a.pt")
    model = mapper()
    for override, message in [
        ({"line_ids": ["FeI_6173.3352"]}, "line list"),
        ({"bounds": [6301.0, 6306.0]}, "support"),
    ]:
        path = tmp_path / "mapper.pt"
        torch.save(
            dict(
                format=FORMAT,
                spec={**model.spec, **override},
                mapper=model.state_dict(),
            ),
            path,
        )
        with pytest.raises(ValueError, match=message):
            make_system(coefficient_backend="nn_surrogate", mapper_checkpoint=path)
