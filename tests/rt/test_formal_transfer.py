"""Independent serial, analytic and derivative controls for parallel transfer."""
import pytest
import torch

from prom3theus.rt.formal_transfer import _transmission_above, formal_stokes, affine_transfer_reduce
from prom3theus.validation.reference_transfer import PolarizedFormalSolver, RayDistancePath


@pytest.mark.parametrize("layers", [1, 5, 8, 191])
@pytest.mark.parametrize("batch", [(), (2,), (2, 1)])
def test_parallel_matches_ordered_reference_and_two_derivatives(layers, batch):
    torch.manual_seed(41)
    shape = (*batch, layers + 1, 3)
    k = (torch.eye(4, dtype=torch.float64) + .1*torch.randn(*shape, 4, 4, dtype=torch.float64)).requires_grad_()
    source = torch.randn(*shape, 4, dtype=torch.float64, requires_grad=True)
    intervals = torch.logspace(-8, 1.5, layers, dtype=torch.float64)
    distance = torch.cat((intervals.new_zeros(1), intervals.cumsum(0))).requires_grad_()
    boundary = torch.randn(*batch, 3, 4, dtype=torch.float64, requires_grad=True)
    actual = formal_stokes(k, source, distance, bottom_boundary=boundary, precision="float64")
    expected = PolarizedFormalSolver()(k, source, torch.arange(layers+1).double(),
        path=RayDistancePath(distance), bottom_boundary=boundary)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    inputs = (k, source, distance, boundary)
    gradients = [torch.autograd.grad(v.square().sum(), inputs, create_graph=True, retain_graph=True)
                 for v in (actual, expected)]
    for a, e in zip(*gradients):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-8)
    second = [torch.autograd.grad(sum(g.square().sum() for g in group), inputs, retain_graph=True)
              for group in gradients]
    for a, e in zip(*second):
        torch.testing.assert_close(a, e, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("tau", [1e-9, 1., 1e4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_thermal_slab_matches_analytic_solution(tau, dtype):
    k = torch.eye(4, dtype=dtype).expand(257, 1, 4, 4)
    source = torch.tensor([2., 0., 0., 0.], dtype=dtype).expand(257, 1, 4)
    boundary = torch.tensor([[1., .1, -.2, .03]], dtype=dtype)
    actual = formal_stokes(k, source, torch.linspace(0, tau, 257, dtype=torch.float64),
                           bottom_boundary=boundary, precision="float64" if dtype == torch.float64 else "float32")
    expected = source[0] + torch.exp(torch.tensor(-tau, dtype=dtype)) * (boundary-source[0])
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    equilibrium = formal_stokes(k, source, torch.linspace(0, tau, 257, dtype=torch.float64))
    torch.testing.assert_close(equilibrium, source[0], rtol=0, atol=0)


def test_float32_inputs_support_explicit_double_transfer_for_reference_gradients():
    torch.manual_seed(7)
    k = (torch.eye(4) + .1*torch.randn(192, 2, 4, 4)).requires_grad_()
    source = torch.randn(192, 2, 4, requires_grad=True)
    distance = torch.linspace(0, 20, 192, dtype=torch.float64)
    actual = formal_stokes(k, source, distance, precision="float64")
    expected = PolarizedFormalSolver()(k.double(), source.double(), torch.arange(192).double(),
                                       path=RayDistancePath(distance)).float()
    torch.testing.assert_close(actual, expected)
    g0 = torch.autograd.grad(actual.square().sum(), (k, source))
    g1 = torch.autograd.grad(expected.square().sum(), (k, source))
    for a, e in zip(g0, g1):
        torch.testing.assert_close(a, e)


def test_transfer_gradcheck():
    torch.manual_seed(3)
    k = (torch.eye(4, dtype=torch.float64) + .1*torch.randn(4, 1, 4, 4, dtype=torch.float64)).requires_grad_()
    source = torch.randn(4, 1, 4, dtype=torch.float64, requires_grad=True)
    distance = torch.linspace(0, 2, 4, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(lambda *args: formal_stokes(*args, precision="float64"), (k, source, distance))


def test_nonmonotonic_distance_is_rejected():
    with pytest.raises(ValueError, match="increasing"):
        formal_stokes(torch.eye(4).expand(3, 1, 4, 4), torch.zeros(3, 1, 4), torch.tensor([0., 1., .5]))


@pytest.mark.parametrize("count", [1, 2, 6, 31, 32])
def test_transmission_above_each_layer_excludes_its_own_attenuation(count):
    rng = torch.Generator().manual_seed(42)
    identity = torch.eye(4, dtype=torch.float64)
    layers = identity + 0.1 * torch.randn(count, 2, 3, 4, 4, dtype=identity.dtype, generator=rng)
    above = _transmission_above(layers)
    expected = identity.expand_as(layers[0])
    for i, layer in enumerate(layers):
        torch.testing.assert_close(above[i], expected)
        expected = expected @ layer


def test_prefix_integral_is_sum_of_local_emission_attenuated_by_overlying_layers():
    torch.manual_seed(15)
    k = torch.eye(4, dtype=torch.float64) + .1*torch.randn(7, 2, 4, 4, dtype=torch.float64)
    attenuation = torch.matrix_exp(-k)
    source = torch.randn(7, 2, 4, dtype=torch.float64)
    boundary = torch.randn(2, 4, dtype=torch.float64)
    emissivity = (torch.eye(4)-attenuation) @ source.unsqueeze(-1)
    # Integrate emitted contributions. No intermediate intensity is solved.
    transmission = torch.eye(4, dtype=torch.float64).expand(2, 4, 4)
    emitted = torch.zeros(2, 4, 1, dtype=torch.float64)
    for i in range(7):
        emitted = emitted + transmission @ emissivity[i]
        transmission = transmission @ attenuation[i]
    expected = (emitted + transmission @ boundary.unsqueeze(-1)).squeeze(-1)
    actual = affine_transfer_reduce(attenuation, source, boundary)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_stratified_thermal_source_converges_to_analytic_integral():
    # S(tau)=a+b*tau and I_bottom=S(T): I_out=a+b*(1-exp(-T)).
    a, b, total_tau = 1., .3, 5.
    expected = a + b * (-torch.expm1(torch.tensor(-total_tau, dtype=torch.float64)))
    errors = []
    for count in (17, 33, 65):
        distance = torch.linspace(0, total_tau, count, dtype=torch.float64)
        source = torch.zeros(count, 1, 4, dtype=torch.float64)
        source[:, 0, 0] = a + b*distance
        k = torch.eye(4, dtype=torch.float64).expand(count, 1, 4, 4)
        prediction = formal_stokes(k, source, distance, precision="float64")
        errors.append((prediction[0, 0]-expected).abs())
    assert errors[1] < errors[0] / 3.5
    assert errors[2] < errors[1] / 3.5


@pytest.mark.parametrize('polarized', [False, True])
@pytest.mark.parametrize('tau', [1e-10, 1e-8, 1e-6, 1e-4])
def test_explicit_float32_retains_optically_thin_emission(tau, polarized):
    optical_depth = torch.tensor(tau, dtype=torch.float32, requires_grad=True)
    k = torch.eye(4).expand(2, 1, 4, 4) * optical_depth
    source = torch.tensor([1., 0., 0., 0.]).expand(2, 1, 4)
    actual = formal_stokes(k, source, torch.tensor([0., 1.]),
                           bottom_boundary=torch.zeros(1, 4),
                           precision='float32', polarized=polarized)[0, 0]
    expected = -torch.expm1(-optical_depth)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=0)
    gradient = torch.autograd.grad(actual, optical_depth, create_graph=True)[0]
    torch.testing.assert_close(gradient, (-optical_depth).exp(), rtol=2e-6, atol=0)
    second = torch.autograd.grad(gradient, optical_depth)[0]
    torch.testing.assert_close(second, -(-optical_depth).exp(), rtol=2e-6, atol=0)


def test_polarized_increment_preserves_thin_coupled_emission():
    from prom3theus.rt.polarized_exponential import polarized_matrix_exp
    matrix = torch.tensor([[-2., .2, -.1, .3], [.2, -2., .4, -.2],
                           [-.1, -.4, -2., .1], [.3, .2, -.1, -2.]]) * 1e-8
    _, increment = polarized_matrix_exp(matrix, return_increment=True)
    expected = matrix.double() + matrix.double() @ matrix.double() / 2
    torch.testing.assert_close(increment.double(), expected, rtol=3e-7, atol=1e-15)


@pytest.mark.parametrize('polarized', [False, True])
def test_float32_transfer_preserves_narrow_double_geometry_and_distance_gradients(polarized):
    # Adjacent jittered nodes can be closer than the ULP of their absolute
    # distance; the local separation itself remains representable in float32.
    distance = torch.tensor([1e6, 1e6 + .01, 1e6 + .03, 1e6 + .05],
                            dtype=torch.float64, requires_grad=True)
    assert (torch.diff(distance.float()) == 0).any()
    opacity = torch.tensor(10., requires_grad=True)
    k = torch.eye(4).expand(4, 1, 4, 4) * opacity
    source = torch.tensor([1.7, 0., 0., 0.]).expand(4, 1, 4)
    actual = formal_stokes(k, source, distance, bottom_boundary=torch.zeros(1, 4),
                           precision='float32', polarized=polarized)[0, 0]
    expected = source[0, 0, 0].double() * -torch.expm1(-opacity.double() * (distance[-1] - distance[0]))
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=0)
    actual_gradients = torch.autograd.grad(actual, (distance, opacity), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (distance, opacity))
    # Interior derivatives cancel analytically; allow float32 cancellation
    # relative to the endpoint sensitivity, rather than a relative error to zero.
    tolerance = 8 * torch.finfo(torch.float32).eps * expected_gradients[0].abs().max().item()
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=2e-6, atol=tolerance)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=2e-6, atol=0)


@pytest.mark.parametrize('distance', [[0., 1e-50], [0., float('nan')]])
def test_distance_validation_retains_finite_geometry_and_representable_intervals(distance):
    with pytest.raises(ValueError, match='Finite coefficients and strictly increasing'):
        formal_stokes(torch.eye(4).expand(2, 1, 4, 4), torch.zeros(2, 1, 4),
                      torch.tensor(distance, dtype=torch.float64), precision='float32')


@pytest.mark.parametrize('polarized', [False, True])
def test_float32_local_intervals_avoid_absolute_distance_cancellation(polarized):
    widths = torch.tensor([.01, .02, .02], requires_grad=True)
    distance = 1e6 + torch.cat((widths.new_zeros(1), widths.detach().cumsum(0)))
    assert (torch.diff(distance) == 0).any()
    opacity = torch.tensor(10., requires_grad=True)
    k = torch.eye(4).expand(4, 1, 4, 4) * opacity
    source = torch.tensor([1.7, 0., 0., 0.]).expand(4, 1, 4)
    actual = formal_stokes(k, source, distance, bottom_boundary=torch.zeros(1, 4),
                           intervals_m=widths, polarized=polarized)[0, 0]
    expected = source[0, 0, 0] * -torch.expm1(-opacity * widths.sum())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=0)
    actual_gradients = torch.autograd.grad(actual, (widths, opacity), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (widths, opacity))
    for value, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(value, reference, rtol=2e-6, atol=0)


@pytest.mark.parametrize('widths', [[1., 0.], [1., float('inf')], [1., -1.]])
def test_supplied_intervals_must_be_finite_and_positive(widths):
    with pytest.raises(ValueError, match='Finite coefficients and strictly increasing'):
        formal_stokes(torch.eye(4).expand(3, 1, 4, 4), torch.zeros(3, 1, 4),
                      torch.tensor([0., 1., 2.]), intervals_m=torch.tensor(widths))


def test_supplied_interval_count_must_match_depth():
    with pytest.raises(ValueError, match='intervals_m must match'):
        formal_stokes(torch.eye(4).expand(3, 1, 4, 4), torch.zeros(3, 1, 4),
                      torch.tensor([0., 1., 2.]), intervals_m=torch.ones(3))


@pytest.mark.parametrize('tau', [1e-6, 1., 1e4])
def test_supplied_uniform_intervals_match_zero_jitter_analytic_slab(tau):
    count = 257
    widths = torch.full((count - 1,), tau / (count - 1))
    distance = torch.linspace(0, tau, count)
    source = torch.tensor([1.7, 0., 0., 0.]).expand(count, 1, 4)
    boundary = torch.tensor([[.2, 0., 0., 0.]])
    actual = formal_stokes(torch.eye(4).expand(count, 1, 4, 4), source, distance,
                           bottom_boundary=boundary, intervals_m=widths, polarized=True)
    expected = boundary + (source[0] - boundary) * -torch.expm1(torch.tensor(-tau))
    torch.testing.assert_close(actual, expected, rtol=4e-6, atol=2e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_stiff_polarized_transfer_preserves_profiles_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    # Passive, noncommuting layers with dichroism and dispersion spanning thin
    # through opaque conditions. Compare arithmetic at identical input values,
    # keeping each Stokes component visible instead of letting I dominate.
    torch.manual_seed(4)
    count = 257
    coefficients = torch.randn(3, count, 3, 7)
    coefficients[..., 0] = coefficients[..., 1:4].norm(dim=-1) + .2
    coefficients *= torch.logspace(-8, 4, count)[None, :, None, None]
    coefficients = coefficients.to(device).requires_grad_()
    diagonal, a, b, c, x, y, z = coefficients.unbind(-1)
    propagation = torch.stack((diagonal, a, b, c, a, diagonal, z, -y,
                               b, -z, diagonal, x, c, y, -x, diagonal), -1)
    propagation = propagation.reshape(*coefficients.shape[:-1], 4, 4)
    source = torch.zeros(3, count, 3, 4)
    source[..., 0] = torch.linspace(.5, 2, count)[None, :, None]
    source = source.to(device).requires_grad_()
    distance = torch.linspace(0, 10, count)
    values, gradients = [], []
    for precision in ('float32', 'float64'):
        result = formal_stokes(propagation, source, distance,
                               precision=precision, polarized=True)
        values.append(result)
        gradients.append(torch.autograd.grad(result.square().mean(),
                                             (coefficients, source), retain_graph=True))
    actual, expected = values
    component_errors = (actual - expected).norm(dim=(0, 1)) / expected.norm(dim=(0, 1))
    assert torch.all(component_errors < 8e-6)
    for actual_gradient, expected_gradient in zip(*gradients):
        assert torch.isfinite(actual_gradient).all()
        assert (actual_gradient - expected_gradient).norm() < 8e-6 * expected_gradient.norm()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ordered_transmission_adjoint_and_higher_derivative(device):
    from prom3theus.rt.transmission import _ordered_products
    if device == "cuda":
        from torch.utils._triton import has_triton
        if not torch.cuda.is_available() or not has_triton():
            pytest.skip("CUDA transmission kernel requires a Triton-capable GPU")

    torch.manual_seed(14)
    # Ray-major storage, an odd layer count, and an incomplete GPU tile.
    layers = (torch.eye(4, dtype=torch.float64, device=device) +
              .1 * torch.randn(2, 7, 3, 4, 4, dtype=torch.float64, device=device))
    layers = layers.movedim(1, 0).requires_grad_()
    reference = [torch.eye(4, dtype=layers.dtype, device=device).expand_as(layers[0])]
    for layer in layers[:-1]:
        reference.append(reference[-1] @ layer)
    reference = torch.stack(reference)
    actual = _transmission_above(layers)
    torch.testing.assert_close(actual, reference)
    weights = torch.randn_like(actual).transpose(-1, -2)

    def gradient(value, create_graph):
        return torch.autograd.grad((weights * value.square()).sum(), layers,
                                   create_graph=create_graph, retain_graph=True)[0]

    expected = gradient(reference, True)
    torch.testing.assert_close(gradient(actual, False), expected)
    # Exercise the adjoint recurrence even on machines without CUDA.
    adjoints = _ordered_products(layers, 2 * weights * reference)
    torch.testing.assert_close(reference.transpose(-1, -2) @ adjoints, expected)
    differentiable = gradient(actual, True)
    torch.testing.assert_close(differentiable, expected)
    second = torch.autograd.grad(differentiable.square().sum(), layers)[0]
    torch.testing.assert_close(second, torch.autograd.grad(expected.square().sum(), layers)[0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixed_boundary", [False, True])
def test_packed_polarized_sweep_spectra_and_derivatives(device, fixed_boundary):
    from prom3theus.rt.polarized_exponential import propagation_matrix

    if device == "cuda":
        from torch.utils._triton import has_triton
        if not torch.cuda.is_available() or not has_triton():
            pytest.skip("CUDA Stokes sweep requires a Triton-capable GPU")

    torch.manual_seed(18)
    # Noncommuting layers, unequal widths, and 70 ray-wavelengths across a GPU tile boundary.
    shape = (2, 6, 35)
    options = {"device": device, "dtype": torch.float64}
    coefficients = .4 * torch.randn(*shape, 7, **options)
    coefficients[..., 0] = 1 + coefficients[..., 1:4].norm(dim=-1)
    coefficients.requires_grad_()
    source = torch.randn(*shape, 4, **options).requires_grad_()
    distance = torch.tensor([0., .001, .13, .4, 1., 2.], **options).requires_grad_()
    inputs = (coefficients, source, distance)
    boundary = None
    if fixed_boundary:
        boundary = torch.randn(2, 35, 4, **options).requires_grad_()
        inputs += (boundary,)
    result = formal_stokes(coefficients, source, distance, bottom_boundary=boundary,
                           precision="float64", polarized=True)
    reference = PolarizedFormalSolver()(propagation_matrix(coefficients), source, distance,
                                        path=RayDistancePath(distance), bottom_boundary=boundary)
    torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-11)
    direction = torch.randn_like(result)
    expected = torch.autograd.grad(reference, inputs, direction, create_graph=True)
    for create_graph in (False, True):
        gradients = torch.autograd.grad(result, inputs, direction, create_graph=create_graph, retain_graph=True)
        for actual, target in zip(gradients, expected):
            torch.testing.assert_close(actual, target, atol=1e-10, rtol=1e-8)
    def second_derivative(first):
        return torch.autograd.grad(sum(g.square().sum() for g in first), inputs)
    for actual, target in zip(second_derivative(gradients), second_derivative(expected)):
        torch.testing.assert_close(actual, target, atol=1e-8, rtol=1e-7)
