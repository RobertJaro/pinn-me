import torch

from pme.lte.transfer import PolarizedFormalSolver, scalar_formal_solution


def test_constant_layer_matches_matrix_exponential_solution():
    dtype = torch.float64
    log_tau500 = torch.tensor([-2.0, 0.0], dtype=dtype)
    matrix_value = torch.tensor(
        [
            [1.4, 0.12, -0.04, 0.09],
            [0.12, 1.4, 0.03, 0.02],
            [-0.04, -0.03, 1.4, 0.05],
            [0.09, -0.02, -0.05, 1.4],
        ],
        dtype=dtype,
    )
    matrix = matrix_value.expand(2, 3, 4, 4).clone()
    source_value = torch.tensor([0.8, 0.0, 0.0, 0.0], dtype=dtype)
    source = source_value.expand(2, 3, 4).clone()
    bottom = torch.tensor(
        [[1.1, 0.02, -0.01, 0.03], [0.9, -0.03, 0.02, -0.01], [1.0, 0.01, 0.04, 0.02]],
        dtype=dtype,
    )
    mu = 0.73

    actual = PolarizedFormalSolver()(matrix, source, log_tau500, mu=mu, bottom_boundary=bottom)
    delta_tau = (10.0 ** log_tau500[1] - 10.0 ** log_tau500[0]) / mu
    attenuation = torch.matrix_exp(-matrix_value * delta_tau)
    expected = torch.stack(
        [attenuation @ ray + (torch.eye(4, dtype=dtype) - attenuation) @ source_value for ray in bottom]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_nonuniform_depth_partition_uses_actual_optical_depth_intervals():
    """A constant layer has the same solution for any interior partition."""

    dtype = torch.float64
    uniform = torch.linspace(-3.0, 0.0, 25, dtype=dtype)
    nonuniform = torch.tensor(
        [-3.0, -2.91, -2.77, -2.43, -2.31, -1.96, -1.88, -1.53,
         -1.21, -1.04, -0.73, -0.51, -0.38, -0.14, 0.0],
        dtype=dtype,
    )

    def solve(grid):
        opacity = torch.full((grid.numel(), 3), 1.7, dtype=dtype)
        source = torch.full_like(opacity, 0.35)
        bottom = torch.full((3,), 1.2, dtype=dtype)
        return scalar_formal_solution(
            opacity,
            source,
            grid,
            mu=0.73,
            bottom_boundary=bottom,
        )

    optical_path = 1.7 * (10.0 ** uniform[-1] - 10.0 ** uniform[0]) / 0.73
    expected = 0.35 + (1.2 - 0.35) * torch.exp(-optical_path)
    torch.testing.assert_close(solve(uniform), expected.expand(3), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(solve(nonuniform), expected.expand(3), rtol=1e-12, atol=1e-12)


def test_scalar_and_diagonal_polarized_solutions_agree():
    dtype = torch.float64
    log_tau500 = torch.linspace(-4.0, 1.0, 17, dtype=dtype)
    opacity = torch.linspace(0.7, 1.8, 17, dtype=dtype)[:, None].expand(17, 5)
    source = torch.linspace(0.4, 1.2, 17, dtype=dtype)[:, None].expand(17, 5)
    bottom = torch.full((5,), 1.35, dtype=dtype)
    expected = scalar_formal_solution(
        opacity, source, log_tau500, mu=0.82, bottom_boundary=bottom
    )

    identity = torch.eye(4, dtype=dtype)
    matrix = opacity[..., None, None] * identity
    polarized_source = torch.zeros(17, 5, 4, dtype=dtype)
    polarized_source[..., 0] = source
    polarized_bottom = torch.zeros(5, 4, dtype=dtype)
    polarized_bottom[..., 0] = bottom
    actual = PolarizedFormalSolver()(
        matrix,
        polarized_source,
        log_tau500,
        mu=0.82,
        bottom_boundary=polarized_bottom,
    )

    torch.testing.assert_close(actual[..., 0], expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual[..., 1:], torch.zeros_like(actual[..., 1:]))


def test_geometric_and_optical_depth_solutions_are_equivalent_for_constant_alpha500():
    dtype = torch.float64
    log_tau500 = torch.tensor([-2.0, -1.0, 0.0], dtype=dtype)
    tau500 = torch.pow(torch.tensor(10.0, dtype=dtype), log_tau500)
    alpha500 = torch.full((2, 3), 2.5, dtype=dtype)
    # z increases upward and q increases inward. This mapping exactly satisfies
    # d tau500 = -alpha500 dz for the constant reference extinction.
    geometric_height_m = -(tau500 - tau500[0]) / alpha500[:, :1]
    matrix = torch.eye(4, dtype=dtype).expand(2, 3, 2, 4, 4).clone()
    matrix = matrix * torch.tensor(1.4, dtype=dtype)
    source = torch.zeros(2, 3, 2, 4, dtype=dtype)
    source[..., 0] = torch.tensor((0.6, 0.8, 1.0), dtype=dtype)[None, :, None]
    bottom = torch.zeros(2, 2, 4, dtype=dtype)
    bottom[..., 0] = 1.2
    solver = PolarizedFormalSolver()

    optical = solver(
        matrix, source, log_tau500, mu=0.71, bottom_boundary=bottom
    )
    geometric = solver(
        matrix,
        source,
        log_tau500,
        mu=0.71,
        bottom_boundary=bottom,
        geometric_height_m=geometric_height_m,
        alpha500=alpha500,
    )

    torch.testing.assert_close(geometric, optical, rtol=2e-12, atol=2e-12)


def test_exact_ray_distances_replace_mu_path_scaling():
    dtype = torch.float64
    q = torch.tensor([-2.0, -1.0, 0.0], dtype=dtype)
    height = torch.tensor([[4.0, 1.0, -5.0]], dtype=dtype)
    alpha500 = torch.full((1, 3), 0.2, dtype=dtype)
    matrix = 1.3 * torch.eye(4, dtype=dtype).expand(1, 3, 2, 4, 4).clone()
    source = torch.zeros(1, 3, 2, 4, dtype=dtype)
    source[..., 0] = torch.tensor((0.5, 0.7, 0.9), dtype=dtype)[None, :, None]
    mu = 0.6
    ray_distance = 100.0 - height / mu
    solver = PolarizedFormalSolver()

    plane_parallel = solver(
        matrix,
        source,
        q,
        mu=mu,
        geometric_height_m=height,
        alpha500=alpha500,
    )
    ray_traced = solver(
        matrix,
        source,
        q,
        mu=torch.tensor(float("nan")),
        geometric_height_m=height,
        alpha500=alpha500,
        ray_distance_m=ray_distance,
    )
    torch.testing.assert_close(ray_traced, plane_parallel, rtol=2e-12, atol=2e-12)


def test_formal_solver_allows_temporary_negative_ray_layer_distance():
    dtype = torch.float64
    q = torch.tensor([-2.0, -1.0, 0.0], dtype=dtype)
    height = torch.tensor([[4.0, 1.0, -5.0]], dtype=dtype)
    alpha500 = torch.full((1, 3), 0.2, dtype=dtype)
    matrix = 1.3 * torch.eye(4, dtype=dtype).expand(1, 3, 1, 4, 4).clone()
    source = torch.zeros(1, 3, 1, 4, dtype=dtype)
    source[..., 0] = torch.tensor((0.5, 0.7, 0.9), dtype=dtype)[None, :, None]
    ray_distance = torch.tensor(
        [[100.0, 99.0, 102.0]], dtype=dtype, requires_grad=True
    )

    emergent = PolarizedFormalSolver()(
        matrix,
        source,
        q,
        geometric_height_m=height,
        alpha500=alpha500,
        ray_distance_m=ray_distance,
    )
    ordered_equivalent = PolarizedFormalSolver()(
        matrix,
        source,
        q,
        geometric_height_m=height,
        alpha500=alpha500,
        ray_distance_m=torch.tensor([[100.0, 101.0, 104.0]], dtype=dtype),
    )
    emergent.square().sum().backward()

    assert torch.isfinite(emergent).all()
    torch.testing.assert_close(emergent, ordered_equivalent)
    assert ray_distance.grad is not None
    assert torch.isfinite(ray_distance.grad).all()


def test_geometric_solution_propagates_height_and_extinction_gradients():
    dtype = torch.float64
    q = torch.tensor([-2.0, -1.0, 0.0], dtype=dtype)
    height = torch.tensor([[0.0, -0.2, -0.7]], dtype=dtype, requires_grad=True)
    alpha500 = torch.full((1, 3), 1.3, dtype=dtype, requires_grad=True)
    matrix = torch.eye(4, dtype=dtype).expand(1, 3, 1, 4, 4).clone()
    source = torch.zeros(1, 3, 1, 4, dtype=dtype)
    source[..., 0] = 0.8

    emergent = PolarizedFormalSolver()(
        matrix,
        source,
        q,
        geometric_height_m=height,
        alpha500=alpha500,
    )
    emergent.square().sum().backward()

    assert height.grad is not None and torch.isfinite(height.grad).all()
    assert alpha500.grad is not None and torch.isfinite(alpha500.grad).all()


def test_formal_solution_propagates_finite_gradients():
    dtype = torch.float64
    log_tau500 = torch.linspace(-3.0, 0.5, 9, dtype=dtype)
    diagonal = torch.full((9, 4), 1.1, dtype=dtype, requires_grad=True)
    matrix = diagonal[..., None, None] * torch.eye(4, dtype=dtype)
    source = torch.zeros(9, 4, 4, dtype=dtype)
    source[..., 0] = torch.linspace(0.5, 1.0, 9, dtype=dtype)[:, None]
    source.requires_grad_()
    stokes = PolarizedFormalSolver()(matrix, source, log_tau500)
    stokes.square().sum().backward()

    assert diagonal.grad is not None and torch.isfinite(diagonal.grad).all()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert torch.count_nonzero(diagonal.grad) > 0
    assert torch.count_nonzero(source.grad) > 0


def test_substepped_polarized_attenuation_preserves_matrix_exponential_gradient():
    dtype = torch.float64
    grid = torch.tensor([-2.0, 0.0], dtype=dtype)
    matrix_value = torch.tensor(
        [
            [9.4, 0.12, -0.04, 0.09],
            [0.12, 9.4, 0.03, 0.02],
            [-0.04, -0.03, 9.4, 0.05],
            [0.09, -0.02, -0.05, 9.4],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    source = torch.tensor(
        [[[0.7, 0.0, 0.0, 0.0]], [[0.9, 0.0, 0.0, 0.0]]],
        dtype=dtype,
    )
    bottom = torch.tensor([[1.1, 0.03, -0.02, 0.01]], dtype=dtype)
    matrix = matrix_value.expand(2, 1, 4, 4)
    actual = PolarizedFormalSolver()(
        matrix, source, grid, mu=0.82, bottom_boundary=bottom
    )
    actual_gradient = torch.autograd.grad(actual.square().sum(), matrix_value)[0]

    direct_matrix = matrix_value.detach().clone().requires_grad_()
    path = (10.0 ** grid[1] - 10.0 ** grid[0]) / 0.82
    attenuation = torch.matrix_exp(-direct_matrix * path)
    midpoint_source = source.mean(dim=0)
    expected = (
        attenuation @ bottom[0]
        + (torch.eye(4, dtype=dtype) - attenuation) @ midpoint_source[0]
    ).unsqueeze(0)
    expected_gradient = torch.autograd.grad(
        expected.square().sum(), direct_matrix
    )[0]

    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-10, atol=2e-12
    )


def test_optically_thick_float32_polarized_layer_has_finite_backward():
    dtype = torch.float32
    grid = torch.tensor([-2.0, 0.0], dtype=dtype)
    matrix_value = torch.tensor(
        [
            [306.0, 300.0, -5.0, 3.0],
            [300.0, 306.0, 2.0, -4.0],
            [-5.0, -2.0, 306.0, 1.0],
            [3.0, 4.0, -1.0, 306.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    matrix = matrix_value.expand(2, 1, 4, 4)
    source = torch.tensor(
        [[[0.7, 0.0, 0.0, 0.0]], [[0.9, 0.0, 0.0, 0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    bottom = torch.tensor([[1.1, 0.03, -0.02, 0.01]], dtype=dtype)
    stokes = PolarizedFormalSolver()(
        matrix, source, grid, mu=0.59, bottom_boundary=bottom
    )
    stokes.square().sum().backward()

    assert torch.isfinite(stokes).all()
    assert matrix_value.grad is not None
    assert torch.isfinite(matrix_value.grad).all()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
