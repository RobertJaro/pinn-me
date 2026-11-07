"""Analytic exponential of the seven-coefficient polarized transfer matrix.

The constant-layer solution follows Landi Degl'Innocenti & Landi Degl'Innocenti
(1985); see also Appendix A of https://doi.org/10.1093/mnras/stx3162.
Only the local propagator is evaluated here, never an iteration over intensities.
"""

import math

import torch


def polarized_matrix_exp(matrix: torch.Tensor, *, return_increment=False):
    """Evaluate exp(matrix) for the LTE propagation-matrix structure.

    All four diagonal entries are equal, the first row/column are symmetric,
    and the spatial 3x3 off-diagonal block is antisymmetric. In production the
    argument is -K*ds. Arbitrary matrices must use torch.matrix_exp instead.

    Writing matrix=t*I+G, Cayley-Hamilton gives G**4=u*G**2+v*I. Thus its
    exponential is a cubic in G. The small-invariant series avoids singular
    eigenvalue derivatives at zero field and at degenerate/null generators.
    Hyperbolic functions include scalar attenuation before exponentiation, so
    optically thick absorbing layers do not overflow internally. With
    ``return_increment=True``, also return exp(matrix)-I evaluated without
    thin-layer cancellation for the emitted-source contribution.
    """
    scalar = matrix[..., 0, 0]
    identity = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
    generator = matrix - scalar[..., None, None] * identity
    dichroism = generator[..., 0, 1:]
    dispersion = torch.stack((generator[..., 2, 3], -generator[..., 1, 3],
                              generator[..., 1, 2]), dim=-1)
    u = dichroism.square().sum(-1) - dispersion.square().sum(-1)
    v = (dichroism * dispersion).sum(-1).square()
    discriminant = u.square() + 4 * v
    small = discriminant < .01
    # Keep inactive analytic branches finite, including their backward graphs.
    delta = discriminant.clamp_min(.01).sqrt()
    hyperbolic_squared = (.5 * (delta + u)).clamp_min(0.)
    trigonometric_squared = (.5 * (delta - u)).clamp_min(0.)
    h = hyperbolic_squared.clamp_min(1e-8).sqrt()
    r = trigonometric_squared.clamp_min(1e-8).sqrt()
    attenuation = scalar.exp()
    positive, negative = (scalar + h).exp(), (scalar - h).exp()
    cosh = .5 * (positive + negative)
    sinhc = .5 * positive * (-torch.expm1(-2 * h)) / h
    x, y = hyperbolic_squared, trigonometric_squared
    cosh = torch.where(x < 1e-8, attenuation * (1 + x / 2 + x.square() / 24), cosh)
    sinhc = torch.where(x < 1e-8, attenuation * (1 + x / 6 + x.square() / 120), sinhc)
    cosine = torch.where(y < 1e-8, 1 - y / 2 + y.square() / 24, r.cos()) * attenuation
    sinc = torch.where(y < 1e-8, 1 - y / 6 + y.square() / 120, r.sin() / r) * attenuation
    coefficients = ((y * cosh + x * cosine) / delta,
                    (y * sinhc + x * sinc) / delta,
                    (cosh - cosine) / delta, (sinhc - sinc) / delta)

    # Six terms suffice at delta < 0.1 in float64. Mask before polynomial
    # evaluation to avoid overflowing unused series in very thick layers.
    small_u, small_v = torch.where(small, u, 0.), torch.where(small, v, 0.)
    previous, polynomial = torch.zeros_like(u), torch.ones_like(u)
    series = [torch.zeros_like(u), torch.ones_like(u),
              torch.zeros_like(u), torch.zeros_like(u)]
    for order in range(6):
        series[0] = series[0] + small_v * polynomial / math.factorial(2 * order + 4)
        series[1] = series[1] + small_v * polynomial / math.factorial(2 * order + 5)
        series[2] = series[2] + polynomial / math.factorial(2 * order + 2)
        series[3] = series[3] + polynomial / math.factorial(2 * order + 3)
        previous, polynomial = polynomial, small_u * polynomial + small_v * previous
    c0, c1, c2, c3 = (torch.where(small, attenuation * value, coefficient)[..., None, None]
                      for value, coefficient in zip((1 + series[0], *series[1:]), coefficients))
    squared = generator @ generator
    nonidentity = c1 * generator + c2 * squared + c3 * (squared @ generator)
    exponential = c0 * identity + nonidentity
    if not return_increment:
        return exponential
    # exp(M)-I must not be recovered by subtracting two nearly equal matrices.
    # This scalar coefficient retains thin-layer emission in float32.
    scalar_increment = torch.expm1(scalar)
    hyperbolic_increment = .5 * (torch.expm1(scalar + h) + torch.expm1(scalar - h)) - scalar_increment
    hyperbolic_increment = torch.where(
        x < 1e-8, attenuation * (x / 2 + x.square() / 24), hyperbolic_increment)
    trigonometric_increment = -2 * attenuation * (r / 2).sin().square()
    trigonometric_increment = torch.where(
        y < 1e-8, attenuation * (-y / 2 + y.square() / 24), trigonometric_increment)
    c0_increment = scalar_increment + (y * hyperbolic_increment + x * trigonometric_increment) / delta
    c0_increment = torch.where(small, scalar_increment + attenuation * series[0], c0_increment)
    increment = c0_increment[..., None, None] * identity + nonidentity
    return exponential, increment
