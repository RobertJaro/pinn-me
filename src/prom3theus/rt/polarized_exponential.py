"""Analytic exponential of the seven-coefficient polarized transfer matrix.

The constant-layer solution follows Landi Degl'Innocenti & Landi Degl'Innocenti
(1985); see also Appendix A of https://doi.org/10.1093/mnras/stx3162.
Only the local propagator is evaluated here, never an iteration over intensities.
"""

import math

import torch


def independent_coefficients(matrix):
    """Return eta I,Q,U,V and rho Q,U,V in inverse metres."""
    return torch.stack(
        (
            *matrix[..., 0, :].unbind(-1),
            matrix[..., 2, 3],
            matrix[..., 3, 1],
            matrix[..., 1, 2],
        ),
        -1,
    )


def propagation_matrix(coefficients):
    i, q, u, v, x, y, z = coefficients.unbind(-1)
    return torch.stack(
        (
            torch.stack((i, q, u, v), -1),
            torch.stack((q, i, z, -y), -1),
            torch.stack((u, -z, i, x), -1),
            torch.stack((v, y, -x, i), -1),
        ),
        -2,
    )


def _small_invariant_series(u, v):
    """Six Cayley–Hamilton terms shared by the four scalar coefficients."""
    u2 = u.square()
    p2 = u2 + v
    p3 = u * (p2 + v)
    p4 = u * p3 + v * p2
    p5 = u * p4 + v * p3

    def series(start):
        return (1 / math.factorial(start) + u / math.factorial(start + 2)
                + p2 / math.factorial(start + 4) + p3 / math.factorial(start + 6)
                + p4 / math.factorial(start + 8) + p5 / math.factorial(start + 10))

    return v * series(4), 1 + v * series(5), series(2), series(3)


def exponential_coefficients(scalar, dichroism, dispersion):
    """Cubic coefficients of exp(t I + G), including an accurate c0 - 1."""
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
    increment0, series1, series2, series3 = _small_invariant_series(small_u, small_v)
    c0 = torch.where(small, attenuation * (1 + increment0), coefficients[0])
    c1 = torch.where(small, attenuation * series1, coefficients[1])
    c2 = torch.where(small, attenuation * series2, coefficients[2])
    c3 = torch.where(small, attenuation * series3, coefficients[3])
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
    c0_increment = torch.where(small, scalar_increment + attenuation * increment0, c0_increment)
    return c0, c0_increment, c1, c2, c3


def polarized_matrix_exp(matrix: torch.Tensor, *, return_increment=False):
    """Analytic exponential of a polarized transfer matrix using Cayley–Hamilton."""
    scalar = matrix[..., 0, 0]
    identity = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
    generator = matrix - scalar[..., None, None] * identity
    dichroism = generator[..., 0, 1:]
    dispersion = torch.stack((generator[..., 2, 3], generator[..., 3, 1], generator[..., 1, 2]), -1)
    c0, delta0, c1, c2, c3 = exponential_coefficients(scalar, dichroism, dispersion)
    squared = generator @ generator
    nonidentity = (c1[..., None, None] * generator + c2[..., None, None] * squared
                   + c3[..., None, None] * (squared @ generator))
    exponential = c0[..., None, None] * identity + nonidentity
    if return_increment:
        return exponential, delta0[..., None, None] * identity + nonidentity
    return exponential
