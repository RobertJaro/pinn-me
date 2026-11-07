"""Differentiable special functions used by spectral synthesis."""

from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn


def _horner(value: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


class _FixedPolynomial(torch.autograd.Function):
    """Keep the polynomial derivative instead of every Horner activation."""

    @staticmethod
    def forward(ctx, value, coefficients):
        derivative_coefficients = coefficients[:-1] * torch.arange(
            len(coefficients) - 1, 0, -1, device=coefficients.device)
        result = _horner(value, coefficients)
        derivative = _horner(value, derivative_coefficients)
        ctx.save_for_backward(value, derivative_coefficients, derivative)
        return result

    @staticmethod
    def backward(ctx, gradient):
        value, coefficients, derivative = ctx.saved_tensors
        # Preserve double backward when requested. Ordinary training reuses the
        # saved derivative, including the complex autograd conjugation rule.
        if torch.is_grad_enabled():
            derivative = _horner(value, coefficients)
        return gradient * derivative.conj(), None


def polyval(value: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Evaluate coefficients ordered highest degree first, with exact derivatives."""
    if (torch.is_grad_enabled() and value.requires_grad
            and not coefficients.requires_grad and coefficients.numel() >= 2):
        return _FixedPolynomial.apply(value, coefficients)
    return _horner(value, coefficients)


class Faddeeva(nn.Module):
    """Differentiable Weideman approximation of the Faddeeva function."""

    def __init__(self, n_coefs: int = 32):
        super().__init__()
        if isinstance(n_coefs, bool) or not isinstance(n_coefs, Integral):
            raise TypeError("n_coefs must be an integer.")
        if n_coefs < 2:
            raise ValueError("n_coefs must be at least 2.")
        n_coefs = int(n_coefs)
        m = 2 * n_coefs
        k = torch.arange(-m + 1, m, dtype=torch.float64)
        length = torch.sqrt(torch.tensor(n_coefs / math.sqrt(2.0), dtype=torch.float64))
        theta = k * torch.pi / m
        samples_at = length * torch.tan(theta / 2)
        samples = torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                torch.exp(-samples_at.square())
                * (length.square() + samples_at.square()),
            )
        )
        coefficients = torch.fft.fft(torch.fft.fftshift(samples)).real / (2 * m)
        coefficients = torch.flip(coefficients[1 : n_coefs + 1], dims=(0,))
        self.register_buffer("length", length)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer(
            "inverse_sqrt_pi",
            torch.tensor(1.0 / math.sqrt(math.pi), dtype=torch.float64),
        )

    def _upper_half_plane(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.real.dtype
        length = self.length.to(dtype=dtype, device=value.device)
        coefficients = self.coefficients.to(dtype=dtype, device=value.device)
        inverse_sqrt_pi = self.inverse_sqrt_pi.to(dtype=dtype, device=value.device)
        # Typed complex constant also keeps autograd conjugation in model dtype.
        imaginary_value = value.new_tensor(1j) * value
        denominator = length - imaginary_value
        transformed = (length + imaginary_value) / denominator
        return (
            2.0 * polyval(transformed, coefficients) / denominator.square()
            + inverse_sqrt_pi / denominator
        )

    def upper_half_plane(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate ``w(z)`` when the caller guarantees ``Im(z) >= 0``."""

        if not torch.is_complex(value):
            value = torch.complex(value, torch.zeros_like(value))
        return self._upper_half_plane(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(value):
            value = torch.complex(value, torch.zeros_like(value))
        upper = value.imag >= 0
        upper_value = self._upper_half_plane(torch.where(upper, value, -value))
        if bool(torch.all(upper)):
            return upper_value
        reflected_input = torch.where(upper, torch.zeros_like(value), value)
        reflected_value = 2.0 * torch.exp(-reflected_input.square()) - upper_value
        return torch.where(upper, upper_value, reflected_value)


__all__ = ["Faddeeva", "polyval"]
