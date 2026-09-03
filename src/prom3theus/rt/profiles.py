"""Differentiable Voigt and Faraday-Voigt line profiles.

The definitions in this module follow the convention used by
Landi Degl'Innocenti & Landolfi (2004).  For ``z = v + i a`` and the
Faddeeva function ``w(z)`` we define

``phi(v, a) = Re[w(z)] / sqrt(pi)`` and
``psi(v, a) = Im[w(z)] / sqrt(pi)``.

Consequently ``phi`` integrates to unity over the dimensionless frequency
coordinate ``v`` and ``psi`` is odd.  Dividing both profiles by a Doppler
width returns profiles in inverse-frequency (or inverse-wavelength) units.
Keeping this normalization explicit makes the line-opacity convention
independent of any particular inversion parameterization.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from prom3theus.core import Faddeeva


class VoigtFaraday(nn.Module):
    """Evaluate normalized absorption and anomalous-dispersion profiles.

    Parameters
    ----------
    n_coefs:
        Number of coefficients in the differentiable Weideman approximation
        of the Faddeeva function. Thirty-two coefficients are accurate to
        roughly float32 precision over the range relevant to the Fe I 630 nm
        lines.
    """

    def __init__(self, n_coefs: int = 32):
        super().__init__()
        self.faddeeva = Faddeeva(n_coefs=n_coefs)
        self.register_buffer(
            "inverse_sqrt_pi",
            torch.tensor(1.0 / math.sqrt(math.pi), dtype=torch.float64),
        )

    def dimensionless(
        self,
        frequency_offset: torch.Tensor,
        damping: torch.Tensor | float,
        *,
        assume_nonnegative_damping: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dimensionless ``(phi, psi)`` for ``v`` and damping ``a``.

        ``frequency_offset`` and ``damping`` follow normal PyTorch broadcasting
        rules. Production callers provide physically decoded, non-negative
        damping. The result preserves the real dtype and device of
        ``frequency_offset``.
        """

        v = torch.as_tensor(frequency_offset)
        if not v.is_floating_point():
            raise TypeError("frequency_offset must be a floating-point tensor")
        if v.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "Voigt/Faraday profiles support float32 and float64 tensors only"
            )
        a = torch.as_tensor(damping, dtype=v.dtype, device=v.device)
        v, a = torch.broadcast_tensors(v, a)
        if not assume_nonnegative_damping and (
            not torch.isfinite(v).all()
            or not torch.isfinite(a).all()
            or torch.any(a < 0)
        ):
            raise ValueError(
                "frequency_offset must be finite and damping must be finite and non-negative"
            )
        z = torch.complex(v, a)
        profile = (
            self.faddeeva.upper_half_plane(z)
            if assume_nonnegative_damping
            else self.faddeeva(z)
        )
        scale = self.inverse_sqrt_pi.to(dtype=v.dtype, device=v.device)
        return profile.real * scale, profile.imag * scale

    def forward(
        self,
        frequency_offset: torch.Tensor,
        damping: torch.Tensor | float,
        doppler_width: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return absorption and dispersion profiles.

        If ``doppler_width`` is omitted the profiles are dimensionless. When
        supplied, the returned profiles are divided by it, so their units are
        the reciprocal of the supplied width. Production callers construct a
        strictly positive width from bounded atmospheric variables.
        """

        absorption, dispersion = self.dimensionless(frequency_offset, damping)
        if doppler_width is None:
            return absorption, dispersion
        width = torch.as_tensor(
            doppler_width,
            dtype=absorption.dtype,
            device=absorption.device,
        )
        if not torch.isfinite(width).all() or torch.any(width <= 0):
            raise ValueError("doppler_width must be finite and strictly positive")
        return absorption / width, dispersion / width


def voigt_faraday(
    frequency_offset: torch.Tensor,
    damping: torch.Tensor | float,
    doppler_width: torch.Tensor | float | None = None,
    *,
    n_coefs: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional wrapper around :class:`VoigtFaraday`."""

    return VoigtFaraday(n_coefs=n_coefs)(
        frequency_offset,
        damping,
        doppler_width=doppler_width,
    )
