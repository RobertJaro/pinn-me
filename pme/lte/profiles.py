"""Differentiable Voigt and Faraday-Voigt line profiles.

The definitions in this module follow the convention used by
Landi Degl'Innocenti & Landolfi (2004).  For ``z = v + i a`` and the
Faddeeva function ``w(z)`` we define

``phi(v, a) = Re[w(z)] / sqrt(pi)`` and
``psi(v, a) = Im[w(z)] / sqrt(pi)``.

Consequently ``phi`` integrates to unity over the dimensionless frequency
coordinate ``v`` and ``psi`` is odd.  Dividing both profiles by a Doppler
width returns profiles in inverse-frequency (or inverse-wavelength) units.
Keeping this normalization explicit is important: the older Milne-Eddington
code uses dimensionless profiles whose scale is absorbed into ``eta_0``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pme.train.profile_functions import Faddeeva


class VoigtFaraday(nn.Module):
    """Evaluate normalized absorption and anomalous-dispersion profiles.

    Parameters
    ----------
    n_coefs:
        Number of coefficients in the differentiable Weideman approximation
        of the Faddeeva function.  The established PINN-ME implementation uses
        32, which is accurate to roughly float32 precision over the range
        relevant to the Fe I 630 nm lines.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dimensionless ``(phi, psi)`` for ``v`` and damping ``a``.

        ``frequency_offset`` and ``damping`` follow normal PyTorch broadcasting
        rules.  Damping must be non-negative.  The result preserves the real
        dtype and device of ``frequency_offset``.
        """

        v = torch.as_tensor(frequency_offset)
        if not v.is_floating_point():
            raise TypeError("frequency_offset must be a floating-point tensor")
        a = torch.as_tensor(damping, dtype=v.dtype, device=v.device)
        if torch.any(a < 0):
            raise ValueError("Voigt damping must be non-negative")
        v, a = torch.broadcast_tensors(v, a)
        profile = self.faddeeva(torch.complex(v, a))
        scale = self.inverse_sqrt_pi.to(dtype=v.dtype, device=v.device)
        return profile.real * scale, profile.imag * scale

    def forward(
        self,
        frequency_offset: torch.Tensor,
        damping: torch.Tensor | float,
        doppler_width: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return absorption and dispersion profiles.

        If ``doppler_width`` is omitted the profiles are dimensionless.  When
        supplied, it must be positive and the returned profiles are divided by
        it, so their units are the reciprocal of the supplied width.
        """

        absorption, dispersion = self.dimensionless(frequency_offset, damping)
        if doppler_width is None:
            return absorption, dispersion
        width = torch.as_tensor(
            doppler_width,
            dtype=absorption.dtype,
            device=absorption.device,
        )
        if torch.any(width <= 0):
            raise ValueError("doppler_width must be strictly positive")
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
