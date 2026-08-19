import numpy as np
import torch
from scipy.special import voigt_profile, wofz
from torch import nn


def polyval(x, coeffs):
    """Evaluate a polynomial with coefficients ordered highest degree first."""
    value = torch.zeros_like(x)
    for coefficient in coeffs:
        value = value * x + coefficient
    return value


class Faddeeva(nn.Module):
    """Differentiable Weideman approximation of the Faddeeva function.

    The approximation is intended for the upper half-plane used by Voigt
    profiles (non-negative damping). The reflection identity is used for
    lower-half-plane inputs so the module remains a general ``w(z)`` helper.
    """

    def __init__(self, n_coefs=32):
        super().__init__()
        if n_coefs < 2:
            raise ValueError("n_coefs must be at least 2")

        n_coefs = int(n_coefs)
        m = 2 * n_coefs
        m2 = 2 * m
        k = torch.arange(-m + 1, m, dtype=torch.float64)
        length = torch.sqrt(torch.tensor(n_coefs / np.sqrt(2.0), dtype=torch.float64))
        theta = k * torch.pi / m
        t = length * torch.tan(theta / 2)

        # The leading zero is part of Weideman's FFT construction. Omitting it
        # shifts every coefficient and gives the previous ~4% error at z=0.
        samples = torch.cat([
            torch.zeros(1, dtype=torch.float64),
            torch.exp(-t.square()) * (length.square() + t.square()),
        ])
        coefficients = torch.fft.fft(torch.fft.fftshift(samples)).real / m2
        coefficients = torch.flip(coefficients[1:n_coefs + 1], dims=(0,))

        self.register_buffer("length", length)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("inv_sqrt_pi", torch.tensor(1 / np.sqrt(np.pi), dtype=torch.float64))

    def _upper_half_plane(self, z):
        real_dtype = z.real.dtype
        length = self.length.to(dtype=real_dtype)
        coefficients = self.coefficients.to(dtype=real_dtype)
        inv_sqrt_pi = self.inv_sqrt_pi.to(dtype=real_dtype)

        denominator = length - 1j * z
        transformed = (length + 1j * z) / denominator
        polynomial = polyval(transformed, coefficients)
        return 2 * polynomial / denominator.square() + inv_sqrt_pi / denominator

    def forward(self, z):
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))

        upper_half_plane = z.imag >= 0
        upper_z = torch.where(upper_half_plane, z, -z)
        upper_value = self._upper_half_plane(upper_z)
        reflected_value = 2 * torch.exp(-z.square()) - upper_value
        return torch.where(upper_half_plane, upper_value, reflected_value)


class FaradayVoigt(nn.Module):

    def __init__(self):
        super().__init__()
        self.faddeeva = Faddeeva()

    def forward(self, x, sigma, gamma):
        ''' Compute the Faraday-Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''

        gamma_i = 1j * gamma
        z_arr = (x + gamma_i) / (sigma)
        z11 = self.faddeeva(z_arr)
        psi_profile = z11.imag / sigma / ((np.pi) ** 0.5)

        return psi_profile


class Voigt(nn.Module):

    def __init__(self):
        super().__init__()
        self.faddeeva = Faddeeva()

    def forward(self, x, sigma, gamma):
        ''' Compute the Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''
        z = (x + 1j * gamma) / (sigma)
        v = self.faddeeva(z).real / (sigma * (np.pi) ** 0.5)
        return v


def faraday_voigt(nu, sigma, gamma, mu):
    ''' Compute the Faraday-Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''
    z_arr = (nu - mu + 1j * gamma) / (sigma)
    z11 = wofz(z_arr)
    psi_profile = z11.imag / (sigma * (np.pi) ** 0.5)
    return psi_profile


def voigt(nu, sigma, gamma, mu):
    ''' Compute the Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''
    phi_profile = voigt_profile(nu - mu, sigma, gamma)
    return phi_profile
