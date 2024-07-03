import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import voigt_profile, wofz
from torch import nn


def polyval(x, coeffs):
    """Implementation of the Horner scheme to evaluate a polynomial

    taken from https://discuss.pytorch.org/t/polynomial-evaluation-by-horner-rule/67124

    Args:
        x (torch.Tensor): variable
        coeffs (torch.Tensor): coefficients of the polynomial
    """
    curVal = 0
    for curValIndex in range(len(coeffs) - 1):
        curVal = (curVal + coeffs[curValIndex]) * x[0]
    return (curVal + coeffs[len(coeffs) - 1])


class Faddeeva(torch.nn.Module):
    """Class to compute the error function of a complex number (extends torch.special.erf behavior)

    This class is based on the algorithm proposed in:
    Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518
    """

    def __init__(self, n_coefs):
        """Defaul constructor

        Args:
            n_coefs (integer): The number of polynomial coefficients to use in the approximation
        """
        super().__init__()
        # compute polynomial coefficients and other constants
        self.N = n_coefs
        self.i = torch.complex(torch.tensor(0.), torch.tensor(1.))
        self.M = 2 * self.N
        self.M2 = 2 * self.M
        self.k = torch.linspace(-self.M + 1, self.M - 1, self.M2 - 1)
        self.L = torch.sqrt(self.N / torch.sqrt(torch.tensor(2.)))
        self.theta = self.k * torch.pi / self.M
        self.t = self.L * torch.tan(self.theta / 2)
        self.f = torch.exp(-self.t ** 2) * (self.L ** 2 + self.t ** 2)
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real / self.M2
        self.a = torch.flipud(self.a[1:self.N + 1])

    def forward(self, z):
        """Compute the Faddeeva function of a complex number

        The constant coefficients are computed in the constructor of the class.

        Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: w(z) for each element of z
        """
        Z = (self.L + self.i * z) / (self.L - self.i * z)
        p = polyval(Z.unsqueeze(0), self.a)
        w = 2 * p / (self.L - self.i * z) ** 2 + (1 / torch.sqrt(torch.tensor(torch.pi))) / (self.L - self.i * z)
        return w


class FaradayVoigt(nn.Module):

    def __init__(self):
        super().__init__()
        self.faddeeva = Faddeeva(16)

    def forward(self, x, sigma, gamma):
        ''' Compute the Faraday-Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''

        gamma_i = 1j * gamma
        z_arr = (x + gamma_i) / (2 ** 0.5 * sigma)
        z11 = self.faddeeva(z_arr)
        psi_profile = z11.imag / sigma / ((2 * np.pi) ** 0.5)

        return psi_profile


class Voigt(nn.Module):

    def __init__(self):
        super().__init__()
        self.faddeeva = Faddeeva(16)

    def forward(self, x, sigma, gamma):
        ''' Compute the Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''
        z = (x + 1j * gamma) / (2 ** 0.5 * sigma)
        v = self.faddeeva(z).real / (sigma *(2 * np.pi) ** 0.5 ) 
        return v


def faraday_voigt(nu, sigma, gamma, mu):
    ''' Compute the Faraday-Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''
    z_arr = (nu - mu + 1j * gamma) / (2 ** 0.5 * sigma)   
    z11 = wofz(z_arr)
    psi_profile = z11.imag / sigma / 1.41/1.71 
    return psi_profile


def voigt(nu, sigma, gamma, mu):
    ''' Compute the Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''
    phi_profile = voigt_profile(nu - mu, sigma, gamma) 
    return phi_profile


if __name__ == '__main__':
    base_path = '/glade/u/home/mmolnar/Projects/PINNME/'
    # # fit(voigt, base_path, 'voigt')
    # # fit(faraday_voigt, base_path, 'faraday_voigt')
    # model = Voigt()
    # fit(voigt, model, base_path, 'voigt')

    t_sigma, t_gamma, t_mu = 1e-6, 0.05, 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_nu = np.linspace(-2, 2, 10000)
    test_gamma = np.ones_like(test_nu) * t_gamma
    test_mu = np.ones_like(test_nu) * t_mu
    test_sigma = np.ones_like(test_nu) * t_sigma

    test_profile = faraday_voigt(test_nu, test_sigma, test_gamma, test_mu)
    test_profile1 = faraday_voigt(test_nu, test_sigma, test_gamma, test_mu)

    f = FaradayVoigt()

    test_nu_t = torch.tensor(test_nu, dtype=torch.float32)[:, None].to(device)
    test_gamma_t = torch.tensor(test_gamma, dtype=torch.float32)[:, None].to(device)
    test_mu_t = torch.tensor(test_mu, dtype=torch.float32)[:, None].to(device)
    test_sigma_t = torch.tensor(test_sigma, dtype=torch.float32)[:, None].to(device)
    pred_profile = f(test_nu_t - test_mu_t, test_sigma_t, test_gamma_t)
    pred_profile = pred_profile.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(test_nu, test_profile, label='ref')
    ax.plot(test_nu, pred_profile, label='pred', linestyle='--')
    ax.legend()
    ax.grid(alpha=0.4)
    plt.savefig(os.path.join(base_path, f'test_faraday_voigt_{t_gamma:02f}_{t_mu:02f}.jpg'), dpi=150)
    plt.close()
