import os.path
from multiprocessing import Pool

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import voigt_profile, wofz
from torch import nn

from pme.model import Sine, jacobian, Swish, PositionalEncoding


# class Voigt(nn.Module):
#
#         def __init__(self):
#             super().__init__()
#             self.G = GenericFunction(4, dim=32)
#
#         def _gaussian(self, x, sigma):
#             return torch.exp(-1 * (x ** 2 / (2 * sigma ** 2))) / (sigma * (2 * np.pi) ** 0.5)
#
#         def _lorentzian(self, x, gamma):
#             return gamma / (np.pi * (x ** 2 + gamma ** 2))
#
#         def f_exact(self, xp, x, sigma, gamma):
#             f = self._gaussian(xp, sigma) * self._lorentzian(x - xp, gamma)
#             return f
#
#         def f(self, xp, x, sigma, gamma):
#             inp = torch.cat([xp, x, sigma, gamma], dim=-1)
#             out = self.G(inp)
#             jac_matrix = jacobian(out, xp)
#             dG_dxp = jac_matrix[:, 0]
#             return dG_dxp
#
#         def forward(self, x, sigma, gamma):
#             lower = -50 * torch.ones_like(x)
#             upper = 50 * torch.ones_like(x)
#             # TODO check stack vs cat
#             input_upper = torch.stack([upper, x, sigma, gamma], dim=-1)
#             input_lower = torch.stack([lower, x, sigma, gamma], dim=-1)
#             res = (self.G(input_upper) - self.G(input_lower)) * 1.414
#             return res[..., 0]
#
#
# class GenericFunction(nn.Module):
#
#     def __init__(self, input_parameters, dim=64, positional_encoding=True):
#         super().__init__()
#         if positional_encoding:
#             pos_enc = PositionalEncoding(8, 20)
#             inp = nn.Linear(40 * input_parameters, dim)
#             self.input_layer = nn.Sequential(pos_enc, inp)
#         else:
#             self.input_layer = nn.Linear(input_parameters, dim)
#         self.hidden_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(6)])
#         self.output_layer = nn.Linear(dim, 1)
#         self.activation = Swish()
#
#     def forward(self, params):
#         # polynomial = torch.stack([params ** i for i in range(self.polynomial_degree)], dim=-1)
#         # profile = torch.einsum('ijk,jk->i', polynomial, self.weights)
#         x = self.input_layer(params)
#         for layer in self.hidden_layers:
#             x = self.activation(layer(x))
#         profile = self.output_layer(x)
#         return profile
#
#
# def fit(function, model, base_path, name, batch_size=int(2**15)):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model.to(device)
#
#     # x = torch.linspace(-100, 100, 1000, dtype=torch.float32)
#     # nu = np.linspace(-50, 50, 100, dtype=np.float32)
#     # gamma = np.linspace(0, 50, 100, dtype=np.float32)
#     # mu = np.linspace(-50, 50, 100, dtype=np.float32)
#     # x, nu, gamma, mu = np.meshgrid(x, nu, gamma, mu, indexing='ij')
#     # x = x.flatten()
#     # nu = nu.flatten()
#     # gamma = gamma.flatten()
#     # mu = mu.flatten()
#     # sigma = np.ones_like(nu, dtype=np.float32)
#
#     # with Pool(16) as p:
#     #     profile = p.starmap(function, zip(nu, sigma, gamma, mu))
#     # profile = np.array(profile, dtype=np.float32)
#
#     # profile = torch.tensor(profile, dtype=torch.float32)
#     # params = torch.tensor(np.stack([x, nu, gamma, mu], axis=-1), dtype=torch.float32)
#
#     # shuffle
#     # idx = torch.randperm(len(params))
#     # # profile = profile[idx]
#     # params = params[idx]
#
#     # optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     # n_batches = np.ceil(len(params) / batch_size).astype(int)
#
#     for epoch in range(100):
#         total_loss = []
#         for i in range(int(1e3)):
#             # batch = profile[i * batch_size:(i + 1) * batch_size]
#             # params[i * batch_size:(i + 1) * batch_size]
#             batch_params = torch.rand(batch_size, 3, dtype=torch.float32, device=device, requires_grad=True)
#
#             xp = torch.rand(batch_size, 1, dtype=torch.float32, device=device, requires_grad=True) * 100 - 50
#             nu = batch_params[:, 0:1] * 100 - 50
#             gamma = batch_params[:, 1:2] * 50
#             mu = batch_params[:, 2:3] * 100 - 50
#             sigma = torch.ones_like(nu, dtype=torch.float32)
#
#             f_exact = model.f_exact(xp, nu - mu, sigma, gamma)
#             f_pred = model.f(xp, nu - mu, sigma, gamma)
#
#             # with Pool(16) as p:
#             #     ref_profile = p.starmap(function, zip(nu.detach().cpu().numpy(),
#             #                                           sigma.detach().cpu().numpy(),
#             #                                           gamma.detach().cpu().numpy(),
#             #                                           mu.detach().cpu().numpy()))
#             # ref_profile = torch.tensor(np.array(ref_profile), dtype=torch.float32).to(device)
#             # pred_profile = model.forward(nu - mu, sigma, gamma)
#             # profile_loss = torch.max((ref_profile - pred_profile) ** 2)
#             # print(profile_loss)
#
#             f_exact = torch.asinh(f_exact / 1e-3) / np.arcsinh(1 / 1e-3)
#             f_pred = torch.asinh(f_pred / 1e-3) / np.arcsinh(1 / 1e-3)
#
#             loss = torch.mean((f_exact - f_pred) ** 2)
#             assert not torch.isnan(loss), f'Loss is NaN at epoch {epoch + 1}, batch {i + 1}'
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += [loss.cpu().detach().numpy()]
#         print(f'Epoch {epoch + 1}, loss {np.mean(total_loss)}')
#         # test functions
#         xp = torch.linspace(-50, 50, 100, dtype=torch.float32, requires_grad=True)[:, None].to(device)
#         nu = torch.ones_like(xp, dtype=torch.float32) * 10
#         gamma = torch.ones_like(xp, dtype=torch.float32) * 20
#         mu = torch.ones_like(xp, dtype=torch.float32) * 30
#         sigma = torch.ones_like(xp, dtype=torch.float32)
#
#         f_exact = model.f_exact(xp, nu - mu, sigma, gamma)
#         f_pred = model.f(xp, nu - mu, sigma, gamma)
#
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#         ax.plot(xp.cpu().detach().numpy(), f_exact.cpu().detach().numpy(), label='ref')
#         ax.plot(xp.cpu().detach().numpy(), f_pred.cpu().detach().numpy(), label='pred', linestyle='--')
#         # make axis log
#         # ax.set_yscale('log')
#         # ax.set_ylim(1e-8, 1e0)
#         ax.legend()
#         plt.savefig(os.path.join(base_path, f'{name}_{epoch:03d}_test.jpg'), dpi=150)
#         plt.close()
#
#     torch.save(model, os.path.join(base_path, f'{name}.pt'))
#
#     # test plot
#     for t_gamma, t_mu in [(0, 0), (1, 10), (1, -10), (10, 0), (10, 10), (10, -10), (40, 0), (40, 10), (40, -10)]:
#         test_nu = np.linspace(-50, 50, 100)
#         test_gamma = np.ones_like(test_nu) * t_gamma
#         test_mu = np.ones_like(test_nu) * t_mu
#         test_sigma = np.ones_like(test_nu)
#
#         with Pool(16) as p:
#             test_profile = p.starmap(function, zip(test_nu, test_sigma, test_gamma, test_mu))
#         test_profile = np.array(test_profile, dtype=np.float32)
#
#         test_nu_t = torch.tensor(test_nu, dtype=torch.float32)[:, None].to(device)
#         test_gamma_t = torch.tensor(test_gamma, dtype=torch.float32)[:, None].to(device)
#         test_mu_t = torch.tensor(test_mu, dtype=torch.float32)[:, None].to(device)
#         test_sigma_t = torch.tensor(test_sigma, dtype=torch.float32)[:, None].to(device)
#         pred_profile = model.forward(test_nu_t - test_mu_t, test_sigma_t, test_gamma_t)
#         pred_profile = pred_profile.cpu().detach().numpy()
#
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#         ax.plot(test_nu, test_profile, label='ref')
#         ax.plot(test_nu, pred_profile, label='pred', linestyle='--')
#         ax.legend()
#         plt.savefig(os.path.join(base_path, f'{name}_{t_gamma:02d}_{t_mu:02d}.jpg'), dpi=150)
#         plt.close()


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
        self.i = torch.complex(torch.tensor(0.),torch.tensor(1.))
        self.M = 2*self.N
        self.M2 = 2*self.M
        self.k = torch.linspace(-self.M+1, self.M-1, self.M2-1)
        self.L = torch.sqrt(self.N/torch.sqrt(torch.tensor(2.)))
        self.theta = self.k*torch.pi/self.M
        self.t = self.L*torch.tan(self.theta/2)
        self.f = torch.exp(-self.t**2)*(self.L**2 + self.t**2)
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real/self.M2
        self.a = torch.flipud(self.a[1:self.N+1])

    def forward(self, z):
        """Compute the Faddeeva function of a complex number

        The constant coefficients are computed in the constructor of the class.

        Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: w(z) for each element of z
        """
        Z = (self.L+self.i*z)/(self.L-self.i*z)
        p = polyval(Z.unsqueeze(0), self.a)
        w = 2*p/(self.L-self.i*z)**2+(1/torch.sqrt(torch.tensor(torch.pi)))/(self.L-self.i*z)
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
        psi_profile = -1 * z11.imag / 1.772

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
        v = self.faddeeva(z).real / (sigma * (2 * np.pi) ** 0.5) * 1.414
        return v

def faraday_voigt(nu, sigma, gamma, mu):
    ''' Compute the Faraday-Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''
    def z(x, sigma, gamma, mu):
        gamma_i = 1j * gamma
        return (x - mu + gamma_i) / (2 ** 0.5 * sigma)

    z_arr = z(nu, sigma, gamma, mu)
    z11 = wofz(z_arr)
    psi_profile = -1 * z11.imag / 1.772
    return psi_profile

def voigt(nu, sigma, gamma, mu):
    ''' Compute the Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''
    phi_profile = voigt_profile(nu - mu, sigma, gamma) * 1.414
    return phi_profile


if __name__ == '__main__':
    base_path = '/glade/work/rjarolim/pinn_me/profile/'
    # # fit(voigt, base_path, 'voigt')
    # # fit(faraday_voigt, base_path, 'faraday_voigt')
    # model = Voigt()
    # fit(voigt, model, base_path, 'voigt')

    t_gamma, t_mu = 10, 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_nu = np.linspace(-50, 50, 100)
    test_gamma = np.ones_like(test_nu) * t_gamma
    test_mu = np.ones_like(test_nu) * t_mu
    test_sigma = np.ones_like(test_nu)

    test_profile = voigt(test_nu, test_sigma, test_gamma, test_mu)

    f = Voigt()

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
    plt.savefig(os.path.join(base_path, f'test_faraday_voigt_{t_gamma:02d}_{t_mu:02d}.jpg'), dpi=150)
    plt.close()

