'''
Includes all the internal workings of a ME synthesis

The formalism follows the computations on p. 159 in Landi Degl'Innocenti & Landolfi 2004

Also a lot of inspiration is taken from AAR github: https://github.com/aasensio/milne/blob/master/maths.f90

'''
import torch
from astropy import constants as const
from astropy import units as u
from torch import nn

from pme.train.atomic_functions import load_zeeman_lookup, get_zeeman_lookup_id
from pme.train.profile_functions import Voigt, FaradayVoigt


class MEAtmosphere(nn.Module):
    ''' Class to contain the ME atmosphere properties'''

    def __init__(self, lambda0, j_up, j_low, g_up, g_low, lambda_grid):
        super().__init__()

        self.voigt = Voigt()
        self.faraday_voigt = FaradayVoigt()

        self.register_buffer('c', torch.tensor(const.c.to_value(u.m / u.s), dtype=torch.float32))  # Speed of light
        self.register_buffer('lambda0', torch.tensor(lambda0.to_value(u.m), dtype=torch.float32))
        self.register_buffer('j_up', torch.tensor(j_up, dtype=torch.float32))  # Upper level angular momentum
        self.register_buffer('j_low', torch.tensor(j_low, dtype=torch.float32))  # Lower level angular momentum
        self.register_buffer('g_up', torch.tensor(g_up, dtype=torch.float32))  # Lande factor for upper level
        self.register_buffer('g_low', torch.tensor(g_low, dtype=torch.float32))

        lambda_grid = torch.tensor(lambda_grid.to_value(u.m), dtype=torch.float32)
        self.lambda_grid = nn.Parameter(lambda_grid, requires_grad=False)

        zeeman_strength_lookup = load_zeeman_lookup(j_up, j_low)
        zeeman_strength_lookup = {k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
                                  for k, v in zeeman_strength_lookup.items()}
        self.zeeman_strength_lookup = nn.ParameterDict(zeeman_strength_lookup)

    def calculate_voigt_faraday_profiles(self, nu, nu_m, damping, lambda_dop, **kwargs):
        gamma = torch.ones_like(nu) * damping  # [batch, n_lambda]
        sigma = torch.ones_like(nu)

        assert (2 * self.j_up + 1) % 1 == 0, 'nUp must be integer'
        nUp = int(2 * self.j_up + 1)

        phi_b = torch.zeros_like(nu)
        psi_b = torch.zeros_like(nu)
        phi_p = torch.zeros_like(nu)
        psi_p = torch.zeros_like(nu)
        phi_r = torch.zeros_like(nu)
        psi_r = torch.zeros_like(nu)

        for iUp in range(0, nUp):
            MUp = self.j_up - iUp

            iLow = 1
            MLow = MUp - 2 + iLow

            if torch.abs(MLow) <= torch.abs(self.j_low):
                strength = self.zeeman_strength(self.j_up, self.j_low, MUp, MLow)
                splitting = self.g_up * MUp - self.g_low * MLow

                mu = torch.ones_like(nu) * (lambda_dop - 1 * splitting * nu_m)  # [batch, n_lambda]
                phi_b += strength * self.voigt(nu - mu, sigma, gamma)
                psi_b += strength * self.faraday_voigt(nu - mu, sigma, gamma)

            iLow = 2
            MLow = MUp - 2 + iLow

            if torch.abs(MLow) <= torch.abs(self.j_low):
                strength = self.zeeman_strength(self.j_up, self.j_low, MUp, MLow)
                splitting = self.g_up * MUp - self.g_low * MLow
                mu = torch.ones_like(nu) * (lambda_dop - 1 * splitting * nu_m)  # [batch, n_lambda]

                phi_p += strength * self.voigt(nu - mu, sigma, gamma)
                psi_p += strength * self.faraday_voigt(nu - mu, sigma, gamma)

            iLow = 3
            MLow = MUp - 2 + iLow
            if torch.abs(MLow) <= torch.abs(self.j_low):
                strength = self.zeeman_strength(self.j_up, self.j_low, MUp, MLow)
                splitting = self.g_up * MUp - self.g_low * MLow

                mu = torch.ones_like(nu) * (lambda_dop - 1 * splitting * nu_m)  # [batch, n_lambda]
                phi_r += strength * self.voigt(nu - mu, sigma, gamma)
                psi_r += strength * self.faraday_voigt(nu - mu, sigma, gamma)
        return {'phi_b': phi_b, 'psi_b': psi_b, 'phi_p': phi_p, 'psi_p': psi_p, 'phi_r': phi_r, 'psi_r': psi_r}

    def zeeman_strength(self, j_up, j_low, MUp, MLow):
        # avoid recomputing the same strength
        z_id = get_zeeman_lookup_id(j_up, j_low, MUp, MLow)
        return self.zeeman_strength_lookup[z_id]

    # Defining the propagation matrix elements from L^2 book
    def eta_I(self, phi_p, phi_r, phi_b, theta, kl, **kwargs):
        eta_I = (phi_p * torch.sin(theta) ** 2 + (phi_r + phi_b) / 2 * (1 + torch.cos(theta) ** 2))
        eta_I *= kl
        return eta_I

    def eta_Q(self, phi_p, phi_r, phi_b, theta, chi, kl, **kwargs):
        eta_Q = ((phi_p - 0.5 * (phi_r + phi_b)) * torch.sin(theta) ** 2 * torch.cos(2 * chi)) * kl

        return eta_Q

    def eta_U(self, phi_p, phi_r, phi_b, theta, chi, kl, **kwargs):
        eta_U = ((phi_p - 0.5 * (phi_r + phi_b)) * torch.sin(theta) ** 2 * torch.sin(2 * chi)) * kl
        return eta_U

    def eta_V(self, phi_r, phi_b, theta, kl, **kwargs):
        eta_V = (phi_r - phi_b) * torch.cos(theta) * kl
        return eta_V

    def rho_Q(self, psi_p, psi_r, psi_b, theta, chi, kl, **kwargs):
        rho_Q = ((psi_p - 0.5 * (psi_r + psi_b)) * torch.sin(theta) ** 2 * torch.cos(2 * chi)) * kl
        return rho_Q

    def rho_U(self, psi_p, psi_r, psi_b, theta, chi, kl, **kwargs):
        rho_U = ((psi_p - 0.5 * (psi_r + psi_b)) * torch.sin(theta) ** 2 * torch.sin(2 * chi)) * kl

        return rho_U

    def rho_V(self, psi_r, psi_b, theta, kl, **kwargs):
        rho_V = kl * (psi_r - psi_b) * torch.cos(theta)

        return rho_V

    def delta(self, eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V, **kwargs):
        delta = ((1 + eta_I) ** 2 * ((1 + eta_I) ** 2
                                     - eta_Q ** 2 - eta_U ** 2 - eta_V ** 2
                                     + rho_Q ** 2 + rho_U ** 2 + rho_V ** 2)
                 - (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V) ** 2)
        return delta

    def compute_I(self, b0, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, **kwargs):
        I = b0 + mu * b1 / delta * ((1 + eta_I) * ((1 + eta_I) ** 2 + rho_Q ** 2 + rho_U ** 2 + rho_V ** 2))
        return I

    def compute_Q(self, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, eta_Q, eta_V, eta_U, **kwargs):
        Q = - mu * b1 / delta * ((1 + eta_I) ** 2 * eta_Q
                                 + (1 + eta_I) * (eta_V * rho_U - eta_U * rho_V)
                                 + rho_Q * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        return Q

    def compute_U(self, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, eta_Q, eta_V, eta_U, **kwargs):
        U = -1 * mu * b1 / delta * ((1 + eta_I) ** 2 * eta_U
                                    + (1 + eta_I) * (eta_Q * rho_V - eta_V * rho_Q)
                                    + rho_U * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        return U

    def compute_V(self, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, eta_Q, eta_V, eta_U, **kwargs):
        V = - mu * b1 / delta * ((1 + eta_I) ** 2 * eta_V
                                 + rho_V * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        return V

    def nu_m(self, b_field, d_lambda, **kwargs):
        dlambda_B = 4.6686e-3 * (self.lambda0 ** 2) * b_field
        return dlambda_B / d_lambda

    def lambda_dop(self, vdop, d_lambda, **kwargs):
        return self.lambda0 * vdop / self.c / d_lambda

    def d_lambda(self, vmac, **kwargs):
        return self.lambda0 * vmac / self.c

    def nu(self, d_lambda, **kwargs):
        return self.lambda_grid[None, :] / d_lambda

    def forward(self, b_field, theta, chi, vmac, damping, b0, b1, mu, vdop, kl, **kwargs):
        # init state
        state = {'b_field': b_field, 'theta': theta, 'chi': chi, 'vmac': vmac, 'damping': damping,
                 'b0': b0, 'b1': b1, 'mu': mu, 'vdop': vdop, 'kl': kl}

        # base profile properties
        state['d_lambda'] = self.d_lambda(**state)
        state['nu'] = self.nu(**state)
        state['nu_m'] = self.nu_m(**state)
        state['lambda_dop'] = self.lambda_dop(**state)
        # voigt and faraday voigt profiles
        profiles = self.calculate_voigt_faraday_profiles(**state)
        state.update(profiles)
        # eta
        state['eta_I'] = self.eta_I(**state)
        state['eta_Q'] = self.eta_Q(**state)
        state['eta_U'] = self.eta_U(**state)
        state['eta_V'] = self.eta_V(**state)
        # rho
        state['rho_Q'] = self.rho_Q(**state)
        state['rho_U'] = self.rho_U(**state)
        state['rho_V'] = self.rho_V(**state)
        # delta
        state['delta'] = self.delta(**state)

        I = self.compute_I(**state)
        Q = self.compute_Q(**state)
        U = self.compute_U(**state)
        V = self.compute_V(**state)
        return I, Q, U, V
