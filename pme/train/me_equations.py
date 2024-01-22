'''
Includes all the internal workings of a ME synthesis

The formalism follows the computations on p. 159 in Landi Degl'Innocenti & Landolfi 2004

Also a lot of inspiration is taken from AAR github: https://github.com/aasensio/milne/blob/master/maths.f90

'''
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import voigt_profile, wofz
from torch import nn

from pme.train.generic_function import FaradayVoigt, Voigt


class MEAtmosphere(nn.Module):
    ''' Class to contain the ME atmosphere properties'''

    def __init__(self, lambda0, jUp, jLow, gUp, gLow, lambdaGrid):
        super().__init__()

        self.voigt = Voigt()
        self.faraday_voigt = FaradayVoigt()

        self.c = 3e8
        
        self.lambda0 = lambda0 * 1e-10
        self.JUp = jUp
        self.JLow = jLow
        self.gUp = gUp
        self.gLow = gLow

        self.lambdaGrid = nn.Parameter(torch.tensor(lambdaGrid, dtype=torch.float32), requires_grad=False)

    def compute_larmor_freq(self):
        e = 1.6e-19 # C
        me = 9.109e-31 # kg
        hbar = 1.54e-34 # J.s / 4pi
        c = 3e8
        
        dlambda_B = 1e-13 * 4.6686e10 * (self.lambda0 **2) * self.BField
        self.nu_m = dlambda_B/ self.dLambda
    
    def calculate_Voigt_Faraday_profiles(self):
        nu = self.nuArray # [batch, n_lambda]
        gamma = torch.ones_like(nu) * self.a # [batch, n_lambda]
        sigma = torch.ones_like(nu)

        mu = torch.ones_like(nu) * (self.lambdaDop - 1 * self.nu_m) # [batch, n_lambda]
        self.phi_b_arr = self.voigt(nu - mu, sigma, gamma)

        mu = torch.ones_like(nu) * (self.lambdaDop + 1 * self.nu_m) # [batch, n_lambda]
        self.phi_r_arr = self.voigt(nu - mu, sigma, gamma)

        mu = torch.ones_like(nu) * (self.lambdaDop) # [batch, n_lambda]
        self.phi_p_arr = self.voigt(nu - mu, sigma, gamma)

        mu = torch.ones_like(nu) * (self.lambdaDop - 1 * self.nu_m)  # [batch, n_lambda]
        self.psi_b_arr = self.faraday_voigt(nu - mu, sigma, gamma)

        mu = torch.ones_like(nu) * (self.lambdaDop)  # [batch, n_lambda]
        self.psi_p_arr = self.faraday_voigt(nu - mu, sigma, gamma)

        mu = torch.ones_like(nu) * (self.lambdaDop + 1 * self.nu_m)  # [batch, n_lambda]
        self.psi_r_arr = self.faraday_voigt(nu - mu, sigma, gamma)


    # Defining the propagation matrix elements from L^2 book
    def eta_I(self):
        self.eta_I_arr = (self.phi_p_arr * torch.sin(self.theta) ** 2
                          + (self.phi_r_arr + self.phi_b_arr) / 2 * (1 + torch.cos(self.theta) ** 2))


    def eta_Q(self):
        self.eta_Q_arr = (self.phi_p_arr
                         - 0.5 * (self.phi_r_arr + self.phi_b_arr)) * torch.sin(self.theta) ** 2 * torch.cos(2 * self.chi)

    def eta_U(self):
        self.eta_U_arr = (self.phi_p_arr
                          - 0.5 * (self.phi_r_arr + self.phi_b_arr)) * torch.sin(self.theta) ** 2 * torch.sin(2 * self.chi)

    def eta_V(self):
        self.eta_V_arr = (self.phi_r_arr - self.phi_b_arr) * torch.cos(self.theta)

    def rho_Q(self):
        self.rho_Q_arr = (self.psi_p_arr
                          - 0.5 * (self.psi_r_arr + self.psi_b_arr)) * torch.sin(self.theta) ** 2 * torch.cos(2 * self.chi)

    def rho_U(self):
        self.rho_U_arr = (self.psi_p_arr
                          - 0.5 * (self.psi_r_arr + self.psi_b_arr)) * torch.sin(self.theta) ** 2 * torch.sin(2 * self.chi)

    def rho_V(self):
        self.rho_V_arr = (self.psi_r_arr - self.psi_b_arr) * torch.cos(self.theta)

    def calc_Delta(self):
        dd = ((1 + self.eta_I_arr) ** 2
              * ((1 + self.eta_I_arr) ** 2
                 - self.eta_Q_arr ** 2
                 - self.eta_U_arr ** 2
                 - self.eta_V_arr ** 2
                 + self.rho_Q_arr ** 2
                 + self.rho_U_arr ** 2
                 + self.rho_V_arr ** 2)
              - (self.eta_Q_arr * self.rho_Q_arr
                 + self.eta_U_arr * self.rho_U_arr
                 + self.eta_V_arr * self.rho_V_arr) ** 2)
        self.Delta = dd


    def compute_I(self):
        I = (self.B0
             + self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr) 
                                                 * ((1 + self.eta_I_arr)**2
                                                    + self.rho_Q_arr ** 2
                                                    + self.rho_U_arr ** 2
                                                    + self.rho_V_arr**2 )))
        return I

    def compute_Q(self):
        Q = - self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr)**2 * self.eta_Q_arr
                                                     + (1 + self.eta_I_arr)*(self.eta_V_arr*self.rho_U_arr
                                                                             - self.eta_U_arr*self.rho_V_arr)
                                                     + self.rho_Q_arr * (self.eta_Q_arr*self.rho_Q_arr
                                                                         + self.eta_U_arr * self.rho_U_arr
                                                                         + self.eta_V_arr * self.rho_V_arr))
        return Q

    def compute_U(self):
        U = - self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr) ** 2 * self.eta_U_arr
                                                     + (1 + self.eta_I_arr) * (self.eta_Q_arr * self.rho_V_arr
                                                                               - self.eta_V_arr * self.rho_Q_arr)
                                                     + self.rho_U_arr * (self.eta_Q_arr * self.rho_Q_arr
                                                                         + self.eta_U_arr * self.rho_U_arr
                                                                         + self.eta_V_arr * self.rho_V_arr))
        return U

    def compute_V(self):
        V = - self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr) ** 2 * self.eta_V_arr
                                                     + self.rho_V_arr * (self.eta_Q_arr * self.rho_Q_arr
                                                                         + self.eta_U_arr * self.rho_U_arr
                                                                         + self.eta_V_arr * self.rho_V_arr))
        return V
    
    def compute_profiles(self):
        
        self.calculate_Voigt_Faraday_profiles()
        self.eta_I()
        self.eta_Q()
        self.eta_U()
        self.eta_V()
        self.rho_Q()
        self.rho_U()
        self.rho_V()
    
    def forward(self, b_field, theta, chi,
                vmac, damping, b0, b1, mu,
                vdop, kl):

        self.dLambda = self.lambda0 * vmac / self.c * 1e3
        self.nuArray = self.lambdaGrid[None, :] / self.dLambda

        self.a = damping
        self.BField = b_field
        self.theta = theta / 180 * 3.1415
        self.chi = chi / 180 * 3.1415
        self.vmac = vmac
        self.damping = damping
        self.B0 = b0
        self.B1 = b1
        self.mu = mu
        self.vdop = vdop
        self.lambdaDop = self.lambda0 * vdop * 1e3 / self.c
        self.kl = kl

        self.nu_L = 1.3996e6 * self.BField  # in 1/s for Bfield in Gauss
        self.nu_D = 0
        self.Gamma = damping
        self.compute_larmor_freq()


        self.compute_profiles()
        self.calc_Delta()
        I = self.compute_I()
        Q = self.compute_Q()
        U = self.compute_U()
        V = self.compute_V()
        return I, Q, U, V


if __name__ == '__main__':
    lambdaStart = 6300.8 * 1e-10
    lambdaStep = 0.03 * 1e-10
    nLambda = 50
    lambdaEnd = (lambdaStart + lambdaStep * (-1 + nLambda))
    lambda_grid = np.linspace(-.5 * (lambdaEnd - lambdaStart), .5 * (lambdaEnd - lambdaStart), num=nLambda)

    forward_model = MEAtmosphere(lambda0=6301.5080, jUp=2.0, jLow=2.0, gUp=1.5, gLow=1.83,
                                 lambdaGrid=lambda_grid)
    forward_model.cpu()

    b_field = torch.tensor([[1000.]], dtype=torch.float32)
    theta = torch.tensor([[20.]], dtype=torch.float32)
    chi = torch.tensor([[20.]], dtype=torch.float32)
    vmac = torch.tensor([[2.]], dtype=torch.float32)
    damping = torch.tensor([[0.2]], dtype=torch.float32)
    b0 = torch.tensor([[.8]], dtype=torch.float32)
    b1 = torch.tensor([[.2]], dtype=torch.float32)
    mu = torch.tensor([[1.]], dtype=torch.float32)
    vdop = torch.tensor([[0.]], dtype=torch.float32)
    kl = torch.tensor([[5.]], dtype=torch.float32)
    I, Q, U, V = forward_model.forward(b_field, theta, chi,
                            vmac, damping, b0, b1, mu,
                            vdop, kl)

    # plot stokes vectors
    fig, ax = plt.subplots(4, 1, figsize=(10, 5))
    ax[0].plot(lambda_grid, I[0].detach().numpy(), label='I')
    ax[1].plot(lambda_grid, Q[0].detach().numpy(), label='Q')
    ax[2].plot(lambda_grid, U[0].detach().numpy(), label='U')
    ax[3].plot(lambda_grid, V[0].detach().numpy(), label='V')
    [ax.legend() for ax in ax]
    plt.savefig('/glade/work/rjarolim/pinn_me/profile/test_stokes.jpg')
    plt.close('all')

    # eta
    plt.plot(lambda_grid, forward_model.eta_Q_arr[0].detach().numpy(), label='eta_Q')
    plt.plot(lambda_grid, forward_model.eta_U_arr[0].detach().numpy(), label='eta_U')
    plt.plot(lambda_grid, forward_model.eta_V_arr[0].detach().numpy(), label='eta_V')
    plt.legend()
    plt.savefig('/glade/work/rjarolim/pinn_me/profile/test_eta.jpg')
    plt.close('all')

    # phi
    plt.plot(lambda_grid, forward_model.phi_p_arr[0].detach().numpy(), label='phi_p')
    plt.plot(lambda_grid, forward_model.phi_b_arr[0].detach().numpy(), label='phi_b')
    plt.plot(lambda_grid, forward_model.phi_r_arr[0].detach().numpy(), label='phi_r')
    plt.legend()
    plt.savefig('/glade/work/rjarolim/pinn_me/profile/test_phi.jpg')
    plt.close('all')

    # psi
    plt.plot(lambda_grid, forward_model.psi_p_arr[0].detach().numpy(), label='psi_p')
    plt.plot(lambda_grid, forward_model.psi_b_arr[0].detach().numpy(), label='psi_b')
    plt.plot(lambda_grid, forward_model.psi_r_arr[0].detach().numpy(), label='psi_r')
    plt.legend()
    plt.savefig('/glade/work/rjarolim/pinn_me/profile/test_psi.jpg')
    plt.close('all')

    # rho
    plt.plot(lambda_grid, forward_model.rho_Q_arr[0].detach().numpy(), label='rho_Q')
    plt.plot(lambda_grid, forward_model.rho_U_arr[0].detach().numpy(), label='rho_U')
    plt.plot(lambda_grid, forward_model.rho_V_arr[0].detach().numpy(), label='rho_V')
    plt.legend()
    plt.savefig('/glade/work/rjarolim/pinn_me/profile/test_rho.jpg')
    plt.close('all')

    arr = (forward_model.psi_r_arr - forward_model.psi_b_arr) * torch.cos(forward_model.theta)
    plt.plot(lambda_grid, arr[0].detach().numpy(), label='phi_p')
    plt.savefig('/glade/work/rjarolim/pinn_me/profile/test_arr.jpg')
    plt.close('all')