'''
Includes all the internal workings of a ME synthesis

The formalism follows the computations on p. 159 in Landi Degl'Innocenti & Landolfi 2004

Also a lot of inspiration is taken from AAR github: https://github.com/aasensio/milne/blob/master/maths.f90

'''

import torch
from scipy.special import voigt_profile, wofz
from torch import nn

from pme.train.generic_function import GenericFunction

class MEAtmosphere(nn.Module):
    ''' Class to contain the ME atmosphere properties'''

    def __init__(self, lambda0, jUp, jLow, gUp, gLow, loggf,
                 lambdaGrid, voigt_pt, faraday_voigt_pt):
        super().__init__()

        m = GenericFunction(3)
        self.voigt = torch.load(voigt_pt)
        self.faraday_voigt = torch.load(faraday_voigt_pt)

        self.c = 3e8
        
        self.lambda0 = lambda0 * 1e-10
        self.JUp = jUp # Angular momentum of the upper level
        self.JLow = jLow # Angular momentum of the lower level
        self.gUp = gUp # Lande-g factor of the upper level
        self.gLow = gLow # Lande-g factor of the lower level
        self.loggf = loggf # loggf of the transition

        self.lambdaGrid = nn.Parameter(torch.tensor(lambdaGrid, dtype=torch.float32), requires_grad=False)

    def compute_larmor_freq(self):
        
        dlambda_B = 1e-13 * 4.6686e10 * (self.lambda0 **2) * self.BField
        self.nu_m = dlambda_B/ self.dLambda

    def compute_scattering_profiles(self, nu, sigma, gamma):


        self.voigt = voigt_profile(nu, sigma, gamma)
        self.dispersion = 0

    phi_b = lambda self: self.Voigt(self.nuArray, 1, 
                                    self.a, self.lambdaDop - 1*self.nu_m)
    phi_p = lambda self: self.Voigt(self.nuArray, 1, 
                                    self.a, self.lambdaDop)
    phi_r = lambda self: self.Voigt(self.nuArray, 1, 
                                    self.a, self.lambdaDop + 1*self.nu_m)

    psi_b = lambda self: self.Faraday_Voigt(self.nuArray, 1, 
                                            self.a, self.lambdaDop - 1*self.nu_m)
    psi_p = lambda self: self.Faraday_Voigt(self.nuArray, 1, 
                                            self.a, self.lambdaDop)
    psi_r = lambda self: self.Faraday_Voigt(self.nuArray, 1, 
                                            self.a, self.lambdaDop + 1*self.nu_m)
    def fact(self, n):
        result = 1

        for i in range(1, n+1):
            result *= i
        return result

    def w3js(self, J1, J2, J3, M1, M2, M3):
        """ Compute the 3J symbol following the routine in Landi Degl'innocenti & Landolfi"""

        WJS  = 0

        if (M1 + M2 + M3) != 0:
            return WJS
        IA = J1 + J2
        if (J3 > IA):
            return WJS
        IB = J1 - J2
        if (J3 < torch.abs(IB)):
            return WJS
        JSUM = J3 + IA
        IC = J1 - M1
        ID = J2 - M2
        if np.mod(JSUM, 2) != 0:
            return WJS
        if np.mod(IC, 2) != 0:
            return WJS
        if np.mod(ID, 2) != 0:
            return WJS
        if torch.abs(M1) > J1:
            return WJS
        if torch.abs(M2) > J2:
            return WJS
        if torch.abs(M3) > J3:
            return WJS
        IE = J3 - J2 + M1
        IF = J3 - J1 - M2
        ZMIN = torch.max([0, -1*IE, -1*IF])
        IG = IA - J3
        IH = J2 + M2
        ZMAX = torch.min([IG, IH, IC])
        CC = 0

        for Z in range(ZMIN, ZMAX, 2):
            DENOM = (self.fact(Z / 2) * self.fact((IG - Z) / 2)
                     * self.fact((IC - Z) / 2)
                     * self.fact((IH - Z) / 2) * self.fact((IE + Z) / 2) * self.fact((IF + Z) / 2))
            if np.mod(Z, 4) !=0:
                DENOM = -1 * DENOM
            CC += 1/DENOM
        CC1 = (self.fact(IG / 2) * self.fact((J3 + IB) / 2) * self.fact((J3 - IB) / 2)
               / self.fact((JSUM + 2) / 2))
        CC2 = (self.fact((J1 + M1) / 2) * self.fact(IC / 2) * self.fact(IH / 2)
               * self.fact(ID / 2) * self.fact((J3 - M3) / 2) * self.fact((J3 + M3) / 2))
        CC = CC * torch.sqrt(CC1 * CC2)

        if np.mod(IB-M3, 4) != 0:
            CC = -1 * CC
        WJS = CC
        return WJS
    def strength_zeeman(self, MUp, MLow):

        zeeman_strength = 3 * self.w3js(2*self.JUp, 2*self.JLow, 2,
                                        2*MUp, 2*MLow, -2 * (MLow - MUp))

        return zeeman_strength

    def calculate_Voigt_Faraday_profiles(self):
        nu = self.nuArray # [batch, n_lambda]
        gamma = torch.ones_like(nu) * self.a # [batch, n_lambda]

        nUp = 2 * self.Jup + 1
        nLow = 2 * self.Jlow + 1
        self.phi_b_arr = torch.zeros(len(nu))
        self.phi_p_arr = torch.zeros(len(nu))
        self.phi_r_arr = torch.zeros(len(nu))
        self.psi_b_arr = torch.zeros(len(nu))
        self.psi_p_arr = torch.zeros(len(nu))
        self.psi_r_arr = torch.zeros(len(nu))

        for iUp in range(1, nUp):
            MUp = self.Jup + 1 - iUp

            iLow = 1
            MLow = Mup - 2 + iLow
            if torch.abs(MLow) < self.JLow:
                strength = self.strength_zeeman(self.Jup, self.Jlow, MUp, MLow)
                splitting = self.gUp * MUp - self.gLow * MLow

                mu = torch.ones_like(nu) * (self.lambdaDop - 1 * splitting*self.nu_m)  # [batch, n_lambda]
                parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
                self.phi_b_arr += strength * self.voigt(parameters)
                self.psi_b_arr += strength * self.faraday_profiles(parameters)

            iLow = 2
            Mlow = Mup - 2 + iLow
            if torch.abs(Mlow) < self.JLow:
                strength = self.strength_zeeman(self.Jup, self.Jlow, Mup, Mlow)
                splitting = self.gUp * Mup - self.gLow * Mlow

                mu = torch.ones_like(nu) * (self.lambdaDop - 1 * splitting * self.nu_m)  # [batch, n_lambda]
                parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
                self.phi_p_arr += strength * self.voigt(parameters)
                self.psi_p_arr += strength * self.faraday_profiles(parameters)

            iLow = 3
            Mlow = Mup - 2 + iLow
            if torch.abs(Mlow) < self.JLow:
                strength = self.strength_zeeman(self.Jup, self.Jlow, Mup, Mlow)
                splitting = self.gUp * Mup - self.gLow * Mlow

                mu = torch.ones_like(nu) * (self.lambdaDop - 1 * splitting * self.nu_m)  # [batch, n_lambda]
                parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
                self.phi_p_arr += strength * self.Voigt(parameters)
                self.psi_p_arr += strength * self.faraday_profiles(parameters)

        mu = torch.ones_like(nu) * (self.lambdaDop - 1 * self.nu_m) # [batch, n_lambda]
        parameters = torch.stack([nu, gamma, mu], dim=-1) # [batch, n_lambda, 3]
         = self.voigt(parameters)

        mu = torch.ones_like(nu) * (self.lambdaDop) # [batch, n_lambda]
        parameters = torch.stack([nu, gamma, mu], dim=-1) # [batch, n_lambda, 3]
        self.phi_r_arr = self.voigt(parameters)

        mu = torch.ones_like(nu) * (self.lambdaDop + 1 * self.nu_m) # [batch, n_lambda]
        parameters = torch.stack([nu, gamma, mu], dim=-1) # [batch, n_lambda, 3]
        self.phi_p_arr = self.voigt(parameters)

        mu = torch.ones_like(nu) * (self.lambdaDop - 1 * self.nu_m)  # [batch, n_lambda]
        parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
        self.psi_b_arr = self.faraday_voigt(parameters)

        mu = torch.ones_like(nu) * (self.lambdaDop)  # [batch, n_lambda]
        parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
        self.psi_p_arr = self.faraday_voigt(parameters)

        mu = torch.ones_like(nu) * (self.lambdaDop + 1 * self.nu_m)  # [batch, n_lambda]
        parameters = torch.stack([nu, gamma, mu], dim=-1)
        self.psi_r_arr = self.faraday_voigt(parameters)


    # Defining the propagation matrix elements from L^2 book
    def eta_I(self):
        self.eta_I_arr = (self.phi_p_arr * torch.sin(self.theta) ** 2
                          + (self.phi_r_arr + self.phi_b_arr)/ 2 * (1 + torch.cos(self.theta) ** 2))
        self.eta_I_arr *= self.kl

    def eta_Q(self):
        self.eta_Q_arr = ((self.phi_p_arr - 0.5 * (self.phi_r_arr + self.phi_b_arr))
                          * torch.sin(self.theta) ** 2 * torch.cos(2 * self.chi))
        self.eta_Q_arr *= self.kl

    def eta_U(self):
        self.eta_U_arr = ((self.phi_p_arr - 0.5 * (self.phi_r_arr + self.phi_b_arr))
                          * torch.sin(self.theta) ** 2 * torch.sin(2 * self.chi)) * self.kl
        self.eta_U_arr *= self.kl

    def eta_V(self):
        self.eta_V_arr = (self.phi_r_arr - self.phi_b_arr) * torch.cos(self.theta) * self.kl
        self.eta_V_arr *= self.kl

    def rho_Q(self):
        self.rho_Q_arr = ((self.psi_p_arr - 0.5 * (self.psi_r_arr + self.psi_b_arr))
                          * torch.sin(self.theta) ** 2 * torch.cos(2 * self.chi)) * self.kl
        self.rho_Q_arr *= self.kl

    def rho_U(self):
        self.rho_U_arr = ((self.psi_p_arr - 0.5 * (self.psi_r_arr + self.psi_b_arr))
                          * torch.sin(self.theta) ** 2 * torch.sin(2 * self.chi)) * self.kl
        self.rho_U_arr *= self.kl

    def rho_V(self):
        self.rho_V_arr = self.kl * (self.psi_r_arr - self.psi_b_arr) * torch.cos(self.theta)
        self.rho_V_arr *= self.kl

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


