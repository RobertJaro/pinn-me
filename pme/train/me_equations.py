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
        ''' My simplistic way to compute factorial

        Input:
            -- n, int
        Output
            -- results, float
                result == n!

        '''

        result = 1
        n = int(n)
        for i in range(1, n + 1):
            result *= i
        return result

    def _int_or_halfint(self, value):
        """return Python int unless value is half-int (then return float)"""
        if isinstance(value, int):
            return value
        elif type(value) is float:
            if value.is_integer():
                return int(value)  # an int
            if (2 * value).is_integer():
                return value  # a float
        elif isinstance(value, Rational):
            if value.q == 2:
                return value.p / value.q  # a float
            elif value.q == 1:
                return value.p  # an int
        elif isinstance(value, Float):
            return _int_or_halfint(float(value))
        raise ValueError("expecting integer or half-integer, got %s" % value)

    def w3js(self, j_1, j_2, j_3, m_1, m_2, m_3):
        """
        Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

        Parameters
        ==========

        j_1, j_2, j_3, m_1, m_2, m_3 :
            Integer or half integer.

        Returns
        =======

        Rational number times the square root of a rational number.

        Examples
        ========

        It is an error to have arguments that are not integer or half
        integer values::

            sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
            Traceback (most recent call last):
            ...
            ValueError: j values must be integer or half integer
            sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
            Traceback (most recent call last):
            ...
            ValueError: m values must be integer or half integer

        Notes
        =====

        The Wigner 3j symbol obeys the following symmetry rules:

        - invariant under any permutation of the columns (with the
          exception of a sign change where `J:=j_1+j_2+j_3`):

          .. math::

             \begin{aligned}
             \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
              &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
              &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
              &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
              &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
              &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
             \end{aligned}

        - invariant under space inflection, i.e.

          .. math::

             \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
             =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

        - symmetric with respect to the 72 additional symmetries based on
          the work by [Regge58]_

        - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

        - zero for `m_1 + m_2 + m_3 \neq 0`

        - zero for violating any one of the conditions
          `j_1 \ge |m_1|`,  `j_2 \ge |m_2|`,  `j_3 \ge |m_3|`

        Algorithm
        =========

        This function uses the algorithm of [Edmonds74]_ to calculate the
        value of the 3j symbol exactly. Note that the formula contains
        alternating sums over large factorials and is therefore unsuitable
        for finite precision arithmetic and only useful for a computer
        algebra system [Rasch03]_.

        Authors
        =======

        - Jens Rasch (2009-03-24): initial version
        """

        j_1, j_2, j_3, m_1, m_2, m_3 = map(self._int_or_halfint,
                                           [j_1, j_2, j_3, m_1, m_2, m_3])

        if m_1 + m_2 + m_3 != 0:
            return 0
        a1 = j_1 + j_2 - j_3
        if a1 < 0:
            return 0
        a2 = j_1 - j_2 + j_3
        if a2 < 0:
            return 0
        a3 = -j_1 + j_2 + j_3
        if a3 < 0:
            return 0
        if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
            return 0

        argsqrt = int(self.fact(int(j_1 + j_2 - j_3)) *
                      self.fact(int(j_1 - j_2 + j_3)) *
                      self.fact(int(-j_1 + j_2 + j_3)) *
                      self.fact(int(j_1 - m_1)) *
                      self.fact(int(j_1 + m_1)) *
                      self.fact(int(j_2 - m_2)) *
                      self.fact(int(j_2 + m_2)) *
                      self.fact(int(j_3 - m_3)) *
                      self.fact(int(j_3 + m_3)) / self.fact(int(j_1 + j_2 + j_3 + 1)))

        ressqrt = torch.sqrt(argsqrt)
        if torch.is_complex(ressqrt) or torch.isnan(ressqrt):
            ressqrt = ressqrt.imag

        imin = torch.max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
        imax = torch.min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
        sumres = 0
        for ii in range(int(imin), int(imax) + 1):
            den = (self.fact(ii) * self.fact(int(ii + j_3 - j_1 - m_2))
                   * self.fact(int(j_2 + m_2 - ii))
                   * self.fact(int(j_1 - ii - m_1))
                   * self.fact(int(ii + j_3 - j_2 + m_1))
                   * self.fact(int(j_1 + j_2 - j_3 - ii)))
            sumres = sumres + int((-1) ** ii) / den

        prefid = int((-1) ** int(j_1 - j_2 - m_3))
        res = ressqrt * sumres * prefid
        return res

    def strength_zeeman(self, MUp, MLow):
        ''' Compute the strength of the different Zeeman components
        Input:
            -- MUp, float
                -- angular momentum projection along the B-field
                of the upper sublevel
            -- MLow, float
                -- angular momentum projection along the B-field
                of the lower sublevel

        Output:
            -- zeeman_strength, float
                Relative strenght of the zeeman component
        '''
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
            MUp = self.JUp + 1 - iUp

            iLow = 1
            MLow = MUp - 2 + iLow
            if torch.abs(MLow) < self.JLow:
                strength = self.strength_zeeman(self.JUp, self.JLow, MUp, MLow)
                splitting = self.gUp * MUp - self.gLow * MLow

                mu = torch.ones_like(nu) * (self.lambdaDop - 1 * splitting*self.nu_m)  # [batch, n_lambda]
                parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
                self.phi_b_arr += strength * self.voigt(parameters)
                self.psi_b_arr += strength * self.faraday_profiles(parameters)

            iLow = 2
            MLow = MUp - 2 + iLow
            if torch.abs(MLow) < self.JLow:
                strength = self.strength_zeeman(self.JUp, self.JLow, MUp, MLow)
                splitting = self.gUp * MUp - self.gLow * MLow

                mu = torch.ones_like(nu) * (self.lambdaDop - 1 * splitting * self.nu_m)  # [batch, n_lambda]
                parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
                self.phi_p_arr += strength * self.voigt(parameters)
                self.psi_p_arr += strength * self.faraday_profiles(parameters)

            iLow = 3
            MLow = MUp - 2 + iLow
            if torch.abs(MLow) < self.JLow:
                strength = self.strength_zeeman(self.JUp, self.JLow, MUp, MLow)
                splitting = self.gUp * MUp - self.gLow * MLow

                mu = torch.ones_like(nu) * (self.lambdaDop - 1 * splitting * self.nu_m)  # [batch, n_lambda]
                parameters = torch.stack([nu, gamma, mu], dim=-1)  # [batch, n_lambda, 3]
                self.phi_p_arr += strength * self.Voigt(parameters)
                self.psi_p_arr += strength * self.faraday_profiles(parameters)

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


