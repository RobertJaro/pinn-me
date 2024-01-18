'''
Includes all the internal workings of a ME synthesis

The formalism follows the computations on p. 159 in Landi Degl'Innocenti & Landolfi 2004

Also a lot of inspiration is taken from AAR github: https://github.com/aasensio/milne/blob/master/maths.f90

'''

import scipy as sp
from scipy.special import voigt_profile, wofz

class ME_Atmosphere():
    ''' Class to contain the ME atmosphere properties'''

    def __init__(self, lambda0, JUp, JLow, gUp, gLow,
                 lambdaStart, lambdaStep, nLambda, BField = 100.0, theta = 20.0, chi = 20.0,
                 vmac = 0.0, damping = 0.0, B0 = 0.8, B1 = 0.2, mu = 1.0,
                 vdop = 0.0, kl = 5.0):

        self.lambda0 = lambda0
        self.JUp = JUp
        self.JLow = JLow
        self.gUp = gUp
        self.gLow = gLow
        self.lambdaStart = lambdaStart
        self.lambdaStep = lambdaStep
        self.nLambda = nLambda
        self.BField = BField
        self.theta = theta
        self.chi = chi
        self.vmac = vmac
        self.damping = damping
        self.B0 = B0
        self.B1 = B1
        self.mu = mu
        self.vdop = vdop
        self.kl = kl

        self.nu_L =  1.3996e6 * self.BField # in 1/s for Bfield in Gauss
        self.nu_D = 0
        self.Gamma = damping



    def Lorentzian(x, x0, gamma):
        '''
        Compute a Lorentzian profile

        inputs:
            -- x, x0, gamma: floats
            Inputs for a lorentzian distribution following the standard notation

        Outputs:
            -- y: float
                Resulting lorentzian probability density
        '''

        res = gamma / (np.pi * ((x - x0) ** 2 + gamma ** 2))

        return res

    def Faraday_fake(x, x0, gamma):
        '''
        Compute the complimentary simplified Faraday profile
        :param x:
        :param x0:
        :param gamma:
        :return:
            -- res --
        '''

        res = (x0 - x) / (np.pi * ((x0 - x) ** 2 + gamma ** 2))

        return res

    def compute_scattering_profiles(self, nu, sigma, gamma):


        self.voigt = voigt_profile(nu, sigma, gamma)
        self.dispersion = 0

    def Voigt(self, nu_array, sigma, gamma, mu):
        ''' Compute the Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''

        phi_profile = voigt_profile(nu_array - mu, sigma, gamma) * 1.414

        return phi_profile

    def Faraday_Voigt(self, nu_array, sigma, gamma, mu):
        ''' Compute the Faraday-Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''

        def z(x, sigma, gamma, mu):
            gamma_i = complex(0, gamma)
            return (x - mu + gamma_i) / (np.sqrt(2) * sigma)

        z_arr = z(nu_array, sigma, gamma, mu)
        z11 = wofz(z1)
        psi_profile = -1 * z11.imag / 1.772

        return psi_profile

    phi_b = lambda self: Voigt(self.nu_array, self.lambda_d, self.a, -1*self.nu_m)
    phi_r = lambda self: Voigt(self.nu_array, self.lambda_d, self.a, self.nu_m)
    phi_r = lambda self: Voigt(self.nu_array, self.lambda_d, self.a, 0)

    psi_b = lambda self: Faraday_Voigt(self.nu_array, self.lambda_d, self.a, -1*self.nu_m)
    psi_r = lambda self: Faraday_Voigt(self.nu_array, self.lambda_d, self.a, self.nu_m)
    psi_r = lambda self: Faraday_Voigt(self.nu_array, self.lambda_d, self.a, 0)

    def calculate_Voigt_Faraday_profiles(self):

        self.phi_b_arr = phi_b(self)
        self.phi_r_arr = phi_r(self)
        self.phi_p_arr = phi_p(self)

        self.psi_b_arr = psi_b(self)
        self.psi_r_arr = psi_r(self)
        self.psi_p_arr = psi_p(self)

    # Defining the propagation matrix elements from L^2 book
    def eta_I(self):
        self.eta_I_arr = (self.phi_p_arr * np.sin(self.theta) ** 2
                         + (self.phi_r_arr + self.phi_b_arr) / 2 * (1 + np.cos(self.theta) ** 2))

    def eta_Q(self):
        self.eta_Q_arr = (self.phi_p_arr
                         - 0.5 * (self.phi_r_arr + self.phi_b_arr)) * np.sin(self.theta) ** 2 * np.cos(2 * self.chi)

    def eta_U(self):
        self.eta_U_arr = (self.phi_p_arr
                          - 0.5 * (self.phi_r_arr + self.phi_b_arr)) * np.sin(self.theta) ** 2 * np.sin(2 * self.chi)

    def eta_V(self):
        self.eta_V_arr = (self.phi_r_arr - self.phi_b_arr) * np.cos(self.theta)

    def rho_Q(self):
        self.rho_Q_arr = (self.psi_p_arr
                          - 0.5 * (self.psi_r_arr + self.psi_b_arr)) * np.sin(self.theta) ** 2 * np.cos(2 * self.chi)

    def rho_U(self):
        self.rho_U_arr = (self.psi_p_arr
                          - 0.5 * (self.psi_r_arr + self.psi_b_arr)) * np.sin(self.theta) ** 2 * np.sin(2 * self.chi)

    def rho_V(self):
        self.rho_V_arr = (self.psi_r_arr - self.psi_b_arr) * np.cos(self.theta)

    def calc_Delta(self):
        dd = ((1 + eta_I_arr) ** 2
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
        I = self.B0 + self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr) * ((1 + self.eta_I_arr) **2
                                                                                + self.rho_Q_arr ** 2
                                                                                + self.rho_U_arr ** 2
                                                                                + self.rho_V_arr ** 2))                                                                               ))
        self.I = I

    def compute_Q(self):
        self.Q = - self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr)**2 * self.eta_Q_arr
                                                     + (1 + self.eta_I_arr)*(self.eta_V_arr*self.rho_U_arr
                                                                             - self.eta_U_arr*self.rho_V_arr)
                                                     + self.rho_Q_arr * (self.eta_Q_arr*self.rho_Q_arr
                                                                         + self.eta_U_arr * self.rho_U_arr
                                                                         + self.eta_V_arr * self.rho_V_arr))
    def compute_U(self):
        self.U = - self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr) ** 2 * self.eta_U_arr
                                                     + (1 + self.eta_I_arr) * (self.eta_Q_arr * self.rho_V_arr
                                                                               - self.eta_V_arr * self.rho_Q_arr)
                                                     + self.rho_U_arr * (self.eta_Q_arr * self.rho_Q_arr
                                                                         + self.eta_U_arr * self.rho_U_arr
                                                                         + self.eta_V_arr * self.rho_V_arr))
    def compute_V(self):
        self.V = - self.mu * self.B1 / self.Delta * ((1 + self.eta_I_arr) ** 2 * self.eta_V_arr
                                                     + self.rho_V_arr * (self.eta_Q_arr * self.rho_Q_arr
                                                                         + self.eta_U_arr * self.rho_U_arr
                                                                         + self.eta_V_arr * self.rho_V_arr))
    def compute_all_Stokes(self):
        compute_I(self)
        compute_Q(self)
        compute_U(self)
        compute_V(self)

    def synth_atmos(self):

        compute_all_Stokes(self)
