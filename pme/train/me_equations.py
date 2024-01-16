'''
Includes all the internal workings of a ME synthesis

The formalism follows the computations on p. 159 in Landi Degl'Innocenti & Landolfi 2004

Also a lot of inspiration is taken from AAR github: https://github.com/aasensio/milne/blob/master/maths.f90

'''

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

    def Voigt(da, dv):
        ''' Compute the Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''

        return 0

    def Faraday_Voigt(da, dv):
        ''' Compute the Faraday-Voigt and anomalous dispersion profiles
        from See Humlicek (1982) JQSRT 27, 437
        '''

        return 0

    phi_b = lambda self: Lorentzian(self.nu, (self.nu_0 + self.nu_L), self.Gamma)
    phi_r = lambda self: Lorentzian(self.nu, (self.nu_0 - self.nu_L), self.Gamma)
    phi_p = lambda self: Lorentzian(self.nu, (self.nu_0), self.Gamma)

    psi_b = lambda self: Faraday_fake(self.nu, (self.nu_0 + self.nu_L), self.Gamma)
    psi_r = lambda self: Faraday_fake(self.nu, (self.nu_0 - self.nu_L), self.Gamma)
    psi_p = lambda self: Faraday_fake(self.nu, (self.nu_0), self.Gamma)

    # Defining the propagation matrix elements from L^2 book
    def nu_I(nu, nu_0, nu_L, Gamma, theta, chi):
        nuI = (phi_p(nu, nu_0, nu_L, Gamma) * np.sin(theta) ** 2
               + (phi_r(nu, nu_0, nu_L, Gamma) + phi_b(nu, nu_0, nu_L, Gamma)) / 2 * (1 + np.cos(theta) ** 2))

        return 0.5 * nuI

    def eta_Q(nu, nu_0, nu_L, Gamma, theta, chi):
        nuQ = (phi_p(nu, nu_0, nu_L, Gamma) - 0.5 * (phi_r(nu, nu_0, nu_L, Gamma) + phi_b(nu, nu_0, nu_L, Gamma)))
        nuQ = nuQ * np.sin(theta) ** 2 * np.cos(2 * chi)

        return 0.5 * nuQ

    def eta_U(nu, nu_0, nu_L, Gamma, theta, chi):
        nuU = (phi_p(nu, nu_0, nu_L, Gamma) - 0.5 * (phi_r(nu, nu_0, nu_L, Gamma) + phi_b(nu, nu_0, nu_L, Gamma)))
        nuU = nuU * np.sin(theta) ** 2 * np.sin(2 * chi)

        return 0.5 * nu_U

    def eta_V(nu, nu_0, nu_L, Gamma, theta, chi):
        nuV = (phi_r(nu, nu_0, nu_L, Gamma) - phi_b(nu, nu_0, nu_L, Gamma)) * np.cos(theta)

        return 0.5 * nu_V

    def rho_Q(nu, nu_0, nu_L, Gamma, theta, chi):
        rhoQ = (psi_p(nu, nu_0, nu_L, Gamma) - 0.5 * (psi_r(nu, nu_0, nu_L, Gamma) + psi_b(nu, nu_0, nu_L, Gamma)))
        rhoQ = rhoQ * np.sin(theta) ** 2 * np.cos(2 * chi)

        return 0.5 * rhoQ

    def rho_U(nu, nu_0, nu_L, Gamma, theta, chi):
        rhoU = (psi_p(nu, nu_0, nu_L, Gamma) - 0.5 * (psi_r(nu, nu_0, nu_L, Gamma) + psi_b(nu, nu_0, nu_L, Gamma)))
        rhoU = rhoU * np.sin(theta) ** 2 * np.sin(2 * chi)

        return 0.5 * rhoU

    def rho_V(nu, nu_0, nu_L, Gamma, theta, chi):
        rhoV = (psi_r(nu, nu_0, nu_L, Gamma) - psi_b(nu, nu_0, nu_L, Gamma)) * np.cos(theta)

        return 0.5 * rhoV

    def Delta(nu, nu_0, nu_L, Gamma, theta, chi):
        dd = ((1 + eta_I(nu, nu_0, nu_L, Gamma, theta, chi)) ** 2
              * ((1 + eta_I(nu, nu_0, nu_L, Gamma, theta, chi)) ** 2
                 - eta_Q(nu, nu_0, nu_L, Gamma, theta, chi) ** 2
                 - eta_U(nu, nu_0, nu_L, Gamma, theta, chi) ** 2
                 - eta_V(nu, nu_0, nu_L, Gamma, theta, chi) ** 2
                 + rho_Q(nu, nu_0, nu_L, Gamma, theta, chi) ** 2
                 + rho_U(nu, nu_0, nu_L, Gamma, theta, chi) ** 2
                 + rho_V(nu, nu_0, nu_L, Gamma, theta, chi) ** 2)
              - (eta_Q(nu, nu_0, nu_L, Gamma, theta, chi) * rho_Q(nu, nu_0, nu_L, Gamma, theta, chi)
                 + eta_U(nu, nu_0, nu_L, Gamma, theta, chi) * rho_U(nu, nu_0, nu_L, Gamma, theta, chi)
                 + eta_V(nu, nu_0, nu_L, Gamma, theta, chi) * rho_V(nu, nu_0, nu_L, Gamma, theta, chi)) ** 2)

        return dd

        
    def compute_I(self):
        I = (self.B0
             + self.mu * self.B1 / Delta())
        self.I = I
    def compute_Q(self): 

    def compute_U(self): 

    def compute_V(self):

    def compute_all_Stokes(self):
        compute_I(self) 
        compute_Q(self)
        compute_U(self)
        compute_V(self)
        
    def synth_atmos(self):

        compute_all_Stokes(self)
