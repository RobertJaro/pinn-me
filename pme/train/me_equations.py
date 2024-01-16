'''
Includes all the internal workings of a ME synthesis

The formalism follows the computations on p. 159 in Landi Degl'Innocenti & Landolfi 2004

Also a lot of inspiration is taken from AAR github: https://github.com/aasensio/milne/blob/master/maths.f90

'''

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

    res = gamma / (np.pi * ((x - x0)**2 + gamma**2))

    return res

def dispersion_Lorentzian(x, x0, gamma):
def Voigt(da, dv):
    ''' Compute the Voigt and anomalous dispersion profiles
    from See Humlicek (1982) JQSRT 27, 437
    '''



    return 0

lambda phi_b(nu, nu_0, nu_L, Gamma): Lorentzian(nu, (nu_0 + nu_L), Gamma)
lambda phi_r(nu, nu_0, nu_L, Gamma): Lorentzian(nu, (nu_0 - nu_L), Gamma)
lambda phi_p(nu, nu_0, nu_L, Gamma): Lorentzian(nu, (nu_0), Gamma)



