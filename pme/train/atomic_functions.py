import numpy as np
import torch

def fact(n):
    ''' My simplistic way to compute fact

    Input:
        -- n, int
    Output:
        -- results, float
            result == n!

    '''

    result = 1
    assert n % 1 == 0, "n must be an integer"
    n = int(n)
    for i in range(1, n + 1):
        result *= i
    return result


def threej(j1, j2, j3, m1, m2, m3):
    """
    3J function from Rebecca (from VFISV)

    """

    coef_aux = torch.tensor(1, dtype=torch.int)
    coef = torch.tensor(0, dtype=torch.int)

    if ((torch.abs(m1) > torch.abs(j1)) or (np.abs(m2) > torch.abs(j2))
            or (torch.abs(m3) > torch.abs(j3))):
        return 0
    if ((j1 + j2 - j3) < 0):
        return 0

    if ((j2 + j3 - j1)) < 0:
        coef_aux = 0

    if ((j3 + j1 - j2)) < 0:
        return 0

    if ((m1 + m2 + m3)) != 0:
        coef_aux = 0

    if (coef_aux == 1):
        kmin1 = j3 - j1 - m2
        kmin2 = j3 - j2 + m1
        kmin = -1 * torch.minimum(kmin1, kmin2)
        if (kmin < 0):
            kmin = 0
        kmax1 = j1 + j2 - j3
        kmax2 = j1 - m1
        kmax3 = j2 + m2
        kmax = torch.minimum(torch.minimum(kmax1, kmax2), kmax3)

        if (kmin < kmax):
            coef_aux = 0

    term1 = frontl(j1, j2, j3, m1, m2, m3)
    msign = (-1) ** (j1 - j2 - m3)

    coef = 0

    for j in range(kmin, kmax+1):
        term2 = fact(j) * fact(kmin1 + j) * fact(kmin2 + j)
        term2 = term2 * fact(kmax1 - j) * fact(kmax2 - j) * fact(kmax3 - j)
        term = (-1) ** j * msign * term1 / term2
        coef = coef + term


    coef = coef * coef_aux
    return coef
def frontl(x1, x2, x3, y1, y2, y3):
    """
    Helper function from Rebecca

    """

    l1 = x1 + x2 - x3
    l2 = x2 + x3 - x1
    l3 = x3 + x1 - x2
    l4 = x1 + x2 + x3 + 1
    l5 = x1 + y1
    l6 = x1 - y1
    l7 = x2 + y2
    l8 = x2 - y2
    l9 = x3 + y3
    l10 = x3 - y3

    dum = 1
    dum = dum * fact(l1) * fact(l2) * fact(l3) * fact(l5) / fact(l4)
    dum = dum * fact(l6) * fact(l7) * fact(l8) * fact(l9) * fact(l10)

    dum = torch.tensor(dum, dtype=torch.float)

    dum = torch.sqrt(dum)

    return dum

def compute_zeeman_strength(JUp, JLow, MUp, MLow):
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
    zeeman_strength = 3 * threej(JUp, JLow, 1,
                                 -1*MUp, MLow, (MUp - MLow))**2

    return zeeman_strength
