import numpy as np


def fact(n):
    ''' My simplistic way to compute factorial

    Input:
        -- n, int
    Output
        -- results, float
            result == n!

    '''

    result = 1
    assert n % 1 == 0, "n must be an integer"
    n = int(n)
    for i in range(1, n + 1):
        result *= i
    return result


def w3js(J1, J2, J3, M1, M2, M3):
    """ Compute the 3J symbol following the routine in
    Landi Degl'innocenti & Landolfi

    Inputs:
    -- J1, J2, J3, M1, M2, M3, float
        Twice the angular momenta to be studied
    Output:
    -- WJS, float
        -- Wigner 3J symbol value

    """

    if (M1 + M2 + M3) != 0:
        return 0
    IA = J1 + J2
    if (J3 > IA):
        return 0
    IB = J1 - J2
    if (J3 < np.abs(IB)):
        return 0
    JSUM = J3 + IA
    IC = J1 - M1
    ID = J2 - M2
    if np.mod(JSUM, 2) != 0:
        return 0
    if np.mod(IC, 2) != 0:
        return 0
    if np.mod(ID, 2) != 0:
        return 0
    if np.abs(M1) > J1:
        return 0
    if np.abs(M2) > J2:
        return 0
    if np.abs(M3) > J3:
        return 0
    IE = J3 - J2 + M1
    IF = J3 - J1 - M2
    ZMIN = np.max([0, -1 * IE, -1 * IF])
    IG = IA - J3
    IH = J2 + M2
    ZMAX = np.min([IG, IH, IC])
    CC = 0
    for Z in np.arange(ZMIN, ZMAX, 2):
        DENOM = (fact(Z / 2) * fact((IG - Z) / 2) * fact((IC - Z) / 2) * fact((IH - Z) / 2) * fact((IE + Z) / 2) * fact(
            (IF + Z) / 2))
        if np.mod(Z, 4) != 0:
            DENOM = -1 * DENOM
        CC += 1 / DENOM
    CC1 = (fact(IG / 2) * fact((J3 + IB) / 2) * fact((J3 - IB) / 2)
           / fact((JSUM + 2) / 2))
    CC2 = (fact((J1 + M1) / 2) * fact(IC / 2) * fact(IH / 2)
           * fact(ID / 2) * fact((J3 - M3) / 2) * fact((J3 + M3) / 2))
    CC = CC * np.sqrt(CC1 * CC2)

    if np.mod(IB - M3, 4) != 0:
        CC = -1 * CC
    return CC


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
    zeeman_strength = 3 * w3js(2 * JUp, 2 * JLow, 2, 2 * MUp, 2 * MLow, -2 * (MLow - MUp))

    return zeeman_strength
