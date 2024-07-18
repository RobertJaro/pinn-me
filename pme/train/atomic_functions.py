import numpy as np


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

    if ((np.abs(m1) > np.abs(j1)) or (np.abs(m2) > np.abs(j2))
            or (np.abs(m3) > np.abs(j3))):
        return 0
    if ((j1 + j2 - j3) < 0):
        return 0

    if ((j2 + j3 - j1)) < 0:
        return 0

    if ((j3 + j1 - j2)) < 0:
        return 0

    if ((m1 + m2 + m3)) != 0:
        return 0

    kmin1 = j3 - j1 - m2
    kmin2 = j3 - j2 + m1
    kmin = -1 * np.minimum(kmin1, kmin2)
    if (kmin < 0):
        kmin = 0
    kmax1 = j1 + j2 - j3
    kmax2 = j1 - m1
    kmax3 = j2 + m2
    kmax = np.minimum(np.minimum(kmax1, kmax2), kmax3)

    if (kmin < kmax):
        return 0

    term1 = frontl(j1, j2, j3, m1, m2, m3)
    msign = (-1) ** (j1 - j2 - m3)

    coef = 0

    for j in range(int(kmin), int(kmax + 1)):
        term2 = fact(j) * fact(kmin1 + j) * fact(kmin2 + j)
        term2 = term2 * fact(kmax1 - j) * fact(kmax2 - j) * fact(kmax3 - j)
        term = (-1) ** j * msign * term1 / term2
        coef = coef + term

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

    dum = np.sqrt(dum)

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
    zeeman_strength = 3 * threej(JUp, JLow, 1, -1 * MUp, MLow, (MUp - MLow)) ** 2

    return zeeman_strength


def get_zeeman_lookup_id(j_up, j_low, MUp, MLow):
    return f'{j_up}_{j_low}_{MUp}_{MLow}'.replace('.', '_')


def load_zeeman_lookup(j_up, j_low):
    lookup = {}
    assert (2 * j_up + 1) % 1 == 0, 'nUp must be integer'
    nUp = int(2 * j_up + 1)
    for iUp in range(0, nUp):
        MUp = j_up - iUp

        iLow = 1
        MLow = MUp - 2 + iLow

        if np.abs(MLow) <= np.abs(j_low):
            z_id = get_zeeman_lookup_id(j_up, j_low, MUp, MLow)
            strength = compute_zeeman_strength(j_up, j_low, MUp, MLow)

            lookup[z_id] = strength

        iLow = 2
        MLow = MUp - 2 + iLow
        if np.abs(MLow) <= np.abs(j_low):
            z_id = get_zeeman_lookup_id(j_up, j_low, MUp, MLow)
            strength = compute_zeeman_strength(j_up, j_low, MUp, MLow)

            lookup[z_id] = strength

        iLow = 3
        MLow = MUp - 2 + iLow
        if np.abs(MLow) <= np.abs(j_low):
            z_id = get_zeeman_lookup_id(j_up, j_low, MUp, MLow)
            strength = compute_zeeman_strength(j_up, j_low, MUp, MLow)
            lookup[z_id] = strength
    return lookup
