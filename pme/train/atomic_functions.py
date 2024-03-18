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

        - M Molnar 2024 --
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
