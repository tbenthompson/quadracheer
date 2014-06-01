from util import memoized

@memoized
def alpha(k):
    return (2.0 * k + 1.0) / (k + 1.0)

@memoized
def beta(k):
    return k / (k + 1.0)

def legendre_integrals(n_max):
    """
    The integrals of the Legendre polynomials are trivial.
    The integral of P_0 on [-1,1] is 2
    Because 1 is a Legendre polynomial and the set of Legendre polynomials
    is orthogonal, \int_{-1}^1 1 * P_n(x) dx = 0
    """
    integrals = [0.0] * (n_max + 1)
    integrals[0] = 2.0
    return integrals
