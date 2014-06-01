"""Shared methods for the recursive legendre quadrature methods."""
import numpy as np
from util import memoized
from math import sqrt

mu_proc = dict()

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

def next_mu(k, m, prior_moments, ay, by):
    multiplier = 1.0
    multiplier_fnc = mu_proc[m].get("recursion_multiplier", None)
    if multiplier_fnc:
        multiplier = multiplier_fnc(k, m, ay, by)
    coeffs = [f(k, m, ay, by) * multiplier for f
              in mu_proc[m]["recursion_fncs"]]
    next = 0
    for (m, c) in zip(prior_moments, coeffs):
        next += m * c
    return next

def modified_moments(k_max, m, *args):
    """
    These are the integrals
    \int_{-1}^{1} \frac{P_k(x)}{[(x - a_y)^2 + b_y^2]^(m/2) dx
    computed by the recursive property of the Legendre polynomials.

    k_max is forced to be >= 1
    """
    moments = np.zeros(k_max + 1)
    for cur_k in range(k_max + 1):
        if cur_k in mu_proc[m]:
            moments[cur_k] = mu_proc[m][cur_k](*args)
            continue
        moments[cur_k] = next_mu(cur_k - 1, m,
                                     [moments[cur_k - 1],
                                     moments[cur_k - 2],
                                     moments[cur_k - 3],
                                     moments[cur_k - 4]],
                                     *args)
    return moments
