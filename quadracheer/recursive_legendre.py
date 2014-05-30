"""Shared methods for the recursive legendre quadrature methods."""
import numpy as np
from util import memoized
from math import sqrt

init_mu = dict()

def k1(ay, by):
    return sqrt((ay - 1) ** 2 + by ** 2)

def k2(ay, by):
    return sqrt((ay + 1) ** 2 + by ** 2)

@memoized
def alpha(k):
    return (2.0 * k + 1.0) / (k + 1.0)

@memoized
def beta(k):
    return k / (k + 1.0)

def A(k, m, ay, by):
    return ((2 * k - 1) / (m - 2.0)) * (2 * ay / alpha(k - 1)) - ay

def B(k, m, ay, by):
    term1 = (beta(k) * alpha(k - 2) - alpha(k)) / (alpha(k) * alpha(k - 2))
    term2 = -(2 * k - 1) / (m - 2.0) * \
            (((beta(k) * alpha(k - 2) + beta(k - 1) * alpha(k)) /\
                    (alpha(k - 2) * alpha(k - 1) * alpha(k))) + \
                (ay ** 2 + by ** 2))
    return term1 + term2

def C(k, m, ay, by):
    return ay + (2 * k - 1) / (m - 2.0) * (2 * ay / alpha(k - 1) * beta(k - 1))

def D(k, m, ay, by):
    return - ((2 * k - 1) / (m - 2.0) *\
            ((beta(k - 1) * beta(k - 2)) / (alpha(k - 2) * alpha(k - 1)))) - \
            (beta(k - 2) / alpha(k - 2))

def E(k, m, ay, by):
    numer = (m - 2.0) * alpha(k - 1) * alpha(k)
    denom = (2 * k - 1) - (m - 2.0) * alpha(k - 1)
    return numer / denom

def next_mu(k, m, ay, by, mu_k, mu_km1, mu_km2, mu_km3):
    Ak = A(k, m, ay, by)
    Bk = B(k, m, ay, by)
    Ck = C(k, m, ay, by)
    Dk = D(k, m, ay, by)
    Ek = E(k, m, ay, by)
    Ak *= Ek
    Bk *= Ek
    Ck *= Ek
    Dk *= Ek
    next = Dk * mu_km3 + Ck * mu_km2 +\
           Bk * mu_km1 + Ak * mu_k
    return next

def modified_moments(k_max, m, ay, by):
    """
    These are the integrals
    \int_{-1}^{1} \frac{P_k(x)}{[(x - a_y)^2 + b_y^2]^(m/2) dx
    computed by the recursive property of the Legendre polynomials.

    k_max is forced to be >= 1
    """
    moments = np.zeros(k_max + 1)
    k1val = k1(ay, by)
    k2val = k2(ay, by)
    for cur_k in range(k_max + 1):
        if cur_k in init_mu[m]:
            moments[cur_k] = init_mu[m][cur_k](ay, by, k1val, k2val)
            continue
        moments[cur_k] = next_mu(cur_k - 1, m, ay, by,
                                     moments[cur_k - 1],
                                     moments[cur_k - 2],
                                     moments[cur_k - 3],
                                     moments[cur_k - 4])
    return moments
