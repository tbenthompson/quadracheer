import numpy as np
from math import sqrt, log

"""
This is all pretty sloppy. Clean it up before implementing in any real
application!
"""
# The starting value definitions are taken from the Aimi and Diligenti 2002
# paper
def k1(ay, by):
    return sqrt((ay - 1) ** 2 + by ** 2)
def k2(ay, by):
    return sqrt((ay + 1) ** 2 + by ** 2)

def mu_1_0(ay, by):
    numer = k1(ay, by) + 1 - ay
    denom = k2(ay, by) - 1 - ay
    return log(numer / denom)

def mu_1_1(ay, by):
    return ay * mu_1_0(ay, by) - k2(ay, by) + k1(ay, by)

def mu_1_2(ay, by):
    term1 = -0.25 * (3 * by ** 2 - 2 * (3 * ay ** 2 - 1)) * mu_1_0(ay, by)
    term2 = -0.75 * (3 * ay - 1) * k2(ay, by)
    term3 = 0.75 * (3 * ay + 1) * k1(ay, by)
    return term1 + term2 + term3

def mu_1_3(ay, by):
    term1 = -0.25 * ay * (15 * by ** 2 - 2 * (5 * ay ** 2 - 3)) *\
            mu_1_0(ay, by)
    term2 = -(20 * by ** 2 - ((55 * ay + 25) * ay - 8)) * k1(ay, by) / 12
    term3 = +(20 * by ** 2 - ((55 * ay - 25) * ay - 8)) * k2(ay, by) / 12
    return term1 + term2 + term3

# The recursion for the legendre polynomials
k = np.arange(40, dtype=np.float64)
alpha = (2 * k + 1) / (k + 1)
beta = k / (k + 1)

def P0(x):
    return 1

def P1(x):
    return x

def P(n, x):
    """ Only call for n > 2 """
    pass


def A(k, m, ay, by):
    return ((2 * k - 1) / (m - 2)) * (2 * ay / alpha[k - 1]) - ay

def B(k, m, ay, by):
    term1 = (beta[k] * alpha[k - 2] - alpha[k]) / (alpha[k] * alpha[k - 2])
    term2 = -(2 * k - 1) / (m - 2) * \
            (((beta[k] * alpha[k - 2] + beta[k - 1] * alpha[k]) /\
                    (alpha[k - 2] * alpha[k - 1] * alpha[k])) + \
                (ay ** 2 + by ** 2))
    return term1 + term2

def C(k, m, ay, by):
    return ay + (2 * k - 1) / (m - 2) * (2 * ay / alpha[k - 1] * beta[k - 1])

def D(k, m, ay, by):
    return - ((2 * k - 1) / (m - 2) *\
            ((beta[k - 1] * beta[k - 2]) / (alpha[k - 2] * alpha[k - 1]))) - \
            (beta[k - 2] / alpha[k - 2])

def E(k, m, ay, by):
    numer = (m - 2) * alpha[k - 1] * alpha[k]
    denom = (2 * k - 1) - (m - 2) * alpha[k - 1]
    return numer / denom

def mu_1_kp1(k, m, ay, by, mu_1_k, mu_1_km1, mu_1_km2, mu_1_km3):
    Ak = A(k, m, ay, by)
    Bk = B(k, m, ay, by)
    Ck = C(k, m, ay, by)
    Dk = D(k, m, ay, by)
    Ek = E(k, m, ay, by)
    Ak *= Ek
    Bk *= Ek
    Ck *= Ek
    Dk *= Ek
    next = Dk * mu_1_km3 + Ck * mu_1_km2 +\
           Bk * mu_1_km1 + Ak * mu_1_k
    return next

def mu_1(k, m, ay, by):
    mu_1_km3 = mu_1_0(ay, by)
    mu_1_km2 = mu_1_1(ay, by)
    mu_1_km1 = mu_1_2(ay, by)
    mu_1_k = mu_1_3(ay, by)
    for cur_k in range(3, k):
        next_mu = mu_1_kp1(cur_k, 1, ay, by, mu_1_k, mu_1_km1, mu_1_km2, mu_1_km3)
        mu_1_km3 = mu_1_km2
        mu_1_km2 = mu_1_km1
        mu_1_km1 = mu_1_k
        mu_1_k = next_mu
    return mu_1_k
