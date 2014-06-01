from recursive_legendre import alpha, beta
from math import sqrt

def k1(ay, by):
    return sqrt((ay - 1) ** 2 + by ** 2)

def k2(ay, by):
    return sqrt((ay + 1) ** 2 + by ** 2)

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
