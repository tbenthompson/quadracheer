import numpy as np
from math import sqrt, log
from util import memoized
from recursive_legendre import k1, k2, init_mu

# The starting value definitions are taken from the Aimi and Diligenti 2002
# paper
init_mu[1] = dict()

def mu_1_0(ay, by, k1val, k2val):
    numer = k1val + 1 - ay
    denom = k2val - 1 - ay
    return log(numer / denom)
init_mu[1][0] = mu_1_0

def mu_1_1(ay, by, k1val, k2val):
    return ay * mu_1_0(ay, by, k1val, k2val) - k2val + k1val
init_mu[1][1] = mu_1_1

def mu_1_2(ay, by, k1val, k2val):
    term1 = -0.25 * (3 * by ** 2 - 2 * (3 * ay ** 2 - 1)) *\
            mu_1_0(ay, by, k1val, k2val)
    term2 = -0.75 * (3 * ay - 1) * k2val
    term3 = 0.75 * (3 * ay + 1) * k1val
    return term1 + term2 + term3
init_mu[1][2] = mu_1_2

def mu_1_3(ay, by, k1val, k2val):
    term1 = -0.25 * ay * (15 * by ** 2 - 2 * (5 * ay ** 2 - 3)) *\
            mu_1_0(ay, by, k1val, k2val)
    term2 = -(20 * by ** 2 - ((55 * ay + 25) * ay - 8)) * k1val / 12
    term3 = +(20 * by ** 2 - ((55 * ay - 25) * ay - 8)) * k2val / 12
    return term1 + term2 + term3
init_mu[1][3] = mu_1_3
