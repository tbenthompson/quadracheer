import numpy as np
from math import sqrt, log
from util import memoized
from recursive_legendre import mu_proc
from rl135 import k1, k2, A, B, C, D, E

# The starting value definitions are taken from the Aimi and Diligenti 2002
# paper
mu_proc[1] = dict()
mu_proc[1]["recursion_fncs"] = [A, B, C, D]
mu_proc[1]["recursion_multiplier"] = E

def mu_1_0(ay, by):
    numer = k1(ay, by) + 1 - ay
    denom = k2(ay, by) - 1 - ay
    return log(numer / denom)
mu_proc[1][0] = mu_1_0

def mu_1_1(ay, by):
    return ay * mu_1_0(ay, by) - k2(ay, by) + k1(ay, by)
mu_proc[1][1] = mu_1_1

def mu_1_2(ay, by):
    term1 = -0.25 * (3 * by ** 2 - 2 * (3 * ay ** 2 - 1)) *\
            mu_1_0(ay, by)
    term2 = -0.75 * (3 * ay - 1) * k2(ay, by)
    term3 = 0.75 * (3 * ay + 1) * k1(ay, by)
    return term1 + term2 + term3
mu_proc[1][2] = mu_1_2

def mu_1_3(ay, by):
    term1 = -0.25 * ay * (15 * by ** 2 - 2 * (5 * ay ** 2 - 3)) *\
            mu_1_0(ay, by)
    term2 = -(20 * by ** 2 - ((55 * ay + 25) * ay - 8)) * k1(ay, by) / 12
    term3 = +(20 * by ** 2 - ((55 * ay - 25) * ay - 8)) * k2(ay, by) / 12
    return term1 + term2 + term3
mu_proc[1][3] = mu_1_3
