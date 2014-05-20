from math import sqrt, log
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

