from legendre import alpha, beta
from math import sqrt, log, atan

def k1(ay, by):
    return sqrt((ay - 1) ** 2 + by ** 2)

def k2(ay, by):
    return sqrt((ay + 1) ** 2 + by ** 2)

def A(k, ay, by):
    return ((2 * k - 1) / (-1.0)) * (2 * ay / alpha(k - 1)) - ay

def B(k, ay, by):
    term1 = (beta(k) * alpha(k - 2) - alpha(k)) / (alpha(k) * alpha(k - 2))
    term2 = -(2 * k - 1) / (-1.0) * \
            (((beta(k) * alpha(k - 2) + beta(k - 1) * alpha(k)) /\
                    (alpha(k - 2) * alpha(k - 1) * alpha(k))) + \
                (ay ** 2 + by ** 2))
    return term1 + term2

def C(k, ay, by):
    return ay + (2 * k - 1) / (-1.0) * (2 * ay / alpha(k - 1) * beta(k - 1))

def D(k, ay, by):
    return - ((2 * k - 1) / (-1.0) *\
            ((beta(k - 1) * beta(k - 2)) / (alpha(k - 2) * alpha(k - 1)))) - \
            (beta(k - 2) / alpha(k - 2))

def E(k, ay, by):
    numer = (-1.0) * alpha(k - 1) * alpha(k)
    denom = (2 * k - 1) - (-1.0) * alpha(k - 1)
    return numer / denom
# The starting value definitions are taken from the Aimi and Diligenti 2002
# paper
rl1 = dict()
rl1["recursion_fncs"] = [A, B, C, D]
rl1["recursion_multiplier"] = E

def mu_1_0(ay, by):
    numer = k1(ay, by) + 1 - ay
    denom = k2(ay, by) - 1 - ay
    return log(numer / denom)
rl1[0] = mu_1_0

def mu_1_1(ay, by):
    return ay * mu_1_0(ay, by) - k2(ay, by) + k1(ay, by)
rl1[1] = mu_1_1

def mu_1_2(ay, by):
    term1 = -0.25 * (3 * by ** 2 - 2 * (3 * ay ** 2 - 1)) *\
            mu_1_0(ay, by)
    term2 = -0.75 * (3 * ay - 1) * k2(ay, by)
    term3 = 0.75 * (3 * ay + 1) * k1(ay, by)
    return term1 + term2 + term3
rl1[2] = mu_1_2

def mu_1_3(ay, by):
    term1 = -0.25 * ay * (15 * by ** 2 - 2 * (5 * ay ** 2 - 3)) *\
            mu_1_0(ay, by)
    term2 = -(20 * by ** 2 - ((55 * ay + 25) * ay - 8)) * k1(ay, by) / 12
    term3 = +(20 * by ** 2 - ((55 * ay - 25) * ay - 8)) * k2(ay, by) / 12
    return term1 + term2 + term3
rl1[3] = mu_1_3

def mu_2_0(ay, by):
    return (atan((1 - ay) / by) + atan((1 + ay) / by)) / by

def mu_2_1(ay, by):
    return (1.0 / (2.0 * by)) * \
            (2.0 * ay * \
                    (atan((1.0 - ay) / by) + atan((1.0 + ay) / by)) + \
              by * \
                    (log((-1.0 + ay) ** 2 + by ** 2) - \
                     log((1.0 + ay) ** 2 + by ** 2)))

def mu_3_0(ay, by):
    term1 = (1 - ay) / (by ** 2 * k1(ay, by))
    term2 = (1 + ay) / (by ** 2 * k2(ay, by))
    return term1 + term2

def mu_3_1(ay, by):
    return -((ay**2 - ay + by**2)*sqrt(ay**2 + 2*ay + by**2 + 1) - (ay**2 + ay + by**2)*sqrt(ay**2 - 2*ay + by**2 + 1))/(by**2*sqrt(ay**2 - 2*ay + by**2 + 1)*sqrt(ay**2 + 2*ay + by**2 + 1))

def mu_4_0(a, b):
    return ((b - a * b) / ((-1.0 + a) ** 2 + b ** 2) + \
            ((1.0 + a) * b) / ((1.0 + a) ** 2 + b ** 2) +
            atan((1.0 - a) / b) + atan((1 + a) / b))\
            /(2 * b ** 3)

def mu_4_1(a, b):
    return (a * (-((2.0 * b * (-1.0 + a ** 2 + b ** 2)) \
                    /((-1.0 + a ** 2) ** 2 + \
                      2 * (1 + a ** 2) * b ** 2 + b ** 4)) + \
                  atan((1.0 - a) / b) + atan((1.0 + a) / b)))\
            / (2 * b ** 3)

def mu_5_0(ay, by):
    term1 = (1 - ay) * (by ** 2 + 2 * k1(ay, by) ** 2) / (3 * by ** 4 * k1(ay, by) ** 3)
    term2 = (1 + ay) * (by ** 2 + 2 * k2(ay, by) ** 2) / (3 * by ** 4 * k2(ay, by) ** 3)
    return term1 + term2

def mu_5_1(ay, by):
    term1 =  (-2 * (-1 + ay) ** 3 * ay -
               3 * (-1 + ay) * ay * by ** 2 -
               by ** 4) / k1(ay, by) ** 3
    term2 = (2 * ay * (1 + ay) ** 3 + \
            3 * ay * (1 + ay) * by ** 2 + \
            by ** 4) / k2(ay, by) ** 3
    return (term1 + term2) / (3 * by ** 4)

def mu_6_0(ay, by):
    return ((-3.0 * (-1.0 + ay)**3.0 * by - 5.0 * (-1.0 + ay) * by**3.0)/\
            ((-1.0 + ay)**2.0 + by**2.0)**2.0 + \
            (3.0 * (1.0 + ay)**3.0 * by + 5.0 * (1.0 + ay) * by ** 3.0)/\
            ((1.0 + ay)**2.0 + by**2.0)**2.0 -
            3.0 * atan((-1.0 + ay)/by) + 3.0 * atan((1.0 + ay)/by))/\
                    (8.0 * by**5.0)

def mu_6_1(ay, by):
    return 1.0 / (8.0 * by**5.0) * ay * \
        ((2.0 * by * (-3.0 * (-1.0 + ay ** 2.0) ** 3.0 + \
            (11.0 - 2.0 * ay**2.0 - 9.0 * ay ** 4.0) * by ** 2.0 + \
            (5.0 - 9.0 * ay**2.0) * by**4.0
            - 3.0 * by**6.0))\
        /(ay**4.0 + 2.0 * ay**2.0 * (-1.0 + by**2.0) +
        (1.0 + by**2.0)**2.0)**2.0 - 3.0 * atan((-1.0 + ay)/by) +
   3.0 * atan((1.0 + ay)/by))
