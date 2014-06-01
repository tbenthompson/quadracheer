import numpy as np
from recursive_legendre import mu_proc
from rl135 import k1, k2, A, B, C, D, E

mu_proc[5] = dict()
mu_proc[5]["recursion_fncs"] = [A, B, C, D]
mu_proc[5]["recursion_multiplier"] = E

def mu_5_0(ay, by):
    term1 = (1 - ay) * (by ** 2 + 2 * k1(ay, by) ** 2) / (3 * by ** 4 * k1(ay, by) ** 3)
    term2 = (1 + ay) * (by ** 2 + 2 * k2(ay, by) ** 2) / (3 * by ** 4 * k2(ay, by) ** 3)
    return term1 + term2
mu_proc[5][0] = mu_5_0

def mu_5_1(ay, by):
    term1 =  (-2 * (-1 + ay) ** 3 * ay -
               3 * (-1 + ay) * ay * by ** 2 -
               by ** 4) / k1(ay, by) ** 3
    term2 = (2 * ay * (1 + ay) ** 3 + \
            3 * ay * (1 + ay) * by ** 2 + \
            by ** 4) / k2(ay, by) ** 3
    return (term1 + term2) / (3 * by ** 4)
mu_proc[5][1] = mu_5_1

def mu_5_2(ay, by):
    term1a = ((1 - ay) * (3 * ay ** 2 - 1) - 3 * by ** 2 * (1 + ay))
    term1adenom = 3 * by ** 2 * k1(ay, by) ** 3
    term1a /= term1adenom

    term1b = (1 - ay) * (6 * ay ** 2 + 3 * by ** 2 - 2)
    term1bdenom = 3 * by ** 4 * k1(ay, by)
    term1b /= term1bdenom

    term1 = 0.5 * (term1a + term1b)

    term2a = ((1 + ay) * (1 - 3 * ay ** 2) + 3 * by ** 2 * (1 - ay))
    term2adenom = 3 * by ** 2 * k2(ay, by) ** 3
    term2a /= term2adenom

    term2b = (1 + ay) * (2 - 6 * ay ** 2 - 3 * by ** 2)
    term2bdenom = 3 * by ** 4 * k2(ay, by)
    term2b /= term2bdenom

    term2 = -0.5 * (term2a + term2b)

    return term1 + term2
mu_proc[5][2] = mu_5_2

def mu_5_3(ay, by):
    term1a_a = ay * (1 - ay) * (5 * ay ** 2 - 3)
    term1a_b = by ** 2 * (3 - 15 * ay + 5 * by ** 2)
    term1a = term1a_a + term1a_b
    term1adenom = 3 * by ** 2 * k1(ay, by) ** 3
    term1a /= term1adenom

    term1b = ay * (1 - ay) * (10 * ay ** 2 + 15 * by ** 2 - 6)
    term1bdenom = 3 * by ** 4 * k1(ay, by)
    term1b /= term1bdenom

    term1 = 0.5 * (term1a + term1b)

    term2a_a = ay * (1 + ay) * (-5 * ay ** 2 + 3)
    term2a_b = by ** 2 * (3 + 15 * ay + 5 * by ** 2)
    term2a = term2a_a + term2a_b
    term2adenom = 3 * by ** 2 * k2(ay, by) ** 3
    term2a /= term2adenom

    term2b = ay * (1 + ay) * (-10 * ay ** 2 - 15 * by ** 2 + 6)
    term2bdenom = 3 * by ** 4 * k2(ay, by)
    term2b /= term2bdenom

    term2 = -0.5 * (term2a + term2b)

    term3 = -2.5 * (1 / k1(ay, by) - 1 / k2(ay, by))

    return term1 + term2 + term3
mu_proc[5][3] = mu_5_3

def mu_5_4(ay, by):
    return -1.0/24.0*((70*ay**9 + 35*(5*ay + 3)*by**8 - 70*ay**8 - 200*ay**7 + 5*(119*ay**3 + 7*ay**2 + 65*ay + 49)*by**6 + 200*ay**6 + 196*ay**5 + (735*ay**5 - 315*ay**4 + 30*ay**3 - 430*ay**2 + 139*ay + 161)*by**4 - 196*ay**4 - 72*ay**3 + (385*ay**7 - 315*ay**6 - 495*ay**5 + 365*ay**4 + 195*ay**3 - 129*ay**2 - 21*ay + 15)*by**2 + 72*ay**2 - 105*np.sqrt(ay**2 + by**2 - 2*ay + 1)*((by**8 + 2*(ay**2 + 1)*by**6 + (ay**4 - 2*ay**2 + 1)*by**4)*np.arcsinh(np.sqrt(by**2)*(ay + 1)/by**2) - (by**8 + 2*(ay**2 + 1)*by**6 + (ay**4 - 2*ay**2 + 1)*by**4)*np.arcsinh(np.sqrt(by**2)*(ay - 1)/by**2)) + 6*ay - 6)*np.sqrt(ay**2 + by**2 + 2*ay + 1) - (70*ay**9 + 35*(5*ay - 3)*by**8 + 70*ay**8 - 200*ay**7 + 5*(119*ay**3 - 7*ay**2 + 65*ay - 49)*by**6 - 200*ay**6 + 196*ay**5 + (735*ay**5 + 315*ay**4 + 30*ay**3 + 430*ay**2 + 139*ay - 161)*by**4 + 196*ay**4 - 72*ay**3 + (385*ay**7 + 315*ay**6 - 495*ay**5 - 365*ay**4 + 195*ay**3 + 129*ay**2 - 21*ay - 15)*by**2 - 72*ay**2 + 6*ay + 6)*np.sqrt(ay**2 + by**2 - 2*ay + 1))/((by**8 + 2*(ay**2 + 1)*by**6 + (ay**4 - 2*ay**2 + 1)*by**4)*np.sqrt(ay**2 + by**2 + 2*ay + 1)*np.sqrt(ay**2 + by**2 - 2*ay + 1))
mu_proc[5][4] = mu_5_4
