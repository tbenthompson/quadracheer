import numpy as np

from rl1 import mu_kp1, k1, k2

def mu_5_0(ay, by, k1val, k2val):
    term1 = (1 - ay) * (by ** 2 + 2 * k1val ** 2) / (3 * by ** 4 * k1val ** 3)
    term2 = (1 + ay) * (by ** 2 + 2 * k2val ** 2) / (3 * by ** 4 * k2val ** 3)
    return term1 + term2

def mu_5_1(ay, by, k1val, k2val):
    term1 =  (-2 * (-1 + ay) ** 3 * ay -
               3 * (-1 + ay) * ay * by ** 2 -
               by ** 4) / k1val ** 3
    term2 = (2 * ay * (1 + ay) ** 3 + \
            3 * ay * (1 + ay) * by ** 2 + \
            by ** 4) / k2val ** 3
    return (term1 + term2) / (3 * by ** 4)

def mu_5_2(ay, by, k1val, k2val):
    term1a = ((1 - ay) * (3 * ay ** 2 - 1) - 3 * by ** 2 * (1 + ay))
    term1adenom = 3 * by ** 2 * k1val ** 3
    term1a /= term1adenom

    term1b = (1 - ay) * (6 * ay ** 2 + 3 * by ** 2 - 2)
    term1bdenom = 3 * by ** 4 * k1val
    term1b /= term1bdenom

    term1 = 0.5 * (term1a + term1b)

    term2a = ((1 + ay) * (1 - 3 * ay ** 2) + 3 * by ** 2 * (1 - ay))
    term2adenom = 3 * by ** 2 * k2val ** 3
    term2a /= term2adenom

    term2b = (1 + ay) * (2 - 6 * ay ** 2 - 3 * by ** 2)
    term2bdenom = 3 * by ** 4 * k2val
    term2b /= term2bdenom

    term2 = -0.5 * (term2a + term2b)

    return term1 + term2

def mu_5_3(ay, by, k1val, k2val):
    term1a_a = ay * (1 - ay) * (5 * ay ** 2 - 3)
    term1a_b = by ** 2 * (3 - 15 * ay + 5 * by ** 2)
    term1a = term1a_a + term1a_b
    term1adenom = 3 * by ** 2 * k1val ** 3
    term1a /= term1adenom

    term1b = ay * (1 - ay) * (10 * ay ** 2 + 15 * by ** 2 - 6)
    term1bdenom = 3 * by ** 4 * k1val
    term1b /= term1bdenom

    term1 = 0.5 * (term1a + term1b)

    term2a_a = ay * (1 + ay) * (-5 * ay ** 2 + 3)
    term2a_b = by ** 2 * (3 + 15 * ay + 5 * by ** 2)
    term2a = term2a_a + term2a_b
    term2adenom = 3 * by ** 2 * k2val ** 3
    term2a /= term2adenom

    term2b = ay * (1 + ay) * (-10 * ay ** 2 - 15 * by ** 2 + 6)
    term2bdenom = 3 * by ** 4 * k2val
    term2b /= term2bdenom

    term2 = -0.5 * (term2a + term2b)

    term3 = -2.5 * (1 / k1val - 1 / k2val)

    return term1 + term2 + term3

def mu_5_4(ay, by, k1val, k2val):
    return -1.0/24.0*((70*ay**9 + 35*(5*ay + 3)*by**8 - 70*ay**8 - 200*ay**7 + 5*(119*ay**3 + 7*ay**2 + 65*ay + 49)*by**6 + 200*ay**6 + 196*ay**5 + (735*ay**5 - 315*ay**4 + 30*ay**3 - 430*ay**2 + 139*ay + 161)*by**4 - 196*ay**4 - 72*ay**3 + (385*ay**7 - 315*ay**6 - 495*ay**5 + 365*ay**4 + 195*ay**3 - 129*ay**2 - 21*ay + 15)*by**2 + 72*ay**2 - 105*np.sqrt(ay**2 + by**2 - 2*ay + 1)*((by**8 + 2*(ay**2 + 1)*by**6 + (ay**4 - 2*ay**2 + 1)*by**4)*np.arcsinh(np.sqrt(by**2)*(ay + 1)/by**2) - (by**8 + 2*(ay**2 + 1)*by**6 + (ay**4 - 2*ay**2 + 1)*by**4)*np.arcsinh(np.sqrt(by**2)*(ay - 1)/by**2)) + 6*ay - 6)*np.sqrt(ay**2 + by**2 + 2*ay + 1) - (70*ay**9 + 35*(5*ay - 3)*by**8 + 70*ay**8 - 200*ay**7 + 5*(119*ay**3 - 7*ay**2 + 65*ay - 49)*by**6 - 200*ay**6 + 196*ay**5 + (735*ay**5 + 315*ay**4 + 30*ay**3 + 430*ay**2 + 139*ay - 161)*by**4 + 196*ay**4 - 72*ay**3 + (385*ay**7 + 315*ay**6 - 495*ay**5 - 365*ay**4 + 195*ay**3 + 129*ay**2 - 21*ay - 15)*by**2 - 72*ay**2 + 6*ay + 6)*np.sqrt(ay**2 + by**2 - 2*ay + 1))/((by**8 + 2*(ay**2 + 1)*by**6 + (ay**4 - 2*ay**2 + 1)*by**4)*np.sqrt(ay**2 + by**2 + 2*ay + 1)*np.sqrt(ay**2 + by**2 - 2*ay + 1))


mu_init_5 = []
mu_init_5.append(mu_5_0)
mu_init_5.append(mu_5_1)
mu_init_5.append(mu_5_2)
mu_init_5.append(mu_5_3)
mu_init_5.append(mu_5_4)

def calculate_modified_moments_5(k_max, m, ay, by):
    mu = np.zeros(k_max + 1)

    k1val = k1(ay, by)
    k2val = k2(ay, by)

    # Deal with the starting values, k_max must be >=
    mu[0] = mu_init_5[0](ay, by, k1val, k2val)
    mu[1] = mu_init_5[1](ay, by, k1val, k2val)
    if k_max == 1:
        return mu

    mu[2] = mu_init_5[2](ay, by, k1val, k2val)
    if k_max == 2:
        return mu

    mu[3] = mu_init_5[3](ay, by, k1val, k2val)
    if k_max == 3:
        return mu

    mu[4] = mu_init_5[4](ay, by, k1val, k2val)
    if k_max == 4:
        return mu

    for cur_k in range(4, k_max):
        mu[cur_k + 1] = mu_kp1(cur_k, 5.0, ay, by, mu[cur_k], mu[cur_k - 1],
                           mu[cur_k - 2], mu[cur_k - 3])
    return mu

