"""
Recursive legendre quadrature in one dimension.
"""
import numpy as np
from scipy.special import legendre
from gaussian_quad import gaussxw
from recursive_legendre import mu_1_0, mu_1_1, mu_1_2, mu_1_3, mu_1_kp1

def points_weights(n, m, ay, by):
    xg, wg = gaussxw(n)
    mu = calculate_modified_moments(n - 1, m, ay, by)
    x = []
    w = []
    for (gauss_x, gauss_w) in zip(xg, wg):
        sum = 0
        for j in range(n):
            sum += (2 * j + 1) * mu[j] * legendre(j)(gauss_x)
        x.append(gauss_x)
        w.append(0.5 * gauss_w * sum)
    x = np.array(x)
    w = np.array(w)
    return x, w

def calculate_modified_moments(k_max, m, ay, by):
    mu = np.zeros(k_max + 1)
    mu[0] = mu_1_0(ay, by)
    mu[1] = mu_1_1(ay, by)
    mu[2] = mu_1_2(ay, by)
    mu[3] = mu_1_3(ay, by)
    for cur_k in range(3, k_max):
        mu[cur_k + 1] = mu_1_kp1(cur_k, 1, ay, by, mu[cur_k], mu[cur_k - 1],
                           mu[cur_k - 2], mu[cur_k - 3])
    return mu
