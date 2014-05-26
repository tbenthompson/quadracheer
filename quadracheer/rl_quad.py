"""
Recursive legendre quadrature in one dimension.
"""
import numpy as np
import scipy.special as ss
from gaussian_quad import gaussxw
from rl1 import mu_1_0, mu_1_1, mu_1_2, mu_1_3, mu_kp1
from lobatto_quad import lobatto_quad

def rl_quad(points, ay, by = 0, m = 1):
    mu = calculate_modified_moments(points - 1, m, ay, by)
    x = gll_nodes(points)
    vander = modified_vandermonde(x)
    w = np.linalg.solve(vander, mu)

    x = np.array(x)
    w = np.array(w)
    return x, w

def calculate_modified_moments(k_max, m, ay, by):
    """
    These are the integrals
    \int_{-1}^{1} \frac{P_k(x)}{[(x - a_y)^2 + b_y^2]^(1/2) dx
    computed by the recursive property of the Legendre polynomials.

    k_max is forced to be >= 1
    """
    mu = np.zeros(k_max + 1)

    # Deal with the starting values, k_max must be >=
    mu[0] = mu_1_0(ay, by)
    mu[1] = mu_1_1(ay, by)
    if k_max == 1:
        return mu

    mu[2] = mu_1_2(ay, by)
    if k_max == 2:
        return mu

    mu[3] = mu_1_3(ay, by)
    if k_max == 3:
        return mu

    for cur_k in range(3, k_max):
        mu[cur_k + 1] = mu_kp1(cur_k, 1, ay, by, mu[cur_k], mu[cur_k - 1],
                           mu[cur_k - 2], mu[cur_k - 3])
    return mu

def gll_nodes(points):
    """
    Returns the Gauss-Lobatto-Lagrange nodes on the interval [-1, 1]
    """
    x, w = lobatto_quad(points)
    return x

def modified_vandermonde(pts):
    """
    Creates the Legendre polynomial vandermonde matrix
    """
    degree = len(pts)
    W = np.row_stack([ss.eval_legendre(i, pts)
                         for i in range(degree)])
    return W
