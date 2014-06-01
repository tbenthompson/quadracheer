import numpy as np
from math import sqrt
import scipy.special
from lobatto_quad import lobatto_quad

def next_mu(descriptor, k, prior_moments, *args):
    multiplier = 1.0
    multiplier_fnc = descriptor.get("recursion_multiplier", None)
    if multiplier_fnc:
        multiplier = multiplier_fnc(k, *args)
    coeffs = [f(k, *args) * multiplier for f
              in descriptor["recursion_fncs"]]
    next = 0
    for (m, c) in zip(prior_moments, coeffs):
        next += m * c
    return next

def modified_moments(descriptor, k_max, *args):
    """
    These are the integrals
    \int_{-1}^{1} \frac{P_k(x)}{[(x - a_y)^2 + b_y^2]^(m/2) dx
    computed by the recursive property of the Legendre polynomials.

    k_max is forced to be >= 1
    """
    moments = []
    recursion_width = len(descriptor["recursion_fncs"])
    for cur_k in range(k_max + 1):
        if cur_k in descriptor:
            moments.append(descriptor[cur_k](*args))
            continue
        moments.append(next_mu(descriptor, cur_k - 1,
            moments[cur_k - recursion_width:cur_k][::-1],
                                     *args))
    return moments

def recursive_quad(descriptor, n_pts, *args):
    mu = modified_moments(descriptor, n_pts - 1, *args)
    x = gll_nodes(n_pts)
    vander = modified_vandermonde(x)
    w = np.linalg.solve(vander, mu)

    x = np.array(x)
    w = np.array(w)
    return x, w

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
    W = np.row_stack([scipy.special.eval_legendre(i, pts)
                         for i in range(degree)])
    return W
