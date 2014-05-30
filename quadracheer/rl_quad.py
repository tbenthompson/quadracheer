"""
Recursive legendre quadrature in one dimension.
"""
import numpy as np
import scipy.special as ss
from gaussian_quad import gaussxw
from recursive_legendre import modified_moments
from lobatto_quad import lobatto_quad

def rl_quad(points, ay, by = 0, m = 1):
    mu = modified_moments(points - 1, m, ay, by)
    x = gll_nodes(points)
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
    W = np.row_stack([ss.eval_legendre(i, pts)
                         for i in range(degree)])
    return W
