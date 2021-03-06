from math import sqrt, log
import numpy as np
from quadracheer.map import map_pts_wts
from quadracheer import gaussian_quad
from quadracheer import lobatto_quad

def test_gauss():
    """
    Exact values retrieved from the wikipedia page on Gaussian Quadrature
    The main function has been tested by the original author for a wide
    range of orders. But, this is just to check everything is still working
    properly
    """
    x, w = gaussian_quad.gaussxw(3)
    x, w = map_pts_wts(x, w, -1.0, 1.0)
    np.testing.assert_almost_equal(x[0], sqrt(3.0 / 5.0))
    np.testing.assert_almost_equal(x[1], 0.0)
    np.testing.assert_almost_equal(x[2], -sqrt(3.0 / 5.0))
    np.testing.assert_almost_equal(w[0], (5.0 / 9.0))
    np.testing.assert_almost_equal(w[1], (8.0 / 9.0))
    np.testing.assert_almost_equal(w[2], (5.0 / 9.0))


def test_build_sets():
    """
    Tests whether the Gaussian quadrature method is properly storing
    the orders that have already been computed
    """
    x, w = gaussian_quad.gaussxw(3)
    x2, w2 = gaussian_quad.gaussxw(12)
    np.testing.assert_almost_equal(x, gaussian_quad.x_set[3])
    np.testing.assert_almost_equal(w, gaussian_quad.w_set[3])
    np.testing.assert_almost_equal(x2, gaussian_quad.x_set[12])
    np.testing.assert_almost_equal(w2, gaussian_quad.w_set[12])

def test_QuadGauss():
    f = lambda x: 3 * x ** 2
    F = lambda x: x ** 3
    exact = F(1) - F(0)
    x, w = gaussian_quad.gaussxw(2)
    x, w = map_pts_wts(x, w, 0, 1)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(exact, est)

def test_lobatto():
    f = lambda x: 12 * x ** 11
    F = lambda x: x ** 12
    exact = F(1) - F(0)
    x, w = lobatto_quad(7)
    x, w = map_pts_wts(x, w, 0, 1)
    assert(len(x) == 7)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(exact, est)
