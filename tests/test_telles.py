from math import sqrt, log
import numpy as np
from quadracheer.gaussian_quad import gaussxwab, gaussxw
from quadracheer.telles_singular import telles_singular
from quadracheer.telles_quasi_singular import telles_quasi_singular

def test_gauss():
    # test lgwt
    x, w = gaussxwab(3, -1.0, 1.0)
    # Exact values retrieved from the wikipedia page on Gaussian Quadrature
    # The main function has been tested by the original author for a wide
    # range of orders. But, this is just to check everything is still working
    # properly
    np.testing.assert_almost_equal(x[0], sqrt(3.0 / 5.0))
    np.testing.assert_almost_equal(x[1], 0.0)
    np.testing.assert_almost_equal(x[2], -sqrt(3.0 / 5.0))
    np.testing.assert_almost_equal(w[0], (5.0 / 9.0))
    np.testing.assert_almost_equal(w[1], (8.0 / 9.0))
    np.testing.assert_almost_equal(w[2], (5.0 / 9.0))

def test_telles_singular():
    #test telles quadrature with the example from the paper (page 964)
    g = lambda x: np.log(np.abs(0.3 + x))
    exact = -1.908598917
    # compare with the result Telles gets for 10 points to make sure
    # the method is properly implemented.
    telles_paper_10_pts = -1.90328
    tx, tw = telles_singular(10, -0.3)
    est = np.sum(g(tx) * tw)
    np.testing.assert_almost_equal(telles_paper_10_pts, est, 5)

def test_quasi_singular():
    #test telles quasi-singular quadrature with the example on page 966
    # sing_pt = 1.08
    sing_pt = 1.004
    g = lambda x: (sing_pt - x) ** -2

    # for sing_pt = 1.08
    # exact = 12.0192
    # for sing_pt = 1.004
    exact = 249.500998
    x_nearest = 1.0
    D = sing_pt - 1.0
    N = 20
    [tx, tw] = telles_quasi_singular(N, x_nearest, D)
    est_telles = np.sum(g(tx) * tw)
    [gx, gw] = gaussxw(N)
    est_gauss = np.sum(g(gx) * gw)

    gauss_error = abs(est_gauss - exact) / exact * 100
    telles_error = abs(est_telles - exact) / exact * 100
    assert(telles_error < gauss_error / 1000)


