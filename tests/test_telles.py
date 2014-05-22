from math import sqrt, log
import numpy as np
from quadracheer.gaussian_quad import gaussxwab, gaussxw
from quadracheer.telles_singular import telles_singular, telles_singular_ab
from quadracheer.telles_quasi_singular import telles_quasi_singular

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

def test_QuadLogR():
    f = lambda x: np.log(np.abs(x - 0.5))
    exact = -1.0 - np.log(2.0)
    tx, tw = telles_singular_ab(50, 0.5, 0, 1)
    est = np.sum(f(tx) * tw)
    np.testing.assert_almost_equal(exact, est, 4)

def test_QuadLogR2():
    f = lambda x: x ** 2 * np.log(np.abs(x - 0.9))
    exact = -0.764714
    tx, tw = telles_singular_ab(40, 0.9, 0, 1)
    est = np.sum(f(tx) * tw)
    np.testing.assert_almost_equal(exact, est, 4)

def test_anotherLogRDouble_G11():
    f = lambda x, y: (1 / (3 * np.pi)) *\
        np.log(1.0 / np.abs(x - y)) * x * y
    exact = 1 / (3 * np.pi)
    gx, gw = gaussxwab(75, -1.0, 1.0)
    sum = 0.0
    for (pt, wt) in zip(gx, gw):
        x, w = telles_singular(76, pt)
        g = lambda x: f(x, pt)
        sum += np.sum(g(x) * w * wt)
    np.testing.assert_almost_equal(exact, sum, 5)

