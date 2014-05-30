import numpy as np
from quadracheer.recursive_legendre import init_mu, modified_moments,\
        k1, k2, alpha
from quadracheer.rl1 import *

def test_alpha():
    assert(alpha(0) == 1.0 / 1.0)
    assert(alpha(1) == 3.0 / 2.0)
    assert(alpha(2) == 5.0 / 3.0)
    assert(alpha(3) == 7.0 / 4.0)

# Exact integrals computed using mathematica
def test_initm10():
    est = init_mu[1][0](1.2, 1.2, k1(1.2, 1.2), k2(1.2, 1.2))
    exact = 1.20061
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm10_singular():
    est = init_mu[1][0](0.8, 0.8, k1(0.8, 0.8), k2(0.8, 0.8))
    exact = 1.79762
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm11():
    est = init_mu[1][1](1.2, 1.2, k1(1.2, 1.2), k2(1.2, 1.2))
    exact = 0.151292
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm11_singular():
    est = init_mu[1][1](0.6, 0.6, k1(0.6, 0.6), k2(0.6, 0.6))
    exact = 0.411843
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm12():
    est = init_mu[1][2](1.2, 1.2, k1(1.2, 1.2), k2(1.2, 1.2))
    exact = 0.00677433
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm12_singular():
    est = init_mu[1][2](0.01, 0.01, k1(0.01, 0.01), k2(0.01, 0.01))
    exact = -3.79787
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm13():
    est = init_mu[1][3](1.2, 1.2, k1(1.2, 1.2), k2(1.2, 1.2))
    exact = -0.00407733
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm13_singular():
    est = init_mu[1][3](-0.01, -0.01, k1(-0.01, -0.01), k2(-0.01, -0.01))
    exact = 0.103966
    np.testing.assert_almost_equal(est, exact, 5)

def test_value():
    ay = 0.05
    by = 0.05
    pairs = [(4,1.1977), (10,-0.376323),
             (20, 0.0743193), (30, -0.00279826),
             (31, -0.05510287928339709)]
    for n, exact in pairs:
        est = modified_moments(n, 1, ay, by)
        np.testing.assert_almost_equal(est[-1], exact, 5)

def test_distant_value():
    ay = 5.0
    by = 5.0
    exact = 9.68812e-8
    est = modified_moments(9, 1, ay, by)
    np.testing.assert_almost_equal(est[-1], exact, 5)
