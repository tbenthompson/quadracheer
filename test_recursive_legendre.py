import numpy as np
from recursive_legendre import *

def test_init0():
    est = mu_1_0(1.2, 1.2)
    exact = 1.20061
    np.testing.assert_almost_equal(est, exact, 5)

def test_init1():
    est = mu_1_1(1.2, 1.2)
    exact = 0.151292
    np.testing.assert_almost_equal(est, exact, 5)

def test_init2():
    est = mu_1_2(1.2, 1.2)
    exact = 0.00677433
    np.testing.assert_almost_equal(est, exact, 5)

def test_init3():
    est = mu_1_3(1.2, 1.2)
    exact = -0.00407733
    np.testing.assert_almost_equal(est, exact, 5)
