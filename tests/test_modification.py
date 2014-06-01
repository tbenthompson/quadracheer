import numpy as np
from math import log
from quadracheer.recursive_legendre import legendre_integrals
from quadracheer.modification import modify_times_x_minus_a,\
                                     modify_divide_x_minus_a,\
                                     modify_divide_r

def test_mult_x_minus_a():
    moments = legendre_integrals(5)
    est = modify_times_x_minus_a(4, moments, 0.5)
    correct = [-1.0, 2.0 / 3.0, 0.0, 0.0, 0.0]
    np.testing.assert_almost_equal(correct, est)

def test_div_x_minus_a():
    moments = legendre_integrals(5)
    a = 0.5
    first_term = log((1 - a) / (1 + a))
    est = modify_divide_x_minus_a(4, moments, a, first_term)
    correct = [-1.098612288668138,
               1.450693855665931,
               1.637326536083517,
               0.3973095429589644,
               -0.8803490519735331]
    np.testing.assert_almost_equal(correct, est)

def test_div_r():
    moments = legendre_integrals(8)
    a = 0.5
    b = 0.5
    correct = [1.075705420225906,
               0.1474037455542261,
               -0.04366033205431969,
               -0.02023247986691955,
               0.0001466495701568366,
               0.002006397510960772,
               0.0003847499889184325,
               -0.0001317394357582433,
               -0.00006769643570114308]
    est = modify_divide_r(8, moments, a, b, correct[0], correct[1])
    np.testing.assert_almost_equal(correct, est)
