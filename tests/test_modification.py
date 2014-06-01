import numpy as np
from math import log
from quadracheer.legendre import legendre_integrals
from quadracheer.rl135 import rl1, mu_3_0, mu_3_1, mu_5_0, mu_5_1
from quadracheer.recursion import modified_moments
from quadracheer.modification import modify_times_x_minus_a,\
                                     modify_divide_x_minus_a,\
                                     modify_divide_r2

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

def test_div_r2():
    moments = legendre_integrals(8)
    a = 0.5
    b = 1.2
    correct = [1.075705420225906,
               0.1474037455542261,
               -0.04366033205431969,
               -0.02023247986691955,
               0.0001466495701568366,
               0.002006397510960772,
               0.0003847499889184325,
               -0.0001317394357582433,
               -0.00006769643570114308]
    est = modify_divide_r2(8, moments, a, b, correct[0], correct[1])
    np.testing.assert_almost_equal(correct, est)

def test_div_r4():
    moments = legendre_integrals(8)
    a = 0.5
    b = 1.2
    correct = [0.6173845393861046, 0.1483356365409694, -0.03770049188677398, \
    -0.02905919230092682, -0.001801984951851534, 0.003509117545068981, \
    0.001056188292853719, -0.0002247265640304056, -0.0001910931757665019]
    r2 = modify_divide_r2(8, moments, a, b,
                          1.075705420225906,
                          0.1474037455542261)
    est = modify_divide_r2(8, r2, a, b, correct[0], correct[1])
    np.testing.assert_almost_equal(correct, est)

def test_x_minus_a_over_r2():
    moments = legendre_integrals(9)
    a = 0.5
    b = 1.2
    xma = modify_times_x_minus_a(8, moments, a)
    correct = [-0.3904489645587269, 0.2557597125953320, 0.06865217632870157, \
        -0.008511531192587127, -0.007950872775398451, -0.0007266762296354132, \
        0.0006627180067297806, 0.0002093149470004469, -0.00002794901652839316]
    est = modify_divide_r2(8, xma, a, b, correct[0], correct[1])
    np.testing.assert_almost_equal(correct, est)

def test_r3():
    correct = [2.073823077631299, 0.2535989787775645, -0.2911113387007714, \
        -0.08946188432319288, 0.04410566356714894, 0.02448735642709018, \
        -0.005390367432553238, -0.005828732964372520, 0.0002538145298338417]
    a = 0.213
    b = 0.85
    moments = modified_moments(rl1, 8, a, b)
    est = modify_divide_r2(8, moments, a, b, correct[0], correct[1])
    np.testing.assert_almost_equal(correct, est)


def test_r5():
    correct = [2.319371017897694, 0.3821621948981323, -0.4762089061831612, \
            -0.1876037135232636, 0.09309106929258391, 0.06515743698013002, \
            -0.01293167540460619, -0.01866314618738899, 0.0001560375793864460]
    a = 0.213
    b = 0.85
    moments = modified_moments(rl1, 8, a, b)
    r3 = modify_divide_r2(8, moments, a, b,
                          2.073823077631299,
                          0.2535989787775645)
    est = modify_divide_r2(8, r3, a, b, correct[0], correct[1])
    np.testing.assert_almost_equal(correct, est)

def test_r5_more():
    a = 1.2
    b = 1.2
    r1 = modified_moments(rl1, 13, a, b)
    r3 = modify_divide_r2(13, r1, a, b, mu_3_0(a, b), mu_3_1(a, b))
    est = modify_divide_r2(13, r3, a, b, mu_5_0(a, b), mu_5_1(a, b))
    exact = [0.236037, 0.119291, 0.0349751, 0.00321571,
            -0.00239632, -0.00140269, \
            -0.000352242, -4.30148*10**-6, 0.0000340514, 0.0000145451, \
            2.60414*10**-6, -3.46627*10**-7, -2.41911*10**-7,
            -2.52412*10**-6]
    np.testing.assert_almost_equal(est, exact, 5)
