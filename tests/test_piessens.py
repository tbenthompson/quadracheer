import numpy as np
from quadracheer.piessens import piessen_neg_one_to_one_nodes,\
                                 piessen_method, piessens

def test_piessen_neg_1_1():
    # Example 1 from Piessens
    f = lambda x: np.exp(x)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = piessen_neg_one_to_one_nodes(2)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_piessen_0_1():
    # Example 1 from Piessens mapped to [0,1]
    g = lambda x: np.exp(x)
    f = lambda x: g((2 * x) - 1)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = piessen_method(2, 0.0, 1.0, 0.5, False)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_piessen_0_1_with_singularity():
    # Example 1 from Piessens mapped to [0,1] and with singularity
    g = lambda x: np.exp(x) / x
    f = lambda x: 2 * g((2 * x) - 1)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = piessen_method(2, 0.0, 1.0, 0.5)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_QuadOneOverR_1():
    f = lambda x: 1 / (x - 0.4)
    exact = np.log(3.0 / 2.0)
    qx, qw = piessens(2, 0.4, nonsingular_N = 10)
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_QuadOneOverR_2():
    # Example 1 from Piessens
    g = lambda x: np.exp(x) / x
    f = lambda x: 2 * g((2 * x) - 1)
    exact = 2.11450175
    qx, qw = piessens(8, 0.5)
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_QuadOneOverR_3():
    # Example 2 from Piessens
    g = lambda x: np.exp(x) / (np.sin(x) - np.cos(x))
    f = lambda x: np.pi / 2.0 * g(np.pi / 2.0 * x)
    exact = 2.61398312
    # Piessens estimate derived with a two pt rule.
    piessens_est = 2.61398135
    qx, qw = piessens(2, 0.5)
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(piessens_est, est)

# Tests x in the upper half of the interval
def test_QuadOneOverR_4():
    f = lambda x: np.exp(x) / (x - 0.8)
    exact = -1.13761642399
    qx, qw = piessens(2, 0.8, nonsingular_N = 20)
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

# Tests x in the lower half of the interval.
def test_QuadOneOverR_5():
    f = lambda x: np.exp(x) / (x - 0.2)
    exact = 3.139062607254266
    qx, qw = piessens(2, 0.2, nonsingular_N = 50)
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)


