import numpy as np
from quadracheer.recursion import modified_moments
from quadracheer.rl135 import rl1, mu_5_0, mu_5_1, mu_3_0, mu_3_1,\
                                   mu_2_0, mu_2_1, mu_4_0, mu_4_1,\
                                   mu_6_0, mu_6_1

def test_initm10():
    est = rl1[0](1.2, 1.2)
    exact = 1.20061
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm10_singular():
    est = rl1[0](0.8, 0.8)
    exact = 1.79762
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm11():
    est = rl1[1](1.2, 1.2)
    exact = 0.151292
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm11_singular():
    est = rl1[1](0.6, 0.6)
    exact = 0.411843
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm12():
    est = rl1[2](1.2, 1.2)
    exact = 0.00677433
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm12_singular():
    est = rl1[2](0.01, 0.01)
    exact = -3.79787
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm13():
    est = rl1[3](1.2, 1.2)
    exact = -0.00407733
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm13_singular():
    est = rl1[3](-0.01, -0.01)
    exact = 0.103966
    np.testing.assert_almost_equal(est, exact, 5)

def test_initm20():
    exact = 4.068887871591405
    est = mu_2_0(0.5, 0.5)
    np.testing.assert_almost_equal(est, exact, 15)

def test_initm21():
    exact = 1.229724979578653
    est = mu_2_1(0.5, 0.5)
    np.testing.assert_almost_equal(est, exact, 15)


def test_initm30():
    est = mu_3_0(0.213, 0.85)
    exact = 2.073823077631299
    np.testing.assert_almost_equal(est, exact, 11)

def test_initm31():
    est = mu_3_1(0.213, 0.85)
    exact = 0.2535989787775645
    np.testing.assert_almost_equal(est, exact, 11)

def test_initm40():
    exact = 11.33777574318281
    est = mu_4_0(0.5, 0.5)
    np.testing.assert_almost_equal(est, exact, 15)

def test_initm41():
    exact = 4.868887871591406
    est = mu_4_1(0.5, 0.5)
    np.testing.assert_almost_equal(est, exact, 14)

def test_initm60():
    exact = 36.25332722954843
    est = mu_6_0(0.5, 0.5)
    np.testing.assert_almost_equal(est, exact, 15)

def test_initm61():
    exact = 17.16666361477422
    est = mu_6_1(0.5, 0.5)
    np.testing.assert_almost_equal(est, exact, 14)


def test_value():
    ay = 0.05
    by = 0.05
    pairs = [(4,1.1977), (10,-0.376323),
             (20, 0.0743193), (30, -0.00279826),
             (31, -0.05510287928339709)]
    for n, exact in pairs:
        est = modified_moments(rl1, n, ay, by)
        np.testing.assert_almost_equal(est[-1], exact, 5)

def test_distant_value():
    ay = 5.0
    by = 5.0
    exact = 9.68812e-8
    est = modified_moments(rl1, 9, ay, by)
    np.testing.assert_almost_equal(est[-1], exact, 5)

def test_initm5():
    ay, by = (1.2, 1.2)
    exact = [0.236037, 0.119291, 0.0349751, 0.00321571, -0.00239632]
    est0 = mu_5_0(ay, by)
    est1 = mu_5_1(ay, by)
    np.testing.assert_almost_equal([est0, est1], exact[0:2], 5)
