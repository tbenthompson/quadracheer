import numpy as np
from quadracheer.rl1 import k1, k2
from quadracheer.rl5 import *
from quadracheer.recursive_legendre import modified_moments, mu_proc

def test_initm5():
    ay, by = (1.2, 1.2)
    exact = [0.236037, 0.119291, 0.0349751, 0.00321571, -0.00239632]
    for i in range(5):
        est = mu_proc[5][i](ay, by)
        np.testing.assert_almost_equal(est, exact[i], 5)

def test_more_values():
    est = modified_moments(15, 5, 1.2, 1.2)
    exact = [0.236037, 0.119291, 0.0349751, 0.00321571,
            -0.00239632, -0.00140269, \
            -0.000352242, -4.30148*10**-6, 0.0000340514, 0.0000145451, \
            2.60414*10**-6, -3.46627*10**-7, -2.41911*10**-7,
            -2.52412*10**-6]
    for i in range(14):
        np.testing.assert_almost_equal(est[i], exact[i], 5)
