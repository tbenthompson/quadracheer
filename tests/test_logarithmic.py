import numpy as np
from quadracheer.recursion import modified_moments
from quadracheer.logarithmic import log_x_minus_y

def test_log_many():
    correct = [-1.984832619482592, -0.2447556437138158, 0.6365617224898709, \
0.1590602757612699, -0.2272345095098529, -0.1214549048852035, \
0.1063727931042495, 0.09717057757496405, -0.05099522699996534, \
-0.07868250626810564, 0.02037098171372864]
    mu = modified_moments(log_x_minus_y, 10, 0.123)
    np.testing.assert_almost_equal(mu, correct)
