import numpy as np
from quadracheer.rl_quad import points_weights

def test_rl_quad():
    x, w = points_weights(10, 1, 0.5, 0.5)
    exact = [2.69982, 0.475878, 0.826076, 0.238427, 0.470906,
             0.153639, 0.325999, 0.112006, 0.248412, 0.0877162]
    for i in range(10):
        est = np.sum(w * x ** i)
        np.testing.assert_almost_equal(exact[i], est, 5)
