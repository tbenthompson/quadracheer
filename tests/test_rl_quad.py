import numpy as np
from quadracheer.rl_quad import rl_quad, gll_nodes, modified_vandermonde

def test_rl_quad():
    x, w = rl_quad(10, 0.5, m = 1, by = 0.5)
    exact = [2.69982, 0.475878, 0.826076, 0.238427, 0.470906,
             0.153639, 0.325999, 0.112006, 0.248412, 0.0877162]
    for i in range(10):
        est = np.sum(w * x ** i)
        np.testing.assert_almost_equal(exact[i], est, 5)

def test_high_order_rl_quad():
    N = 100
    x, w = rl_quad(N, 0.5, m = 1, by = 0.5)
    exact = dict()
    # I get 10 digits at N = 100
    exact[100] = 0.007920933265917480
    # I get 5 digits at N = 110
    exact[110] = 0.007192320886184629
    exact[121] = 0.01703712555375862
    exact[150] = 0.005257679877042412
    exact[200] = 0.003934647991362962
    est = np.sum(w * x ** (N - 1))
    np.testing.assert_almost_equal(exact[N], est, 10)


def test_gll_nodes():
    nodes = gll_nodes(10)
    np.testing.assert_almost_equal(nodes[0], -1.0)
    np.testing.assert_almost_equal(nodes[-1], 1.0)
    np.testing.assert_almost_equal(nodes[-2],
                    0.9195339081664588138289, 15)

def test_inv_mod_vander():
    x = gll_nodes(100)
    W = modified_vandermonde(x)
    assert(np.linalg.cond(W) < 20.1)
