import numpy as np
from quadracheer.rl_quad import rl_quad, gll_nodes, modified_vandermonde
from quadracheer.map import map_rl_quad

def test_small_order_rl_quad():
    for N in range(2, 4):
        x, w = rl_quad(N, 0.5, m = 1, by = 0.5)
        exact = [2.69982, 0.475878]
        for i in range(len(exact)):
            est = np.sum(w * x ** i)
            np.testing.assert_almost_equal(exact[i], est, 5)

def test_rl_quad():
    x, w = rl_quad(10, 0.5, m = 1, by = 0.5)
    exact = [2.69982, 0.475878, 0.826076, 0.238427, 0.470906,
             0.153639, 0.325999, 0.112006, 0.248412, 0.0877162]
    for i in range(10):
        est = np.sum(w * x ** i)
        np.testing.assert_almost_equal(exact[i], est, 5)

def test_rl_quad2():
    x, w = rl_quad(10, 0.5, m = 1, by = 2.5)
    exact = [0.7675150090814903,
             0.01779131391448664,
             0.2514125580305381,
             0.01042604118726131,
             0.1497329916946964,
             0.007351040226216374,
             0.1065156992309334,
             0.005670755078910097,
             0.08263150653792486,
             0.004613594538086503]
    for i in range(10):
        est = np.sum(w * x ** i)
        np.testing.assert_almost_equal(exact[i], est, 11)

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

def test_mapped_rl_quad():
    x, w = map_rl_quad(rl_quad, 10, 0.5, 0.5, 1.0, 2.0)
    exact = [0.9370728722124352,
             1.342568485003826,
             2.000243585190683, \
             3.091829672374373,
             4.940760583752816,
             8.128175556459884, \
             13.70706664710914,
             23.59959784014733,
             41.33761425072659, \
             73.44841816566141]
    for i in range(10):
        est = np.sum(w * x ** i)
        np.testing.assert_almost_equal(exact[i], est, 10)

def test_mapped_rl_quad2():
    x, w = map_rl_quad(rl_quad, 10, 0.2, 0.3, 0.0, 1.0)
    exact = [2.332556553293539,
             0.9603565576440610,
             0.5636909785950152, \
             0.3894662160471265,
             0.2949533988361775,
             0.2365588120191993, \
             0.1971850086215070,
             0.1689379319389373,
             0.1477219765628722, \
             0.1312177383160300]
    for i in range(len(exact)):
        est = np.sum(w * x ** i)
        np.testing.assert_almost_equal(exact[i], est, 9)
