import numpy as np
from quadracheer.recursion import recursive_quad,\
                gll_nodes, modified_vandermonde, modified_moments
from quadracheer.rl135 import rl1
from quadracheer.map import map_pts_wts, map_singular_pt,\
        map_distance_to_interval, map_weights_by_inv_power

def test_small_order_recursive_quad():
    for N in range(2, 4):
        moments = modified_moments(rl1, N - 1, 0.5, 0.5)
        x, w = recursive_quad(moments)
        exact = [2.69982, 0.475878]
        for i in range(len(exact)):
            est = np.sum(w * x ** i)
            np.testing.assert_almost_equal(exact[i], est, 5)

def test_recursive_quad():
    x, w = recursive_quad(modified_moments(rl1, 9, 0.5, 0.5))
    exact = [2.69982, 0.475878, 0.826076, 0.238427, 0.470906,
             0.153639, 0.325999, 0.112006, 0.248412, 0.0877162]
    for i in range(10):
        est = np.sum(w * x ** i)
        np.testing.assert_almost_equal(exact[i], est, 5)

def test_recursive_quad2():
    x, w = recursive_quad(modified_moments(rl1, 9, 0.5, 2.5))
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

def test_high_order_recursive_quad():
    N = 100
    x, w = recursive_quad(modified_moments(rl1, N - 1, 0.5, 0.5))
    exact = dict()
    # I get 10 digits at N = 100
    exact[100] = 0.007920933265917480
    # I get 5 digits at N = 110
    exact[110] = 0.007192320886184629
    exact[121] = 0.01703712555375862
    exact[150] = 0.005257679877042412
    exact[200] = 0.003934647991362962
    est = np.sum(w * x ** (N - 1))
    np.testing.assert_almost_equal(exact[N], est, 9)


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

def test_mapped_recursive_quad():

    mapped_ay = map_singular_pt(0.5, 1.0, 2.0)
    mapped_by = map_distance_to_interval(0.5, 1.0, 2.0)
    moments = modified_moments(rl1, 10, mapped_ay, mapped_by)
    x, w = recursive_quad(moments)
    x, w = map_pts_wts(x, w, 1.0, 2.0)
    w = map_weights_by_inv_power(w, 1.0, 1.0, 2.0)

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

def test_mapped_recursive_quad2():
    mapped_ay = map_singular_pt(0.2, 0.0, 1.0)
    mapped_by = map_distance_to_interval(0.3, 0.0, 1.0)
    moments = modified_moments(rl1, 10, mapped_ay, mapped_by)
    x, w = recursive_quad(moments)
    x, w = map_pts_wts(x, w, 0.0, 1.0)
    w = map_weights_by_inv_power(w, 1.0, 0.0, 1.0)

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
#
# def test_mapped_recursive_quad_with_m5():
#     x, w = map_rl_quad(recursive_quad, rl1, 10, 0.2, 0.3, 5.0, 0.0, 1.0)
#     exact = [143.2724199247024,
#              35.23159050009458,
#              12.03421465198119, \
#              5.150067213377777,
#              2.641021685444761,
#              1.566290120735974, \
#              1.039722729869763,
#              0.7510326686133133,
#              0.5773568932540537, \
#              0.4646714666702138]
#     est = [np.sum(w * x ** i) for i in range(len(exact))]
#     # print [e0 / e1 for e0, e1 in zip(est, exact)]
#     np.testing.assert_almost_equal(exact, est, 10)
