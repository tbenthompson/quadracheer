from math import sqrt, log
import numpy as np
import scipy.interpolate as spi
from quadracheer.gaussian_quad import gaussxw
from quadracheer.telles_quasi_singular import telles_quasi_singular

def test_interp_telles():
    # The problem I solve:
    # exact is from mathematica
    sing_pt = 1.005;
    denom = lambda x: (sing_pt - x) ** 2;
    numer = lambda x: x ** 3;
    f = lambda x: numer(x) / denom(x);
    exact_f = lambda s: (s * (-4 + 6 * s ** 2 - 3 * s * (-1 + s ** 2) * \
                                (log((-1 - s) / (1 - s))))) / (-1 + s ** 2)
    exact = exact_f(sing_pt)

    # Solved with standard Telles quadrature
    x_nearest = 1.0;
    D = sing_pt - 1.0;
    N = 14;
    [tx, tw] = telles_quasi_singular(N, x_nearest, D);
    est_telles = np.sum(f(tx) * tw)

    # Solved with gauss quadrature
    [gx, gw] = gaussxw(N)
    est_gauss = np.sum(f(gx) * gw)

    # Solved with interpolation and Telles quadrature
    # X = Interpolation Points
    X = gx;
    # Y = Value of function at the interpolation points
    Y = f(gx) * denom(gx);
    # WARNING, WARNING, WARNING: This implementation of lagrange interpolation
    # is super unstable. I just downloaded it from somewhere online. A
    # reimplementation using barycentric Lagrange interpolation is necessary to
    # go above N = appx 20.
    P = spi.lagrange(X, Y)

    est_interp_telles = sum(P(tx) / denom(tx) * tw)
    np.testing.assert_almost_equal(est_telles, est_interp_telles)
