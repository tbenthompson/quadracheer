from numpy import log
from quadracheer import gaussxw, telles_singular
from scipy.special import eval_legendre

def test_subtract_sing():
    degree = 20
    f = lambda x: eval_legendre(degree, x) * log(x + 1)
    almost_exact_x, almost_exact_w = telles_singular(51, -1)
    exact = sum(f(almost_exact_x) * almost_exact_w)
    # print exact

    gauss_x, gauss_w = gaussxw(degree + 1)
    est = sum(f(gauss_x) * gauss_w)
    error = abs(exact - est)

    f_singular_pt = lambda x: (-1.0) ** degree * log(x + 1)
    f_minus_singularity = lambda x: f(x) - f_singular_pt(x)
    addme = 0.6137056388801094 * (-1.0) ** (degree + 1)
    est2 = addme + sum(f_minus_singularity(gauss_x) * gauss_w)
    error2 = abs(exact - est2)

    # print error / exact
    # print error2 / exact

    # subtracting out the singularity is super ineffective on this problem...
    assert(abs(error2 / exact) < 0.3)
