from numpy import log
from quadracheer import gaussxw

def test_subtract_sing():
    f = lambda x: x**3 * log(x + 1)
    gauss_x, gauss_w = gaussxw(4)
    est = sum(f(gauss_x) * gauss_w)
    exact = 2.0 / 3.0
    error = abs(exact - est)

    f_singular_pt = lambda x: -1 * log(x + 1)
    f_minus_singularity = lambda x: f(x) - f_singular_pt(x)
    addme = 0.6137056388801094
    est2 = addme + sum(f_minus_singularity(gauss_x) * gauss_w)
    error2 = abs(exact - est2)
    assert(error2 <= error / 5)
