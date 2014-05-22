import numpy as np
from gaussian_quad import gaussxw, gaussxwab

def piessens(N, x0, nonsingular_N = -1):
    """
    Quadrature points and weights for integrating a function with form
    f(x) / (x - x0)
    on the interval [0, 1]
    Uses the 2N point gauss rule derived in Piessens (1970) Almost certainly
    suboptimal, but it's very simple and it works. Exact for polynomials of
    order 4N.
    """
    if nonsingular_N == -1:
        nonsingular_N = N
    nonsingular_N = nonsingular_N
    N = N
    x0 = x0

    # Split the interval into two sections. One is properly integrable.
    # The other is symmetric about the singularity point and must be
    # computed using as a cauchy principal value integral.
    if x0 < 0.5:
        proper_length = 1 - (2 * x0)
        pv_length = 2 * x0
        pv_start = 0.0
        proper_start = 2 * x0
    else:
        proper_start = 0.0
        pv_start = -1.0 + 2 * x0
        proper_length = -1.0 + 2 * x0
        pv_length = 2.0 - 2 * x0

    # Just check...
    assert(pv_length + proper_length == 1.0)
    assert(pv_start + pv_length == proper_start
        or pv_start + pv_length == 1.0)
    assert(proper_start + proper_length == pv_start
        or proper_start + proper_length == 1.0)
    assert(pv_start + pv_length / 2.0 == x0)


    # the interval without the singularity
    x, w = gaussxwab(nonsingular_N, proper_start, proper_start + proper_length)

    # Get the points for the singular part using Piessen's method
    # Change the code to use Longman's method, but Piessen's is clearly
    # superior.
    x_sing, w_sing = piessen_method(N, pv_start, pv_length, x0)

    # Finished!
    x = np.append(x, x_sing)
    w = np.append(w, w_sing)
    return x,w

def piessen_method(N, pv_start, pv_length, x0, add_singularity = True):
    x_base, w_base = piessen_neg_one_to_one_nodes(N)
    # Convert to the interval from [pv_start, pv_start + pv_length]
    x = (pv_length / 2) * x_base + \
            (2 * pv_start + pv_length) / 2.0
    # No need to scale the weights because the linear factor in the 1/r
    # exactly cancels the jacobian.
    w = w_base

    # If we don't factor out the 1 / (x - x0) of the quadratured function,
    # so we must account for it here.
    if add_singularity:
        w *= x - x0
    return x, w

def piessen_neg_one_to_one_nodes(N):
    """Piessen nodes and weights for [-1, 1]"""
    if N % 2 == 1:
        raise Exception("Piessens method requires an even quadrature " +
                "order")

    gx, gw = gaussxw(2 * N)
    x = gx
    w = gw / gx
    return x, w
