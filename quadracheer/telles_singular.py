import numpy as np
from gaussian_quad import gaussxw

def telles_singular(N, x0):
    """
    Use a cubic polynomial transformation to turn a 1D log singular
    integral into an integrable form.
    This should also be able to accurately integrate terms like
    log(|r|) where r = (x - y).

    This should also be able to accurately integrate terms like 1/r if
    the singularity is outside, but near the domain of integration.

    See
    "A SELF-ADAPTIVE CO-ORDINATE TRANSFORMATION FOR EFFICIENT NUMERICAL
    EVALUATION OF GENERAL BOUNDARY ELEMENT INTEGRALS", Telles, 1987.
    for a description of the method. I use the same notation adopted in that
    paper. The interval of integration is [-1, 1]

    Note there is a printing error in the Jacobian of the transformation
    from gamma coordinates to eta coordinates in the Telles paper. The
    formula in the paper is
    (3 * (gamma - gamma_bar ** 2)) / (1 + 3 * gamma_bar ** 2)
    It SHOULD read:
    (3 * (gamma - gamma_bar) ** 2) / (1 + 3 * gamma_bar ** 2)
    """
    eta_bar = x0
    eta_star = eta_bar ** 2 - 1.0

    # The location of the singularity in gamma space
    term1 = (eta_bar * eta_star + np.abs(eta_star))
    term2 = (eta_bar * eta_star - np.abs(eta_star))

    # Fractional powers of negative numbers are multiply valued and python
    # recognizes this. So, I specify that I want the real valued third root
    gamma_bar = np.sign(term1) * np.abs(term1) ** (1.0 / 3.0) + \
            np.sign(term2) * np.abs(term2) ** (1.0 / 3.0) + \
            eta_bar

    gamma, gamma_weights = gaussxw(N)
    x = ((gamma - gamma_bar) ** 3 + gamma_bar * (gamma_bar ** 2 + 3))\
            / (1 + 3 * gamma_bar ** 2)

    w = gamma_weights * (3 * (gamma - gamma_bar) ** 2) \
            / (1 + 3 * gamma_bar ** 2)

    # If I accidentally choose a Gaussian integration scheme that
    # exactly samples the singularity, this method will fail. This can
    # be easily remedied by simply increasing the order of the method.
    # For example, this happens if x0 == 0 and N is odd. Just use an even
    # order in that case.
    if (np.abs(x - x0) < 1e-12).any():
        raise Exception("Telles integration has sampled the " +
                "singularity. Choose a different order of integration.")

    return x, w
