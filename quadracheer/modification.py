from math import log
from recursive_legendre import alpha, beta


def modify_times_x_minus_a(k_max, moments, a):
    """
    Create the modified moments for k(x,y) * (x - a) * Pn from the
    modified moments for k(x, y) * Pn
    See (i) on page 34 of Diligenti, Monegato 1997
    I assume use of the Legendre Polynomials
    """
    new_list = []
    new_list.append(moments[1] - a * moments[0])
    for k in range(1, k_max + 1):
        term1 = (1.0 / alpha(k)) * moments[k + 1]
        term2 = -a * moments[k]
        term3 = beta(k) / alpha(k) * moments[k - 1]
        new_moment = term1 + term2 + term3
        new_list.append(new_moment)
    return new_list

def modify_divide_x_minus_a(k_max, moments, a, first_term):
    """
    Create the modified moments for k(x,y) * Pn / (x - a) from the
    modified moments for k(x,y) * Pn
    The first term depends on k(x,y) and thus, the correct value
    must be provided
    See (ii) on pages 34 and 35 of Diligenti, Monegato 1997
    I assume use of the Legendre polynomials
    """
    new_list = []

    new_list.append(first_term)
    new_list.append(a * new_list[0] + moments[0])
    for k in range(1, k_max):
        term1 = alpha(k) * a * new_list[k]
        term2 = -beta(k) * new_list[k - 1]
        term3 = alpha(k) * moments[k]
        new_moment = term1 + term2 + term3
        new_list.append(new_moment)
    return new_list

def initial_q_r(a, new_list):
    """Helper method for modify_divide_r"""
    # Compute q_0^R and q_1^R explicitly
    q_r = []
    q_r.append(1 / alpha(0) * new_list[1] - a * new_list[0])
    q_r.append(1 / alpha(1) * new_list[2] -
               a * new_list[1] +
               beta(1) / alpha(1) * new_list[0])
    return q_r

def initial_q_i(b, new_list):
    """Helper method for modify_divide_r"""
    # Compute q_0^I and q_1^I explicitly
    q_i = []
    q_i.append(b * new_list[0])
    q_i.append(b * new_list[1])
    return q_i

def next_q_r(k, a, b, moments, q_i, q_r):
    """Helper method for modify_divide_r"""
    term1 = alpha(k) * a * q_r[k]
    term2 = -alpha(k) * b * q_i[k]
    term3 = -beta(k) * q_r[k - 1]
    term4 = alpha(k) * moments[k]
    return term1 + term2 + term3 + term4

def next_q_i(k, a, b, q_i, q_r):
    """Helper method for modify_divide_r"""
    term1 = alpha(k) * a * q_i[k]
    term2 = alpha(k) * b * q_r[k]
    term3 = -beta(k) * q_i[k - 1]
    return term1 + term2 + term3

def modify_divide_r2(k_max, moments, a, b, first_term, second_term):
    """
    Create the modified moments for k(x,y) * Pn / ((x - a)^2 + b^2) from
    the modified moments for k(x, y) * Pn
    The first two terms must be computed explicitly from the
    expression for the kernel, k(x, y).
    See (iii) on page 35 of Diligenti, Monegato 1997
    I assume use of the Legendre Polynomials
    """
    new_list = []

    # The first two terms are computed explicitly
    new_list.append(first_term)
    new_list.append(second_term)

    # Compute m_2 explicitly
    term1 = alpha(1) * (moments[0] + 2 * a * new_list[1])
    term2 = -(alpha(1) * (a ** 2 + b ** 2) + beta(1)) * new_list[0]
    m_2 = term1 + term2
    new_list.append(m_2)

    # Compute q_r and q_i (the real and imaginary parts of the helper
    # integral)
    q_r = initial_q_r(a, new_list)
    q_i = initial_q_i(b, new_list)
    for k in range(1, k_max):
        q_r.append(next_q_r(k, a, b, moments, q_i, q_r))
        q_i.append(next_q_i(k, a, b, q_i, q_r))

    # Compute m_j through the relation to q_j^I
    for k in range(3, k_max + 1):
        new_list.append(q_i[k] / b)

    return new_list
