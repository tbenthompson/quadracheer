from math import log
# From pages 32 and 33 in Diligenti, Monegato 1997
def mu_log_0(y):
    return (1.0 + y) * log(1.0 + y) + (1.0 - y) * log(1.0 - y) - 2.0

def mu_log_1(y):
    return 0.5 * (1.0 - y ** 2) * log((1.0 - y) / (1.0 + y)) - y

def mu_log_2(y):
    return 1.0 * y * mu_log_1(y) + (2.0 / 3.0)

def multiplier(k, y):
    value = (k + 1.0) / (2.0 * (k + 1.0) * k * (k + 2.0))
    return value

def alpha(k, y):
    return (2.0 * k + 1.0) * (2.0 * k) * y

def beta(k, y):
    return -k * (2.0 * k - 2.0)

log_x_minus_y = dict()
log_x_minus_y["recursion_multiplier"] = multiplier
log_x_minus_y["recursion_fncs"] = [alpha, beta]
log_x_minus_y[0] = mu_log_0
log_x_minus_y[1] = mu_log_1
log_x_minus_y[2] = mu_log_2

def mu_log_r_0(ay, by):
    term1 = (1.0 - ay) * log((1.0 - ay) ** 2 + by ** 2)
    term1 = (1.0 + ay) * log((1.0 + ay) ** 2 + by ** 2)
    term3 = 2.0 * by * (arctan((1.0 + ay) / by) + arctan((1.0 - ay) / by))
    term4 = -4.0
    return term1 + term2 + term3 + term4
