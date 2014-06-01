
# From pages 32 and 33 in Diligenti, Monegato 1997
def mu_log_0(y):
    return (1.0 + y) * log(1.0 + y) + (1.0 - y) * log(1.0 - y) - 2.0

def mu_log_1(y):
    return (1.0 - y ** 2) * log((1.0 - y) / (1.0 + y)) - 2 * y

def mu_log_2(y):
    return 2 * y * mu_log_1(y) + (8.0 / 3.0)

def multiplier(k, y):
    return (k + 1) / (2 * k ** 2 * (k + 2))

def alpha(k, y):
    return (2 * k + 1) * (2 * k)

def beta(k, y):
    return - k * (2 * k - 2)

log_x_minus_y = dict()
log_x_minus_y["recursion_multiplier"] = multiplier
log_x_minus_y["recursion_fncs"] = [alpha, beta]
log_x_minus_y[0] = mu_log_0
log_x_minus_y[1] = mu_log_1
log_x_minus_y[2] = mu_log_2
