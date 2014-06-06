import numpy as np

def map_pts_wts(x, w, a, b):
    return 0.5 * (b - a) * x + 0.5 * (b + a), \
           0.5 * (b - a) * w

def map_singular_pt(x0, a, b):
    return (2 * (x0 - a) / (b - a)) - 1

def map_distance_to_interval(by, a, b):
    """
    Used for mapping the by in the RL quadrature rules from
    physical to reference space.
    """
    return 2 * by / (b - a)

def map_weights_by_inv_power(w, m, a, b):
    return w / ((0.5 * (b - a)) ** m)
