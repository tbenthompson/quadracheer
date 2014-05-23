import numpy as np

def map_nonsing(quad_fnc, N, a, b):
    x, w = quad_fnc(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), \
           0.5 * (b - a) * w

def map_singular(quad_fnc, N, x0, a, b, **kwargs):
    moved_x0 = (2 * (x0 - a) / (b - a)) - 1
    x, w = quad_fnc(N, moved_x0, **kwargs)
    return 0.5 * (b - a) * x + 0.5 * (b + a), \
           0.5 * (b - a) * w

def map_rl_quad(quad_fnc, N, ay, by, a, b):
    moved_ay = (2 * (ay - a) / (b - a)) - 1
    moved_by = 2 * by / (b - a)
    x, w = quad_fnc(N, moved_ay, by = moved_by)
    # w *= np.sqrt((x - moved_ay) ** 2 + by ** 2)
    out_x = 0.5 * (b - a) * x + 0.5 * (b + a)
    out_w = w
    # out_w /= np.sqrt((out_x - ay) ** 2 + by ** 2)
    return out_x, out_w
