def map_nonsing(quad_fnc, N, a, b):
    x, w = quad_fnc(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), \
           0.5 * (b - a) * w

def map_singular(quad_fnc, N, x0, a, b, **kwargs):
    moved_x0 = (2 * (x0 - a) / (b - a)) - 1
    x, w = quad_fnc(N, moved_x0, **kwargs)
    return 0.5 * (b - a) * x + 0.5 * (b + a), \
           0.5 * (b - a) * w
