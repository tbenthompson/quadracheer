% test lgwt
[x, w] = gaussian_quad.gaussxwab(3, -1.0, 1.0)
% Exact values retrieved from the wikipedia page on Gaussian Quadrature
% The main function has been tested by the original author for a wide
% range of orders. But, this is just to check everything is still working
% properly
assert(x[1], sqrt(3.0 / 5.0))
assert(x[2], 0.0)
assert(x[3], -sqrt(3.0 / 5.0))
assert(w[1], 5.0 / 9.0)
assert(w[2], 8.0 / 9.0)
assert(w[3], 5.0 / 9.0)
