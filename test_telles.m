% test lgwt
[x, w] = lgwt(3, -1.0, 1.0);
% Exact values retrieved from the wikipedia page on Gaussian Quadrature
% The main function has been tested by the original author for a wide
% range of orders. But, this is just to check everything is still working
% properly
assert(x(1) == sqrt(3.0 / 5.0));
assert(x(2) == 0.0);
assert(x(3) == -sqrt(3.0 / 5.0));
assert(abs(w(1) - (5.0 / 9.0)) < 0.000001);
assert(abs(w(2) - (8.0 / 9.0)) < 0.000001);
assert(abs(w(3) - (5.0 / 9.0)) < 0.000001);

%test telles quadrature with the example from the paper (page 964)
g = @(x) log(abs(0.3 + x));
f = @(y) 2 * g(2 * y - 1);
exact = -1.908598917;
% compare with the result Telles gets for 10 points to make sure
% the method is properly implemented.
telles_paper_10_pts = -1.90328;
[tx, tw] = telles_singular(10, 0.35);
est = sum(f(tx) .* tw);
assert(abs(telles_paper_10_pts - est) < 0.00001);

%test telles quasi-singular quadrature with the example on page 966
% sing_pt = 1.08
sing_pt = 1.004;
g = @(x) (sing_pt - x) .^ -2;
f = @(y) 2 * g(2 * y - 1);

% for sing_pt = 1.08
% exact = 12.0192;
% for sing_pt = 1.004
exact = 249.500998;
x_nearest = 1.0;
D = sing_pt - 1.0;
N = 20;
[tx, tw] = telles_quasi_singular(N, x_nearest, D);
est_telles = sum(f(tx) .* tw);
[gx, gw] = lgwt(N, 0.0, 1.0);
est_gauss = sum(f(gx) .* gw);

gauss_error = abs(est_gauss - exact) / exact * 100
telles_error = abs(est_telles - exact) / exact * 100


