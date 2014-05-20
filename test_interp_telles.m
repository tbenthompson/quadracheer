% The problem I solve:
% exact is from mathematica
sing_pt = 1.005;
denom = @(x) (sing_pt - x) .^ 2;
numer = @(x) x.^3;
f = @(x) numer(x) ./ denom(x);
exact_f = @(s) 3/2 + 1/(-1 + s) + 3 * s + 3 * s ^ 2 * (log(1 - s) - log(-s));
exact = exact_f(sing_pt)

% Solved with standard Telles quadrature
x_nearest = 1.0;
D = sing_pt - 1.0;
N = 10;
[tx, tw] = telles_quasi_singular(N, x_nearest, D);
est_telles = sum(f(tx) .* tw)

% Solved with gauss quadrature
[gx, gw] = lgwt(N, 0.0, 1.0);
est_gauss = sum(f(gx) .* gw)

% Solved with interpolation and Telles quadrature
% X = Interpolation Points
X = gx;
% Y = Value of function at the interpolation points
Y = f(gx) .* denom(gx);
% WARNING, WARNING, WARNING: This implementation of lagrange interpolation
% is super unstable. I just downloaded it from somewhere online. A
% reimplementation using barycentric Lagrange interpolation is necessary to
% go above N = appx 20.
P = lagrangepoly(X,Y);

est_interp_telles = sum(polyval(P, tx) ./ denom(tx) .* tw)
