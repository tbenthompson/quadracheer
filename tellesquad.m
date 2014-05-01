function [x, w] = tellesquad(N, x0)
    % Use a cubic polynomial transformation to turn a 1D log singular
    % integral into an integrable form.
    % This should also be able to accurately integrate terms like
    % log(|r|) where r = (x - y).

    % This should also be able to accurately integrate terms like 1/r if
    % the singularity is outside, but near the domain of integration.

    % See
    % "A SELF-ADAPTIVE CO-ORDINATE TRANSFORMATION FOR EFFICIENT NUMERICAL
    % EVALUATION OF GENERAL BOUNDARY ELEMENT INTEGRALS", Telles, 1987.
    % for a description of the method. I use the same notation adopted in that
    % paper. Because the reference segment is [0, 1] here and the reference
    % segment is [-1, 1] in the Telles paper, a small extra transformation
    % is performed.

    % Note there is a printing error in the Jacobian of the transformation
    % from gamma coordinates to eta coordinates in the Telles paper. The
    % formula in the paper is
    % (3 * (gamma - gamma_bar ** 2)) / (1 + 3 * gamma_bar ** 2)
    % It SHOULD read:
    % (3 * (gamma - gamma_bar) ** 2) / (1 + 3 * gamma_bar ** 2)

    % TODO: Implement the near-by singularity method from the Telles paper.

    % The location of the singularity in eta space
    eta_bar = 2 * x0 - 1.0;

    eta_star = eta_bar ^ 2 - 1.0;

    % The location of the singularity in gamma space
    term1 = (eta_bar * eta_star + abs(eta_star));
    term2 = (eta_bar * eta_star - abs(eta_star));

    % Fractional powers of negative numbers are multiply valued and python
    % recognizes this. So, I specify that I want the real valued third root
    gamma_bar = sign(term1) * abs(term1) ^ (1.0 / 3.0) + ...
                sign(term2) * abs(term2) ^ (1.0 / 3.0) + eta_bar;

    [x_gamma, w_gamma] = lgwt(N, -1.0, 1.0);
    x = ((x_gamma - gamma_bar) .^ 3 + gamma_bar * (gamma_bar ^ 2 + 3))...
            / (2 * (1 + 3 * gamma_bar ^ 2)) + 0.5;

    w = w_gamma .* (3 .* (x_gamma - gamma_bar) .^ 2) ...
            / (2 * (1 + 3 * gamma_bar ^ 2));

    % If we accidentally choose a Gaussian integration scheme that
    % exactly samples the singularity, this method will fail. This can
    % be easily remedied by simply increasing the order of
    % For example, this happens if x0 == 0 and N is odd.
    % Instead of throwing an error, the code could be modified to just 
    % increase the accuracy by one point and restart recursively
    if abs(x - x0) < 1e4 * eps
        error('Telles integration has sampled the singularity. Choose a different number of points.');
    end
end
