function [x, w] = telles_quasi_singular(N, x_nearest, D)
    % Inputs:
    % N = the number of quadrature points to produce
    % x_nearest = the location of the nearest point  to the singularity 
    %             within the domain of integration.
    % D = the distance to the singularity.
    %
    % See
    % "A SELF-ADAPTIVE CO-ORDINATE TRANSFORMATION FOR EFFICIENT NUMERICAL
    % EVALUATION OF GENERAL BOUNDARY ELEMENT INTEGRALS", Telles, 1987.
    % for a description of this method. I use the same notation adopted in that
    % paper. Because the reference segment is [0, 1] here and the reference
    % segment is [-1, 1] in the Telles paper, a small extra transformation
    % is performed. 
    % This quasi-singular quadrature method is described on page 9 of that 
    % paper and is based on a cubic polynomial transformation designed to
    % reduce the degree of singularity and allow standard Gaussian quadrature
    % method to accurately integrate an arbitrary function.

    % The value of r_bar depends on the distance to the singularity. The value
    % should increase further from the singularity.

    % TODO: Think about deriving a theoretical value of r_bar as a function
    % of D

    % The location of the singularity in eta space
    eta_bar = 2 * x_nearest - 1.0;

    D_eta = D * 2;
    % If we are less than 0.05 from the singularity, just use the singular
    % telles quadrature method.
    if D < 0.05
        [x, w] = telles_singular(N, x_nearest);
        return;
    end

    % This empirical formula was determined in the paper for singularities
    % like 1/r or 1/r^2. I should redo this for 1/r^3?
    if D < 1.3
        r_bar = 0.85 + 0.24 * log(D_eta);
    elseif D < 3.618
        r_bar = 0.893 + 0.0832 * log(D_eta);
    else
        % Far enough away that we should just use gauss quadrature.
        [x, w] = lgwt(N, 0.0, 1.0);
        return;
    end

    % The following is an almost direct transcription of the equation in
    % the article cited above.
    q_factor = (1.0 / (2 * (1 + 2 * r_bar)));
    q_term1 = ((eta_bar * (3 - 2 * r_bar)) - ...
              ((2 * eta_bar ^ 3) / (1 + 2 * r_bar))) * ...
              (1.0 / (1 + 2 * r_bar));
    q = q_factor * (q_term1 - eta_bar);

    p_factor = 1.0 / (3 * (1 + 2 * r_bar) ^ 2);
    p_term1 = 4 * r_bar * (1 - r_bar);
    p_term2 = 3 * (1 - eta_bar ^ 2);
    p = p_factor * (p_term1 + p_term2);

    arg_minus = -q + sqrt(q ^ 2 + p ^ 3);
    arg_plus = -q - sqrt(q ^ 2 + p ^ 3);
    gamma_bar = sign(arg_plus) * abs(arg_plus) ^ (1.0 / 3.0) + ...
                sign(arg_minus) * abs(arg_minus) ^ (1.0 / 3.0) + ...
                (eta_bar / (1 + 2 * r_bar));

    Q = 1 + 3 * gamma_bar ^ 2;
    a = (1 - r_bar) / Q;
    b = -3 * (1 - r_bar) * gamma_bar / Q;
    c = (r_bar + 3 * gamma_bar ^ 2) / Q;
    d = -b;

    % The Gaussian points and weights in gamma space. 
    [x_gamma, w_gamma] = lgwt(N, -1.0, 1.0);

    % The cubic transformation back into the original [-1, 1]
    eta = ((x_gamma * a + b) .* x_gamma + c) .* x_gamma + d;
    w_eta = w_gamma .* (((3 * a * x_gamma + 2 * b) .* x_gamma) + c);

    % Now transform back to [0, 1]
    x = eta / 2.0 + 0.5;
    w = w_eta / 2.0;
