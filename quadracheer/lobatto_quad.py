from orthopoly import lobatto, rec_jacobi

def lobatto_quad(N):
    """
    Gauss-Lobatto quadrature points and weights on the interval [-1, 1]
    """
    alpha, beta = rec_jacobi(N, 0, 0)
    x, w = lobatto(alpha, beta, -1.0, 1.0)
    return x, w
