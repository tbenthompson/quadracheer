import numpy as np
from quadracheer.legendre import alpha, legendre_integrals

def test_alpha():
    assert(alpha(0) == 1.0 / 1.0)
    assert(alpha(1) == 3.0 / 2.0)
    assert(alpha(2) == 5.0 / 3.0)
    assert(alpha(3) == 7.0 / 4.0)

def test_legendre_integrals():
    abc = legendre_integrals(5)
    np.testing.assert_almost_equal(abc, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
