# I was just curious how well I could approximate the rl5 starting values
# with an interpolative scheme

from quadracheer.rl5 import *
import numpy as np
import matplotlib.pyplot as plt

def test_interpolate_rl5():
    fnc = lambda ay, by: mu_5_4(ay, by, k1(ay, by), k2(ay, by))

    ay = np.linspace(-2, 2, 100)
    by = np.linspace(1.0, 2.0, 100)
    Ay, By = np.meshgrid(ay, by)
    f = fnc(Ay, By)
    plt.imshow(f)
    plt.colorbar()
    plt.show()
