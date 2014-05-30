# I was just curious how well I could approximate the rl5 starting values
# with an interpolative scheme

from quadracheer.rl5 import *
import numpy as np
import matplotlib.pyplot as plt

fnc = lambda ay, by: mu_5_4(ay, by, k1(ay, by), k2(ay, by))

ay = np.linspace(-2, 2, 100)
by = np.linspace(-2.0, 2.0, 100)
Ay, By = np.meshgrid(ay, by)
f = np.log(np.abs(fnc(Ay, By)))
plt.imshow(f)
plt.title(r'$\log(|\mu_4^5(a_y, b_y)|)$')
plt.xlabel(r'$a_y$')
plt.ylabel(r'$b_y$')
plt.colorbar()
plt.show()
