import matplotlib
matplotlib.use('Qt4Agg')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=50, cstride=50)

plt.show()

