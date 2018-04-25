import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches
import numpy as np

def visual_circle(X, B, eqs, title):
    """отображение результатов в виде кругов"""
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    pol = [36, 52, 37, 46]
    m = Basemap(llcrnrlat=pol[2], urcrnrlat=pol[3],
                llcrnrlon=pol[0], urcrnrlon=pol[1],
                resolution='h')
    # m.fillcontinents(color='white', lake_color='aqua',zorder=0, alpha=.5)
    m.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True, zorder=0)

    m.drawcountries(zorder=1, linewidth=0.6)
    m.drawcoastlines(zorder=1, linewidth=0.6)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False], zorder=1, linewidth=0.4)
    meridians = np.arange(0., 360, 4)
    m.drawmeridians(meridians, labels=[True, False, False, True], zorder=1, linewidth=0.4)


    plt.scatter(X[:, 0], X[:, 1], c='m', marker='.', lw=0, zorder=3, s=13)

    for x, y, r in zip(B[:, 0], B[:, 1], [0.225 for i in range(len(B))]):
        color = 'g'

        # if coord_in_sample((x, y), self.sample_coord): color = 'g'

        # круги без проекции
        # circle_B = ax.add_artist( Circle(xy=(x, y),radius=r, alpha=0.9, linewidth=0.75, zorder=2, facecolor=color, edgecolor="k"))

        # эллипсы с проекцией
        m.tissot(x, y, r, 50, alpha=0.9, linewidth=0.75, zorder=2, facecolor=color, edgecolor="k")

        plt.scatter(x, y, c='b', marker='.', lw=0, zorder=4, s=15)

    # исторические - треугольник инструментальные - круг
    plt.scatter(eqs[0][:, 0], eqs[0][:, 1], c='r', marker='^', linewidths=0.45, zorder=4, s=17, alpha=0.8)
    plt.scatter(eqs[1][:, 0], eqs[1][:, 1], c='r', marker='o', linewidths=0.45, zorder=4, s=17, alpha=0.8)

    # все - круги
    # plt.scatter(self.eq_all[:, 0], self.eq_all[:, 1], c='r', marker='o', linewidths=0.45, zorder=4, s=20, alpha=0.8)


    plt.title(title)
    plt.show()