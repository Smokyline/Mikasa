import matplotlib

#matplotlib.use('TkAgg')
matplotlib.use('Qt4Agg')
#matplotlib.use('GTK3Cairo')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from matplotlib import cm

def visual_two_predict(X_test, X_train, sampleAll, mp_s, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    m = Basemap(llcrnrlat=mp_s[2], urcrnrlat=mp_s[3],
                llcrnrlon=mp_s[0], urcrnrlon=mp_s[1], resolution='l')
    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360., 2)
    m.drawmeridians(meridians, labels=[True, False, False, True])

    #ax.axis([mp_s[0] - 1., mp_s[1] + 1., mp_s[2] - 1., mp_s[3] + 1.])
    #m.scatter(B[:, 0], B[:, 1], marker='o', s=45, c='b', linewidth=0.6)

    m.scatter(X_test[:, 0], X_test[:, 1], c='b', marker='o', linewidth=0.0, zorder=1, alpha=1, label='B')
    try:
        m.scatter(X_train[:, 0], X_train[:, 1], c='g', marker='o', linewidth=0.0, zorder=2, alpha=1, label='train')
    except:
        pass
    m.scatter(sampleAll[:, 0], sampleAll[:, 1], marker='v', s=14, c='r', linewidth=0.1, zorder=3, label='eq M>5.4')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc=1)
    plt.savefig('/Users/Smoky/Documents/workspace/result/skMLA/'+title+'_10.png', dpi=500)
    plt.show()

def create_3d_map(data, title):
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.gca()
    x,y,z = data[:, 0], data[:, 1], data[:, 3]
    #ax.plot_wireframe(X, Y, Z, rstride=1000, cstride=1000)

    #ax.plot_surface(X, Y, Z)
    #ax.scatter(X,Y,Z,marker='.')
    #ax.plot_surface(X, Y, Z, rstride=1000, cstride=1000)

    xi = np.linspace(x.min(), x.max(), 10)
    yi = np.linspace(y.min(), y.max(), 10)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='nearest')  # create a uniform spaced grid
    xig, yig = np.meshgrid(xi, yi)
    #surf = ax.plot_wireframe(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1)  # 3d plot
    #surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.coolwarm)
    surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.cool)
    #surf = ax.plot_surface(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1,cmap=cm.coolwarm)  # 3d plot

    fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.scatter(eq_data[0], eq_data[1], marker='.', c='k', linewidths=0)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    #plt.savefig('/Users/Smoky/Documents/workspace/result/univ/crimea_'+title+'.png', dpi=400)

    plt.show()