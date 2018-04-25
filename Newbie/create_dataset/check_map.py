from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.interpolate import griddata
from matplotlib import rc

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)


def read_csv(path, param):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")

    for i, title in enumerate(param):
        try:
            array.append(frame[title].values)
        except:
            print(title, 'error')

    return np.array(array)


def create_grap(data):
    eff_total = data[0]
    eff_count = data[1]
    title = data[2]
    x_title = np.array(range(0, len(title)))
    # plt.hist(eff_total,x_title)
    # plt.scatter(e5ff_count[np.where(eff_count>55)], eff_total[np.where(eff_total>150)], linewidths=0, c='b', s=4)
    plt.scatter(eff_count, eff_total, linewidths=0, c='b', s=4)
    # plt.xticks(x_title, title, rotation='vertical')
    for i, xy in enumerate(zip(eff_count, eff_total)):
        # plt.text(xy[0],xy[1],title[i],fontsize=2)
        # if xy[0]>55 and xy[1]>150:
        plt.annotate(title[i], xy=xy, fontsize=2, xytext=(xy[0] + 0.1, xy[1]))
    plt.grid(True)
    plt.xlabel('eff count')
    plt.ylabel('total point')
    plt.savefig('/Users/Ivan/Documents/workspace/result/epsilon/eff.png', dpi=500)
    plt.show()


# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
# 40,54,35,47
def create_map(data, title):

    x = data[0]
    y = data[1]

    plt.scatter(x, y, marker='.', c='g', linewidths=0.1)
    # m.scatter(47.57, 52.05, marker='p', c='y', s=100)

    # check = read_csv("/Users/Smoky/Documents/workspace/resourses/csv/balac.csv")

    # m.scatter(check[0], check[1], marker='.', c='w')

    # kvz [40,54,35,47]
    # balac [44, 53, 48, 54]
    # crimea 29., 40., 41., 49.
    # calif -127, -113, 30, 42

    lnm = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/newEPA/Caucasus_coord.csv', ['x', 'y']).T
    plt.scatter(lnm[:, 0], lnm[:, 1], c='r')

    map_size = [38, 54, 35, 47]
    # m.drawrivers()
    # m.bluemarble()
    m = Basemap(llcrnrlat=map_size[2], urcrnrlat=map_size[3],
                llcrnrlon=map_size[0], urcrnrlon=map_size[1], resolution='l')
    m.drawcountries()
    m.drawcoastlines()
    map_text = {'Черное море': [36.5, 42.6], 'Каспийское море': [47.7, 43.4], 'Баку': [49.5, 40.21],
                'Тбилиси': [44.48, 41.43],
                'Ереван': [44.31, 40.11], 'Владикавказ': [44.41, 43.01], 'Сочи': [39.43, 43.35]}

    for i, key in enumerate(map_text):
        t, xy = key, map_text[key]
        if t in ['Черное море', 'Каспийское море']:
            plt.annotate(u'{}'.format(t), (xy[0], xy[1]), fontsize=11)
        else:
            plt.annotate(u'{}'.format(t), (xy[0], xy[1]), fontsize=10)
            plt.scatter(xy[0], xy[1], marker='o', c='k', s=5)



    plt.grid(True)

    # m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)
    #plt.savefig('/Users/Ivan/Documents/workspace/result/sk/kvz_rlf.png', dpi=400)

    plt.show()


def create_3d_map(data, title):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.gca()
    x, y, z = data[0], data[1], data[2]
    # ax.plot_wireframe(X, Y, Z, rstride=1000, cstride=1000)
    # m_r = [40., 54., 35., 47.]  # kvz
    m_r = [36, 52, 37, 46]  # kvzln
    m = Basemap(llcrnrlat=m_r[2], urcrnrlat=m_r[3],
                llcrnrlon=m_r[0], urcrnrlon=m_r[1],
                resolution='l', )
    # draw a land-sea mask for a map background.
    # lakes=True means plot inland lakes with ocean color.


    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360, 2)
    m.drawmeridians(meridians, labels=[True, False, False, True])

    # ax.plot_surface(X, Y, Z)
    # ax.scatter(X,Y,Z,marker='.')
    # ax.plot_surface(X, Y, Z, rstride=1000, cstride=1000)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='nearest')  # create a uniform spaced grid
    xig, yig = np.meshgrid(xi, yi)
    # surf = ax.plot_wireframe(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1)  # 3d plot
    surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.coolwarm)
    # surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.Spectral)
    # surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.cool)
    # surf = ax.plot_surface(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1,cmap=cm.coolwarm)  # 3d plot

    fig.colorbar(surf, shrink=0.5, aspect=5)
    eq_6 = read_csv("/Users/Ivan/Documents/workspace/resourses/csv/sk/kvzLN/kvz_eqB.csv", ['x', 'y'])
    eq_55 = read_csv("/Users/Ivan/Documents/workspace/resourses/csv/sk/kvzLN/kvz_eqX.csv", ['x', 'y'])
    plt.scatter(eq_55[0], eq_55[1], marker='v', c='k', linewidths=0)
    plt.scatter(eq_6[0], eq_6[1], marker='v', c='w', linewidths=0.4)

    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.savefig('/Users/Ivan/Documents/workspace/result/sk/kvz_' + title + '.png', dpi=400)
    print('done')

    plt.show()


title = 'kvz_reliefS4'
data = read_csv("/Users/Ivan/Documents/workspace/resourses/csv/newEPA/" + title + ".csv", ['x', 'y'])
# data = read_csv('/Users/Smoky/Documents/workspace/resourses/csv/geop/crimea/crimea_'+title+'.csv')
create_map(data, title)
# create_3d_map(data, 'reliefAll')
# create_grap(data)
