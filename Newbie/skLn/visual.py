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
from skLn.tools import read_csv
from matplotlib import rc

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)

def visual_ln(B, X, H,sample, mp_s):
    #fig = plt.figure(2, figsize=(8.6, 7))
    plt.cla()
    #title_w = 'mks' + str(param)
    #fig.canvas.set_window_title(title_w)
    # fig.savefig('test2png.png', dpi=100)

    m = Basemap(llcrnrlat=mp_s[2], urcrnrlat=mp_s[3],
                llcrnrlon=mp_s[0], urcrnrlon=mp_s[1], resolution='l')
    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360., 2)
    m.drawmeridians(meridians, labels=[True, False, False, True])

    m.scatter(B[:, 0], B[:, 1], marker='o', s=20, c='r', linewidth=0, label='B [%i]'%len(B))
    try:
        m.scatter(X[:, 0], X[:, 1], marker='o', s=20, c='g', linewidth=0, label='X [%i]'%len(X))
        m.scatter(H[:, 0], H[:, 1], marker='s', s=6, c='k', linewidth=0, label='H [%i]'%len(H))
    except:
        pass

    m.scatter(sample[:, 0], sample[:, 1], marker='v', s=13, c='b', linewidth=0, label='M>=6')
    plt.grid(True)
    plt.legend(loc=1)
    title = 'ln_25'
    plt.title(title+'; count eq ln total:{}'.format(len(B)+len(X)))
    plt.savefig('/Users/Smoky/Documents/workspace/result/sk/'+title+'.png', dpi=500)

    #plt.show()

def sk_visual(A, B, C, D, H, sample, mp_s, r, title,legend, visual=True, TotalB=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    m = Basemap(llcrnrlat=mp_s[2], urcrnrlat=mp_s[3],
                llcrnrlon=mp_s[0], urcrnrlon=mp_s[1], resolution='i')
    m.drawcountries()
    m.drawcoastlines()
    #m.shadedrelief()
    #m.drawlsmask(land_color='coral', ocean_color='aqua', lakes=True)
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360., 2)
    m.drawmeridians(meridians, labels=[True, False, False, True])

    if TotalB:

        ln_array = []
        for l in [A, B, C, D]:
            try:
                for i in l:
                    ln_array.append(i)
            except:
                print('empty')
        ln_array = np.array(ln_array)


        try:
            for x, y, r in zip(ln_array[:, 0], ln_array[:, 1], [r for i in range(len(ln_array))]):
                Bobj = ax.add_artist(Circle(xy=(x, y),
                                               radius=r, linewidth=0.45, zorder=2, facecolor='#e1e1e1',
                                               edgecolor="k", label=u'Высокосейсмичные области'))
                ##e1e1e1 gray
        except:
            print('error')

        try:
            m.scatter(H[:, 0], H[:, 1], marker='.', s=12, c='k', linewidth=0.3, zorder=1)
        except:
            print('no data in H')

    else:
        try:
            for x, y, r in zip(A[:, 0], A[:, 1], [r for i in range(len(A))]):
                circleA = ax.add_artist(Circle(xy=(x, y),
                                               radius=r, alpha=0.5, linewidth=0.7, zorder=1, facecolor="none",
                                               edgecolor="k"))
            m.scatter(A[:, 0], A[:, 1], c='b', linewidth='0.5', label=legend[0], zorder=1)
        except:
            print('no data in A')

        try:
            for x, y, r in zip(B[:, 0], B[:, 1], [r for i in range(len(B))]):
                circleB = ax.add_artist(Circle(xy=(x, y),
                                               radius=r, alpha=0.5, linewidth=0.7, zorder=2, facecolor="none",
                                               edgecolor="k"))
            m.scatter(B[:, 0], B[:, 1], c='c', linewidth='0.5', label=legend[1], zorder=2)
        except:
            print('no data in B')

        try:
            for x, y, r in zip(C[:, 0], C[:, 1], [r for i in range(len(C))]):
                circleC = ax.add_artist(Circle(xy=(x, y),
                                               radius=r, alpha=0.5, linewidth=0.7, zorder=3, facecolor="none",
                                               edgecolor="k"))
            m.scatter(C[:, 0], C[:, 1], c='y', linewidth='0.5', label=legend[2], zorder=3)
        except:
            print('no data in C')

        try:
            for x, y, r in zip(D[:, 0], D[:, 1], [r for i in range(len(D))]):
                circleD = ax.add_artist(Circle(xy=(x, y),
                                               radius=r, alpha=0.5, linewidth=0.7, zorder=4, facecolor="none",
                                               edgecolor="k"))
            m.scatter(D[:, 0], D[:, 1], c='r', linewidth='0.5', label=legend[3], zorder=4)
        except:
            print('no data in D')

        try:
            m.scatter(H[:, 0], H[:, 1], marker='.', s=11, c='k', linewidth=0, zorder=1)
        except:
            print('no data in H')

    dps_data = read_csv('/Users/Smoky/Documents/workspace/resourses/csv/sk/kvzLN/kvz_dps.csv', ['x','y']).T
    m_three = m.scatter(dps_data[:, 0], dps_data[:, 1], marker='.', s=13, c='g', linewidth=0.0, zorder=5, alpha=0.6, label='Землетрясения 3.0 ≤ M')

    #m.scatter(sample[:, 0], sample[:, 1], marker='o', s=17, c='r', linewidth=0.1, zorder=5, label='M ≥ 6.0')
    check_eq= m.scatter(sample[:, 0], sample[:, 1], marker='o', s=17, c='r', linewidth=0.6, zorder=5,label='Землетрясения M ≥ 6.0')

    map_text = {'Черное море': [36.5, 42.6], 'Каспийское море': [48.45, 42.2], 'Баку': [49.5, 40.21],
                'Тбилиси': [44.48, 41.43],
                'Ереван': [44.31, 40.11], 'Владикавказ': [44.41, 43.01], 'Сочи': [39.43, 43.35]}


    for i, key in enumerate(map_text):
        t, xy = key, map_text[key]
        if t in ['Черное море', 'Каспийское море']:
            plt.annotate(u'{}'.format(t), (xy[0], xy[1]), fontsize=11)
        else:
            plt.annotate(u'{}'.format(t), (xy[0], xy[1]), fontsize=8)
            plt.scatter(xy[0], xy[1], marker='4', c='k', s=5)
    plt.title(title)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8, borderaxespad=0., handles=[Bobj, check_eq, m_three])
    plt.savefig('/Users/Smoky/Documents/workspace/result/skLn/'+title+'.png', dpi=500)
    if visual == True:
        plt.show()

def visual_data(data_hi, data_low, mp_s, title):
    m = Basemap(llcrnrlat=mp_s[2], urcrnrlat=mp_s[3],
                llcrnrlon=mp_s[0], urcrnrlon=mp_s[1], resolution='l')
    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360., 2)
    m.drawmeridians(meridians, labels=[True, False, False, True])

    plt.scatter(data_low[:, 0], data_low[:, 1], c ='b', marker='.', linewidths=0)
    plt.scatter(data_hi[:, 0], data_hi[:, 1], c='r', linewidths=0, marker='.')
    eq_6 = read_csv("/Users/Smoky/Documents/workspace/resourses/csv/sk/kvzLN/kvz_eqB.csv", ['x', 'y'])
    eq_55 = read_csv("/Users/Smoky/Documents/workspace/resourses/csv/sk/kvzLN/kvz_eqX.csv", ['x', 'y'])
    plt.scatter(eq_55[0], eq_55[1], marker='v', c='y', linewidths=0.3, s=8)
    plt.scatter(eq_6[0], eq_6[1], marker='v', c='w', linewidths=0.4, s=8)
    plt.title(title)
    #plt.savefig('/Users/Smoky/Documents/workspace/result/sk/'+title+'.png', dpi=500)
    plt.show()


def create_3d_map(data, title):
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.gca()
    x,y,z = data[:, 0], data[:, 1], data[:, 2]
    #m_r = [40., 54., 35., 47.]  # kvz
    m_r = [36, 52, 37, 46]  # kvzln
    m = Basemap(llcrnrlat=m_r[2], urcrnrlat=m_r[3],
                llcrnrlon=m_r[0], urcrnrlon=m_r[1],
                resolution='l', )
    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360, 2)
    m.drawmeridians(meridians, labels=[True, False, False, True])

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='nearest')  # create a uniform spaced grid
    xig, yig = np.meshgrid(xi, yi)
    #surf = ax.plot_wireframe(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1)  # 3d plot
    surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.coolwarm)
    #surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.cool)
    #surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.Spectral)
    #surf = ax.plot_surface(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1,cmap=cm.coolwarm)  # 3d plot
    eq_6 = read_csv("/Users/Smoky/Documents/workspace/resourses/csv/sk/kvzLN/kvz_eqB.csv", ['x', 'y'])
    eq_55 = read_csv("/Users/Smoky/Documents/workspace/resourses/csv/sk/kvzLN/kvz_eqX.csv", ['x', 'y'])
    plt.scatter(eq_55[0], eq_55[1], marker='v', c='k', linewidths=0, s=10)
    plt.scatter(eq_6[0], eq_6[1], marker='v', c='w', linewidths=0.4, s=10)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.scatter(eq_data[0], eq_data[1], marker='.', c='k', linewidths=0)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.savefig('/Users/Smoky/Documents/workspace/result/sk/'+title+'3D.png', dpi=500)

    plt.show()