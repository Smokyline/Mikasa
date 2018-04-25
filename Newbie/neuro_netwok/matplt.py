import matplotlib

#matplotlib.use('TkAgg')
matplotlib.use('Qt4Agg')
#matplotlib.use('GTK3Cairo')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from matplotlib import cm




#plt.ion()
def draw_plot():
    # plt.grid(True)

    print('map created')
    plt.show()

def visual_clust(clusters, check_data, param, mp_s,centroinds, k):
    fig = plt.figure(2, figsize=(8.6, 7))
    plt.cla()
    title_w = 'mks'+str(param)
    fig.canvas.set_window_title(title_w)
    #fig.savefig('test2png.png', dpi=100)

    m = Basemap(llcrnrlat=mp_s[2], urcrnrlat=mp_s[3],
                llcrnrlon=mp_s[0], urcrnrlon=mp_s[1],resolution='l')
    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2.5)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360., 2.5)
    m.drawmeridians(meridians, labels=[True, False, False, True])
    #m.shadedrelief()

    color_a = ['b', 'c',  'g', 'y', 'r']
    if k == 7:
        color_a = ['#b0e0e6', '#319696', '#003366', '#008000', '#ffff00','#FFA500', '#ff0000']
        #color_a = ['#660066', '#950096', '#8a2be2', '#bf1796','#cc641e', '#dd9f2a', '#eed543']
        #color_a = ['#eed543', '#dd9f2a', '#cc641e', '#bf1796','#8a2be2', '#950096', '#660066']
    for i, cluster in enumerate(clusters):

        clust = np.array(cluster).T
        label = None
        try:
            label = str(round(centroinds[i], 4))
            #print('clust[{}] min:{}; max:{};'.format((i + 1), min(cluster[:, index_param]), max(cluster[:, index_param])))
        except Exception as ex:
            print(ex)
            label = 'error'
        x,y = m(clust[0],clust[1])
        m.scatter(x, y, 2, marker='.', color=color_a[i], label=label, linewidth=1)
    m.scatter(check_data[0], check_data[1], marker='v', s=14, c='w', linewidth=0.7)
    #m.scatter(47.57, 52.05, marker='o', c='w', s=20)
    plt.legend(bbox_to_anchor=(1, 1), loc=9, borderaxespad=0., markerscale=10)
    plt.title(title_w)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('/Users/Smoky/Documents/workspace/result/epsilon/' + title_w + '.png', dpi=500)


def visual_array(data, param):
    plt.figure(1)
    plt.figure(1).canvas.set_window_title(str(param))
    #ax1 = plt.figure(1).add_subplot(211)
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('syn0')

    color = [np.random.rand(3, 1) for i in range(len(data[0]))]
    """data = data.T
    for c_index, foo in enumerate(data[0]):
        plt.plot(range(len(data.T)), foo, c=color[c_index], label=name_param[param[c_index]], linewidth=2)
        # plt.plot(range(0,len(data[index])), data[index],) """

    for i, p in enumerate(param):
        ax1.plot(range(len(data)), data[:, i], c=color[i], label=param[i], linewidth=2)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


def visual_sk_map(data,sample,  param,mp_s,sk_param,label):
    fig = plt.figure(3, figsize=(8.6, 7))
    plt.cla()
    title_w = 'mks:' + str(param) + ' skLn:' + str(sk_param)
    fig.canvas.set_window_title(title_w)
    # fig.savefig('test2png.png', dpi=100)

    m = Basemap(llcrnrlat=mp_s[2], urcrnrlat=mp_s[3],
                llcrnrlon=mp_s[0], urcrnrlon=mp_s[1], resolution='l')
    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2.5)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360., 2.5)
    m.drawmeridians(meridians, labels=[True, False, False, True])


    plt.scatter(data[0], data[1],  3, marker='.', color='b',label='skLn cluster')
    plt.scatter(sample[0], sample[1], marker='v', c='r',s=13,linewidths=0,label='simple eq')
    #m.scatter(47.57, 52.05, marker='o', c='w', s=20, label='balacNPP')
    #plt.legend(bbox_to_anchor=(1, 1), loc=9, borderaxespad=0.)
    plt.grid(True)
    plt.title(label)
    #plt.savefig('/Users/Smoky/Documents/workspace/result/test/'+label+'.png', dpi=400)
    #plt.cla()


def create_3d_map(data, sample, mp_s, param):
    fig = plt.figure(1414)
    #ax = fig.gca(projection='3d')
    ax = fig.gca()
    x,y,z = data[:,1], data[:,2], data[:,0]
    #ax.plot_wireframe(X, Y, Z, rstride=1000, cstride=1000)

    m = Basemap(llcrnrlat=mp_s[2], urcrnrlat=mp_s[3],
                llcrnrlon=mp_s[0], urcrnrlon=mp_s[1], resolution='l')


    m.drawcountries()
    m.drawcoastlines()
    parallels = np.arange(0., 90, 2.5)
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 360., 2.5)
    m.drawmeridians(meridians, labels=[True, False, False, True])

    #ax.plot_surface(X, Y, Z)
    #ax.scatter(X,Y,Z,marker='.')
    #ax.plot_surface(X, Y, Z, rstride=1000, cstride=1000)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='nearest')  # create a uniform spaced grid
    xig, yig = np.meshgrid(xi, yi)
    #surf = ax.plot_wireframe(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1)  # 3d plot
    surf = ax.contourf(xig, yig, zi, zdir='z', cmap=cm.RdYlBu)
    #surf = ax.plot_surface(X=xig, Y=yig, Z=zi, rstride=1, cstride=1, linewidth=1,cmap=cm.coolwarm)  # 3d plot

    #plt.savefig('/Users/Smoky/Documents/workspace/result/univ/test.png', dpi=400)

    ax.scatter(sample[0], sample[1], marker='v', c='k', s=5)

    fig.colorbar(surf,fraction=0.046, pad=0.04, label='error')
    plt.title(param)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    #plt.show()
#
def visual_error(data):
    plt.figure(1)
    #ax1 = plt.figure(1).add_subplot(231)
    ax1 = plt.subplot(2,1,2)
    ax1.set_title('error')
    iteration = np.arange(0, len(data))
    er = data
    ax1.semilogy(data)

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('mean error')
    ax1.grid(True)