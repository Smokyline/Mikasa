from mpl_toolkits.basemap import Basemap
import matplotlib
import numpy as np

#matplotlib.use('TkAgg')
matplotlib.use('Qt4Agg')
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
#40,54,35,47
m = Basemap(llcrnrlat=50,urcrnrlat=53,
            llcrnrlon=43,urcrnrlon=51,
            resolution='l',)
# draw a land-sea mask for a map background.
# lakes=True means plot inland lakes with ocean color.
m.drawcountries()
m.drawcoastlines()
parallels = np.arange(20.,60,5.)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(20.,60,5.)
m.drawmeridians(meridians, labels=[True,False,False,True])
x = [51.15,51.38,50.90,51.74,47.03,46.80,46.95]
y = [52.80,50.87,50.70,50.61,49.98,49.80,49.34]
m.scatter(x,y)


#m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)

plt.show()