from neuro_netwok import matplt, reader
from neuro_netwok.learning import recognition, omega
# from neuro_netwok.learning_re import recognition, omega
from neuro_netwok.tools import norm
from neuro_netwok.scikit_learning import sk_learning
from neuro_netwok import kmeans2, kmeans
import itertools
from scipy.interpolate import griddata
import matplotlib as cm


from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.base import Bunch
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids
from sklearn import svm, metrics
from mpl_toolkits.basemap import Basemap

name_param = {2: 'dps', 3: 'grav', 4: 'em', 5: 'ln', 6: 'em_disp', 7: 'relief_disp', 8: 'grav_disp', 9: 'dpsP',
              10: 'tecton'}
# index_param = range(2,9)
index_param = [2]
param_all = [name_param[x] for x in range(2, 10)]
param = [name_param[x] for x in index_param]

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/geop/' + place_name + '/'
sample = reader.read_csv(directory + place_name + "_simple.csv", param_all).T
data = reader.read_csv(directory + place_name + "_grid.csv", param_all).T
sample, data = norm(sample, data, param_all)

x = data[:, 0]
y = data[:, 1]
xi = np.linspace(x.min(), x.max(), 500)
yi = np.linspace(y.min(), y.max(), 500)
xig, yig = np.meshgrid(xi, yi)

Y = sample[:, index_param]
X = data[:, index_param]

clf = svm.OneClassSVM()
clf.fit(Y)
#y_pred = clf.predict(Y)
x_pred = clf.predict(X)

#print(print('classifier', list(set(y_pred))))
#print(print('classifier', list(set(x_pred))))
#print(len(x_pred))
print(x_pred)


array = []
for i, item in enumerate(x_pred):
    if item == 1:
        array.append(data[i, [0, 1]])

title = 'mks' + str(param) + ' skLn' + str(param)
map_size = [40, 54, 35, 47]
matplt.visual_sk_map(np.array(array).T, sample.T, param, map_size, param, title)
matplt.draw_plot()