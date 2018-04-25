from neuro_netwok import matplt, reader
#from neuro_netwok.learning import recognition, omega
#from neuro_netwok.learning_re import recognition, omega
from neuro_netwok.learning_epsilon import recognition, omega
#from neuro_netwok.learning_theta import recognition, omega
from neuro_netwok.tools import norm, efficiency
from neuro_netwok.scikit_learning import sk_learning
from neuro_netwok import kmeans2, kmeans
import itertools
import numpy as np
import pandas as pd

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/geop/' + place_name + '/'
# kvz [40,54,35,47]
# balac [44, 53, 48, 54]
# crimea 29., 40., 41., 49.
# calif -127, -113, 30, 42
map_size = [40, 54, 35, 47]
#map_size = [29., 40., 41., 49.]
#map_size = [44, 53, 48, 54]
#map_size = [-127, -113, 30, 42]

name_param = {2: 'dps', 3: 'grav', 4: 'em', 5: 'ln', 6: 'em_disp', 7: 'relief_disp', 8: 'grav_disp', 9: 'dpsP', 10: 'tecton'}
#name_param = {2: 'dps', 3: 'grav', 4: 'em',  5: 'em_disp', 6: 'relief_disp', 7: 'grav_disp', 8: 'dpsP', 9: 'tecton'}
#index_param = range(2, 9)
index_param = [7]
index_sk_param = range(2,9)
#index_sk_param = [7]
param_all = [name_param[x] for x in range(2, 10)]
param = [name_param[x] for x in index_param]
sk_param = [name_param[x] for x in index_sk_param]

sample = reader.read_csv(directory + place_name + "_simple.csv", param_all).T
data = reader.read_csv(directory + place_name + "_grid.csv", param_all).T
sample, data = norm(sample, data, param_all)

def make_combinations(index_param):
    eff_array = []
    #for e in range(2, 9):
    for e in [4]:
        combin_array = np.array(list(itertools.combinations(index_param, e))) # кол-во комбинаций
        for i, index in enumerate(combin_array):
            print("\n%i/%i" % (i + 1, len(combin_array)))
            param = [name_param[x] for x in index]
            eff = run(param, index, sk_param)
            eff_array.append([str(eff[0]), str(eff[1]), ','.join(map(str, [name_param[x] for x in index])) ])
    my_df = pd.DataFrame(np.asarray(eff_array), columns=['total','count','title'])
    my_df.to_csv(directory + place_name + "_eff.csv", index=False, header=True,
                 sep=';', decimal=',')

def check_param(index_param):
    errors = {}
    for e in range(2,9):
        combin_array = np.array(list(itertools.combinations(index_param, e)))  # кол-во комбинаций
        for i, index in enumerate(combin_array):
            print("\n%i/%i" % (i + 1, len(combin_array)))
            title = '{}'.format([name_param[x] for x in index])
            print(title)
            error, syn0, syn1 = omega(sample[:, index], len(sample), index)
            errors[title] = error
    for key in sorted(errors, key=errors.get, reverse=False):
        print(key, errors[key])

def run(param, index_param, sk_param):
    print('\nparam:', param)
    error, syn0, syn1 = omega(sample[:, index_param], len(sample), param)
    res = recognition(data, syn0, syn1, index_param)

    k = 5
    #clust, centroids = kmeans.createClust(res, k)
    clust, centroids = kmeans2.create_clust(res, k)
    #return efficiency(clust[-1], sample, param)

    #matplt.create_3d_map(res, sample.T[:2], map_size, param)
    matplt.visual_clust(clust, sample.T[:2], param, map_size, centroids, k)

    #sk_learning(sample, clust, data, index_sk_param, map_size, param, sk_param)

    matplt.draw_plot()





run(param, index_param, sk_param)
#make_combinations(index_param)
#check_param(index_param)

