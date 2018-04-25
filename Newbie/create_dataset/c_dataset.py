import numpy as np
from itertools import product
import pandas as pd
import sys
from create_dataset import DPS
from sklearn.metrics.pairwise import euclidean_distances
import time




def read_csv(title, path):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i, title in enumerate(title):
        array.append(frame[title].values)
    return np.array(array)


def link_value(data_coord, value_data, r=10., range=False):
    array = []
    eps = sys.float_info.epsilon
    if range:
        #evk_array = euclidean_distances(data_coord, value_data)
        for it, i in enumerate(data_coord):
            #evk_i = [x for x in evk_array[it] if x > eps]
            #evk_array=np.sum(abs(value_data - i),axis=1 )
            evk_array = np.sqrt((i[0] - value_data[:, 0]) ** 2 + (i[1] - value_data[:, 1]) ** 2)

            evk_array = [x for x in evk_array if x > eps]

            array.append(np.append(i, np.min(evk_array)))
    else:
        simple_data = value_data[:, [0, 1]]
        #evk_array = euclidean_distances(data_coord, simple_data)
        for it, i in enumerate(data_coord):
            evk_array = np.sqrt((i[0] - simple_data[:,0])**2+(i[1] - simple_data[:,1])**2)

            index = np.argmin(evk_array)
            if evk_array[index] <= r:
                array.append(np.append(i, value_data[index][2]))
            else:
                array.append(np.append(i, [eps]))


    return np.array(array)





def create_simple_set(place_name, prm):
    #time1 = int(round(time.time() * 1000)) print(int(round(time.time() * 1000))-time1,'time')
    print('\nsetup simples')
    directory = '/Users/Smoky/Documents/workspace/resourses/csv/geop/' + place_name + '/'
    param_set = {}
    dps_set = None
    data_coord = read_csv(['x', 'y'], directory + place_name + '_nEQ.csv').T

    for i in prm[2:]:
        print(i)

        if i == 'dps':

            eq_data = read_csv(['x', 'y'], directory + place_name + '_dps.csv').T
            # balac beta=-0.1, q=-2
            # kvz beta=-0.3, q=-3
            # crimea beta = 0.3 q = -0.5
            # calif -0.3 -2
            dps_set = DPS.dps_clust(eq_data, -0.3, -3)
            param_set["dps"] = link_value(data_coord, dps_set[0], range=True)
            param_set["dpsP"] = link_value(data_coord, dps_set[1], r=0.3)

        if i=='ln':
            ln_data = read_csv(['x', 'y'], directory + place_name + '_' + i + '.csv').T
            param_set['ln'] = link_value(data_coord, ln_data, range=True)

        if i=='tecton':
            tecton_data = read_csv(['x', 'y'], '/Users/Smoky/Documents/workspace/resourses/csv/geop/tecton.csv').T
            param_set['tecton'] = link_value(data_coord, tecton_data, range=True)

        if i in ['grav','em','relief_disp','grav_disp','em_disp']:
                value_data = read_csv(['x', 'y', 'value'], directory + place_name + '_' + i + '.csv').T
                param_set[i] = link_value(data_coord, value_data)

   # print(param_set['dps'][0])
   # print(param_set['grav'][0])
   # print(param_set['em'][0])
    simple_data = []
    for n, j in enumerate(data_coord):
        for p in prm[2:]:
            j = np.append(j, param_set[p][n][2])
        simple_data.append(j)

    my_df = pd.DataFrame(np.asarray(simple_data), columns=prm)
    my_df.to_csv(directory + place_name+"_simple.csv", index=False, header=True,
                 sep=';', decimal=',')

    return dps_set


def create_grid_set(grid_data, dps_set,  place_name, prm):
    print('\nsetup grid')
    directory = '/Users/Smoky/Documents/workspace/resourses/csv/geop/' + place_name + '/'
    param_set = {}
    print('grid param:', prm)

    for p in prm[2:]:
        print(p, 'start')
        if p =='dps':
            param_set['dps'] = link_value(grid_data, dps_set[0], range=True)
            param_set['dpsP'] = link_value(grid_data, dps_set[1], r=0.3)
        if p == 'ln':
            ln_data = read_csv(['x', 'y'], directory + place_name + '_' + p + '.csv').T
            param_set['ln'] = link_value(grid_data, ln_data, range=True)

        if p == 'tecton':
            tecton_data = read_csv(['x', 'y'], '/Users/Smoky/Documents/workspace/resourses/csv/geop/tecton.csv').T
            param_set['tecton'] = link_value(grid_data, tecton_data, range=True)

        if p in ['grav','em','relief_disp','grav_disp','em_disp']:
            value_data = read_csv(['x', 'y', 'value'], directory + place_name + '_' + p + '.csv').T
            param_set[p] = link_value(grid_data, value_data)


    grid = []

    for n, j in enumerate(grid_data):
        for pr in prm[2:]:
            j = np.append(j, param_set[pr][n][2])
        grid.append(j)

    my_df = pd.DataFrame(np.asarray(grid), columns=prm)
    my_df.to_csv(directory + place_name + "_grid.csv", index=False, header=True,
                 sep=';', decimal=',')



def create_param(param):

    #param = ['em']
    dps_set = create_simple_set(place_name, param)
    return dps_set


def create_grid(param):
    # x_min, x_max, y_min, y_max, step
    # balac 44, 53, 48, 54, 0.015
    # kvz 40, 54, 35, 47 025
    # calif -127, -113, 30, 42, 025
    # crimea 29., 40., 41., 49. 020
    #map_size = [40, 54, 35, 47, 0.025]
    #map_size = [29., 40., 41., 49., 0.020]
    #map_size = [44, 53, 48, 54, 0.015]
    map_size = [-127, -113, 30, 42, 0.025]
    coordinates = list(product(np.arange(map_size[0], map_size[1], map_size[4]), np.arange(map_size[2], map_size[3], map_size[4])))
    grid = np.array(coordinates)
    #directory = '/Users/Smoky/Documents/workspace/resourses/csv/geop/' + place_name + '/'
    #grid = read_csv(['x', 'y'], directory + place_name + '_ln.csv').T
    print(len(grid), 'grid data')

    dps_set = create_param(param)
    create_grid_set(grid, dps_set,place_name,param)


param = ['x','y','dps','dpsP','grav','em','tecton','ln','relief_disp','grav_disp', 'em_disp']
#param = ['x','y','dps','dpsP','grav','em', 'tecton','relief_disp','grav_disp', 'em_disp']
#param = ['x','y','dps']
place_name = 'kvz' #dps!!!



#create_param(param)
create_grid(param)


