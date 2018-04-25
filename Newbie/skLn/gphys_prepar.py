import numpy as np
import pandas as pd
from itertools import product
from skLn.visual import create_3d_map, visual_data
from skLn.tools import read_csv, evk

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/sk/' + place_name + 'LN/'
type_data = 'relief'
data = read_csv(directory + place_name + "_"+type_data+".csv", ['x', 'y', 'value']).T


def giveDataInRadius(grid, data, r, print_title):
    print(print_title)
    array = []
    for n, i in enumerate(grid):
        evk_array = evk(i, data)
        r_array = data[np.where(evk_array <= r)]
        # disp = np.sum((r_array[:, 2] - np.mean(r_array[:, 2])) ** 2) / len(r_array)
        #disp = np.sqrt(np.sum((r_array[:, 2] - np.mean(r_array[:, 2])) ** 2) / len(r_array))
        disp = np.std(r_array[:, 2])
        array.append(disp)

    return np.asarray(array)

R = 2 * np.sqrt(2) * 0.0167
#x1, x2, y1, y2, d = [36, 52, 37, 46, 1]
#coordinates = list(product(np.arange(x1, x2, d), np.arange(y1, y2, d)))
#grid = np.array(coordinates)
grid = data

disp_d = giveDataInRadius(grid, data, R, '').reshape(len(grid),1)



data_set = np.append(grid[:, :2], disp_d, axis=1)
p = 1.75
Mp = (np.sum(np.power(data_set[:, 2], p))/len(data_set))**(1/p)
print(Mp)
data_set_hi = data_set[np.where(data_set[:, 2] >= Mp)]
data_set_low = data_set[np.where(data_set[:, 2] < Mp)]
#print(data_set)
# 2:max 3:min 4:delta 5:grad 6:disp
print(type_data+' done')
m_r = [36, 52, 37, 46]

my_df = pd.DataFrame(np.asarray(data_set), columns=['x','y','value'])
my_df.to_csv(directory + place_name + type_data+"_1.csv", index=False, header=True,
             sep=';', decimal=',')
visual_data(data_set_hi, data_set_low , m_r, type_data)
#create_3d_map(data_set, 'relief_dAll')


