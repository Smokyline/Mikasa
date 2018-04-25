import numpy as np
import pandas as pd
from itertools import product
from skLn.visual import create_3d_map, visual_data
from skLn.tools import read_csv, evk

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/sk/' + place_name + 'LN/'
type_data = 'kvzrelief_1'
data_set = read_csv(directory+type_data+".csv", ['x', 'y', 'value']).T

m_r = [36, 52, 37, 46]
p = 1.5
Mp = (np.sum(np.power(data_set[:, 2], p))/len(data_set))**(1/p)
print(Mp)
data_set_p = data_set[np.where(data_set[:, 2] >= Mp)]
data_set_m = data_set[np.where(data_set[:, 2] < Mp)]
title_v = type_data+'_p='+str(p)
visual_data(data_set_p, data_set_m , m_r, title_v)
#create_3d_map(data_set, type_data)
