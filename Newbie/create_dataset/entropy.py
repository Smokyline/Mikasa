import pandas as pd
import numpy as np
import math

def read_csv(title, path):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")

    for i, title in enumerate(title):
        array.append(frame[title].values)

    return np.array(array)

def calculate_dispersion(x, mean_x, len_x):
    return 1/len_x*(x-mean_x)**2

math.log(p,2)
path = '/Users/Smoky/Documents/workspace/resourses/csv/geop/kvz/kvz_relief.csv'
relief_data = read_csv(['x', 'y', 'value'], path).T
print(relief_data[0])
mean_x = np.mean(relief_data[:, 2])
len_data = len(relief_data)
dispersion_data = relief_data
dispersion_data[:, 2] = calculate_dispersion(dispersion_data[:, 2], mean_x, len_data)
print(dispersion_data[0])

my_df = pd.DataFrame(dispersion_data, columns=['x', 'y', 'value'])
my_df.to_csv('/Users/Smoky/Documents/workspace/resourses/csv/geop/kvz_disp.csv', index=False, header=True,
                 sep=';', decimal=',')