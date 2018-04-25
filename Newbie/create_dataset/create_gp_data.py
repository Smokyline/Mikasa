import pandas as pd
import numpy as np


def read_csv(title, path):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")

    for i, title in enumerate(title):
        array.append(frame[title].values)

    return np.array(array)


def calculate_dispersion(x, mean_x, len_x):
    return 1 / len_x * (x - mean_x) ** 2


def calculate_Kurtosis(len_x, x, mean_x):
    return 1 / len_x * (((x - mean_x) ** 4) / (calculate_dispersion(x, mean_x, len_x) ** 4)) - 3

def calculate_asymmetry(len_x, x, mean_x):
    return 1 / len_x * ((x-mean_x)**3 / calculate_dispersion(x, mean_x, len_x)**3)

place = 'calif'
type_data = 'em'
path = '/Users/Smoky/Documents/workspace/resourses/csv/geop/{}/{}_{}.csv'.format(place, place, type_data)
relief_data = read_csv(['x', 'y', 'value'], path).T
print(relief_data[0])
mean_x = np.mean(relief_data[:, 2])
len_data = len(relief_data)
dispersion_data = relief_data
dispersion_data[:, 2] = calculate_dispersion(dispersion_data[:, 2], mean_x, len_data)
print(dispersion_data[0])

#kurtosis_data = relief_data
#kurtosis_data[:, 2] = calculate_Kurtosis(len_data, kurtosis_data[:, 2], mean_x)
#print(kurtosis_data[0])

#asymmetry_data = relief_data
#asymmetry_data[:, 2] = calculate_Kurtosis(len_data, asymmetry_data[:, 2], mean_x)
#print(asymmetry_data[0])


my_df = pd.DataFrame(dispersion_data, columns=['x', 'y', 'value'])
# my_df = pd.DataFrame(kurtosis_data, columns=['x', 'y', 'value'])
path_to_save = '/Users/Smoky/Documents/workspace/resourses/csv/geop/{}/{}_{}_disp.csv'.format(place, place, type_data)
my_df.to_csv(path_to_save, index=False, header=True, sep=';', decimal=',')
