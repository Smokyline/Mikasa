import pandas as pd
import numpy as np
import os

res_path = os.path.expanduser('~' + os.getenv("USER") +
                              '/Documents/workspace/resources/csv/Barrier/kvz/')


def read_csv(path, col=['x', 'y']):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i, title in enumerate(col):
        cell = frame[title].values

        try:
            cell = cell[~np.isnan(cell)]
        except Exception as ex:
            print(ex)
            for j, c in enumerate(cell):
                try:
                    np.float(c.replace(',', '.'))
                except:
                    print('Error in row:%s "%s"' % (j, c))

        array.append(cell)

    return np.array(array).T

def norm_data(data):
    _, cell = data.shape
    for c in range(cell):
        cell = data[:, c]
        cell_positive = cell-cell.min()
        data[:, c] = cell_positive/cell_positive.max()
    return data

def get_data_XY(norm=False):


    col = ['idx', 'Hmax', 'Hmin', 'DH', 'Top', 'Q', 'HR', 'Nl',
           'Rint', 'DH/l', 'Nlc', 'R1', 'R2', 'Bmax', 'Bmin',
           'DB', 'Mmax', 'Mmin', 'DM', ]

    data_X = read_csv(res_path + 'khar.csv', col)
    data_Y = read_csv(res_path + 'sample.csv', col)

    if norm:
        data_X, data_Y = norm_data(data_X), norm_data(data_Y)

    return data_X,data_Y

def get_coordX():
    data_coord = read_csv(res_path+'coord.csv', ['x', 'y'])
    return data_coord

def get_eq():
    data_ist = read_csv(res_path+'_eq_istor.csv', ['x', 'y'])
    data_instr = read_csv(res_path+'_eq_instr.csv', ['x', 'y'])
    return [data_ist, data_instr]