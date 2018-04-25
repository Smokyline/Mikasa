import numpy as np
import pandas as pd
import sys
from itertools import product
from skLn.tools import read_csv, evk
from skLn.data_prepar import giveDataInRadius, giveLnData
from create_dataset import DPS

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/sk/' + place_name + 'MLA/'
name_param = {0: 'x', 1: 'y', 2: 'Hmax', 3: 'Hmin', 4: 'dH', 5: 'Hgrad', 6: 'Hdisp', 7: 'Gmax', 8: 'Gmin', 9: 'dG', 10: 'Ggrad',
              11: 'Gdisp', 12: 'Mmax', 13: 'Mmin', 14: 'dM', 15: 'Mgrad', 16: 'Mdisp', 17: 'Lnint', 18: 'DPSc', 19: 'DPSint'}
eq_6 = read_csv(directory + place_name + "_eq6.csv", ['x', 'y']).T
eq_55 = read_csv(directory + place_name + "_eq55.csv", ['x', 'y']).T
eq_dps = read_csv(directory + place_name + "_eqDPS.csv", ['x', 'y']).T
data_ln = read_csv(directory + place_name + "_ln.csv", ['x', 'y']).T
data_grav = read_csv(directory + place_name + '_gravB.csv', ['x', 'y', 'value']).T
data_mag = read_csv(directory + place_name + '_mag.csv', ['x', 'y', 'value']).T
data_relief = read_csv(directory + place_name + '_relief.csv', ['x', 'y', 'value']).T

calc_r = lambda x: x / 111
eps = sys.float_info.epsilon


# x_min, x_max, y_min, y_max, step
# balac 44, 53, 48, 54, 0.015
# kvz 40, 54, 35, 47 025
# calif -127, -113, 30, 42, 025
# crimea 29., 40., 41., 49. 020
x1, x2, y1, y2, d = [40, 54, 35, 47, 0.05]
coordinates = list(product(np.arange(x1, x2, d), np.arange(y1, y2, d)))
grid = np.array(coordinates)
print(len(grid), 'grid data')
R = calc_r(25)

def giveDPSdata(grid, dps_d, r, print_title):
    print(print_title)
    array = []
    for i in grid:
        evk_array = evk(i, dps_d)
        dps_min_range = np.min(evk_array)
        r_array = dps_d[np.where(evk_array <= r)]
        dps_count = len(r_array)
        b = np.array([dps_count, dps_min_range])
        array.append(b)
    return np.asarray(array)

h_set = giveDataInRadius(grid, data_relief, R, 'relief')
data_Hmax = h_set[:, 0]
data_Hmin = h_set[:, 1]
data_dH = h_set[:, 2]
data_Hgrad = h_set[:, 3]
data_Hdisp = h_set[:, 4]
g_set = giveDataInRadius(grid, data_grav, R, 'grav')
data_Gmax = g_set[:, 0]
data_Gmin = g_set[:, 1]
data_dG = g_set[:, 2]
data_Ggrad = g_set[:, 3]
data_Gdisp = g_set[:, 4]
mag_set = giveDataInRadius(grid, data_mag, R, 'mag')
data_Mmax = mag_set[:, 0]
data_Mmin = mag_set[:, 1]
data_dM = mag_set[:, 2]
data_Mgrad =mag_set[:, 3]
data_Mdisp = mag_set[:, 4]
ln_set = giveLnData(grid, data_ln)
dps_clustrs, _ = DPS.dps_clust(eq_dps, -0.45, -3)
dps_set = giveDPSdata(grid, dps_clustrs, R, 'dps')
data_dpsC = dps_set[:, 0]
data_dpsInt = dps_set[:, 1]

final_grid = []
#['x', 'y', 'Hmax', 'Hmin', 'dH', 'Hgrad', 'Hdisp', 'Gmax', 'Gmin', 'dG', 'Ggrad', 'Gdisp', 'Mmax', 'Mmin',
# 'dM', 'Mgrad', 'Mdisp', 'Lnint', 'DPSc', 'DPSint']
for n, j in enumerate(grid):
    p = np.array([j[0], j[1], data_Hmax[n], data_Hmin[n], data_dH[n], data_Hgrad[n], data_Hdisp[n],
                  data_Gmax[n], data_Gmin[n], data_dG[n], data_Ggrad[n], data_Gdisp[n],
                  data_Mmax[n], data_Mmin[n], data_dM[n], data_Mgrad[n], data_Mdisp[n],
                  ln_set[n], data_dpsC[n], data_dpsInt[n]])
    final_grid.append(p)

prm = list(name_param.values())
grid_df = pd.DataFrame(np.asarray(final_grid), columns=prm)
grid_df.to_csv(directory + place_name + "_grid.csv", index=False, header=True, sep=';', decimal=',')

def separation_data(data, sampleB, sampleX, r):
    B_class = []
    H_class = []
    X_class = []
    b_count = 0
    for i in data:
        evk_array = evk(i[:2], sampleB)
        evk_array = evk_array[np.where(evk_array <= r)]
        evk_array_X = evk(i[:2], sampleX)
        evk_array_X = evk_array_X[np.where(evk_array_X <= r)]
        if len(evk_array) > 0:
            B_class.append(i)
        elif len(evk_array_X) > 0:
            X_class.append(i)
        if len(evk_array) > 0 or len(evk_array_X) > 0:
            b_count += 1
        if len(evk_array) == 0 and len(evk_array_X) == 0:
            H_class.append(i)
    return np.asarray(B_class), np.asarray(X_class), np.asarray(H_class), b_count

B, X, H, count = separation_data(final_grid, eq_6, eq_55, R)
print(count, 'sample count')

b_df = pd.DataFrame(B, columns=prm)
b_df.to_csv(directory + place_name + "_gridB.csv", index=False, header=True, sep=';', decimal=',')

x_df = pd.DataFrame(X, columns=prm)
x_df.to_csv(directory + place_name + "_gridX.csv", index=False, header=True, sep=';', decimal=',')

h_df = pd.DataFrame(H, columns=prm)
h_df.to_csv(directory + place_name + "_gridH.csv", index=False, header=True, sep=';', decimal=',')