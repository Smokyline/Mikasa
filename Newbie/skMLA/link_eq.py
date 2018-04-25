import numpy as np
import pandas as pd
from skLn.tools import evk, read_csv

calc_r = lambda x: x / 111


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
        if len(evk_array_X) > 0:
            X_class.append(i)
        if len(evk_array) > 0 or len(evk_array_X) > 0:
            b_count += 1
        if len(evk_array) == 0 and len(evk_array_X) == 0:
            H_class.append(i)
    return np.asarray(B_class), np.asarray(X_class), np.asarray(H_class), b_count

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/sk/' + place_name + 'MLA/'
name_param = {0: 'x', 1: 'y', 2: 'Hmax', 3: 'Hmin', 4: 'dH', 5: 'Hgrad', 6: 'Hdisp', 7: 'Gmax', 8: 'Gmin', 9: 'dG', 10: 'Ggrad',
              11: 'Gdisp', 12: 'Mmax', 13: 'Mmin', 14: 'dM', 15: 'Mgrad', 16: 'Mdisp', 17: 'Lnint', 18: 'DPSc', 19: 'DPSint'}
param_all = [name_param[x] for x in range(0, 20)]

grid_data = read_csv(directory + place_name + '_grid.csv', param_all).T
sampleB = read_csv(directory + place_name + '_eq55-6.csv', ['x', 'y']).T
sampleX = read_csv(directory + place_name + '_eq5-55.csv', ['x', 'y']).T

R = calc_r(10)
print(R, 'r')
B, X, H, count = separation_data(grid_data, sampleB, sampleX, R)

print(count, 'sample count')
prm = list(name_param.values())

b_df = pd.DataFrame(B, columns=prm)
b_df.to_csv(directory + place_name + "_gridB.csv", index=False, header=True, sep=';', decimal=',')

x_df = pd.DataFrame(X, columns=prm)
x_df.to_csv(directory + place_name + "_gridX.csv", index=False, header=True, sep=';', decimal=',')

h_df = pd.DataFrame(H, columns=prm)
h_df.to_csv(directory + place_name + "_gridH.csv", index=False, header=True, sep=';', decimal=',')