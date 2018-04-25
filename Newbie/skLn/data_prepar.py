import numpy as np
import pandas as pd
import sys
from skLn.tools import read_csv, evk

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/sk/' + place_name + 'Ln/'
# name_param = {0: 'x', 1: 'y', 2: 'Hmax', 3: 'Hmin', 4: 'dH', 5: 'Hgrad', 6: 'Rint', 7: 'Bmax', 8: 'Bmin', 9: 'dB'}




calc_r = lambda x: x / 111
eps = sys.float_info.epsilon


def giveLnData(grid, data):
    print('ln')
    array = []
    for i in grid:
        evk_array = evk(i, data)
        array.append(np.min(evk_array[np.where(evk_array >= eps)]))
    return np.asarray(array).reshape(len(grid),1)


def giveDataInRadius(grid, data, r, print_title):
    print(print_title)
    array = []
    for n, i in enumerate(grid):
        evk_array = evk(i, data)
        r_array = data[np.where(evk_array <= r)]
        max_b = np.max(r_array[:, 2])
        min_b = np.min(r_array[:, 2])
        #disp = np.sqrt(np.sum((r_array[:, 2] - np.mean(r_array[:, 2])) ** 2) / len(r_array))
        disp = np.std(r_array[:, 2])

        if not max_b == min_b:
            delta = max_b - min_b
            xy_max = r_array[np.argmax(r_array[:, 2]), :2]
            xy_min = r_array[np.argmin(r_array[:, 2]), :2]
            evk_range = np.sqrt(np.power(xy_max[0] - xy_min[0],2) + np.power(xy_max[1] - xy_min[1],2))
            grad = delta / (evk_range*111000)
            b = np.array([max_b, min_b, delta, grad, disp])
            array.append(b)
        else:
            b = np.array([max_b, min_b, 0, 0, disp])
            array.append(b)

    return np.asarray(array)




def run():
    grid = read_csv(directory + place_name + "_ln.csv", ['x', 'y']).T
    data_relief = read_csv(directory + place_name + '_relief.csv', ['x', 'y', 'value']).T
    data_grav = read_csv(directory + place_name + '_gravB.csv', ['x', 'y', 'value']).T
    data_mag = read_csv(directory + place_name + '_mag.csv', ['x', 'y', 'value']).T
    R = calc_r(25)
    print(R)

    h_set = giveDataInRadius(grid, data_relief, R, 'relief')
    g_set = giveDataInRadius(grid, data_grav, R, 'grav')[:, [0,1,2,4]]
    mag_set = giveDataInRadius(grid, data_mag, R, 'mag')[:, [0,1,2,4]]
    ln_set = giveLnData(grid, grid)



    prm = ['x', 'y', 'Hmax', 'Hmin', 'dH', 'Hgrad', 'Hdisp', 'Gmax', 'Gmin', 'dG', 'Gdisp', 'Mmax', 'Mmin',
              'dM', 'Mdisp', 'Lnint']
    print(prm)

    for i in [h_set, g_set, mag_set, ln_set]:
        grid = np.append(grid, i, axis=1)

    my_df = pd.DataFrame(np.asarray(grid), columns=prm)
    my_df.to_csv(directory + place_name + "_gridv6.csv", index=False, header=True,
                 sep=';', decimal=',')



run()