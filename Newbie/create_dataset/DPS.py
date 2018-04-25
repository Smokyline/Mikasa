import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time


def read_csv(path):
    array = []
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i, title in enumerate(['x', 'y']):
        try:
            array.append(frame[title].values)
        except:
            print('no_' + title, end=' ')

    return np.array(array)


def visual_data(clusters, data):
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c='k', marker='.', s=10, linewidths=0)
    plt.scatter(clusters[:, 0], clusters[:, 1], c='r', marker='.', s=20, linewidths=0)
    plt.show()


def calc_r(data, q):
    eps = sys.float_info.epsilon
    evk = 0
    count = 0

    for i, j in enumerate(data):
        evk_array = np.sqrt((j[0] - data[i + 1:, 0]) ** 2 + (j[1] - data[i + 1:, 1]) ** 2)
        # evk_array = np.sqrt(np.sum(np.power(data[i+1:]-j, 2), axis=1))
        # array = np.sum(abs(data[i + 1:]-j), axis=1)
        # array = np.sum([x**q for x in evk_array if x > eps])
        array = np.sum(np.power(evk_array[np.where(evk_array > eps)], q))
        count += len(evk_array)
        if array > eps:
            evk += array

    r = (evk / count) ** (1 / q)
    print(r, 'r')
    return r


def calc_p(data, r):
    array = []
    for i in data:
        evk_array = np.sqrt((i[0] - data[:, 0]) ** 2 + (i[1] - data[:, 1]) ** 2)
        # evk_array = np.sqrt(np.abs(i-data))
        #array_i = np.sum(1 - x / r for x in evk_array if x <= r)
        evk_array = evk_array[np.where(evk_array <= r)]
        array_i = np.sum(1-evk_array/r)
        array.append(array_i)

    arrayP = array / np.max(array)
    return arrayP


def calc_a(Pxa, beta):
    min_x = 0
    max_x = 1
    eps = 0.00001
    alpha = 0

    def foo(arr, a, beta):
        EPx = 0
        for b in arr:
            EPx += ((a - b) / np.max([a, b]))
        return EPx / len(arr) - beta

    while True:
        half_x = (max_x + min_x) / 2
        fA_min = foo(Pxa, min_x, beta)
        fA_half = foo(Pxa, half_x, beta)
        if fA_min * fA_half < 0:
            max_x = half_x
        else:
            min_x = half_x
        alpha = half_x
        if max_x - min_x < eps:
            break
    print(alpha, 'alpha')
    return alpha


def compare(data, Pxa, a):
    fail_data = []
    DPS_data = np.array([np.append(data[i], x) for i, x in enumerate(Pxa) if x >= a])
    DPS_clust = DPS_data[:, [0, 1]]
    return DPS_clust, DPS_data


def dps_clust(data, beta=0.15, q=-3):
    print('beta={} q={}'.format(beta, q))
    it = 1
    r = None
    a = None
    p_data = data
    dps_set = None
    old_dps_set = None
    while True:
        if it == 1:
            r = calc_r(p_data, q)
        Pxa = calc_p(p_data, r)
        if it == 1:
            a = calc_a(Pxa, beta)
        dps_set = compare(p_data, Pxa, a)
        if np.array_equal(dps_set[0], p_data):
            break
        else:
            #break
            it += 1
            p_data = dps_set[0]
    print(len(dps_set[0]), 'dps_data')
    print(it, 'dps iteration\n')
    return dps_set



data = read_csv('/Users/Ivan/Documents/workspace/resourses/csv/geop/kvz/kvz_dps.csv').T
dps_set = dps_clust(data)
visual_data(dps_set[0], data)
# dps_clust(geop_data.read_csv(['x', 'y'], '/Users/Smoky/Documents/workspace/resourses/csv/Kavkaz.csv').T)
