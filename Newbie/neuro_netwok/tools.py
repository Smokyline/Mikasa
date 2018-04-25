import numpy as np
import sys
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fmin


def norm(sample, data, param):
    all_data_array = np.append(sample, data, axis=0)

    #for m in range(len(sample[0])):
     #   if (m > 1):
      #      max_value = np.max(all_data_array[:, m])
       #     min_value = np.min(all_data_array[:, m])
        #    all_data_array[:, m] = np.true_divide(all_data_array[:, m]-min_value, max_value-min_value)

    #max = np.max(all_data_array[:, 2:])
    #min = np.min(all_data_array[:, 2:])
    #all_data_array[:, 2:] = np.true_divide(all_data_array[:, 2:]-min,max-min)
    #all_data_array = np.append(all_data_array[:,:2],preprocessing.normalize(all_data_array[:, 2:],axis=0),axis=1)
    #all_data_array[:,2: ] = preprocessing.scale(all_data_array[:, 2:],axis=0)
    #all_data_array[:,2: ] = preprocessing.normalize(all_data_array[:, 2:], axis=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    all_data_array[:, 2:] = min_max_scaler.fit_transform(all_data_array[:,2: ])

    print(np.min(all_data_array[:, 2:]), 'min')
    print(np.max(all_data_array[:, 2:]), 'max')

    #return all_data_array.T[:len(sample)], all_data_array.T[len(sample):]
    return all_data_array[:len(sample)], all_data_array[len(sample):]


def nonlin(x, deriv=False):
    if deriv:
        return nonlin(x, deriv=False) * (1 - nonlin(x, deriv=False)) #f sigm
        #return (1+nonlin(x,deriv=False))*(1-nonlin(x,deriv=False))
        #return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        #return x * np.sqrt(np.exp(x ** 2))  # f gaus

    return 1 / (1 + np.exp(-x))  # f sign
    #return np.exp((-x ** 2) / 2)  # f gaus


def check_error(y, l1):
    #return (y-l1) ** 2
    #return np.abs(y-l1)
    #return np.sqrt((y-l1)**2)
    return y-l1


def efficiency(data, check_eq, param):
    eff_total = 0
    eff_count = 0
    delta = np.sqrt((data[0][0] - data[1][0]) ** 2 + (data[0][1] - data[1][1]) ** 2)
    for dot in check_eq:
        evk_array = np.sqrt((dot[0] - data[:, 0]) ** 2 + (dot[1] - data[:, 1]) ** 2)
        size = len(evk_array[np.where(evk_array <= delta)])
        eff_total += size
        if size > 0:
            eff_count+=1
    eff_total = eff_total/len(data)
    print(','.join(map(str,param)), 'total:'+str(eff_total), 'count:'+str(eff_count)+'/'+str(len(check_eq)))
    return eff_total, eff_count


def calc_ni(syn0, l2_delta, l0):
    range_x = np.arange(0.0000001, 1., 0.00005)
    function_value = []

    def foo(x):
        X = l0.T * syn0
        Xni = x * nonlin(X, True)
        #syn = syn0 + x*l2_delta
        return np.mean(np.abs(X - Xni))
    for x in range_x:
        function_value.append(foo(x))

    ni = range_x[np.array(function_value).argmin()]
    #print(ni, 'ni')


    #fmin(foo, np.array([min_x, max_x]))
    return ni
