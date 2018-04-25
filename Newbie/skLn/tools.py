import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cross_validation, svm

from scipy.optimize import fmin
import operator


def read_csv(path, param):
    array=[]
    frame = pd.read_csv(path, header=0, sep=';',decimal=",")
    for i,title in enumerate(param):
        try:
            array.append(frame[title].values)
        except:
            print('no_'+title)

    return np.array(array)

def norm(data):


    for m in range(len(data[0])):
        if (m > 1):
            max_value = np.max(data[:, m])
            min_value = np.min(data[:, m])
            data[:, m] = np.true_divide(data[:, m]-min_value, max_value-min_value)

    #max = np.max(all_data_array[:, 2:])
    #min = np.min(all_data_array[:, 2:])
    #all_data_array[:, 2:] = np.true_divide(all_data_array[:, 2:]-min,max-min)
    #all_data_array = np.append(all_data_array[:,:2],preprocessing.normalize(all_data_array[:, 2:],axis=0),axis=1)
    #all_data_array[:,2: ] = preprocessing.scale(all_data_array[:, 2:],axis=0)
    #data[:,2: ] = preprocessing.normalize(data[:, 2:], axis=0)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #data[:, 2:] = min_max_scaler.fit_transform(data[:,2: ])

    #print(np.min(data[:, 2:]), 'min')
    #print(np.max(data[:, 2:]), 'max')

    #return all_data_array.T[:len(sample)], all_data_array.T[len(sample):]
    return data

def evk(p, data):
    return np.sqrt((p[0] - data[:, 0]) ** 2 + (p[1] - data[:, 1]) ** 2)

def separation_data(data, sample, sampleX, r):
    B_class = []
    H_class = []
    X_class = []
    count = 0
    for i in data:
        evk_array = evk(i[:2], sample)
        evk_array = evk_array[np.where(evk_array <= r)]
        evk_array_X = evk(i[:2], sampleX)
        evk_array_X = evk_array_X[np.where(evk_array_X <= r)]
        if len(evk_array) > 0:
            B_class.append(i)
        elif len(evk_array_X) > 0:
            X_class.append(i)
        if len(evk_array) > 0 or len(evk_array_X) > 0:
            count += 1
        if len(evk_array) == 0 and len(evk_array_X) == 0:
            H_class.append(i)
    return np.asarray(B_class), np.asarray(X_class), np.asarray(H_class), count


def cross_v(alg_set, X, y, scoring):
    cv = cross_validation.ShuffleSplit(len(y), n_iter=10, test_size=0.1, random_state=0)
    itog_val = {}
    for name, i in alg_set:
        scores = cross_validation.cross_val_score(i, X, y, cv=cv, scoring=scoring)
        itog_val[name] = scores.mean()
        print('%s: %0.3f %s' % (scoring, itog_val[name], name))
    itog_val = sorted(itog_val.items(), key=operator.itemgetter(1))
    #for key, value in itog_val:
     #   print('%s: %0.3f %s' % (scoring, value, key))

