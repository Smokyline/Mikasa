import numpy as np
import pandas as pd

def read_csv(path, param):
    array=[]
    frame = pd.read_csv(path, header=0, sep=';', decimal=",")
    for i,title in enumerate(param):
        try:
            array.append(frame[title].values)
        except:
            print('no_'+title)

    return np.array(array)

def normal(B, X, H):
    from sklearn import preprocessing
    all_data = np.append(np.append(B, X, axis=0), H, axis=0)

    for m in range(len(all_data[0])):
        if (m > 1):
            max_value = np.max(all_data[:, m])
            min_value = np.min(all_data[:, m])
            all_data[:, m] = np.true_divide(all_data[:, m]-min_value, max_value-min_value)

    #all_data[:, 2:] = preprocessing.normalize(all_data[:, 2:], axis=0)
    return all_data[:len(B)], all_data[len(B):len(X)], all_data[len(X):]

