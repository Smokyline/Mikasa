import numpy as np
from neuro_netwok import matplt
from neuro_netwok import matplt, reader
from neuro_netwok.learning import recognition, omega
from neuro_netwok.tools import norm
from neuro_netwok import kmeans2, kmeans

from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pylab as pl
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/geop/'+place_name+'/'

name_param = {2: 'tecton', 3: 'dps', 4: 'grav', 5: 'em', 6: 'ln', 7: 'dpsP'}
#param = [name_param[x] for x in range(3,6)]
param = [name_param[x] for x in [3,4,5]]

sample = reader.read_csv(directory + place_name+"_simple.csv", param).T
data = reader.read_csv(directory + place_name+"_grid.csv", param).T
eq_sample =  reader.read_csv(directory + place_name+"_simple.csv", param)
sample_p = preprocessing.normalize(sample.T[2:].T)
data_p = preprocessing.normalize(data.T[2:].T)
#data_learn = [data_p[np.random.randint(len(data_p))] for x in range(150000)]

X = np.append(sample_p, data_p[0:100],axis=0)
#X = np.append(sample_p, data_p,axis=0)
y = np.append([[1], ] * len(sample), [[0], ] * 100,axis=0)
#y = np.append([[1], ] * len(sample), [[0], ] * len(data_p),axis=0)
#print(X[0],y[0])

model = GaussianNB()
model.fit(X, y)
#print(model)
# make predictions
expected = y
predicted = model.predict(data_p)
print(list(set(predicted)))
array = []
for i, item in enumerate(predicted):
    if item==1:
       array.append(data[i, [0, 1]])
print(len(array))
matplt.visual_map(np.array(array).T, eq_sample)
# summarize the fit of the model
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))