import numpy as np
from neuro_netwok import matplt
from neuro_netwok import matplt, reader
from neuro_netwok.learning import recognition, omega
from neuro_netwok.tools import norm
from neuro_netwok import kmeans2, kmeans

from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation, svm
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pylab as pl
from sklearn.ensemble import GradientBoostingClassifier
import warnings

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB


def sk_learning(sample, clust, data_grid, index_param, map_size,param,sk_param):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sample_param = sample[:, index_param]


        false_data_param = np.empty((1, len(index_param)))
        for i in range(0,4):
            false_data_param = np.append(false_data_param,clust[i][:,index_param],axis=0)
        #false_data_param = clust[3][:, index_param]
        data_grid_param = data_grid[:, index_param]


        X = np.append(sample_param, false_data_param, axis=0)
        # X = np.append(sample_p, data_p,axis=0)
        y = np.append([[1], ] * len(sample), [[0], ] * len(false_data_param), axis=0)
        # y = np.append([[1], ] * len(sample), [[0], ] * len(data_p),axis=0)
        # print(X[0],y[0])

        model_1 = GaussianNB()  # bayes
        model_2 = KNeighborsClassifier()
        model_3 = LogisticRegression()
        # model = SVC()
        model_4 = DecisionTreeClassifier()
        model_5 = RandomForestClassifier()
        model_6 = ExtraTreesClassifier()  # !!!
        model_7 = AdaBoostClassifier()
        model_8 = GradientBoostingClassifier()
        model_9 = LinearSVC(random_state=0)
        """
        eclf = VotingClassifier(estimators=[('gnb', model_1), ('knc', model_2), ('lrg', model_3), ('dtc', model_4),
                                            ('rfc', model_5), ('etc', model_6), ('abc', model_7), ('gbc', model_8)], voting='soft')
        cv = cross_validation.ShuffleSplit(len(y), n_iter=5, test_size=0.1, random_state=0)
        for clf, label in zip([model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, eclf],
                              ['GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier',
                               'RandomForestClassifier', 'ExtraTreesClassifier',
                               'AdaBoostClassifier', 'GradientBoostingClassifier']):
            scores = cross_validation.cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        """"""
        for model_type, label in zip([model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8],
                              ['GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier',
                               'RandomForestClassifier', 'ExtraTreesClassifier',
                               'AdaBoostClassifier', 'GradientBoostingClassifier']):
            try:
                model = model_type
                model.fit(X, y.ravel())
                predicted = model.predict(data_grid_param)

                print('check classifier', list(set(predicted)))
                array = []
                for i, item in enumerate(predicted):
                    if item == 1:
                        array.append(data_grid[i, [0, 1]])

                title = 'mks' + str(param) + ' sk_'+label + str(sk_param)
                matplt.visual_sk_map(np.array(array).T, sample.T, param, map_size, sk_param, title)
            except:
                continue

        """
        model = model_8
        model.fit(X, y.ravel())
        # print(model)
        # make predictions
        print('param eff', str(model.feature_importances_))
        # expected = y
        predicted = model.predict(data_grid_param)

        #print('classifier', list(set(predicted)))
        array = []
        for i, item in enumerate(predicted):
            if item == 1:
                array.append(data_grid[i, [0, 1]])

        title = 'mks' + str(param) + ' skLn' + str(sk_param)
        matplt.visual_sk_map(np.array(array).T, sample.T, param,map_size,sk_param,title)



