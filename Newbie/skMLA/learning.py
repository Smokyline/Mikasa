import warnings

import numpy as np
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from skMLA.visual import visual_two_predict, create_3d_map
from skLn.tools import cross_v


def sk_learn(B, X, H, param, sampleAll, mp_s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        BH = np.append(B, H, axis=0)
        BX = np.append(B, X, axis=0)
        XH = np.append(X, H, axis=0)
        T = np.append(BX, H, axis=0)

        X_test = T[:, 2:]
        #X_test = BH[:, 2:]
        #y = np.append(np.append([[2], ] * len(B), [[0], ] * len(H), axis=0), [[1], ] * len(X), axis=0)
        y = np.append([[1], ] * len(BX), [[0], ] * len(H), axis=0)


        model_0 = GaussianNB()  # bayes
        model_1 = KNeighborsClassifier()
        model_2 = LogisticRegression(penalty='l1', tol=0.01)
        model_3 = DecisionTreeClassifier()
        model_4 = RandomForestClassifier(n_estimators=1)
        model_5 = ExtraTreesClassifier(n_estimators=1)  # !!!
        model_6 = AdaBoostClassifier(n_estimators=1, base_estimator=ExtraTreesClassifier(n_estimators=100))
        model_7 = GradientBoostingClassifier()
        model_8 = LinearSVC()
        model_9 = SVC(probability=True)
        model_10 = BernoulliNB()
        model_11 = SGDClassifier(loss="log", penalty="l2", shuffle=True)
        model_12 = LinearDiscriminantAnalysis()
        model_13 = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0001, store_covariances=False, tol=0.0001)
        model_14 = RadiusNeighborsClassifier()
        model_15 = MultinomialNB()

        model_set = [model_0, model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9,
                     model_10,
                     model_11, model_12, model_13, model_14, model_15]
        model_name = ['GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier',
                      'RandomForestClassifier',
                      'ExtraTreesClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'LinearSVC', 'SVC',
                      'BernoulliNB()',
                      'SGDClassifier', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis',
                      'RadiusNeighborsClassifier',
                      'MultinomialNB']

        alg_set = zip(model_name, model_set)
        #cross_v(alg_set, X_test, y, 'accuracy')

        #for i in range(4, 16):
        for i in [1]:
            model = model_set[i]
            model.fit(X_test, y)


            print('\n{} score {}'.format(model.__class__.__name__, model.score(X_test, y)))
            try:
                imp = zip(param, list(model.feature_importances_))
                print('\nfeature_importances')
                for p, i in sorted(imp, key=lambda x: x[1]):
                    print('%0.3f %s' % (i, p))
            except Exception as ex:
                print(ex)

            predicted_test = model.predict(X_test)

            #predicted_train = np.append(X_

            # print('param eff', str(model.feature_importances_))

            print('\nc_test', list(set(predicted_test)))
            #print('c_train', list(set(predicted_train)))

            X_test = T[np.where(predicted_test == 1)][:, :2]
            #X_train = grid[np.where(predicted_train == 1)][:, :2]

            print(len(X_test), 'xtest')
            #print(len(X_train), 'xtrain')

            # inB0_array = np.append(testInB0, trainInB0, axis=0)
            title = str(model.__class__.__name__)
            if len(X_test) > 0:
                pass
                #visual_two_predict(X_test, None, sampleAll, mp_s, title)