import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from skLn.tools import cross_v
from skLn.visual import sk_visual, visual_ln


def sk_learning(B, X, H, sampleAll, mp_s, r, param):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # visual_ln(B, X, H, sampleAll, mp_s)

        BH = np.append(B, H, axis=0)
        BX = np.append(B, X, axis=0)
        T = np.append(BX, H, axis=0)
        bh_count = np.zeros((len(BH), 1))
        x_count = np.zeros((len(X), 1))
        X_test = BH[:, 2:]
        X_train = X[:, 2:]
        y = np.append([[1], ] * len(B), [[0], ] * len(H), axis=0)

        model_0 = GaussianNB()  # bayes
        model_1 = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        model_2 = LogisticRegression(penalty='l2', max_iter=1,random_state=1)
        model_3 = DecisionTreeClassifier(max_depth=10, max_features=len(X_test[0]),random_state=1)
        model_4 = RandomForestClassifier(max_depth=10, n_estimators=1, max_features=len(X_test[0]),random_state=6)
        model_5 = ExtraTreesClassifier(n_estimators=1, max_depth=50, max_features=len(X_test[0]),random_state=1)  # !!!
        model_6 = AdaBoostClassifier(n_estimators=1, base_estimator=model_4,random_state=1)
        #model_7 = GradientBoostingClassifier(loss='exponential', learning_rate=0.085,subsample=0.02, n_estimators=73,random_state=3, max_depth=5)
        #model_7 = GradientBoostingClassifier(n_estimators=10, learning_rate=1000, random_state=3)
        model_7 = GradientBoostingClassifier(n_estimators=10, learning_rate=600, random_state=3)
        model_8 = LinearSVC(C=1)
        model_9 = SVC(probability=True, kernel='poly', random_state=1, C=1)
        model_10 = BernoulliNB()
        #model_11 = SGDClassifier(loss="perceptron", penalty="l2", shuffle=True, random_state=3, learning_rate='optimal')
        model_11 = SGDClassifier(loss="perceptron", penalty="l2", shuffle=True, random_state=6, learning_rate='optimal')
        model_12 = LinearDiscriminantAnalysis()
        model_13 = QuadraticDiscriminantAnalysis()
        model_14 = RadiusNeighborsClassifier(weights='distance', p=2, algorithm='auto')
        model_15 = MultinomialNB()

        model_set = [model_0, model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9,
                     model_10,
                     model_11, model_12, model_13, model_14, model_15]
        model_set2 = [model_0, model_1, model_3, model_4, model_5, model_6, model_7, model_8,
                     model_10, model_11, model_12, model_13, model_14, model_15]
        model_set3 = [model_2, model_7, model_9, model_11]


        names = [x.__class__.__name__ for x in model_set]
        alg_set = zip(names, model_set)
        #cross_v(alg_set, X_test, y, 'accuracy')

        def check_b(i, B):
            for b in B:
                if i[0] == b[0] and i[1] == b[1]:
                    return True

        #for i in range(0, 16):
        for i in range(len(model_set3)):
        #for i in [0]:
            model = model_set3[i]
            model.fit(X_test, y)
            print('\n{} score {}'.format(model.__class__.__name__, model.score(X_test, y)))
            try:
                imp = zip(param, list(model.feature_importances_))
                print('\nfeature_importances')
                for p, i in sorted(imp, key=lambda x: x[1]):
                    print('%0.4f %s' % (i, p))
            except Exception as ex:
                print(ex)
            predicted_test = model.predict(X_test)
            predicted_train = model.predict(X_train)

            if model.__class__.__name__ in ['SVC', 'LogisticRegression']:
                proba_test = model.predict_proba(X_test)
                proba_train = model.predict_proba(X_train)


                if model.__class__.__name__ == 'SVC':
                    u = -190
                else:
                    u = -90
                #print(proba_test)
                mean_w = np.mean(np.power(proba_test[:len(B), 0], u)) ** (1 / u)
                #print(np.mean(proba_test[:len(B), 0]),'meanB')
                #print(np.mean(proba_test[len(B):, 0]),'meanH')
                #mean_w = np.min(proba_test[:len(B), 0])
                #print(mean_w, 'mean')
                index_test = np.where(proba_test[:, 0] >= mean_w)
                index_train = np.where(proba_train[:, 0] >= mean_w)
            else:
                index_test = np.where(predicted_test == 1)
                index_train = np.where(predicted_train == 1)
            #print(metrics.classification_report(y, predicted_test))
            #print(metrics.confusion_matrix(y, predicted_test))
            X_test_B = BH[index_test][:, :2]
            bh_count[index_test] += 1
            X_train_B = X[index_train][:, :2]
            x_count[index_train] += 1
            X_H = np.append(BH[np.where(predicted_test == 0)][:, :2], X[np.where(predicted_train == 0)][:, :2], axis=0)
            testBinB0 = []
            testBinH0 = []
            trainB = X_train_B




            for i in X_test_B:
                if check_b(i, B):
                    testBinB0.append(i)

                else:
                    testBinH0.append(i)


            # print('B in X_test {}/{}'.format(len(X_test_B), len(X_test)))
            b_inB0 = 'B in B0  {}/{}'.format(len(testBinB0), len(B))
            b_inH0 = 'B in H0 {}/{}'.format(len(testBinH0), len(H))
            b_inX0 = 'B in X {}/{}'.format(len(trainB), len(X))
            legend = [b_inB0, b_inH0, b_inX0]
            print(b_inB0)
            print(b_inH0)
            print(b_inX0)
            print('Total B', len(X_test_B) + len(X_train_B))

            # inB0_array = np.append(testInB0, trainInB0, axis=0)
            title = str(model.__class__.__name__)
            legend = [b_inB0, b_inH0, b_inX0, '']

            #if len(testBinH0)+len(testBinB0)+len(trainB) > 0:
             #   sk_visual(np.asarray(testBinB0), np.asarray(testBinH0), np.asarray(trainB), None, X_H, sampleAll, mp_s, r,
              #        title, legend, visual=False, TotalB=True)

        cls_a = []  # 1-2
        cls_b = []  # 3-4
        cls_c = []  # 5-7
        cls_d = []  # 8-10
        cls_h = []  # <1

        def add_count(count, i):
            if count <3:
                cls_h.append(i)
            elif 1 <= count <2:
                cls_a.append(i[:2])
            elif 3 <= count < 4:
                cls_b.append(i[:2])
            elif 3 <= count < 4:
                cls_c.append(i[:2])
            elif 4 <= count:
                cls_d.append(i[:2])

        for n, i in enumerate(BH):
            count = bh_count[n]
            #add_count(count, i[:2])
            if count <3:
                cls_h.append(i[:2])
            else:
                cls_b.append(i[:2])
        for n, i in enumerate(X):
            count = x_count[n]
            if count == 0:
                cls_h.append(i[:2])
            else:
                cls_b.append(i[:2])

        print(len(cls_b)+len(cls_c)+len(cls_d), 'len B')


        '''x_a = []
        b_a = []
        h_a = []
        for a in [cls_b, cls_c, cls_d]:
            for i in a:
                if check_b(i, B):
                    b_a.append(i)
                if check_b(i, H):
                    h_a.append(i)
                if check_b(i, X):
                    x_a.append(i)


        print(len(b_a))
        print(len(h_a))
        print(len(x_a))'''

                    # np.asarray(trainB)
        legend = ['1','2','3','4']
        print(len(cls_b)+len(cls_c)+len(cls_d))
        if len(cls_a)+len(cls_b)+len(cls_c)+len(cls_d) > 0:
            sk_visual(None, np.asarray(cls_b), np.asarray(cls_c), np.asarray(cls_d), np.asarray(cls_h), sampleAll, mp_s, r,
                  '3.0 ≤ М', legend, visual=True, TotalB=True)
           # np.asarray(cls_a), np.asarray(cls_b), np.asarray(cls_c), np.asarray(cls_d)

