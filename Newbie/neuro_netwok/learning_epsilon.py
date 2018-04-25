from neuro_netwok import matplt, reader
import numpy as np
from neuro_netwok.tools import nonlin, norm, check_error, calc_ni
import sys


def omega(samples, size, param):
    print('Ω start')
    #print(np.mean(simples,axis=0))
    X = []
    directory = '/Users/Smoky/Documents/workspace/resourses/csv/'
    np.random.seed(1)
    # np.random.shuffle(simples)
    #x1 = norm(reader.read_csv(directory + "kvzSimple5-55.csv",param)).T
    #x2 = norm(reader.read_csv(directory + "kvzSimple55-6.csv",param)).T
    #x3 = norm(reader.read_csv(directory + "kvzSimple6-7.csv",param)).T

    #pers = [50, 40, 10]
    pers = [100]
    # x_set = [int(pers[0] * size / 100), int(pers[1] * size / 100), int(pers[2] * size / 100)]
    # X.append(simples[:, param][:x_set[0]])
    # X.append(simples[:, param][x_set[0]:x_set[0] + x_set[1]])
    # X.append(simples[:, param][size - x_set[2]:])
    X.append(samples)

    #X.append(x1[:, param])
    #X.append(x2[:, param])
    #X.append(x3[:, param])

    syn0 = 2 * np.random.random((len(param), 1)) - 1
    #syn0 = np.random.random((len(param), 1))

    array = np.array([np.empty((len(param), 1))])
    array_n = np.array([])
    eps = sys.float_info.epsilon
    l1_s = None
    ni = 0.001
    final_error = None
    it = 1
    for i in range(len(pers)):
        y = np.array([[1], ] * len(X[i]))

        #q = (y - nonlin(np.dot(X[i], syn0)))
        q = check_error(y, nonlin(np.dot(X[i], syn0)))

        while True:

            # прямое распространение
            l0 = X[i]
            l1 = l0.T * syn0
            #l2 = np.sum(l1, axis=1).reshape((len(l0), 1))
            l2 = nonlin(np.sum(l1, axis=0).reshape(len(l0), 1))
            #print(l2, 'l2\n')

            # насколько мы ошиблись?
            l1_error = check_error(y, l2)
            #print(l1_error, 'l1_e\n')
            # l1_error = y - l1

            error = np.mean(np.abs(l1_error))
            q_new = ((size - 1) / size * q) + ((1 / size) * (l1_error ** 2))
            #diff =np.mean(abs(q-q_new))
            if np.allclose(q, q_new):
            #if abs(old_error-error)<eps:
            #if error<0.20:
            #if it>=40000:
                #if diff<eps:
                # if np.array_equal(q, q.dot((len(l0) - 1) / len(l0)) + (1 / len(l0) * error)):
                # if it>15000:
                print(error, 'error')
                #print(syn0)
                #print(l2.T)
                l1_s = l1
                final_error = error
                break
            else:
                q = q_new
            # перемножим это с наклоном сигмоиды
            # на основе значений в l1
            l1_delta = l1_error * nonlin(l2, True)  # !!!

            #l1_delta = nonlin(l1_error) * nonlin(l1, True)


            #l1_delta = nonlin(l1_error, True)*nonlin(l1, True)

            # обновим веса
            #syn0 += 0.01 * np.dot(l0.T, l1_delta)  # !!!
            #print(l1_delta)


            #ni = (1/it)**2
            #ni=1
            l2_delta = np.dot(l0.T, l1_delta)
            #l2_delta = np.sum(l0.T*l1_delta.T,axis=1)

            if it == 1 or (it in [100,1000,10000,25000,60000,100000,150000,250000]):
                ni = calc_ni(syn0, l2_delta, l0)
                #ni = 0.0001
                print(ni, 'ni')
                print(error, 'error')


            syn0 +=ni*l2_delta
            #syn0 -=ni*(l0.T* l1_delta)
            #syn0 -= ni * ()

            #array_n = np.append(array_n, np.mean(q_new))
            #array_n = np.append(array_n, error)
            if it % 1000 == 0:
                #ni = calc_ni(syn0,l2_delta , l0.T, y)
                pass
                #array = np.append(array, [syn0], axis=0)
            if it % 10000 == 0:
                print(it, 'omega iteration')
                print(error, 'error')
            it += 1


    print(it, 'iter')

    print('Ω done \n')
    #np.delete(array, 0, 0)
    #matplt.visual_array(array, param)
    #matplt.visual_error(array_n)
    #matplt.draw_plot()
    return final_error, syn0, l1_s


def recognition(data, syn0, l1_s, index_param):
    print('recognition eq')

    l0 = data[:, index_param]
    l1 = (l0.T * syn0).T
    #y = np.array([[1], ] * len(l0))
    #print(l1_s)
    l1_s=l1_s.T

    arrayR = []

    error_l0_array = []

    #print(l1_s[0], 'l1s[0]\n')
    for i in l1:
        #error_l0_array.append(np.mean(check_error(l1_s, i)))
        error_l0_array.append(np.mean(np.abs(l1_s - i)))
    #l1_error = check_error(l1, l1_s)
    #print(error_l0_array[0])
    #print(l1_error,'error')
    #print(l1[0], 'l[0]')
    #print(l1[134721], 'l[134721]')
    #print(l1[80135], 'hi l1')
    #print(l1_error[0], 'er[0]')
    #print(l1_error[80135], 'hi er')
    #l=0
    for i in range(len(data)):
        #arrayR.append(np.append(l1[i],data[i], axis=0).tolist())
        #arrayR.append(np.append(l1_error[i], data[i], axis=0).tolist())
        arrayR.append(np.append([error_l0_array[i]], data[i], axis=0).tolist())
    arrayR = np.array(arrayR)
    #print(len(arrayR), ' grid points \n')

    return arrayR
