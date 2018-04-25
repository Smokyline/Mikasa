import numpy as np
from neuro_netwok.tools import nonlin, norm, check_error, calc_ni
import sys


def omega(samples, size, param):
    print('Ω start')
    np.random.seed(1)

    syn0 = 2 * np.random.random((len(param), len(samples))) - 1
    syn1 = 2 * np.random.random((len(samples), 1)) - 1

    array = np.array([np.empty((len(param), 1))])
    array_n = np.array([])
    eps = sys.float_info.epsilon


    y = np.array([[0], ] * len(samples))
    q = check_error(y, nonlin(np.dot(nonlin(np.dot(samples, syn0)), syn1)))

    it = 1
    while True:
        l0 = samples
        l1 = nonlin(np.dot(l0, syn0))
        l2 = nonlin(np.dot(l1, syn1))

        # как сильно мы ошиблись относительно нужной величины?
        #l2_error = y - l2
        l2_error = check_error(y,l2)
        mean_error = np.mean(np.abs(l2_error))

        q_new = ((size - 1) / size * q) + ((1 / size) * (l2_error ** 2))

        if (it % 10000) == 0:
            print('iteration', it)
            print("Error:" + str(mean_error))
            print('q error', str(np.mean(np.abs(q_new))))


        if np.allclose(q, q_new):
        #if it>=30000:
            print('q error', str(np.mean(np.abs(q_new))))
            print("Error:" + str(np.mean(np.abs(l2_error))))
            break
        else:
            q = q_new

        # в какую сторону нужно двигаться?
        # если мы были уверены в предсказании, то сильно менять его не надо
        l2_delta = l2_error * nonlin(l2, deriv=True)

        # как сильно значения l1 влияют на ошибки в l2?
        l1_error = l2_delta.dot(syn1.T)

        # в каком направлении нужно двигаться, чтобы прийти к l1?
        # если мы были уверены в предсказании, то сильно менять его не надо
        l1_delta = l1_error * nonlin(l1, deriv=True)

        #ni_0 = 0.01
        #ni = 1/len(param)**2
        ni = 0.1

        syn0 -= l0.T.dot(l1_delta)
        syn1 -= ni*l1.T.dot(l2_delta)

        it += 1


    print(it, 'iter')

    print('Ω done \n')
    #np.delete(array, 0, 0)
    #matplt.visual_array(array, param)
    #matplt.visual_num(array_n)
    #matplt.draw_plot()
    return syn0, syn1


def recognition(data, syn0, syn1, index_param):
    #print('recognition eq, omega =', omegaW)
    u_data = data[:, index_param]
    y = np.array([[0], ] * len(u_data))

    arrayR = []
    l0 = u_data
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    #print(l1, 'l1\n')
    error = check_error(y, l2)

    for i in range(len(data)):
        #arrayR.append(np.append(l1[i],data[i], axis=0).tolist())
        arrayR.append(np.append(error[i], data[i], axis=0).tolist())
    arrayR = np.array(arrayR)
    #print(len(arrayR), ' grid points \n')

    return arrayR