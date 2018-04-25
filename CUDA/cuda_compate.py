from numbapro import guvectorize
from numpy import arange, dot
import time

@guvectorize(['void(float64[:,:], float64[:,:], float64[:,:])'],
             '(m,n),(n,p)->(m,p)')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

def numpy_matmul(A, B):
    return dot(A, B).astype(float)


def test_spd(itr, foo, A, B):
    time_start = int(round(time.time() * 1000))
    for i in range(itr):
        foo(A, B)
    return int(round(time.time() * 1000)) - time_start  # ms


w = 100
A = arange(w**2).reshape(w, w)
B = arange(w**2).reshape(w, w)

itr = 1000
nb_speed = test_spd(itr, matmul, A, B)
np_speed = test_spd(itr, numpy_matmul, A, B)

print('numba speed:%i' % nb_speed)
print('numpy speed:%i' % np_speed)