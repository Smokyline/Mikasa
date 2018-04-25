import numpy as np
import time

a = np.arange(10000).astype(float).reshape(1000, 10)
b = np.arange(10000).astype(float).reshape(10, 1000)

time_start = int(round(time.time() * 1000))





c = np.matmul(a, b)
print(c)

time_stop = int(round(time.time() * 1000)) - time_start
print(time_stop)