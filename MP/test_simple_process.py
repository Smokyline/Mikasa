import time

def power(name, x):
    for i in range(x):
        u = 10**10
    print(name, 'done')

time_start = int(round(time.time() * 1000))

for i, x in enumerate([10**9, 10**9, 10**9]):
    power(str(i+1),x)


time_stop = int(round(time.time() * 1000)) - time_start

print(time_stop)
