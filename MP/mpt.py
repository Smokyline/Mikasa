import time
import multiprocessing as mp


def power(q, name, x):
    for i in range(x):
        u = 10**10
    print(name, 'done')
    return q.put(u)

time_start = int(round(time.time() * 1000))

procs = []
Qs = []
if __name__ == '__main__':
    for i, x in enumerate([10**8 for i in range(32)]):
        q = mp.Queue()
        p = mp.Process(target=power, args=(q, str(i+1), x))
        #procs.append(p)
        Qs.append(q)
        #print('process %s start' % (i+1))
        p.start()
        #print(q.get())

print('\nprocesses join 8')
Qs = [q.get() for q in Qs]
print(Qs)





time_stop = int(round(time.time() * 1000)) - time_start

print(time_stop)