import threading
import time

def writer(x, event_for_wait, event_for_set):

    event_for_wait.wait() # wait for event
    event_for_wait.clear() # clean event for future
    print(x)
    event_for_set.set() # set event for neighbor thread

def power(name, x):
    for i in range(x):
        u = 10**10
    print(name, 'done')

time_start = int(round(time.time() * 1000))
# init events
e1 = threading.Event()
e2 = threading.Event()
e3 = threading.Event()

foo = power
# init threads
#foo = writer
#t1 = threading.Thread(target=writer, args=(0, e1, e2))


t1 = threading.Thread(target=foo, args=('1', 10**9,))
t2 = threading.Thread(target=foo, args=('2', 10**9,))
t3 = threading.Thread(target=foo, args=('3', 10**9,))

t1.setDaemon(True)
t2.setDaemon(True)
t3.setDaemon(True)

# start threads
t1.start()
t2.start()
t3.start()

e1.set() # initiate the first event

# join threads to the main thread
t1.join()
t2.join()
t3.join()

#t1.kill_received = True
#t2.kill_received = True
#t3.kill_received = True

time_stop = int(round(time.time() * 1000)) - time_start

print(time_stop)
