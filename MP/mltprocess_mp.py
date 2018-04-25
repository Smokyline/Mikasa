import multiprocessing as mp
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def foo(q):
    info('foo')
    q.put([[1],[2]])


if __name__ == '__main__':
    mp.set_start_method('spawn')
    info('main')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()