from multiprocessing import Pool

def f(x):
    a = x**10
    for i in range(a):
        a=a**10

if __name__ == '__main__':
    with Pool(2) as p:
        p.map(f, [10, 20, 30])
    with Pool(1) as p:
        p.map(f, [5, 7, 9])
