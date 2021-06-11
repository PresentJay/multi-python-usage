from numba import cuda, jit
import numpy as np

from utils import monitoring
from timeit import default_timer as timer

# To run on CPU
def func(a, b):
    for i in range(1000):
        for j in range(500):
            for k in range(500):
                a[i][j][k] -= b[i][j][k]
        
        
# To run on GPU
@jit
def func2(x,y):
    return x + y

@jit
def func3(x,y):
    return x * y


@jit(forceobj=True)
def func4(y):
    z = 1
    for x in y:
        z += func5(x)
    return z
    
def func5(x):
    return x**2

@jit(forceobj=True)
def init4():
    fake_Arr = []
    for i in range(5000):
        fake_Arr.append(np.random.rand(3, 500, 500))
        if i%1000==0:
            monitoring.show_short_info()
    return reversed(fake_Arr)


# using simple cpu test
@jit(nopython=True)
def cpu_F(x, y, z):
    t1 = np.matmul(x, y)
    t2 = np.matmul(t1, z)
    return t2


def test():
    a = np.ones([1000, 500, 500], dtype = np.float64)
    b = np.ones([1000, 500, 500], dtype = np.float64)
    # start = timer()
    # func(a)
    # print(f'without GPU: {timer()-start:.2f} s elapsed')
    
    start = timer()
    c = func2(a, b)
    print(c.shape, c.size, "computated")
    d = func3(c, c)
    print(d.shape, c.size, "computated")
    print(d[0][0][0])
    print(f'with CPU test1: {timer()-start:.2f}s elapsed\n\n')
    
    monitoring.get_gpu_info()
    
    start = timer()
    fake_Arr = init4()
    e = func4(fake_Arr)
    print(e.shape, e.size * 5000, "computated")
    print(e[0][0][0])
    print(f'with CPU test2: {timer()-start:.2f}s elapsed\n\n')
    

    monitoring.get_gpu_info()
    
    cuda.profile_stop()