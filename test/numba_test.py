import numpy as np
from numba import cuda, njit, prange

class M:
    def __init__(self):
        self.arr = np.zeros((17, 8025472),dtype=np.uint32)
        self.d_arr = None
        
@njit(parallel=True)
def test(x):
    n = x.shape[0]
    a = np.sin(x)
    b = np.cos(a * a)
    acc = 0
    for i in prange(n - 2):
        for j in prange(n - 1):
            acc += b[i] + b[j + 1]
    return acc

  

if __name__ == '__main__': 
    
    
    test(np.arange(10))

    test.parallel_diagnostics(level=4)      
    
    # ms = [M() for _ in range(26)]
    # for m in ms:
    #     m.d_arr = cuda.to_device(m.arr)
    #     # do whatever it is you want to do with m.d_arr here
    #     m.d_arr = None