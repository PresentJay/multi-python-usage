from numba import cuda, float32, njit, void
import math
import numpy as np
import time

from utils import monitoring


@cuda.jit
def initialize_array(x):
    start, stride = cuda.grid(1), cuda.gridsize(1)
    for i in range(start, len(x), stride):
        x[i] = 0xAB
        
        
@njit
def check(x):
    difference = False
    for i in range(len(x)):
        if x[i] != 0xAB:
            difference = True
    
    if difference:
        print("Difference detected!")
    else:
        print("All values as expected!")
        
        
@cuda.jit('void()')
def some_kernel_1():
    return

@cuda.jit('void(float32[:])')
def some_kernel_2(arr1):
    return

@cuda.jit('void(float32[:],float32[:])')
def some_kernel_3(arr1,arr2):
    return

@cuda.jit('void(float32[:],float32[:],float32[:])')
def some_kernel_4(arr1,arr2,arr3):
    return

@cuda.jit('void(float32[:],float32[:],float32[:],float32[:])')
def some_kernel_5(arr1,arr2,arr3,arr4):
    return


def overhead_test():
    # A small array
    arr = cuda.managed_array(100, dtype=np.float64)
    
    # If you have more than, or a lot less than, 8 of GPU RAM then edit this:
    GB = 5

    n_elements = (GB + 1) * (1024 * 1024 * 1024)

    # Create a very large array
    big_arr = cuda.managed_array(n_elements, dtype=np.uint8)
    initialize_array[1024, 1024](big_arr)

    check(big_arr)
    
    
    arr = cuda.device_array(10000, dtype=np.float32)
    
    start = time.perf_counter()
    some_kernel_1[1, 1]()
    print(f'{time.perf_counter()-start:.2f} seconds elapsed.')
    
    start = time.perf_counter()
    some_kernel_2[1, 1](arr)
    print(f'{time.perf_counter()-start:.2f} seconds elapsed.')
    
    start = time.perf_counter()
    some_kernel_3[1, 1](arr, arr)
    print(f'{time.perf_counter()-start:.2f} seconds elapsed.')
    
    start = time.perf_counter()
    some_kernel_4[1, 1](arr, arr, arr)
    print(f'{time.perf_counter()-start:.2f} seconds elapsed.')
    
    # overheaded below
    
    # start = time.perf_counter()
    # some_kernel_5[1, 1](arr, arr, arr, arr)
    # print(f'{time.perf_counter()-start:.2f} seconds elapsed.')