from numba import cuda, float32, njit, void
from numba.cuda import libdevice
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



# Implementation using the standard trigonometric functions
@cuda.jit(void(float32[::1], float32[::1], float32[::1]))
def trig_functions(r, x, y):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = math.sin(x[i]) * math.cos(y[i]) + math.tan(x[i] + y[i])
        

# Implementation using the fast trigonometric functions
@cuda.jit(void(float32[::1], float32[::1], float32[::1]))
def fast_trig_functions(r, x, y):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = libdevice.fast_sinf(x[i]) * libdevice.fast_cosf(y[i]) + libdevice.fast_tanf(x[i] + y[i])


def overhead_test():
    # A small array
    arr = cuda.managed_array(100, dtype=np.float64)
    
    # If you have more than, or a lot less than, 8 of GPU RAM then edit this:
    GB = 4
    n_elements = (GB + 1) * (1024 * 1024 * 1024)

    # Create a very large array
    big_arr = cuda.managed_array(n_elements, dtype=np.uint8)
    initialize_array[1024, 1024](big_arr)

    check(big_arr)
    
    arr = cuda.device_array(100, dtype=np.float32)
    
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
    
    start = time.perf_counter()
    some_kernel_5[1, 1](arr, arr, arr, arr)
    print(f'{time.perf_counter()-start:.2f} seconds elapsed.')
    
    
def LibdeviceTest():
    # Create some random input
    N = 2000000
    np.random.seed(1)
    x = np.random.random(N).astype(np.float32)
    y = np.random.random(N).astype(np.float32)

    # Copy input to the device and allocate space for output
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    r_math = cuda.device_array_like(x)
    r_libdevice = cuda.device_array_like(x)

    n_runs = 100000
    n_threads = 1024
    n_blocks = math.ceil(N / n_threads)

    # Run and time the normal version
    start_math = time.perf_counter()
    for i in range(n_runs):
        trig_functions[n_blocks, n_threads](r_math, d_x, d_y)
    end_math = time.perf_counter()
    cuda.synchronize()

    # Run and time the version using fast trig functions
    start_libdevice = time.perf_counter()
    for i in range(n_runs):
        fast_trig_functions[n_blocks, n_threads](r_libdevice, d_x, d_y)
    cuda.synchronize()
    end_libdevice = time.perf_counter()


    # Note that the fast versions of the functions sacrifice accuracy for speed,
    # so a lower-than-default relative tolerance is required for this sanity check.
    np.testing.assert_allclose(r_math.copy_to_host(), r_libdevice.copy_to_host(), rtol=1.0e-2)

    # Note that timings will be fairly similar for this example, as the execution time will be
    # dominated by the kernel launch time.
    print(f"Standard version time {(end_math - start_math):.4f}s")
    print(f"Libdevice version time {(end_libdevice - start_libdevice):.4f}s")
    
    

@cuda.jit
def subtract_example(x, values):
    i = cuda.grid(1)
    cuda.atomic.sub(x, 0, values[i])
    

@cuda.jit
def cuda_frexp_ldexp(x, y):    
    i = cuda.grid(1)
    if i < len(x):
        fractional, exponent = math.frexp(x[i])
        y[i] = math.ldexp(fractional, exponent)
        

@cuda.jit
def complex_power(r, x, y):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = x[i] ** y[i]
        

def random_complex(n_values):
    # "Generate an array of random complex values"
    real = np.random.random(n_values)
    imag = np.random.random(n_values)
    return real + imag * 1j
   

# subtracts several values from an element of an array with every thread contending on the same location:
def automic_substract_test():
    initial = 12345.0
    n_blocks = 4
    n_threads = 32
    n_values = n_blocks * n_threads
    values = np.arange(n_values, dtype=np.float32)

    x = np.zeros(1, dtype=np.float32)
    x[0] = initial

    start = time.perf_counter()
    
    # for i in range(500):
    subtract_example[n_blocks, n_threads](x, values)
        
    print(f'\n{time.perf_counter()-start:.2f} seconds elapsed.')
        
    # Floating point subtraction is not associative - the order in which subtractions
    # occur can cause a slight variation, so we use assert_allclose instead of checking
    # for exact equality.
    np.testing.assert_allclose(x, [initial - np.sum(values)])
    print(f"Result: {x[0]}\n")
    
    
# Math library functions
def math_test():
    np.random.seed(2)
    n_values = 16384
    n_threads = 256
    n_blocks = n_values // n_threads

    values = np.random.random(16384).astype(np.float32)
    results = np.zeros_like(values)
    
    start = time.perf_counter()
    
    cuda_frexp_ldexp[n_blocks, n_threads](values, results)

    print(f'\n{time.perf_counter()-start:.2f} seconds elapsed.')
    
    # Sanity check
    np.testing.assert_equal(values, results)

    # Print the first few values and results
    print(values[:10])
    print(results[:10])
    

def power_complex_test():
    np.random.seed(3)
    n_values = 98304
    n_threads = 1024
    n_blocks = n_values // n_threads

    start = time.perf_counter()
    x = random_complex(n_values)
    y = random_complex(n_values)
    r = np.zeros_like(x)

    for i in range(10000):
        complex_power[n_blocks, n_threads](r, x, y)

    print(f'\n{time.perf_counter()-start:.2f} seconds elapsed.')
    
    # Sanity check
    np.testing.assert_allclose(r, x ** y)

    # Print the first few results and the same computed on the CPU for comparison
    print(r[:10])
    print(x[:10] ** y[:10])