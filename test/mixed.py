import math
import numpy as np

from utils import monitoring
from numba import cuda, jit, prange, vectorize, guvectorize
from sys import getsizeof
from multiprocessing import cpu_count, Pool
from timeit import default_timer as timer


def naive_target_function(matrixesA, matrixesB):
    result_matrix = []
    for i in range(len(matrixesA)):
        result_matrix.append(np.matmul(matrixesA[i], matrixesB[i]))
        if (i+1)%(len(matrixesA)/10) ==0:
            monitoring.show_short_info()
    result_matrix
    return result_matrix


def multiprocessing_target_function(matrixesA, matrixesB):
    
    # https://github.com/uqfoundation/pathos/issues/150 
    # There is a bug in Python C core code that prevents data responses bigger than 2GB return correctly to the main thread.
    # you need to either split the data into smaller chunks as suggested in the previous answer or not use multiprocessing for this function
    
    # 그리고 객체를 복사해서 Pool에 넣다보니 더 느려짐
    # CPU 코어를 더 잘 쓰기 위한 멀티프로세싱
    
    cores = int(cpu_count())
    
    matrixesA_split = np.array_split(matrixesA, cores)
    starmap_args = [(split, matrixesB) for split in matrixesA_split]
    with Pool(cores) as p:
        data = np.concatenate(p.starmap(naive_target_function, starmap_args))
        print(f'calculation done with {cores} of cores.')
        return data

            
def test():
    ITERATION = 500
    AXIS_SIZE = 400
    
    matrixesA = []
    matrixesB = []
    
    for i in range(ITERATION):
        matrixesA.append(np.random.rand(AXIS_SIZE, AXIS_SIZE))
    for i in range(ITERATION):
        matrixesB.append(np.random.rand(AXIS_SIZE, AXIS_SIZE))
    
    matrixesA = np.array(matrixesA)
    matrixesB = np.array(matrixesB)
    
    print(f'matrixes are loades.\n')
    
    
    # naive
    start = timer()
    result1 = naive_target_function(matrixesA, matrixesB)
    print(f'with naive CPU test: {timer()-start:.2f}s elapsed\n')
    print(result1[0].shape, result1[0].size*len(result1), "computation")
    
    
    # multiprocessing
    start = timer()
    result2 = multiprocessing_target_function(matrixesA, matrixesB)
    print(f'with naive CPU test : {timer()-start:.2f}s elapsed\n')
    print(result2[0].shape, result2[0].size*len(result2), "computation")
   
    
    # Numba-CPU
    
    
    monitoring.show_short_info()