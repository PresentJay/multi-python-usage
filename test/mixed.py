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
        if (i+1)%1000 ==0:
            monitoring.show_short_info()
    return result_matrix
            
            
def test():
    ITERATION = 5000
    AXIS_SIZE = 600
    
    matrixesA = []
    matrixesB = []
    
    for i in range(ITERATION):
        matrixesA.append(np.random.rand(AXIS_SIZE, AXIS_SIZE))
    for i in range(ITERATION):
        matrixesB.append(np.random.rand(AXIS_SIZE, AXIS_SIZE))
    
    matrixesA = np.array(matrixesA)
    matrixesB = np.array(matrixesB)
    
    print(f'matrixes are loades.\n')
    monitoring.show_info()
    
    start = timer()
    naive_target_function(matrixesA, matrixesB)
    print(f'with naive CPU test: {timer()-start:.2f}s elapsed\n\n')
    
    monitoring.show_short_info()
    
    
    