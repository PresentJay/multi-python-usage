import numpy as np

from utils import monitoring
from numba import cuda, jit
from multiprocessing import cpu_count, Pool
from timeit import default_timer as timer


def naive_target_function(matrixesA, matrixesB):
    result_matrix = []
    for i in range(len(matrixesA)):
        result_matrix.append(np.matmul(matrixesA[i], matrixesB[i]))
        if (i+1)%(len(matrixesA)/10) ==0:
            monitoring.show_short_info()
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
    
    
@jit(forceobj=True)
def matmul_append(matrixesA, matrixesB, reslist):
    result = np.matmul(matrixesA, matrixesB)
    reslist.append(result)
    return reslist


@jit(parallel=True, forceobj=True)
def NumbaCPU_target_function(matrixesA, matrixesB, result_matrix=[]):
    for i in range(len(matrixesA)):
        result_matrix = matmul_append(matrixesA, matrixesB, result_matrix)
        if (i+1)%(len(matrixesA)/10) ==0:
            monitoring.show_short_info()
            
    return result_matrix


            
def test():
    ITERATION = 200
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
    print('< start naive CPU >')
    result1 = naive_target_function(matrixesA, matrixesB)
    print(f'with naive CPU test: {timer()-start:.2f}s elapsed')
    print(result1[0].shape, '*', len(result1), "computation\n")
    
    
    # multiprocessing
    start = timer()
    print('< start multiprocessing CPU >')
    result2 = multiprocessing_target_function(matrixesA, matrixesB)
    print(f'with multiprocessing CPU test : {timer()-start:.2f}s elapsed')
    print(result2[0].shape, '*', len(result2), "computation\n")
   
    
    # Numba-CPU
    start = timer()
    print('< start Numba-CPU >')
    result3 = NumbaCPU_target_function(matrixesA, matrixesB)
    print(f'with Numba-CPU test : {timer()-start:.2f}s elapsed')
    print(result3[0].shape, "computation\n")
    
    monitoring.show_short_info()