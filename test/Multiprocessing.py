import time
import itertools
import multiprocessing
from utils.monitoring import show_short_info, show_full_cpuinfo

""" Sequential nested loop test """

def test(iter1 = 500, iter2 = 800, iter3 = 800, benchmark=25000000):   
    z = range(iter1)
    y = range(iter2)
    x = range(iter3)
    
    start = time.perf_counter()
    temp = time.time()
    
    total = iter1 * iter2 * iter3
    
    #Generate a list of tuples where each tuple is a combination of parameters.
    #The list will contain all possible combinations of parameters.
    paramlist = list(itertools.product(z, y, x))
    
    #Generate processes equal to the number of cores
    pool = multiprocessing.Pool()

    #Distribute the parameter sets evenly across the cores
    res = pool.map(func, paramlist)
    
    print(f'Duration: {time.perf_counter()-start:.2f} s elapsed.')

# A function which will process a tuple of parameters
def func(params):
    z = params[0]
    y = params[1]
    x = params[2]