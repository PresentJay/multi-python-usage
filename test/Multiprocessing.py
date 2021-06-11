import time
import itertools
import multiprocessing
from utils.monitoring import show_short_info

""" Sequential nested loop test """

def test(iter1 = 1000, iter2 = 800, iter3 = 800, benchmark=20000000):   
    
    z = range(iter1)
    y = range(iter2)
    x = range(iter3)
    
    start = time.perf_counter()
    temp = time.time()
    cnt = 0
    
    
    
    for z in range(iter1):
        for y in range(iter2):
            for x in range(iter3):
                cnt+=1
                if cnt%benchmark == 0:
                    temp = time.time() - temp
                    print(f'{(cnt / (iter1 * iter2 * iter3) * 100):.2f}% progress : elapsed time {time.perf_counter()-start:.2f} s, ', end='')
                    show_short_info()
                    
                    temp = time.time()
    
    print(f'Duration: {time.perf_counter()-start:.2f} s elapsed.')
    
