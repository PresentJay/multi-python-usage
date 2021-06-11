import time
from utils.monitoring import show_short_info, show_full_cpuinfo

""" Sequential nested loop test """

def test(iter1 = 500, iter2 = 800, iter3 = 800, benchmark=25000000):   
    start = time.perf_counter()
    temp = time.time()
    cnt = 0
    
    for z in range(iter1):
        for y in range(iter2):
            for x in range(iter3):
                cnt+=1
                if cnt%benchmark == 0:
                    temp = time.time() - temp
                    print(f'{(cnt / (iter1 * iter2 * iter3) * 100):05.2f}% progress, ', end='')
                    show_short_info()
                    
                    show_full_cpuinfo()
                    
                    temp = time.time()
    
    print(f'Duration: {time.perf_counter()-start:.2f} s elapsed.')
    
