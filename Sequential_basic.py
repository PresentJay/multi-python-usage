from urllib import request
from ast import literal_eval
import time, json

TRIAL_MAX = 100000000

# for loop computing basic test
if __name__ == '__main__':
    
    """ Sequential test """
    
    start = time.perf_counter()
    for i in range(1, TRIAL_MAX+1):
        if i%1000000 == 0:
            print(f'~ {i:d} trials : elapsed time {time.perf_counter()-start:.2f} s.')
    
    print(f'Duration: {time.perf_counter()-start:.2f} s elapsed.')