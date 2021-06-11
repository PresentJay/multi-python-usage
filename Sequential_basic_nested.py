from urllib import request
from ast import literal_eval
import time, json

TRIAL_MAX_NESTED_1 = 10000
TRIAL_MAX_NESTED_2 = 10000

# for loop computing basic test
if __name__ == '__main__':
    
    """ Sequential nested test """
    
    start = time.perf_counter()
    cnt = 0
    for x in range(TRIAL_MAX_NESTED_1):
        for y in range(TRIAL_MAX_NESTED_2):
            cnt += 1
            if cnt%10000000 == 0:
                print(f'{cnt:d}/{TRIAL_MAX_NESTED_1 * TRIAL_MAX_NESTED_2} trials . . .')
    
    print(f'Duration: {time.perf_counter()-start:.2f} s elapsed.')
    
