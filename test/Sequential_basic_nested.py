import time

TRIAL_MAX_NESTED_1 = 1000

TRIAL_MAX_NESTED_2 = 800
TRIAL_MAX_NESTED_3 = 800

# for nested loop computing basic test
if __name__ == '__main__':
    
    """ Sequential basic test """
    
    start = time.perf_counter()
    temp = time.time()
    cnt = 0
    for z in range(TRIAL_MAX_NESTED_1):
        for y in range(TRIAL_MAX_NESTED_2):
            for x in range(TRIAL_MAX_NESTED_3):
                if cnt%10000000 == 0:
                    temp = time.time() - temp
                    print(f'{(cnt / (TRIAL_MAX_NESTED_1 * TRIAL_MAX_NESTED_2 * TRIAL_MAX_NESTED_3) * 100):.2f}% progress : elapsed time {time.perf_counter()-start:.2f} s.')
                    temp = time.time()
    
    print(f'Final Duration: {time.perf_counter()-start:.2f} s elapsed.')
    
