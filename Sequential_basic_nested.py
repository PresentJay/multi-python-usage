import time

TRIAL_MAX_NESTED_1 = 10000
TRIAL_MAX_NESTED_2 = 10000

# for loop computing basic test
if __name__ == '__main__':
    
    """ Sequential nested test """
    
    start = time.perf_counter()
    temp = time.time()
    cnt = 0
    for x in range(TRIAL_MAX_NESTED_1):
        for y in range(TRIAL_MAX_NESTED_2):
            cnt += 1
            if cnt%10000000 == 0:
                temp = time.time() - temp
                print(f'{cnt:d}/{TRIAL_MAX_NESTED_1 * TRIAL_MAX_NESTED_2} trials : elapsed time {temp:.2f} s.')
                temp = time.time()
    
    print(f'Duration: {time.perf_counter()-start:.2f} s elapsed.')
    
