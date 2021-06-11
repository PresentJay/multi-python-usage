import time

TRIAL_MAX = 100000000

# for loop computing basic test
if __name__ == '__main__':
    
    """ Sequential test """
    
    start = time.perf_counter()
    temp = time.time()
    
    for i in range(1, TRIAL_MAX+1):
        if i%10000000 == 0:
            temp = time.time() - temp
            print(f'{i:d}/{TRIAL_MAX} trials : elapsed time {temp:.2f} s.')
            temp = time.time()
    
    print(f'Duration: {time.perf_counter()-start:.2f} s elapsed.')