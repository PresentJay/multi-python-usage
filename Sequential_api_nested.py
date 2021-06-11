from urllib import request
from ast import literal_eval
import time, json

FAKE_RESTAPI_URL = 'https://jsonplaceholder.typicode.com/posts'
TRIAL_MAX_NESTED_1 = 10
TRIAL_MAX_NESTED_2 = 10
DATA = {"msg" : "Test"}


def post_request(data):
    req = request.Request(FAKE_RESTAPI_URL, method='POST')
    req.add_header('Content-Type', 'application/json')
    response = request.urlopen(req, data= (json.dumps(data)).encode())
    str_content = response.read().decode('utf-8')
    dict_content = literal_eval(str_content)
    return dict_content


# for loop computing API test
if __name__ == '__main__':
    
    """ Sequential nested test """
    
    start = time.perf_counter()
    temp = time.time()
    cnt = 0
    for x in range(TRIAL_MAX_NESTED_1):
        for y in range(TRIAL_MAX_NESTED_2):
            response = post_request(DATA)
            cnt += 1
            if cnt%10 == 0:
                temp = time.time() - temp
                print(f'{cnt:d}/{TRIAL_MAX_NESTED_1 * TRIAL_MAX_NESTED_2} trials : elapsed time {temp:.2f} s.')
                temp = time.time()
    
    print(f'Duration: {time.perf_counter() - start:.2f} s elapsed.')