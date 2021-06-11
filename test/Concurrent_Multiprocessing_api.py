from urllib import request
from ast import literal_eval
from concurrent import futures
import time, json


FAKE_RESTAPI_URL = 'https://jsonplaceholder.typicode.com/posts'
TRIAL_MAX = 100
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
    
    """ Concurrent MultiProcess test """
    
    with futures.ProcessPoolExecutor() as executor:
        start = time.perf_counter()
        processes = []
        
        for i in range(1, TRIAL_MAX+1):
            processes.append(executor.submit(post_request, DATA))
    
    print(f'Duration: {time.perf_counter() - start:.2f} s elapsed.')
    
    # for process in processes[0:10]:
    #     print(process.result())