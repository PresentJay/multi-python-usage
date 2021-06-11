from test import Sequential, Multiprocessing, numba_simple, numba_cuda, mixed
from utils.monitoring import show_info


if __name__ == '__main__':
    show_info()
    # Sequential.test()  # 28.35s / Only CPU, no efficiency
    # Multiprocessing.test()  # 122.75s / unrecommended method - - - unstateful computation, weak to large numerical data
    
    # numba_simple.test() # CPU지만 매우 빠름
    
    mixed.test()