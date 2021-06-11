from test import Sequential, Multiprocessing
from utils.monitoring import show_info



if __name__ == '__main__':
    show_info()
    # Sequential.test()  # 28.35s / Only CPU, no efficiency
    # Multiprocessing.test()  # 122.75s / unrecommended method - - - unstateful computation, weak to large numerical data
    
    