from test import Sequential, Multiprocessing, numba_simple, mixed, Numba_Mandelbrot, Numba_052releaseCUDA
from utils.monitoring import show_info


if __name__ == '__main__':
    show_info()
    # Sequential.test()  # 28.35s / Only CPU, no efficiency
    # Multiprocessing.test()  # 122.75s / unrecommended method - - - unstateful computation, weak to large numerical data
    
    # numba_simple.test() # CPU지만 매우 빠름
       
    # mixed.test()
    
    # pyCUDA_gpu.test()  # CUDA version issue --> segmentation fault error
    
    # Numba GPU parallel으로 가야함 (NVIDIA에서도 numba 추천 : https://developer.nvidia.com/cuda-python)
    # 추가적 reference : https://developer.nvidia.com/blog/numba-python-cuda-acceleration/
    # Numba_Mandelbrot.naive_test()
    # Numba_Mandelbrot.Faster_test()
    
    # Numba_052releaseCUDA.overhead_test()
    Numba_052releaseCUDA.LibdeviceTest()
    
    # Numba_052releaseCUDA.automic_substract_test()
    
    # Numba_052releaseCUDA.math_test()
    
    # Numba_052releaseCUDA.power_complex_test()