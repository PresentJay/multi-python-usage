import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from numba import jit
from utils import monitoring

from numba import cuda
from numba import *


def mandel(x, y, max_iters):
    """
        Given the real and imaginary parts of a complex number,
        determine if it is a candidate for membership in the Mandelbrot
        set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    """ 
        create_fractal iterates over all the pixels in the image,
        computing the complex coordinates from the pixel coordinates,
        and calls the mandel function at each pixel.
        The return value of mandel is used to color the pixel.
    """
    
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    
    cnt=0
    
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            cnt+=1
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color
            
            if (cnt % (width*height)/10)==0:
                monitoring.show_short_info()


def naive_test():
    image = np.zeros((1024, 1536), dtype = np.uint8)
    start = timer()
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) 
    dt = timer() - start
    
    print("Mandelbrot created in %f s" % dt)
    plt.imshow(image)
    plt.show(block=True)
    
    start = timer()
    create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20) 
    dt = timer() - start
    print("fractal created in %f s" % dt)
    plt.imshow(image)
    plt.show(block=True)


@jit(forceobj=True)
def mandel(x, y, max_iters):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return max_iters

@jit(forceobj=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]
    
    cnt = 0

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
        
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            if (cnt % (height*width/10))==0:
                monitoring.show_short_info()
                
            cnt +=1
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color
            
            
    
    
def Faster_test():
    image = np.zeros((1024, 1536), dtype = np.uint8)
    start = timer()
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) 
    dt = timer() - start

    print("Mandelbrot created in %f s" % dt)
    plt.imshow(image)
    plt.show(block=True)


@cuda.jit(parallel=True, forceobj=True)
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    cnt = 0

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            cnt+=1
            imag = min_y + y * pixel_size_y 
            image[y, x] = cuda.jit(device=True)(mandel)(real, imag, iters)
            
            if (cnt % (width*height/10)) ==0:
                monitoring.show_short_info()
            
            
# def cuda_test():
#     gimage = np.zeros((1024, 1536), dtype = np.uint8)
#     blockdim = (32, 8)
#     griddim = (32,16)
    
    

#     start = timer()
#     d_image = cuda.to_device(gimage)
#     mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20) 
#     d_image.to_host()
#     dt = timer() - start

#     print("Mandelbrot created on GPU in %f s" % dt)

#     plt.imshow(gimage)
#     plt.show(block=True)