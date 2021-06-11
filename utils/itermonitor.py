import monitoring
import time
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--delay', type=float, default=1.0)
    opt = parser.parse_args()
    
    while(True):
        # monitoring.simple_gpu_info()
        monitoring.show_short_info()
        time.sleep(opt.delay)