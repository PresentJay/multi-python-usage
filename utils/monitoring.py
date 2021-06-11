import pandas as pd
import GPUtil
import time
import cpuinfo
import psutil

MAX_ITERATIONS = 10000000 # (set a maximum number of acquisition, to prevent explosions recorded text)
DELAY = 1 # sampling interval information

def get_gpu_info():
    Gpus = GPUtil.getGPUs()
    gpulist = []
    # GPUtil.showUtilization() 
    for gpu in Gpus:
        print(f'[GPU] {gpu.name}\t(index {gpu.id} /{len(Gpus)-1} total) \n\t==> {gpu.memoryUsed/1024:.2f} GB/ {gpu.memoryTotal/1024:.2f} GB({gpu.memoryUtil*100:.2f}%)')
        
        # Press# to add information one by one GPU
        gpulist.append([ gpu.id, gpu.memoryTotal, gpu.memoryUsed,gpu.memoryUtil * 100])

    return gpulist


def get_cpu_info():
    ''' :return:
         memtotal: Total Memory
         memfree: free memory
         memused: Linux: total - free, used memory
         mempercent: the proportion of memory used
         cpu: CPU usage of each accounting
    '''
    cpu = cpuinfo.get_cpu_info()
    mem = psutil.virtual_memory()
    memtotal = mem.total
    mempercent = mem.percent
    memused = mem.used
    cpuPercent = psutil.cpu_percent(percpu=True)
    print(f'[CPU] {cpu["brand_raw"]}\t[{cpu["count"]} cores] \n\t==> {memused/1024/1024/1024:.2f} GB/ {memtotal/1024/1024/1024:.2f} GB({mempercent}%)')
    for index, core in enumerate(cpuPercent):
        print(f'[{index}]{core}%  \t|\t', end='')
        if (index+1)%8==0:
            print('')
    print('')

    return memtotal, memused, mempercent, cpuPercent


if __name__ == '__main__':
    times = 0
    while True:
                 # The maximum number of cycles
        if times < MAX_ITERATIONS:
                         # Prints the current time
            time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            now = time.localtime()
            print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                         # Obtain information CPU
            cpu_info = get_cpu_info()
                         # Obtain information GPU
            gpu_info = get_gpu_info()
                         # Added gap
            print('- - - -')
            time.sleep(DELAY)
            times += 1
        else:
            break
