import GPUtil
import time
import cpuinfo
import psutil

def show_short_info():
    Gpus = GPUtil.getGPUs()
    for gpu in Gpus:
        print(f'[GPU({gpu.id}) Usage {gpu.load*100:.2f}%|VRAM {gpu.memoryUtil*100:.2f}%]',end=' ')
    mem=psutil.virtual_memory()
    full=psutil.cpu_percent()
    print(f'[CPU Usage {full:.2f}%|RAM {mem.percent:.2f}%]')
    
    return gpu.memoryUtil*100, mem.percent


def get_gpu_info():
    Gpus = GPUtil.getGPUs()
    gpulist = []
    # GPUtil.showUtilization() 
    for gpu in Gpus:
        print(f'[GPU] {gpu.name}\t(index {gpu.id} /{len(Gpus)-1} total)')
        print(f'\t==> Usage: {gpu.load*100:.2f}%')
        print(f'[VRAM] {gpu.memoryUsed/1024:2.2f} GB/ {gpu.memoryTotal/1024:2.2f} GB({gpu.memoryUtil*100:2.2f}%)')
        
        # Press# to add information one by one GPU
        gpulist.append([ gpu.id, gpu.memoryTotal, gpu.memoryUsed,gpu.memoryUtil * 100])

    return gpulist


def show_full_cpuinfo():
    cpu = psutil.cpu_percent(percpu=True)
    for core in cpu:
        print(f'[{core:05.2f}%]', end=' ')
    print('\n')
    return cpu


# TODO: 코어별 사용량 변동폭을 기록할만한 것 같음
def get_cpu_info():
    cpu = cpuinfo.get_cpu_info()
    mem = psutil.virtual_memory()
    memtotal = mem.total
    mempercent = mem.percent
    memused = mem.used
    totalUsage = psutil.cpu_percent()
    coreUsage = psutil.cpu_percent(percpu=True)
    print(f'[CPU] {cpu["brand_raw"]}\t[{cpu["count"]} cores]')
    print(f'\t==> Usage: {totalUsage:.2f}%')
    print(f'[RAM] {memused/1024/1024/1024:.2f} GB/ {memtotal/1024/1024/1024:.2f} GB ({mempercent}%)')
    for index, core in enumerate(coreUsage):
        print(f'[{index:2d}]{core:05.2f}%', end='  ')
        if (index+1)%8==0:
            print('')
    print('')
    return coreUsage, totalUsage, mempercent, mempercent


def show_monitor_iter(max_iter=10000000, delay=2):
    times = 0
    while True:
        if times < max_iter:
            
            time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            now = time.localtime()
            print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
            
            cpu_info = get_cpu_info()
            gpu_info = get_gpu_info()
            print('- - - -')
            time.sleep(delay)
            times += 1
        else:
            break
        
def simple_gpu_info():
    Gpus = GPUtil.getGPUs()
    # GPUtil.showUtilization() 
    for gpu in Gpus:
        print(f'[GPU] {gpu.name} ==> Usage: {gpu.load*100:.2f}%  /[VRAM] {gpu.memoryUsed/1024:2.2f} GB/ {gpu.memoryTotal/1024:2.2f} GB({gpu.memoryUtil*100:2.2f}%)')


def show_info():
    time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    now = time.localtime()
    print("\n\t- - - - %04d/%02d/%02d %02d:%02d:%02d - - - -" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    
    
    # TODO: 코어별 변동량 기록준비
    coreUsage, totalUsage, mempercent, mempercent = get_cpu_info()
    gpu_info = get_gpu_info()
    print('\n\t- - - - start experiment')
    
