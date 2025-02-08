import os
import glob
import torch as th

def smooth(datas, win_size=10000):
    ret = []
    tmp_sum = 0
    for i in range(len(datas)):
        tmp_sum += datas[i]
        if i >= win_size - 1:
            ret.append(tmp_sum / win_size)
            tmp_sum -= datas[i - win_size + 1]
    return ret
        

def cleanup_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        files = glob.glob(os.path.join(path, '*'))
        for f in files:
            try:
                os.remove(f)
            except:
                pass      


def normalize_TM_maxthroughput(mcf_mlu, TM):
    return TM / mcf_mlu * 1.2

def normalize_TM_mincost(sp_mlu, TM):
    return TM / sp_mlu





def print_gpu_usage():
    if th.cuda.is_available():
        # Get the GPU device count
        num_gpus = th.cuda.device_count()

        print(f"Number of GPUs: {num_gpus}")

        for gpu_id in range(num_gpus):
            gpu = th.cuda.get_device_properties(gpu_id)
            print(f"GPU {gpu_id}: {gpu.name}")
            print(f"Memory Usage - Allocated: {th.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB | Cached: {th.cuda.memory_cached(gpu_id) / 1e9:.2f} GB | Peak Allocated: {th.cuda.max_memory_allocated(gpu_id) / 1e9:.2f} GB")
        print("\n")
    else:
        print("No GPU available.")

