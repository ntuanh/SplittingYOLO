import psutil
import torch

def get_ram():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # MB

def get_vram():
    vram_peak = torch.cuda.max_memory_reserved() / 1024 ** 2
    return vram_peak

def reset_vram():
    torch.cuda.reset_peak_memory_stats()
    return 0
