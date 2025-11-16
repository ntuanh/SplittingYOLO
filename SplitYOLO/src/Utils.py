import psutil

def check_ram():
    ram = psutil.virtual_memory()
    # print(f"Total RAM: {ram.total / (1024 ** 3):.2f} GB")
    print(f"Used RAM: {ram.used / (1024 ** 3):.2f} GB")
    # print(f"Available RAM: {ram.available / (1024 ** 3):.2f} GB")
    # print(f"RAM Usage: {ram.percent}%")
    return ram.used
