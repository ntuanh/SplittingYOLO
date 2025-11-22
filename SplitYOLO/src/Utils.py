import psutil
import torch , yaml

def get_ram():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # MB

def get_vram():
    vram_peak = torch.cuda.max_memory_reserved() / 1024 ** 2
    return vram_peak

def reset_vram():
    torch.cuda.reset_peak_memory_stats()
    return 0

def extract_input_layer(file_name ):
    file = "./cfg/" + file_name
    cfg = yaml.safe_load(open(file, 'r', encoding='utf-8'))

    config = yaml.safe_load(open("./cfg/config.yaml", 'r', encoding='utf-8'))
    cut_layer = config["cut_layer"]

    res_dict = {
        "output" : [cut_layer - 1] ,
        "res_head" : [] ,
        "res_tail" : []
    }

    lst = []

    for index , layer in enumerate(cfg["head"]) :
        if layer[0] != -1 :
            for j in range(len(layer[0])):
                if layer[0][j] != -1 :
                    new_val = [layer[0][j] , index + 11]
                    lst.append(new_val)

    for pair in lst :
        if pair[0] < cut_layer and  pair[1] < cut_layer :
            res_dict["res_head"].append(pair[0])
        elif pair[0] < cut_layer and pair[1] > cut_layer :
            res_dict["output"].append(pair[0])
        else:
            res_dict["res_tail"].append(pair[0])

    # print(res_dict)
    return res_dict




