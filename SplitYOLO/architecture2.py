from ultralytics import YOLO
from collections import OrderedDict
import torch,yaml

model = YOLO("yolo11n.pt")
full_model = model.model

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

with open('cfg/head.yaml') as file:
    new_yaml = yaml.safe_load(file)


# get split layer
split_index = int(config["cut_layer"])
# split_index = 12
# print(config["cut_layer"])
print(f"\nSplitting model at layer index = {split_index}")


#   Tách layers
kept_layers = torch.nn.ModuleList(list(full_model.model[:split_index+1]))  # 0→12

#   Cập nhật layers của model
full_model.model = kept_layers
# Tách layers tương ứng sẽ tách được weights tương ứng
len_model = len(full_model.model)
#   Cập nhật save
full_model.save = [i < split_index+1 for i in range(len_model)]
# print(full_model.save)
# print(full_model.model.state_dict())
print(len(full_model.state_dict()))



# --- SAVE TOÀN BỘ MODEL theo chuẩn Ultralytics ---
torch.save({
    "model": full_model,                  # nn.Module
    "model_state_dict": full_model.state_dict(),  # state_dict đầy đủ sau khi update
    "yaml": "head.yaml",                # YAML của model
    "names": model.names,                  # class names
    "task": "detect"
}, "model_new.pt")