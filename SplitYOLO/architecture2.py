
from ultralytics import YOLO
from torch import nn
import torch, yaml
from copy import deepcopy
from src.Utils import get_ram, get_vram, reset_vram, extract_input_layer

# Load model
model = YOLO("yolo11n.pt")
full_model = model.model

# Load config
with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)
with open('cfg/head.yaml') as file:
    head_yaml = yaml.safe_load(file)
split_index = int(config["cut_layer"])


output = extract_input_layer("yolo11n.yaml")["output"]
res_head = extract_input_layer("yolo11n.yaml")["res_head"]


print(f"\nSplitting model at layer index = {split_index}")
# print(type(full_model.model))
# ---- TÁCH MODEL ----
new_model = deepcopy(full_model)  # deepcopy để tránh phá model gốc
# print(type(new_model))
# Giữ lại layers 0 → split_index
new_model.model = nn.Sequential(*list(full_model.model[:split_index + 1]))

# Reset danh sách layer cần save
output.sort()
new_model.save = output

print(new_model.save)

# Lấy state_dict sau khi cắt
new_state = new_model.state_dict()
# print(new_state)
print("Số lượng tham số sau khi cắt:", len(new_state))

