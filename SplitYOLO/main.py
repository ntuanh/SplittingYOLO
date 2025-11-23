import torch
from ultralytics import YOLO
import yaml

# Load config (nếu muốn lưu lại yaml)
with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

# Load model gốc
print("Loading original YOLOv11n model...")
model = YOLO("yolo11n.pt")

# Load state_dict phần đầu
state_dict_part1 = torch.load('part1.pt', map_location='cpu', weights_only=True)
# print("Keys in part1.pt:", list(state_dict_part1.keys()))

# Gán trọng số
missing, unexpected = model.model.load_state_dict(state_dict_part1, strict=False)
# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)

# --- SAVE TOÀN BỘ MODEL theo chuẩn Ultralytics ---
torch.save({
    "model": model.model,                  # nn.Module
    "model_state_dict": model.model.state_dict(),  # state_dict đầy đủ sau khi update
    "yaml": "yolo11n.yaml",                # YAML của model
    "names": model.names,                  # class names
    "task": "detect"
}, "model_new.pt")

print("Saved new model as 'model_new.pt'. You can now load it with YOLO('model_new.pt')")

# ==============================
# from ultralytics import YOLO
# import torch , yaml
# from PIL import Image
# import torchvision.transforms as T
#
# with open('./cfg/config.yaml') as file:
#     config = yaml.safe_load(file)
#
# # Load model
# model = YOLO("./cfg/yolo11n.yaml")
#
# # Load + transform image
# img = Image.open("./data/image.png").convert('RGB')
# transform = T.Compose([
#     T.Resize((640, 640)),
#     T.ToTensor(),
# ])
# x = transform(img)  # [3, 640, 640]
#
# batch = x.unsqueeze(0).repeat(int(config["batch_size"]), 1, 1, 1)  # [30, 3, 640, 640]
#
# batch = batch.to(model.device).float()
#
# # time.sleep(10)
#
# for _ in range(int(config["nums_round"])):
#     results = model(batch, verbose=False)
#
# # logits = results.pred[0]
# #
# # print(f"[Type of logits] { type(logits)}")
# print(f"[Length] {len(results)}")
# print(f"[Type of index 0 ] {type(results[0])}")
# torch.save(results, 'feature_map.pt')