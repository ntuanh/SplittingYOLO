import torch, yaml, time, gc
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

from SplittingYOLO.SplitYOLO.architecture2 import full_model
from src.Utils import get_ram, get_vram, reset_vram, extract_input_layer


# ==========================
# 1. SETUP
# ==========================
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")


# ==========================
# 2. INIT MODEL
# ==========================
output = extract_input_layer("yolo11n.yaml")["output"]
res_head = extract_input_layer("yolo11n.yaml")["res_head"]
print(f"Res Head: {res_head}")
print(f"Output: {output}")

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

model = YOLO("model_new.pt")
model.to(device)
model.eval()

gc.collect()
torch.cuda.empty_cache()

time.sleep(config["time_sleep"])

# ==========================
# 3. PREPARE INPUT
# ==========================
img = Image.open('data/image.png').convert('RGB')
w, h = img.size
print(f"[Image size] {w}x{h}")

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

x_single = transform(img).unsqueeze(0)

x_single = x_single.to(device).float()

x = x_single.repeat(int(config["batch_size"]), 1, 1, 1)




# ==========================
# 4. RUN LOOP
# ==========================
# time.sleep(config["time_sleep"])
print("Starting inference...")


# 2. Tạo dict để lưu feature map
features = {}

# 3. Hàm hook để bắt feature
def hook_fn(name):
    def hook(module, input, output):
        features[name] = output
    return hook

# 4. Gắn hook vào tất cả layer có output
for name, layer in full_model.named_modules():
    # chỉ lấy những layer có forward (Conv, C2f, SPPF, Detect,...)
    layer.register_forward_hook(hook_fn(name))



# 5. Forward ảnh
with torch.inference_mode():
    for i in range(int(config["nums_round"])):
        _ = model(x)

# 7. Xem tên các feature map
print(features.keys())
# Dọn rác lần cuối
gc.collect()


# ==========================
# 6. SAVE
# ==========================
print(f"[Type] {type(features)}")
print(f"[Keys] {features.keys()}")

# Save
torch.save(features, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")
