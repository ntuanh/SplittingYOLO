import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
import yaml, time
from src.Utils import get_ram, get_vram, reset_vram

# ==========================
# 1. Pick device and load model architecture
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

# start_ram = get_ram()
# start_vram = reset_vram()
# t0 = time.time()

cfg = yaml.safe_load(open('cfg/head.yaml', 'r', encoding='utf-8'))
# yolo11.yaml
model = DetectionModel(cfg, verbose=False   ).to(device)

# t1 = time.time()
# ram_after_arch = get_ram()
# vram_after_arch = get_vram()
# print(f"[Architecture] Time: {t1 - t0:.4f} s")
# print(f"[Architecture] RAM used: {ram_after_arch - start_ram:.2f} MB")
# print(f"[Architecture] VRAM used: {vram_after_arch - start_vram:.2f} MB")

# ==========================
# 2. Load weights (part1)
# ==========================
# ram_before = get_ram()
# vram_before = reset_vram()
# t0 = time.time()

state_dict_part1 = torch.load('part1.pt', map_location=device, weights_only=True)
model.load_state_dict(state_dict_part1, strict=False)
model.eval()

# t1 = time.time()
# ram_after = get_ram()
# vram_after = get_vram()
# print(f"[Weights] Time: {t1 - t0:.4f} s")
# print(f"[Weights] RAM used: {ram_after - ram_before:.2f} MB")
# print(f"[Weights] VRAM used: {vram_after - vram_before:.2f} MB")

# ==========================
# 3. Load and preprocess image
# ==========================
# print("Loading and preprocessing image...")
img = Image.open('data/image.png').convert('RGB')
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
x = transform(img).unsqueeze(0).repeat(30 , 1, 1, 1).to(device)
# print(f"   Input image tensor shape: {x.shape}")

# ==========================
# 4. Forward head function
# ==========================
def forward_head(head_model, x_in):
    split_index = 4
    y = {}  # store features map

    for layer in head_model.model[:split_index]:
        if layer.f != -1:  # get input from previous layer
            pass
            x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]

        x_in = layer(x_in)  # forward
        y[layer.i] = x_in

    return x_in

# ==========================
# 5. Forward pass + measure RAM and VRAM
# ==========================
# print("Performing custom forward pass on head...")
# ram_before = get_ram()
# vram_before = reset_vram()
# t0 = time.time()

with torch.inference_mode():
    for _ in range(1000):
        feature_map = forward_head(model, x)

# t1 = time.time()
# ram_after = get_ram()
# vram_after = get_vram()
#
# print(f"[Inference] Time: {t1 - t0:.4f} s")
# print(f"[Inference] RAM used: {ram_after - ram_before:.2f} MB")
# print(f"[Inference] VRAM used: {vram_after - vram_before:.2f} MB")

# ==========================
# 6. Save feature map
# ==========================
print(f"   Output feature map shape: {feature_map.shape}")
torch.save(feature_map, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")
