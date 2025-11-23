import torch, yaml, time, gc
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
from src.Utils import get_ram, get_vram, reset_vram, extract_input_layer

# ==========================
# 1. SETUP
# ==========================
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")


# ==========================
# HELPER:
# ==========================
def load_weights_optimized(model, path):
    print(f"[Weights] Loading {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except:
        print("[Warning] mmap failed, using standard load")
        ckpt = torch.load(path, map_location='cpu', weights_only=True)

    state_dict = ckpt['model'].state_dict() if 'model' in ckpt else ckpt

    model.load_state_dict(state_dict, strict=False)

    del ckpt, state_dict
    gc.collect()
    print("[Weights] Loaded & RAM cleaned.")


# ==========================
# 2. INIT MODEL
# ==========================
output = extract_input_layer("yolo11n.yaml")["output"]
res_head = extract_input_layer("yolo11n.yaml")["res_head"]
print(f"Res Head: {res_head}")
print(f"Output: {output}")

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

if config["head_architect"] == "head":
    yaml_file = 'cfg/head.yaml'
else:
    yaml_file = 'cfg/yolo11n.yaml'

print(f"YAML file {yaml_file}")
cfg = yaml.safe_load(open(yaml_file, 'r', encoding='utf-8'))

model = DetectionModel(cfg, verbose=False)

load_weights_optimized(model, 'part1.pt')

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
# 4. FORWARD HEAD FUNCTION
# ==========================
def forward_head(head_model, x_in):
    split_index = config["cut_layer"]
    y = {}  # store features map
    y[-1] = x_in
    state = {}

    for layer in head_model.model[:split_index]:
        x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]
        x_in = layer(x_in)  # forward

        if layer.i in output:
            state[layer.i] = x_in
        elif layer.i in res_head:
            y[layer.i] = x_in
        y[-1] = x_in

    return state


# ==========================
# 5. RUN LOOP
# ==========================
time.sleep(config["time_sleep"])
print("Starting inference...")

# Dọn rác lần cuối
gc.collect()

with torch.inference_mode():
    for i in range(int(config["nums_round"])):
        state_dict = forward_head(model, x)

        # if i % 20 == 0: torch.cuda.empty_cache()

# ==========================
# 6. SAVE
# ==========================
print(f"[Type] {type(state_dict)}")
print(f"[Keys] {state_dict.keys()}")

# Save
torch.save(state_dict, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")