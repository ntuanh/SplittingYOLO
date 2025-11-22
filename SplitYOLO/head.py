import torch , yaml , time
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
from src.Utils import get_ram, get_vram, reset_vram

# ==========================
# 1. Pick device and load model architecture
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)


if config["head_architect"] == "head" :
    yaml_file = 'cfg/head.yaml'
else :
    yaml_file = 'cfg/yolo11n.yaml'
cfg = yaml.safe_load(open(yaml_file, 'r', encoding='utf-8'))
model = DetectionModel(cfg, verbose=False ).to(device)


state_dict_part1 = torch.load('part1.pt', map_location=device, weights_only=True)
model.load_state_dict(state_dict_part1, strict=False)
model.eval()

img = Image.open('data/image.png').convert('RGB')
w, h = img.size
print(f"[Image size] {w}x{h}")
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
x = transform(img).unsqueeze(0).repeat(int(config["batch_size"]), 1, 1, 1).to(device)

# ==========================
# 4. Forward head function
# ==========================
def forward_head(head_model, x_in):
    split_index = config["cut_layer"]
    # y = {}  # store features map

    for layer in head_model.model[:split_index]:
        # if layer.f != -1:  # get input from previous layer
        #     pass
        #     x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]

        x_in = layer(x_in)  # forward
        # y[layer.i] = x_in

    return x_in

# ==========================
# 5. Forward pass + measure RAM and VRAM
# ==========================

with torch.inference_mode():
    for _ in range(int(config["nums_round"])):
        feature_map = forward_head(model, x)

# ==========================
# 6. Save feature map
# ==========================
print(f"   Output feature map shape: {feature_map.shape}")
torch.save(feature_map, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")
