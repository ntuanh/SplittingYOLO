import torch , yaml , time, gc
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
from src.Utils import get_ram, get_vram, reset_vram , extract_input_layer

# ==========================
# 1. Pick device and load model architecture
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

output = extract_input_layer("yolo11n.yaml")["output"]
res_head = extract_input_layer("yolo11n.yaml")["res_head"]

print(res_head)
print(output)

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)


if config["head_architect"] == "head" :
    yaml_file = 'cfg/head.yaml'
else :
    yaml_file = 'cfg/yolo11n.yaml'

print(f"YAML file {yaml_file}")
cfg = yaml.safe_load(open(yaml_file, 'r', encoding='utf-8'))
model = DetectionModel(cfg, verbose=False ).to(device)

time.sleep(10)

state_dict_part1 = torch.load('part1.pt', map_location=device, weights_only=True)
model.load_state_dict(state_dict_part1, strict=False)
# garbage memory
del state_dict_part1
gc.collect()
torch.cuda.empty_cache()

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
    y = {}  # store features map
    y[-1] = x_in
    state =  {}

    for layer in head_model.model[:split_index]:
        x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]
        x_in = layer(x_in)  # forward
        if layer.i in output :
            state[layer.i] = x_in
        elif layer.i in res_head :
            y[layer.i] = x_in
        y[-1] = x_in

    return state

# ==========================
# 5. Forward pass + measure RAM and VRAM
# ==========================

time.sleep(10)
with torch.inference_mode():
    for i in range(int(config["nums_round"])):
        state_dict = forward_head(model, x)
        # if i % 20 == 0 :
        # gc.collect()  # dọn RAM CPU
        # torch.cuda.empty_cache() # dọn VRAM GPU

# ==========================
# 6. Save feature map
# ==========================

print(type(state_dict))
print(f"[Type] {type(state_dict)}")
torch.save(state_dict, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")
