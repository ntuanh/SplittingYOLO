import torch, yaml, time, gc
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
from src.Utils import get_ram, get_vram, reset_vram, extract_input_layer

# Setup and check type of devices

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

def load_weights_optimized(model, path):
    """Load weights and remove architecture with random weights.

    Args:
        model : model after load by DetectionModel class
        path : path of weights ( only weights )

    Return :
        model : model with weights loaded .

    """
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

# Init model

output = extract_input_layer("yolo11n.yaml")["output"]
print(f"Output: {output}")

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

# ========= Select architecture yaml ============

yaml_file = 'cfg/yolo11n.yaml'
print(f"YAML file {yaml_file}")
cfg = yaml.safe_load(open(yaml_file, 'r', encoding='utf-8'))

model = DetectionModel(cfg, verbose=False)

load_weights_optimized(model, 'part1.pt')

model.to(device)
model.eval()
time.sleep(config["time_sleep"])        # 1 st

# gc.collect()
# torch.cuda.empty_cache()

time.sleep(config["time_sleep"])        # 2 nd
# Prepare Input for model

img = Image.open('data/image.png').convert('RGB')
w, h = img.size
print(f"[Image size] {w}x{h}")

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

x_single = transform(img).unsqueeze(0)

x_single = x_single.to(device).half()   # convert to FP16
x = x_single.repeat(int(config["batch_size"]), 1, 1, 1)


# def forward_head(head_model, x_in):
#     """ Forward throughout head layers .
#     Arguments :
#         head_model : model with truth weights .
#         x_in : input ( image and n batches )
#
#     return :
#         feature_map : dict - include last layer output and short-cut block
#     """
#     split_index = config["cut_layer"]
#     y = {}  # store features map
#     y[-1] = x_in
#     state = {}
#
#     for layer in head_model.model[:split_index]:
#         x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]
#         x_in = layer(x_in)  # forward
#
#         if layer.i in output:
#             state[layer.i] = x_in
#         elif layer.i in res_head:
#             y[layer.i] = x_in
#         y[-1] = x_in
#
#     return state
def forward_head_optimized(model, x, output_layers):
    """
    Optimize the final forward function to reduce CPU dependency
    by minimizing CPU-side control logic and offloading execution
    flow and tensor operations to the device.
    :param model: model FP16
    :param x: input  FP16
    :param output_layers: list short cut output layers
    :return: state dict ( include shortcut output layers )
    """
    feature_maps = {}

    def hook_fn(layer_id):
        def fn(_, __, out):
            feature_maps[layer_id] = out
        return fn

    handles = []
    for i in output_layers:
        handles.append(model.model[i].register_forward_hook(hook_fn(i)))

    with torch.inference_mode():
        _ = model(x)
        torch.cuda.synchronize()

    for h in handles:
        h.remove()
    return feature_maps


# clean and run loops
# time.sleep(config["time_sleep"])        # 3rd
print("Starting inference...")

# gc.collect()

time.sleep(config["time_sleep"])        # 4th

with torch.inference_mode():
    model = model.half()
    for i in range(int(config["nums_round"]) ):
        state_dict = forward_head_optimized(model, x , output_layers=output)
        # gc.collect()
        # torch.cuda.empty_cache()

# Save to feature_map.pt
print(f"[Type] {type(state_dict)}")
print(f"[Keys] {state_dict.keys()}")

torch.save(state_dict, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")