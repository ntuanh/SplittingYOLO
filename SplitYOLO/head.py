import torch, yaml, time, gc
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
from src.Utils import get_ram, get_vram, reset_vram, extract_input_layer


# ==========================
# HELPER: Cắt gọt Config để giảm RAM khởi tạo
# ==========================
def prune_config(cfg, cut_layer):
    """
    Cắt bỏ các định nghĩa layer thừa trong config dictionary.
    Giúp DetectionModel không khởi tạo các layer phía sau cut_layer.
    """
    # Tính tổng số layer hiện tại trong backbone
    len_backbone = len(cfg['backbone'])

    if cut_layer <= len_backbone:
        # Nếu cắt ngay trong backbone -> Xóa hết head, cắt bớt backbone
        cfg['backbone'] = cfg['backbone'][:cut_layer]
        cfg['head'] = []
    else:
        # Nếu cắt ở head -> Giữ backbone, cắt bớt head
        # Index trong head bắt đầu từ 0, nên cần trừ đi độ dài backbone
        head_keep_count = cut_layer - len_backbone
        cfg['head'] = cfg['head'][:head_keep_count]

    print(f"[Config] Pruned model structure. Keep first {cut_layer} layers.")
    return cfg


# ==========================
# HELPER: Load trọng số an toàn (Tránh kẹt RAM)
# ==========================
def load_weights_safe(model, path, device):
    print(f"[Weights] Loading {path} ...")
    # Load lên CPU trước để tránh đầy VRAM đột ngột
    ckpt = torch.load(path, map_location='cpu', weights_only=True)

    # Chỉ lấy state_dict (xử lý nếu file là checkpoint full)
    state_dict = ckpt['model'].state_dict() if 'model' in ckpt else ckpt

    # Lọc trọng số: Chỉ nạp những layer có trong model đã cắt gọt
    model_state = model.state_dict()
    # Chỉ giữ lại key nào khớp cả tên lẫn kích thước
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}

    model.load_state_dict(filtered_dict, strict=False)

    # Xóa ngay biến tạm
    del ckpt, state_dict, filtered_dict
    gc.collect()
    print("[Weights] Loaded and RAM cleaned.")


# ==========================
# 1. Pick device and load model architecture
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

# Lấy thông tin layer cần output (Vẫn đọc từ file gốc để lấy index chuẩn)
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

# --- TỐI ƯU: Load YAML -> Cắt gọt -> Mới tạo Model ---
raw_cfg = yaml.safe_load(open(yaml_file, 'r', encoding='utf-8'))
cut_layer_idx = config["cut_layer"]

# Cắt config trước khi đưa vào DetectionModel -> Tiết kiệm RAM nhất
pruned_cfg = prune_config(raw_cfg, cut_layer_idx)

# Model được tạo ra chỉ chứa đúng số layer cần thiết
model = DetectionModel(pruned_cfg, verbose=False).to(device)

time.sleep(config["time_sleep"])

# --- TỐI ƯU: Load weights qua hàm helper ---
load_weights_safe(model, 'part1.pt', device)

# Dọn dẹp lần cuối
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
    # Model đã được cắt gọt, nên len(head_model.model) chính bằng split_index
    # Ta duyệt qua tất cả các layer hiện có trong model

    y = {}  # store features map
    y[-1] = x_in
    state = {}

    # Lưu ý: head_model.model bây giờ ngắn hơn model gốc, nhưng thuộc tính layer.i (index)
    # và layer.f (from) vẫn được Ultralytics giữ đúng theo thứ tự topo.
    for layer in head_model.model:
        x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]
        x_in = layer(x_in)  # forward

        # Logic giữ nguyên để tương thích tail.py
        if layer.i in output:
            state[layer.i] = x_in
        elif layer.i in res_head:
            y[layer.i] = x_in

        y[-1] = x_in

    return state


# ==========================
# 5. Forward pass + measure RAM and VRAM
# ==========================

time.sleep(config["time_sleep"])
print("Starting inference loop...")
gc.collect()

with torch.inference_mode():
    for i in range(int(config["nums_round"])):
        state_dict = forward_head(model, x)
        # if i % 20 == 0 :
        # gc.collect()  # dọn RAM CPU
        # torch.cuda.empty_cache() # dọn VRAM GPU

# ==========================
# 6. Save feature map
# ==========================

print(f"[Type] {type(state_dict)}")
print(f"[Keys] {state_dict.keys()}")
torch.save(state_dict, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")