import torch, yaml, time, gc
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
from src.Utils import get_ram, get_vram, reset_vram, extract_input_layer

# ==========================
# 1. SETUP
# ==========================
# Tắt tính toán Gradient để tiết kiệm RAM
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")


# ==========================
# HELPER: Load trọng số tối ưu (Dùng mmap)
# ==========================
def load_weights_optimized(model, path):
    print(f"[Weights] Loading {path}...")
    try:
        # mmap=True: Đọc file từ ổ cứng, không copy vào RAM
        ckpt = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except:
        # Fallback nếu mmap không hỗ trợ (hiếm gặp)
        print("[Warning] mmap failed, using standard load")
        ckpt = torch.load(path, map_location='cpu', weights_only=True)

    # Lấy state_dict
    state_dict = ckpt['model'].state_dict() if 'model' in ckpt else ckpt

    # Nạp vào model (strict=False để bỏ qua các layer không khớp nếu có)
    model.load_state_dict(state_dict, strict=False)

    # Xóa ngay biến tạm
    del ckpt, state_dict
    gc.collect()
    print("[Weights] Loaded & RAM cleaned.")


# ==========================
# 2. INIT MODEL (GIỮ NGUYÊN LOGIC GỐC)
# ==========================
# Lấy thông tin layer output
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

# Khởi tạo model đầy đủ (Để đảm bảo layer index đúng tuyệt đối)
model = DetectionModel(cfg, verbose=False)

# Load weights bằng mmap
load_weights_optimized(model, 'part1.pt')

# Đẩy model sang GPU
model.to(device)
model.eval()

# Dọn dẹp RAM sau khi khởi tạo
gc.collect()
torch.cuda.empty_cache()

time.sleep(config["time_sleep"])

# ==========================
# 3. PREPARE INPUT (FIX QUAN TRỌNG NHẤT: GIẢM 300MB RAM)
# ==========================
img = Image.open('data/image.png').convert('RGB')
w, h = img.size
print(f"[Image size] {w}x{h}")

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

# BƯỚC QUAN TRỌNG: Đẩy 1 ảnh sang GPU TRƯỚC KHI nhân bản
# 1. Transform 1 ảnh
x_single = transform(img).unsqueeze(0)  # Size: [1, 3, 640, 640] - Rất nhẹ

# 2. Đẩy sang GPU ngay lập tức
x_single = x_single.to(device)

# 3. Nhân bản (Repeat) ngay trên GPU
# RAM CPU sẽ KHÔNG bị tốn thêm 300MB để chứa batch size 30
x = x_single.repeat(int(config["batch_size"]), 1, 1, 1)


# ==========================
# 4. FORWARD HEAD FUNCTION (GIỮ NGUYÊN LOGIC GỐC)
# ==========================
def forward_head(head_model, x_in):
    split_index = config["cut_layer"]
    y = {}  # store features map
    y[-1] = x_in
    state = {}

    # Chạy đúng số layer cần thiết
    # Lưu ý: Vì ta dùng Model gốc (không cắt YAML), nên dùng slice [:split_index] là chuẩn xác nhất
    for layer in head_model.model[:split_index]:
        x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]
        x_in = layer(x_in)  # forward

        # Logic lưu output y hệt code cũ của bạn
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

        # Tùy chọn: Nếu VRAM bị đầy, bỏ comment dòng dưới (nhưng sẽ chậm hơn)
        # if i % 20 == 0: torch.cuda.empty_cache()

# ==========================
# 6. SAVE
# ==========================
print(f"[Type] {type(state_dict)}")
print(f"[Keys] {state_dict.keys()}")  # In ra để kiểm tra có đúng key không

# Save
torch.save(state_dict, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")