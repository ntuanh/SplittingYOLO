# head.py (Phiên bản cuối cùng)

import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
import yaml

# ======== 1. Load TOÀN BỘ kiến trúc, nhưng chỉ load trọng số part1 ========
print("🚀 Loading full architecture for head...")
# Sử dụng file yaml gốc
cfg = yaml.safe_load(open('head.yaml', 'r', encoding='utf-8'))
model = DetectionModel(cfg)

print("   Loading state_dict for part1...")
state_dict_part1 = torch.load('part1.pt', map_location='cpu' , weights_only=True)
model.load_state_dict(state_dict_part1, strict=False)  # Bỏ qua các key bị thiếu của part2
model.eval()
print("   ...Done.")

# ======== 2. Load and preprocess image ========
print("🖼️ Loading and preprocessing image...")
img = Image.open('image.png').convert('RGB')
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
x = transform(img).unsqueeze(0)
print(f"   Input image tensor shape: {x.shape}")


# ======== 3. Viết lại hàm forward để chỉ chạy qua part1 ========
def forward_head(head_model, x_in):
    # Chỉ chạy qua các lớp 0 và 1
    split_index = 4
    y = {}  # Lưu output trung gian

    # Chạy qua các lớp của head
    for layer in head_model.model[:split_index]:
        if layer.f != -1:  # Lấy input từ các lớp trước nếu cần
            x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]

        x_in = layer(x_in)  # Chạy forward
        y[layer.i] = x_in  # Lưu output

    return x_in  # Trả về output của lớp cuối cùng trong head


# ======== 4. Thực hiện forward và lưu feature map ========
print("🧠 Performing custom forward pass on head...")
with torch.no_grad():
    feature_map = forward_head(model, x)

print(f"   Output feature map shape: {feature_map.shape}")
torch.save(feature_map, 'feature_map.pt')
print("\n✅ Saved single feature map to 'feature_map.pt'")