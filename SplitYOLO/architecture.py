# architecture.py (Phiên bản cuối cùng)

import torch
from ultralytics import YOLO
from collections import OrderedDict

# ================================
# ⚙️ 1. Load model YOLO11 gốc
# ================================
print("🚀 Loading original YOLOv11n model...")
model = YOLO("yolo11n.pt").model

# ================================
# ✂️ 2. Chọn điểm split
# ================================
split_index = 4
print(f"\n🔧 Splitting model at layer index = {split_index}")

# ================================
# 🧩 3. Tách state_dict mà KHÔNG THAY ĐỔI KEY
# ================================
full_state_dict = model.state_dict()
part1_state_dict = OrderedDict()
part2_state_dict = OrderedDict()

first_key = next(iter(full_state_dict))
first_value = full_state_dict[first_key]

print("[KEYS OF FULL STATE DICT ")
keys = list(full_state_dict.keys())
for key in keys :
    print( key)

print(f"Full state dict length :{len(full_state_dict)}")
print(f"Full state dict type :{type(full_state_dict)}")
# print(f"Part1 state dict length :{len(part2_state_dict)}")
# print(f"Part2 state dict length :{len(part1_state_dict)}")
print("   Processing state_dict keys...")
for key, value in full_state_dict.items():
    if not key.startswith('model.'):
        continue  # Bỏ qua các key không thuộc model

    try:
        layer_index = int(key.split('.')[1])

        if layer_index < split_index:
            part1_state_dict[key] = value
        else:
            part2_state_dict[key] = value

    except (ValueError, IndexError):
        # Key của Detect head có thể không theo quy tắc
        # Giả sử chúng luôn thuộc phần cuối
        part2_state_dict[key] = value

print(f"   Part 1 has {len(part1_state_dict)} keys.")
print(f"   Part 2 has {len(part2_state_dict)} keys.")

# ================================
# 💾 4. Lưu 2 state_dict thành file riêng
# ================================
torch.save(part1_state_dict, "part1.pt")
torch.save(part2_state_dict, "part2.pt")

print("\n✅ State dictionaries saved with original keys:")
print(f" - part1.pt (layers 0 → {split_index - 1})")
print(f" - part2.pt (layers {split_index} → end)")