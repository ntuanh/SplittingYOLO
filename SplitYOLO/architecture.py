import torch
from ultralytics import YOLO
from collections import OrderedDict

# load origin model
print("Loading original YOLOv11n model...")
model = YOLO("yolo11s.pt").model

# get split layer
split_index = 4
print(f"\nSplitting model at layer index = {split_index}")

# separate weight dict
full_state_dict = model.state_dict()
# print(full_state_dict.keys())
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
        continue

    try:
        layer_index = int(key.split('.')[1])

        if layer_index < split_index:
            part1_state_dict[key] = value
        else:
            part2_state_dict[key] = value

    except (ValueError, IndexError):
        # Load key of detect ( purpose that it in the end of progress )
        part2_state_dict[key] = value

print(f"   Part 1 has {len(part1_state_dict)} keys.")
print(f"   Part 2 has {len(part2_state_dict)} keys.")

# save 2 dicts
torch.save(part1_state_dict, "part1.pt")
torch.save(part2_state_dict, "part2.pt")

print("\nState dictionaries saved with original keys:")
print(f" - part1.pt (layers 0 → {split_index - 1})")
print(f" - part2.pt (layers {split_index} → end)")
