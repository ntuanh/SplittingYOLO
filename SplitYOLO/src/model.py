import torch , yaml
from ultralytics import YOLO

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

print("Loading original YOLOv11n model...")
model = YOLO("yolo11n.pt")

state_dict_part1 = torch.load('part1.pt', map_location='cpu', weights_only=True)

missing, unexpected = model.model.load_state_dict(state_dict_part1, strict=False)

# Save file weight follow yolo11n.pt
torch.save({
    "model": model.model,                  # nn.Module
    "model_state_dict": model.model.state_dict(),  # state_dict after update
    "yaml": "head.yaml",                # YAML of model
    "names": model.names,                  # class names
    "task": "detect"
}, "model_new.pt")

print("Saved new model as 'model_new.pt' . ")