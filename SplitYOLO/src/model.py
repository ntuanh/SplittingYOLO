import torch , yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


print("Loading original YOLOv11n model...")
model = YOLO("yolo11n.pt")

state_dict_part1 = torch.load('part1.pt', map_location='cpu', weights_only=True)
detection_model = DetectionModel("cfg/head.yaml" )

# Save file weight follow yolo11n.pt
torch.save({
    "model": detection_model,                  # nn.Module
    "model_state_dict": state_dict_part1 ,   # state_dict after update
    "yaml": "head.yaml",                # YAML of model
    "names": model.names,                  # class names
    "task": "detect"
}, "model_new.pt")

print("Saved new model as 'model_new.pt' . ")