import time
from ultralytics import YOLO
import torch , yaml , cv2
from PIL import Image
import torchvision.transforms as T
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.nn.tasks import DetectionModel

# with open('./cfg/config.yaml') as file:
#     config = yaml.safe_load(file)

# Load model
# model = YOLO("yolo11n.pt")
#
# detection_model = DetectionModel("../cfg/head.yaml" )
#
# print(f"[Type] {type(model)}")
#
# print(f"model of model {type(model.model)}")

import torch

ckpt = torch.load("yolo11n.pt", map_location="cpu")
print(type(ckpt['model']))
print(type(ckpt['model'].state_dict().keys()))
print(ckpt['model'].state_dict().keys())

# print(type(ckpt))
# print(ckpt.keys())

