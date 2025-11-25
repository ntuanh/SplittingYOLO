import torch , yaml , time
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

with open('./cfg/config.yaml') as file:
    config = yaml.safe_load(file)

# Load model
model = YOLO("model_new.pt")

# Load + transform image
img = Image.open("./data/image.png").convert('RGB')
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
x = transform(img)  # [3, 640, 640]

batch = x.unsqueeze(0).repeat(int(config["batch_size"]), 1, 1, 1)  # [30, 3, 640, 640]

batch = batch.to(model.device).float()

time.sleep(config["time_sleep"])
# Inference a lot of times
for _ in range(int(config["nums_round"])):
    results = model(batch, verbose=False)

print("Inference head done !")