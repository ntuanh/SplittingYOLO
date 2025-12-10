import torch
import yaml
import time
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

# -------------------------
# Load config
# -------------------------
with open("cfg/config.yaml") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load HEAD checkpoint
# -------------------------
ckpt = torch.load("model_new.pt", map_location="cpu")

model = YOLO(ckpt["yaml"], task="detect")
model.model.load_state_dict(ckpt["model_state_dict"])
model.model.names = ckpt.get("names")
model.model.to(DEVICE).eval()

print("HEAD loaded on:", DEVICE)

# -------------------------
# Load + preprocess image
# -------------------------
img = Image.open("data/image.png").convert("RGB")

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor()
])

x = transform(img)  # [3, 640, 640]

batch_size = int(config["batch_size"])

# ✅ NO DATA COPYING
batch = x.unsqueeze(0).expand(batch_size, -1, -1, -1).to(DEVICE).float()

# -------------------------
# Run HEAD forward
# -------------------------
time.sleep(config["time_sleep"])

print("Running head inference...")

with torch.no_grad():
    for _ in range(int(config["nums_round"])):
        features = model.model(batch)
        del features
        torch.cuda.empty_cache()

# last run for saving
with torch.no_grad():
    features = model.model(batch)

features = features.half().cpu()
torch.save(features, "features.pt")

print("Feature tensor shape:", features.shape)
print("Head inference finished, features saved.")

