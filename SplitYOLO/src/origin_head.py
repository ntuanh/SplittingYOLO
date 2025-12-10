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
# Load model HEAD
# -------------------------
ckpt = torch.load("model_new.pt", map_location="cpu")

model = YOLO(ckpt["yaml"], task="detect")
model.model.load_state_dict(ckpt["model_state_dict"])
model.model.names = ckpt.get("names")

# ✅ chuyển model sang FP16
model.model = model.model.to(DEVICE).half().eval()

# -------------------------
# Prepare batch
# -------------------------
img = Image.open("data/image.png").convert("RGB")

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor()
])

x = transform(img)

N = int(config["batch_size"])

# ✅ expand() → KHÔNG copy memory
batch = x.unsqueeze(0).expand(N, -1, -1, -1).to(DEVICE).half()

time.sleep(config["time_sleep"])

# -------------------------
# HEAD inference
# -------------------------
with torch.inference_mode():
    features = model.model(batch)

# ✅ offload sang RAM
features = features.cpu()
torch.save(features, "features.pt")

print("Feature shape:", features.shape)
print("Head inference done.")
