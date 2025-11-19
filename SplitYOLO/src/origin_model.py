# from ultralytics import YOLO
# import cv2
# import time
# import psutil
# import torch , yaml , tracemalloc
# from Utils import get_ram , get_vram , reset_vram
# from ultralytics.nn.tasks import DetectionModel
# from ultralytics.utils.plotting import Annotator, colors
# from ultralytics.models.yolo.detect import DetectionPredictor
# import torchvision.transforms as T
# from PIL import Image
#
# # pick device
# # device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[DEVICE] { device}")
#
# ram_before = get_ram()
# vram_before =  reset_vram()
# t0 = time.time()
#
# cfg = yaml.safe_load(open('./cfg/yolo11n.yaml', 'r', encoding='utf-8'))
# model = DetectionModel(cfg, verbose=False).to(device)
#
# t1 = time.time()
# ram_after = get_ram()
# vram_after = get_vram()
# print(f"[Architecture] Time: {t1 - t0:.4f} s")
# print(f"[Architecture] RAM used:  {(ram_after - ram_before):.4f} MB")
# print(f"[Architecture] VRAM used: {(vram_after):.4f} MB")
#
# # ==========================
# # 2. LOAD WEIGHTS
# # ==========================
# ram_before = get_ram()
# vram_before = reset_vram()
# t0 = time.time()
#
# state_dict = torch.load('./yolo11weights.pt', map_location=device, weights_only=True)
# model.load_state_dict(state_dict, strict=False)
# model.eval()
#
# t1 = time.time()
# ram_after = get_ram()
# vram_after = reset_vram()
#
# print(f"[Weights] Time: {t1 - t0:.4f} s")
# print(f"[Weights] RAM used:  {(ram_after - ram_before):.4f} MB")
# print(f"[Weights] VRAM used: {(vram_after - vram_before):.4f} MB")
#
# # ============================================================
# # forward function
# # ============================================================
# img = Image.open('./data/image.png').convert('RGB')
# transform = T.Compose([
#     T.Resize((640, 640)),
#     T.ToTensor(),
# ])
# x = transform(img).unsqueeze(0)
# def forward(model, x):
#     y = {}             # store outputs of layers
#     current_x = x      # input image
#
#     for layer in model.model:
#         if isinstance(layer.f, int):
#             x_in = current_x if layer.f == -1 else y[layer.f]
#         else:
#             x_in = [(current_x if j == -1 else y[j]) for j in layer.f]
#
#         current_x = layer(x_in)
#
#         y[layer.i] = current_x
#
#     return current_x
#
#
# # 3.2 Check cuda available
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = model.to(device)
# x = x.to(device)
# # print(f"[DEVICE] : {device}")
#
# # ============================================================
# # 4. FORWARD TAIL
# # ============================================================
#
# ram_before = get_ram()
# vram_before = reset_vram()
# t0 = time.time()
#
# with torch.inference_mode():
#     for _ in range(1000):
#         preds = forward(model,x)
#
# t1 = time.time()
# ram_after = get_ram()
# vram_after = get_vram()
#
# print(f"[Forward] Time: {t1 - t0:.4f} s")
# print(f"[Forward] RAM used:  {(ram_after - ram_before):.4f} MB")
# print(f"[Forward] VRAM used: {(vram_after - vram_before):.4f} MB")
#
# print("\nInference done.")

# # ------------------------------
# # Load model
# # ------------------------------
# start_ram = get_ram()
# start_vram = get_vram()
#
# model = YOLO("yolo11n.pt")
#
# load_ram = get_ram()
# load_vram = get_vram()
#
# print(f"[Model Load] RAM used: {load_ram - start_ram:.2f} MB")
# print(f"[Model Load] VRAM used: {load_vram - start_vram:.2f} MB")
#
# # ------------------------------
# # Inference on image
# # ------------------------------
# img = cv2.imread("./data/image.png")
#
# t0 = time.time()
# ram_before = get_ram()
# vram_before = get_vram()
#
# results = model(img)[0]
#
# t1 = time.time()
# ram_after = get_ram()
# vram_after = get_vram()
#
# print(f"[Inference Time] {t1 - t0:.4f} seconds")
# print(f"[RAM Increase] {ram_after - ram_before:.2f} MB")
# print(f"[VRAM Increase] {vram_after - vram_before:.2f} MB")
#
# # ------------------------------
# # Draw results
# # ------------------------------
# for box in results.boxes:
#     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#     conf = float(box.conf[0])
#     cls_id = int(box.cls[0])
#     label = f"{model.names[cls_id]} {conf:.2f}"
#
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(img, label, (x1, y1 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
# # Show result
# cv2.imshow("YOLO11n Detection", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ========================
# SIMPLE Code with YOLO
# ========================
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")   # or yolov8n.pt

for _ in range(1000):
    results = model("./data/image.png" , verbose=False)[0]

# Print only class names
names = model.names

printed = set()  # avoid duplicate names
for box in results.boxes:
    cls_id = int(box.cls[0])
    printed.add(names[cls_id])

for name in printed:
    print(name)

