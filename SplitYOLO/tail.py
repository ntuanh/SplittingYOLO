import torch , psutil , yaml , time , cv2
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG
import numpy as np
from src.Utils import get_ram, get_vram, reset_vram

# ============================================================
# 0. Pick device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

# ============================================================
# 1. LOAD ARCHITECTURE
# ============================================================

ram_before = get_ram()
vram_before = reset_vram()
t0 = time.time()

cfg = yaml.safe_load(open('cfg/yolo11n.yaml', 'r', encoding='utf-8'))
tail_model = DetectionModel(cfg, verbose=False).to(device)

t1 = time.time()
ram_after = get_ram()
vram_after = get_vram()
print(f"[Architecture] Time: {t1 - t0:.4f} s")
print(f"[Architecture] RAM used: {ram_after - ram_before:.4f} MB")
print(f"[Architecture] VRAM used: {vram_after - vram_before:.4f} MB")

# ============================================================
# 2. LOAD PART2 WEIGHTS
# ============================================================

ram_before = get_ram()
vram_before = reset_vram()
t0 = time.time()

state_dict_part2 = torch.load('part2.pt', map_location=device, weights_only=True)
tail_model.load_state_dict(state_dict_part2, strict=False)
tail_model.eval()

t1 = time.time()
ram_after = get_ram()
vram_after = get_vram()
print(f"[Weights] Time: {t1 - t0:.4f} s")
print(f"[Weights] RAM used: {ram_after - ram_before:.4f} MB")
print(f"[Weights] VRAM used: {vram_after - vram_before:.4f} MB")

# ============================================================
# 3. LOAD FEATURE MAP
# ============================================================

ram_before = get_ram()
vram_before = reset_vram()
t0 = time.time()

feature_map = torch.load('feature_map.pt', map_location=device, weights_only=True)

t1 = time.time()
ram_after = get_ram()
vram_after = get_vram()
print(f"[Feature Map] Time: {t1 - t0:.4f} s")
print(f"[Feature Map] RAM used: {ram_after - ram_before:.4f} MB")
print(f"[Feature Map] VRAM used: {vram_after - vram_before:.4f} MB")

# ============================================================
# 4. FORWARD TAIL
# ============================================================
def forward_tail(model, feature_map_in):
    split_index = 4
    y = {}
    current_x = feature_map_in
    y[split_index - 1] = current_x
    for layer in model.model[split_index:]:
        if isinstance(layer.f, int):
            if layer.f == -1:
                x_in = current_x
            else:
                x_in = y[layer.f]
        else:
            x_in = []
            for from_index in layer.f:
                if from_index == -1:
                    x_in.append(current_x)
                else:
                    x_in.append(y[from_index])
        current_x = layer(x_in)
        y[layer.i] = current_x
    return current_x

ram_before = get_ram()
vram_before = reset_vram()
t0 = time.time()

with torch.no_grad():
    preds = forward_tail(tail_model, feature_map)

t1 = time.time()
ram_after = get_ram()
vram_after = get_vram()
print(f"[Tail Forward] Time: {t1 - t0:.4f} s")
print(f"[Tail Forward] RAM used: {ram_after - ram_before:.4f} MB")
print(f"[Tail Forward] VRAM used: {vram_after - vram_before:.4f} MB")
# ============================================================
# 5. POSTPROCESS
# ============================================================
ram_before = get_ram()
vram_before = reset_vram()
t0 = time.time()

args = DEFAULT_CFG
args.imgsz = 640
custom_predictor = DetectionPredictor(overrides=vars(args))
custom_predictor.model = tail_model

# load origin img and parameters
original_img_path = 'data/image.png'
img_to_draw = cv2.imread(original_img_path)
if img_to_draw is None:
    print(f"[Error]: Could not read the original image at '{original_img_path}'")
    exit()

orig_imgs = [img_to_draw]
dummy_im = torch.zeros(1, 3, 640, 640)
custom_predictor.batch = [original_img_path], orig_imgs, dummy_im, None

results = custom_predictor.postprocess(preds, dummy_im, orig_imgs)
result = results[0]

t1 = time.time()
ram_after = get_ram()
vram_after = get_vram()
print(f"[Postprocess] Time: {t1 - t0:.4f} s")
print(f"[Postprocess] RAM used: {ram_after - ram_before:.4f} MB")
print(f"[Postprocess] VRAM used: {vram_after - vram_before:.4f} MB")

# ============================================================
# 6. DRAW OUTPUT
# ============================================================
boxes = result.boxes
if len(boxes) > 0:
    annotator = Annotator(orig_imgs[0], line_width=2, example=str(result.names))
    for box in boxes:
        class_id = int(box.cls)
        coords = box.xyxy[0].tolist()
        conf = float(box.conf)
        class_name = result.names[class_id]
        label = f'{class_name} {conf:.2f}'
        print(f"   - Object: {class_name}, Confidence: {conf:.2f}")
        annotator.box_label(coords, label, color=colors(class_id + 1, True))
    output_image = annotator.result()
else:
    output_image = orig_imgs[0]

cv2.imshow("Detection Result", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
