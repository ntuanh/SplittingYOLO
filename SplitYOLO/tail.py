import torch
import psutil
from ultralytics.nn.tasks import DetectionModel
import yaml
from ultralytics.utils.plotting import Annotator, colors
import cv2
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG
import numpy as np

def check_ram():
    return psutil.virtual_memory().used / (1024 ** 3)


ram_at_init = check_ram()
print("🚀 Loading full architecture for tail...")
cfg = yaml.safe_load(open('cfg/yolo11n.yaml', 'r', encoding='utf-8'))
tail_model = DetectionModel(cfg, verbose=False)
ram_after_full_archi = check_ram()
print(f"[RAM usage after loading architecture: {ram_after_full_archi - ram_at_init:.4f} GB]")
print("   Loading state_dict for part2...")
state_dict_part2 = torch.load('part2.pt', map_location='cpu', weights_only=True)
tail_model.load_state_dict(state_dict_part2, strict=False)
tail_model.eval()
print("   ...Done.")
print("🗺️ Loading feature map from 'feature_map.pt'...")
feature_map = torch.load('feature_map.pt', map_location='cpu', weights_only=True)
print(f"   Input feature map shape: {feature_map.shape}")


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


print("🧠 Performing custom forward pass on tail...")
with torch.no_grad():
    preds = forward_tail(tail_model, feature_map)
print("\n✅ Inference done.")

# Post-processing
print("🔍 Post-processing and drawing bounding boxes...")

args = DEFAULT_CFG
args.imgsz = 640

custom_predictor = DetectionPredictor(overrides=vars(args))
custom_predictor.model = tail_model

# load origin img and parameters
original_img_path = 'data/image.png'
img_to_draw = cv2.imread(original_img_path)
if img_to_draw is None:
    print(f"❌ Error: Could not read the original image at '{original_img_path}'")
    exit()

orig_imgs = [img_to_draw]
dummy_im = torch.zeros(1, 3, 640, 640)

# self.batch : (paths, images, preprocessed_images, None)
custom_predictor.batch = [original_img_path], orig_imgs, dummy_im, None

results = custom_predictor.postprocess(preds, dummy_im, orig_imgs)
result = results[0]

boxes = result.boxes
if len(boxes) > 0:
    print(f"\n✅ Found {len(boxes)} objects. Drawing them on the image...")

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
    print("\n✅ No objects found.")
    output_image = orig_imgs[0]

print("\n🖼️ Displaying result image. Press any key to close.")
cv2.imshow("Detection Result", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()