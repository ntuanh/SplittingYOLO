# tail.py (Phiên-bản-sửa-lỗi-NoneType-và-hiển-thị)

import torch
import psutil
from ultralytics.nn.tasks import DetectionModel
import yaml
from ultralytics.utils.plotting import Annotator, colors
import cv2
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG
import numpy as np


# ... (Phần 1, 2, 3, 4 giữ nguyên y hệt như phiên bản trước) ...
def check_ram():
    return psutil.virtual_memory().used / (1024 ** 3)


ram_at_init = check_ram()
print("🚀 Loading full architecture for tail...")
cfg = yaml.safe_load(open('yolo11n.yaml', 'r', encoding='utf-8'))
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

# ================================================================= #
# ======== 5. Hậu xử lý và vẽ Bounding Box (ĐÃ SỬA) ========
print("🔍 Post-processing and drawing bounding boxes...")

# 5.1. Tạo một đối tượng predictor tùy chỉnh
args = DEFAULT_CFG
# args.model = 'yolo11n.pt'
args.imgsz = 640

custom_predictor = DetectionPredictor(overrides=vars(args))
custom_predictor.model = tail_model

# 5.2. Load ảnh gốc và chuẩn bị các tham số cần thiết
original_img_path = 'image.png'
img_to_draw = cv2.imread(original_img_path)
if img_to_draw is None:
    print(f"❌ Error: Could not read the original image at '{original_img_path}'")
    exit()

# Ảnh gốc cần được chuyển thành một list numpy array để truyền vào postprocess
orig_imgs = [img_to_draw]
# Ảnh đã được tiền xử lý (resize, to tensor, v.v.)
# Chúng ta sẽ tạo một phiên bản giả lập
dummy_im = torch.zeros(1, 3, 640, 640)

# ✅ SỬA LOGIC Ở ĐÂY: "Giả lập" thuộc tính `batch` cho predictor
# self.batch cần có cấu trúc: (paths, images, preprocessed_images, None)
custom_predictor.batch = [original_img_path], orig_imgs, dummy_im, None

# 5.3. Gọi hàm postprocess từ predictor đã tạo
# preds là đầu ra thô từ mô hình, orig_imgs là ảnh gốc chưa resize
results = custom_predictor.postprocess(preds, dummy_im, orig_imgs)
result = results[0]

# 5.4. Vẽ kết quả lên ảnh
boxes = result.boxes
if len(boxes) > 0:
    print(f"\n✅ Found {len(boxes)} objects. Drawing them on the image...")

    # Không cần load lại img_to_draw vì đã có orig_imgs[0]
    annotator = Annotator(orig_imgs[0], line_width=2, example=str(result.names))

    for box in boxes:
        class_id = int(box.cls)
        # Tọa độ box.xyxy đã được scale về kích thước ảnh gốc bởi hàm postprocess
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

# 5.5. Hiển thị ảnh kết quả
print("\n🖼️ Displaying result image. Press any key to close.")
cv2.imshow("Detection Result", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ================================================================= #