import time
from ultralytics import YOLO
import torch, yaml
from PIL import Image
import torchvision.transforms as T

with open('./cfg/config.yaml') as file:
    config = yaml.safe_load(file)


model = YOLO("yolo11n.pt")

# Load + transform image
img = Image.open("./data/image.png").convert("RGB")
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
x = transform(img)

batch = x.unsqueeze(0).repeat(int(config["batch_size"]), 1, 1, 1)

# Convert ONLY input to FP16
if torch.cuda.is_available():
    batch = batch.to("cuda").half()
else:
    batch = batch.float()

time.sleep(config["time_sleep"])

# Inference
for _ in range(int(config["nums_round"])):
    result = model(batch, verbose=False)




# 5. POSTPROCESS


# print(type(config["post_process"]))
if config["post_process"] == True :
    args = DEFAULT_CFG
    args.imgsz = 640
    preds= result
    custom_predictor = DetectionPredictor(overrides=vars(args))
    custom_predictor.model = model

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

    
    # 6. DRAW OUTPUT
    
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

print("Inference done ")


# ====================================
feature_maps = {}

def save_feature_map(name):
    def hook(module, input, output):
        feature_maps[name] = output
    return hook

# Ví dụ: gán hook vào các lớp bạn quan tâm
target_layers = [10, 15, 20]  # chỉ số layer trong model.model
for idx in target_layers:
    model.model[idx].register_forward_hook(save_feature_map(f"layer_{idx}"))

# Gọi inference như thường
output = model(batch)

# Sau khi inference xong, bạn có thể truy xuất:
for name, fmap in feature_maps.items():
    print(f"{name}: {fmap.shape}")


