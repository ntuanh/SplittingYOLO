import torch , psutil , yaml , time , cv2
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG
import numpy as np
from src.Utils import get_ram, get_vram, reset_vram , extract_input_layer

# ============================================================
# 0. Pick device and config
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

# ============================================================
# 1. LOAD ARCHITECTURE
# ============================================================
if config["head_architect"] == "tail" :
    yaml_file = 'cfg/tail.yaml'
else :
    yaml_file = 'cfg/yolo11n.yaml'

print(f"YAML file {yaml_file}")

cfg = yaml.safe_load(open(yaml_file, 'r', encoding='utf-8'))
tail_model = DetectionModel(cfg, verbose=False).to(device)

res_tail = extract_input_layer("yolo11n.yaml")["res_tail"]
out_put =  extract_input_layer("yolo11n.yaml")["output"]
# print(res_tail)


# ============================================================
# 2. LOAD PART2 WEIGHTS
# ============================================================

state_dict_part2 = torch.load('part2.pt', map_location=device, weights_only=True)
tail_model.load_state_dict(state_dict_part2, strict=False)
tail_model.eval()

# ============================================================
# 3. LOAD FEATURE MAP
# ============================================================

state_dict  = torch.load('feature_map.pt', map_location=device, weights_only=True)

# ============================================================
# 4. FORWARD TAIL
# ============================================================
def forward_tail(model, state_dict):
    split_index = config["cut_layer"]
    y = state_dict
    y[-1] = y[split_index - 1]
    for layer in model.model[split_index:]:
        if isinstance(layer.f, int):
            x_in = y[layer.f]
        else:
            x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]
            # y.pop(from_index , None)
        x_in = layer(x_in)
        if layer.i in res_tail :
            y[layer.i] = x_in
        y[-1] = x_in
    return x_in

with torch.no_grad():
    for _ in range(int(config["nums_round"])):
        preds = forward_tail(tail_model, state_dict)

# ============================================================
# 5. POSTPROCESS
# ============================================================

# print(type(config["post_process"]))
if config["post_process"] == True :
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

    # cv2.imshow("Detection Result", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

print("Inference done ")
