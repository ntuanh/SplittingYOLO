import torch, yaml, time, gc
import torchvision.transforms as T
from PIL import Image
from SplittingYOLO.SplitYOLO.architecture2 import new_model


"""
Vì khi gọi YOLO("yolo11n.pt") để chạy forward người ta chỉ lấy cái model bên trong của file
nên mình chỉ cần thay đổi cái model và vài tham số bên trong và dùng luôn cái model đấy
không cần thay đổi các tham số khác
"""
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")
# Code lại hàm forward của ultralytics
# Return về features_map của các layers cần save
def my_new_predict_once(self, x):
    y = []# outputs
    features_map = {}
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
    for i in self.save:
        features_map[i] = y[i]
    return features_map

#Thay đổi hàm _predict_once của người ta thành của mình
new_model._predict_once = my_new_predict_once.__get__(new_model, new_model.__class__)


with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)


gc.collect()
torch.cuda.empty_cache()

time.sleep(config["time_sleep"])


# ==========================
#  PREPARE INPUT
# ==========================
img = Image.open('data/image.png').convert('RGB')
w, h = img.size
print(f"[Image size] {w}x{h}")

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

x_single = transform(img).unsqueeze(0).to(device).float()
x = x_single.repeat(int(config["batch_size"]), 1, 1, 1)


# ==========================
#  RUN LOOP
# ==========================
print("Starting inference...")


# ----- FORWARD -----
with torch.inference_mode():
    for _ in range(int(config["nums_round"])):
        state_dict = new_model._predict_once(x)

print(state_dict)


gc.collect()


# ==========================
# 6. SAVE FEATURES
# ==========================
print(f"[Type] {type(state_dict)}")
torch.save(state_dict, 'feature_map.pt')
print("\nSaved single feature map to 'feature_map.pt'")
