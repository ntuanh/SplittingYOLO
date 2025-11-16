import gc, torch, psutil
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
import yaml

lst_RAM = []

def check_ram():
    return psutil.virtual_memory().used / (1024 ** 3)

def measure_load_ram(model_path="yolo11n.pt", device="cuda"):
    ram_before = check_ram()

    # Load model
    model = YOLO(model_path).to(device)
    torch.cuda.synchronize()

    ram_after = check_ram()
    # print(f"RAM used: {ram_after - ram_before:.3f} GB")
    return round(ram_after - ram_before , 3)
    # 🔥 XÓA HOÀN TOÀN MODEL
    del model                  # xóa tham chiếu chính
    gc.collect()               # ép garbage collector dọn rác
    torch.cuda.emty_cache()   # dọn VRAM cache
    ram_after_cleanup = check_ram()

    print(f"RAM after cleanup: {ram_after_cleanup:.3f} GB")
    print("=" * 40)

def head():
    #
    ram_before = check_ram()
    cfg = yaml.safe_load(open('tail.yaml', 'r', encoding='utf-8'))
    model = DetectionModel(cfg, verbose = False)
    print("   Loading state_dict for model...")
    state_dict_part1 = torch.load('part2.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict_part1, strict=False)  # Bỏ qua các key bị thiếu của part2
    model.eval()
    ram_after = check_ram()
    return round(ram_after - ram_before, 3)


# Thử đo nhiều lần
for i in range(1):
    ram = head()
    print(ram)
