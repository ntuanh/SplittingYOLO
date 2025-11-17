import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
import yaml

# Load full model architecture but just really get part 1 weight
print("🚀 Loading full architecture for head...")
cfg = yaml.safe_load(open('cfg/head.yaml', 'r', encoding='utf-8'))
model = DetectionModel(cfg)

print("   Loading state_dict for part1...")
state_dict_part1 = torch.load('part1.pt', map_location='cpu' , weights_only=True)
model.load_state_dict(state_dict_part1, strict=False)  # Bỏ qua các key bị thiếu của part2
model.eval()
print("   ...Done.")

# Load and preprocess image
print("🖼️ Loading and preprocessing image...")
img = Image.open('data/image.png').convert('RGB')
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
x = transform(img).unsqueeze(0)
print(f"   Input image tensor shape: {x.shape}")


# rebuild forward function : just only run head
def forward_head(head_model, x_in):
    split_index = 4
    y = {}  # store features map

    # Forward layers of head
    for layer in head_model.model[:split_index]:
        if layer.f != -1:  # get input from previous layer
            x_in = y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]

        x_in = layer(x_in)  # Forward
        y[layer.i] = x_in  # save output

    return x_in  # Return last layer input


# forward and save
print("🧠 Performing custom forward pass on head...")
with torch.no_grad():
    feature_map = forward_head(model, x)

print(f"   Output feature map shape: {feature_map.shape}")
torch.save(feature_map, 'feature_map.pt')
print("\n✅ Saved single feature map to 'feature_map.pt'")