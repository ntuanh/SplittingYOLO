# SplitYOLO

SplitYOLO is a project designed to optimize YOLO (You Only Look Once) model inference on resource-constrained edge devices by splitting the model weights into two parts. This approach significantly reduces RAM and VRAM usage, enabling efficient deployment on devices like the NVIDIA Jetson Nano.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Traditional YOLO models require substantial memory for inference, which can be prohibitive on edge devices with limited RAM and VRAM. SplitYOLO addresses this by dividing the model into a head and tail component, allowing distributed processing across devices connected via Wi-Fi LAN. This not only lowers memory consumption but also potentially speeds up inference by parallelizing computations.

### Key Features
- **Model Splitting**: Divides YOLO11n weights into `part1.pt` (head layers) and `part2.pt` (tail layers).
- **Memory Optimization**: Reduces RAM usage by 200-300 MB depending on the cut layer.
- **Edge Device Support**: Optimized for devices like the Jetson Nano.
- **Distributed Inference**: Supports processing across multiple devices over a network.

## Prerequisites

- Python 3.8 or higher
- PyTorch (compatible with your device)
- Ultralytics YOLO library
- OpenCV (for image processing)
- PIL (Pillow) for image handling
- YAML for configuration
- Access to edge devices (e.g., Jetson Nano) with Wi-Fi connectivity

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/SplitYOLO.git
   cd SplitYOLO
   ```

2. Install dependencies:
   ```bash
   pip install torch ultralytics opencv-python pillow pyyaml
   ```

3. Download the YOLO11n model weights (`yolo11n.pt`) and place it in the `SplitYOLO/` directory.

## Usage

1. **Split the Model**:
   Run `architecture.py` to split the model weights:
   ```bash
   python SplitYOLO/architecture.py
   ```
   This generates `part1.pt` and `part2.pt`.

2. **Run Head Inference**:
   On the first device, execute `head.py` to process the input image through the head layers:
   ```bash
   python SplitYOLO/head.py
   ```
   This produces `feature_map.pt`.

3. **Transfer Feature Map**:
   Send `feature_map.pt` to the second device via Wi-Fi LAN.

4. **Run Tail Inference**:
   On the second device, execute `tail.py` to complete the inference:
   ```bash
   python SplitYOLO/tail.py
   ```
   This outputs the detection results and bounding boxes.

For batch processing or multiple rounds, use `run_many.py` or modify `main.py` accordingly.

## Configuration

Configure the splitting and inference parameters in `SplitYOLO/cfg/config.yaml`:

- `cut_layer`: Layer index to split the model (default: 12)
- `batch_size`: Number of images to process in a batch (default: 1)
- `head_architect`: Architecture for head (default: yolo11)
- `tail_architect`: Architecture for tail (default: tail)
- `nums_round`: Number of inference rounds (default: 1)
- `post_process`: Enable post-processing for bounding boxes (default: False)

Additional configurations are in `cfg/head.yaml`, `cfg/tail.yaml`, and `cfg/yolo11n.yaml`.

## How It Works

- **[architecture.py](SplitYOLO/architecture.py)**: Splits the YOLO model weights into two parts: `part1.pt` for head layers and `part2.pt` for tail layers based on the specified cut layer.
- **[head.py](SplitYOLO/head.py)**: Takes raw image data as input, performs inference through the head layers, and outputs `feature_map.pt` (including skip connection feature maps).
- **[tail.py](SplitYOLO/tail.py)**: Receives `feature_map.pt`, processes it through the tail layers, and returns detection results and bounding boxes if enabled.

The process leverages distributed computing over Wi-Fi LAN to balance load and reduce memory per device.

## Results

### RAM Usage
- Full model at cut_layer 12: Approximately 12 (units, e.g., GB or as per system).
- Optimized at cut_layer 4: Reduces RAM by 200-300 MB.

### Inference Time
- Forward time on each device: Approximately half the time of the full model inference.

These results demonstrate significant memory savings and potential speed improvements, making SplitYOLO ideal for edge deployments.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

