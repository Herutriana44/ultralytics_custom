import os
from ultralytics import YOLO
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='-1', help='cpu, 0, 0,1,2... atau -1 (auto)')
args = parser.parse_args()

# Path dataset
DATASET_DIR = '../dataset_medis'
IMAGE_DIR = os.path.join(DATASET_DIR, 'image')
MASK_DIR = os.path.join(DATASET_DIR, 'mask')
DATA_YAML = os.path.join(DATASET_DIR, 'data_medis.yaml')  # pastikan sudah sesuai format YOLO

# 1. Training
# Pastikan model swin transformer sudah tersedia di ultralytics, misal: 'yolov8s-swin.pt'
# Jika tidak, gunakan model YOLOv8 biasa: 'yolov8s.pt'
model = YOLO('yolov8s.pt')  # Ganti dengan 'yolov8s-swin.pt' jika ada

# Training
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    project='runs/train',
    name='yolov8_swin_custom',
    device=args.device
)

# 2. Export Model
# Export ke format ONNX, TorchScript, atau lainnya
model.export(format='onnx', dynamic=True)
print("Model exported to ONNX format.")
