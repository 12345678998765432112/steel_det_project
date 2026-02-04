import torch
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())
model = YOLO("/home/ubuntu/Project/models/yolov8s.pt")  # 用你已有的模型路径
model = model.to('cuda')
print("Model on device:", next(model.parameters()).device)
dummy_input = torch.randn(1, 3, 800, 800).to('cuda')
print("Dummy forward pass OK")