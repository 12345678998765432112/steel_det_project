import torch
import numpy as np

print("NumPy version:", np.__version__)
print("ndarray exists:", hasattr(np, 'ndarray'))
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))