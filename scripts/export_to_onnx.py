from ultralytics import YOLO
import os
from pathlib import Path

# ===================== 配置部分 =====================
MODEL_PATH = "/home/ubuntu/Project/train_models/steel_yolov8s_seg_debug_gpu_weight/best.pt"          # 你的模型权重路径（可改）
EXPORT_DIR = "/home/ubuntu/Project/train_models/steel_yolov8s_seg_debug_gpu_weight"
EXPORT_NAME = "best.onnx"   # 导出的文件名（可自定义）

# 确保导出目录存在
os.makedirs(EXPORT_DIR, exist_ok=True)

# 完整导出路径
export_path = str(Path(EXPORT_DIR) / EXPORT_NAME)

# ===================== 加载并导出 =====================
print(f"加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"开始导出 ONNX 到: {export_path}")

success = model.export(
    format="onnx",
    imgsz=1024,
    dynamic=True,           # 支持动态 batch / 尺寸
    simplify=True,          # 简化模型图
    opset=12,               # 常用兼容性好的 opset 版本
    project=EXPORT_DIR,     # 项目目录（ultralytics 会生成子文件夹）
    name=Path(EXPORT_NAME).stem,  # 只取文件名（不带 .onnx），避免多层目录
)

if success:
    print("\n导出成功！")
    print(f"ONNX 文件路径: {export_path}")
else:
    print("\n导出失败，请检查日志或路径权限。")