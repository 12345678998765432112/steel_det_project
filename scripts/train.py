import os
import torch
from ultralytics import YOLO
from datetime import datetime
import time

# ====== CUDA 初始化与清理 ======
torch.cuda.init()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # 加速，但可能略微增加首次加载时间

# 配置
DATA_YAML = "/home/ubuntu/Project/data/severstal.yaml"
MODEL_PATH = "/home/ubuntu/Project/models/yolov8m.pt"  # 如果想用更新模型，可换 yolov8m.pt / yolo11l.pt 等
EPOCHS = 250          
BATCH_SIZE = 8     # 5090 建议从 16 开始，显存富裕可逐步加到 32
IMG_SIZE = 1280
SAVE_DIR = f"/home/ubuntu/Project/train_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def print_gpu_status():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
        print(f"当前显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB / 已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GiB")

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"✅ 训练设备：{device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print_gpu_status()
    print(f"✅ 数据集配置：{DATA_YAML}")
    print(f"✅ 训练结果将保存至：{SAVE_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ 成功加载模型：{MODEL_PATH}")
        print(f"模型参数量: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")

        # 开始训练
        print("\n开始训练...")
        start_time = time.time()

        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=device,
            project=SAVE_DIR,
            name="yolov8m_steel_det",
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            cos_lr=True,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            save=True,
            save_period=10,

            hsv_h=0.015,   # 色调增强保持小（钢板颜色变化不大）
            hsv_s=0.8,     # 饱和度增强强一点，帮助低对比缺陷
            hsv_v=0.6,     # 亮度增强
            degrees=10.0,  # 旋转 ±10°
            translate=0.15,# 平移
            scale=0.9,     # 关键！随机缩放范围更大，帮助小目标
            shear=5.0,     # 剪切
            perspective=0.0002,
            flipud=0.0,    # 上下翻转关闭（钢板方向有意义）
            fliplr=0.5,    # 左右翻转开启
            mosaic=1.0,    # 全开 mosaic
            mixup=0.3,     # mixup 加强，对类别不平衡很有效
            copy_paste=0.3,# copy_paste 也加强，少数类缺陷收益大
            close_mosaic=10,  # 最后 10 epoch 关闭 mosaic，避免过拟合

            val=True,
            plots=True,
            augment=True,
            overlap_mask=True,
            amp=True,              # 混合精度训练，节省显存 + 加速
            workers=12,             # 数据加载线程数
            cache='ram',            # 如果内存够，缓存数据集到 RAM
            patience=50,           # 早停（可选）
            seed=42,
            single_cls=False,
            mask_ratio=4,
            dropout=0.0,

            box=7.5,               # box loss 权重默认 7.5，保持
            cls=0.5,
            dfl=1.5,
            pose=0.0,
            kobj=1.0,
        )

        print(f"\n训练完成！耗时: {(time.time() - start_time)/3600:.2f} 小时")

        print("\n🎉 开始验证最终模型...")
        val_results = model.val(
            data=DATA_YAML,
            imgsz=IMG_SIZE,
            device=device,
            batch=16,
            plots=True,
        )

        print(f"✅ 最终模型 mAP@50    : {val_results.box.map50:.4f}")
        print(f"✅ 最终模型 mAP@50:95 : {val_results.box.map:.4f}")
        print(f"✅ 所有结果保存至：{SAVE_DIR}/yolov8m_steel_det")

    except Exception as e:
        print("\n训练过程中发生错误：")
        print(e)
        import traceback
        traceback.print_exc()
        print("\n建议检查：")
        print("1. 数据集路径和 yaml 是否正确")
        print("2. labels 文件是否都存在且格式正确")
        print("3. 显存是否足够（可降低 batch 或 imgsz）")
        print("4. 运行 nvidia-smi 查看显存占用")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()