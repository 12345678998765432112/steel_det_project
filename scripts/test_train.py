from ultralytics import YOLO
import os

def train_steel_defect():
    # ====================== 核心配置（只改这里）======================
    config_path = "/home/ubuntu/Project/steel_dataset/data.yaml"
    # 测试阶段用 yolov8s.pt，正式训练再换成 yolov8m.pt
    model_name = "/home/ubuntu/Project/models/yolov8s.pt"  # 或 "yolov8m.pt"
    epochs = 10          # 测试用10轮
    batch_size = 8       # 根据显存调整，测试用8即可
    imgsz = 1600         # 你的图片宽度是1600
    device = 0 if os.path.exists("/dev/nvidia0") else "cpu"
    # ================================================================

    print(f"📌 加载模型：{model_name}")
    model = YOLO(model_name)

    print(f"\n🚀 开始训练（{epochs}轮，{batch_size}批次，设备：{device}）")
    results = model.train(
        data=config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        patience=3,
        save=True,
        project="runs",
        name="v8s_test",  # 用m模型时改成 "v8m_test"
        exist_ok=True,
        verbose=True,
        rect=False,
        mosaic=0.0,
        overlap_mask=True,
        mask_ratio=4
    )

    print("\n✅ 训练完成！")
    print(f"📊 最佳模型路径：{results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_steel_defect()