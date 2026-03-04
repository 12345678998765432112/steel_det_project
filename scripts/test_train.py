"""用seg的模型训练"""

from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import torch

# 强制 CUDA 错误同步，便于调试（GPU 时有用）
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def diagnose_labels(data_yaml_path):
    """
    全量扫描所有 labels 文件，检查：
    - class id 是否在 0 ~ nc-1 范围内
    - 坐标是否为偶数个、是否都在 [0,1] 范围内
    - 是否存在退化 polygon（面积 ≈ 0）
    """
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        nc = cfg.get('nc', 0)
        print(f"data.yaml nc = {nc}")
        if nc != 4:
            print("!!! nc 不是 4，请检查 yaml !!!")
            return
    except Exception as e:
        print(f"读取 data.yaml 失败: {e}")
        return

    dataset_root = Path(data_yaml_path).parent
    label_roots = [
        dataset_root / "train" / "labels",
        dataset_root / "val" / "labels",
    ]

    bad_classes = {}
    format_issues = []
    total_files = 0
    total_lines = 0

    for root in label_roots:
        if not root.exists():
            print(f"目录不存在，跳过: {root}")
            continue

        for txt_path in root.rglob("*.txt"):
            total_files += 1
            if txt_path.stat().st_size == 0:
                continue  # 空文件 = 无缺陷，正常

            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        total_lines += 1
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            format_issues.append(
                                f"格式异常（列数 < 5）: {txt_path} 行{line_num} → {line}"
                            )
                            continue

                        try:
                            cls = int(float(parts[0]))
                            if cls < 0 or cls >= nc:
                                bad_classes.setdefault(cls, []).append(str(txt_path))
                                print(
                                    f"越界类别 {cls} (合法 0~{nc-1}): {txt_path} 行{line_num} → {line}"
                                )
                                continue  # 可以选择跳过或继续

                            # 检查坐标部分
                            coords = [float(p) for p in parts[1:]]
                            if len(coords) % 2 != 0:
                                format_issues.append(
                                    f"坐标不是偶数个: {txt_path} 行{line_num} → {line}"
                                )
                                continue

                            xs = coords[0::2]
                            ys = coords[1::2]
                            if not xs:
                                format_issues.append(
                                    f"无坐标点: {txt_path} 行{line_num}"
                                )
                                continue

                            # 范围检查 [0,1]
                            invalid_coords = any(
                                x < 0 or x > 1 or y < 0 or y > 1
                                for x, y in zip(xs, ys)
                            )
                            if invalid_coords:
                                format_issues.append(
                                    f"坐标超出 [0,1]: {txt_path} 行{line_num} → {line}"
                                )

                            # 退化 polygon 检查（简单 min-max bbox 面积）
                            if len(xs) >= 3:
                                w = max(xs) - min(xs)
                                h = max(ys) - min(ys)
                                if w < 1e-5 or h < 1e-5:
                                    format_issues.append(
                                        f"退化 polygon（面积≈0）: {txt_path} 行{line_num} → {line}"
                                    )

                        except ValueError as ve:
                            format_issues.append(
                                f"解析失败（class 或坐标非数字）: {txt_path} 行{line_num} → {line} ({ve})"
                            )

            except Exception as e:
                print(f"读取文件失败 {txt_path}: {e}")

    print(f"\n扫描完成：共 {total_files} 个 txt 文件，处理 {total_lines} 行有效标注")

    if format_issues:
        print("\n发现格式问题（前 20 条显示）：")
        for issue in format_issues[:20]:
            print(issue)
        if len(format_issues) > 20:
            print(f"... 还有 {len(format_issues)-20} 条类似问题")
        print("\n请先修复以上问题文件/行，再开始训练。")
    else:
        print("所有检查通过：class id 和坐标格式看起来正常")

    if bad_classes:
        print("\n!!! 发现越界类别 !!!")
        for cls, files in sorted(bad_classes.items()):
            print(f"  class = {cls} 出现在 {len(files)} 个文件中（示例前3个）：")
            for f in files[:3]:
                print(f"    - {f}")
        print("\n请修正这些文件中的 class id")


def train_steel_defect():
    # ====================== 核心配置 ======================
    config_path = "/home/ubuntu/Project/steel_dataset/data.yaml"
    model_name   = "/home/ubuntu/Project/models/yolov8s-seg.pt"
    epochs       = 30
    batch_size   = 12               # CPU 调试建议小一些
    imgsz        = 1024
    device       = 0 if torch.cuda.is_available() else "cpu"           # 先用 CPU 看清晰报错
    # ================================================================

    base_save_path = os.path.expanduser("/home/ubuntu/Project/runs")
    exp_name = "steel_yolov8s_seg_debug_gpu"

    os.makedirs(base_save_path, exist_ok=True)

    print(f"📌 加载模型：{model_name}")
    model = YOLO(model_name)

    print(f"\n🚀 开始训练（{epochs} epochs, batch={batch_size}, imgsz={imgsz}, device={device}）")

    results = model.train(
        data=config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        patience=15,
        save=True,
        project=base_save_path,
        name=exp_name,
        exist_ok=True,
        verbose=False,
        task="segment",
        rect=True,
        mosaic=0.0,
        mixup=0.0,
        overlap_mask=True,
        mask_ratio=4,
        workers=8,               # CPU 建议关闭多线程加载
        single_cls=False,
        amp=True,               # CPU 上关闭混合精度
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

    final_save_path = os.path.join(base_save_path, exp_name)
    print("\n✅ 训练完成！")
    print(f"📊 最佳模型：{final_save_path}/weights/best.pt")
    print(f"📌 训练日志目录：{results.save_dir}")


if __name__ == "__main__":
    config_path = "/home/ubuntu/Project/steel_dataset/data.yaml"

    print("=== 第一步：诊断标注档案 ===")
    diagnose_labels(config_path)

    print("\n=== 第二步：开始训练 YOLOv8s-seg (CPU 调试模式) ===")
    train_steel_defect()