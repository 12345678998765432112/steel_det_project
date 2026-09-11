# -*- coding: utf-8 -*-
"""
改进版训练脚本 (train_improved.py)
=====================================
针对当前模型的问题(类别极度不平衡 + 过拟合)做针对性优化:

1. 类别过采样:对稀有类别(尤其是 defect_2 / class=1)大幅离线过采样,
   并支持按类别配置不同倍率。
2. 强化增强:亮度/对比度/HSV 抖动、水平翻转(标签同步)、平移(标签同步)、
   轻微缩放、轻微旋转。
3. Copy-Paste 增强:把稀有类缺陷的 mask 区域裁剪出来,随机粘贴到其他
   位置/其他图像,从而在保持真实缺陷形态的前提下成倍扩充稀有样本。
4. 训练参数优化:cos_lr 余弦退火、dropout 正则、更合理的 lr0 / warmup /
   epochs,以缓解过拟合(val loss 停滞)。

用法:
    python scripts/train_improved.py

只改下方「配置区」即可。
"""

from ultralytics import YOLO
import torch
import os
import yaml
import random
import numpy as np
import cv2
from pathlib import Path

# 注意:不设置 CUDA_LAUNCH_BLOCKING,它会强制 CUDA 内核同步,
# 大幅拖慢训练速度(仅调试时才需要)。

# ============================================================
# 配置区(只改这里)
# ============================================================
CONFIG = {
    "original_config_path": "/home/ubuntu/Project/steel_dataset/data.yaml",
    # 起始权重:可换成 yolov8m-seg 或上一轮 best.pt 继续微调
    "model_name": "/home/ubuntu/Project/models/yolov8s-seg.pt",
    "epochs": 100,
    "batch_size": 12,
    "imgsz": 1024,
    "device": 0 if torch.cuda.is_available() else "cpu",
    "project": "/home/ubuntu/Project/runs",
    "name": "steel_yolov8s_seg_improved",

    # ---- 过采样配置:类别ID -> 增强倍率(每个原始样本生成几份) ----
    # defect_2(class=1) 只有 267 个,重点增强 8 倍 → 约 2100+
    # defect_4(class=3) 1509 个,增强 2 倍 → 约 3000
    "oversample_map": {1: 8, 3: 2},

    # ---- Copy-Paste 增强(针对稀有类) ----
    "copy_paste": True,            # 是否开启 copy-paste
    "copy_paste_classes": [1, 3],  # 只对哪些类别做 copy-paste
    "copy_paste_times": 3,         # 每个稀有类缺陷额外粘贴几次

    # ---- 训练超参数 ----
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "patience": 25,
    "dropout": 0.1,
    "cos_lr": True,
    "optimizer": "auto",
    "weight_decay": 0.0005,
    "close_mosaic": 10,   # 关闭 mosaic 前的 epoch 数(此处 mosaic 已关,占位)
}
# ============================================================


def parse_label(label_path):
    """
    解析 YOLO 分割标签。
    返回 list[(cls, np.array([[x,y],...]) )],坐标保持归一化。
    """
    instances = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            coords = np.array([float(p) for p in parts[1:]], dtype=np.float32)
            coords = coords.reshape(-1, 2)
            instances.append((cls, coords))
    return instances


def instances_to_lines(instances):
    """把实例列表转回 YOLO 标签字符串。"""
    lines = []
    for cls, coords in instances:
        coords = np.clip(coords, 0.0, 1.0)
        flat = coords.reshape(-1).tolist()
        vals = " ".join(f"{v:.6f}" for v in flat)
        lines.append(f"{cls} {vals}")
    return lines


# ---------------- 基础增强(标签同步) ----------------
def augment_image_and_labels(img, instances, rng):
    """
    对图像和标签做同步增强。
    返回 (aug_img, instances)。
    """
    h, w = img.shape[:2]
    new_instances = [(c, co.copy()) for c, co in instances]
    out = img.copy()

    # 1. 亮度/对比度(不影响标签)
    alpha = rng.uniform(0.7, 1.4)
    beta = rng.uniform(-30, 30)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # 2. HSV 抖动(不影响标签)
    if rng.random() < 0.7:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-15, 15)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.3), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. 水平翻转(标签 x -> 1-x)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        new_instances = [
            (c, np.column_stack([1.0 - co[:, 0], co[:, 1]]))
            for c, co in new_instances
        ]

    # 4. 轻微平移(标签同步平移,越界裁剪到边界)
    if rng.random() < 0.7:
        dx = rng.uniform(-0.08, 0.08)
        dy = rng.uniform(-0.08, 0.08)
        M = np.float32([[1, 0, dx * w], [0, 1, dy * h]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        new_instances = [
            (c, np.clip(co + np.array([dx, dy], dtype=np.float32), 0.0, 1.0))
            for c, co in new_instances
        ]

    # 5. 轻微缩放(缩放后 resize 回原尺寸,标签近似不变)
    if rng.random() < 0.6:
        scale = rng.uniform(0.85, 1.15)
        resized = cv2.resize(out, (int(w * scale), int(h * scale)))
        if scale < 1.0:
            # 缩小后 pad 回原尺寸
            out = np.zeros_like(out)
            y0, x0 = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
            out[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        else:
            # 放大后居中裁剪回原尺寸
            y0, x0 = (resized.shape[0] - h) // 2, (resized.shape[1] - w) // 2
            out = resized[y0:y0 + h, x0:x0 + w]

    # 6. 轻微旋转(细长缺陷旋转幅度小,标签近似不变)
    if rng.random() < 0.5:
        angle = rng.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return out, new_instances


# ---------------- Copy-Paste 增强 ----------------
def copy_paste_instance(img, instances, rng, target_cls=None):
    """
    从当前图像裁剪一个缺陷区域,随机粘贴到同图其它位置。
    简化实现:从当前图挑一个目标类别的缺陷,复制其 mask 区域,粘贴到随机位置。

    返回 (out_img, instances)。
    """
    h, w = img.shape[:2]
    out = img.copy()

    # 挑选一个可粘贴的实例
    candidates = [(i, c, co) for i, (c, co) in enumerate(instances)]
    if target_cls is not None:
        candidates = [x for x in candidates if x[1] == target_cls]
    if not candidates:
        return out, instances

    # 随机挑选一个候选实例(randrange 左闭右开,避免越界)
    idx = rng.randrange(len(candidates))
    _, cls, co = candidates[idx]
    # 计算 bbox(归一化)
    x_min, y_min = co.min(axis=0)
    x_max, y_max = co.max(axis=0)
    x0, y0 = int(x_min * w), int(y_min * h)
    x1, y1 = int(x_max * w), int(y_max * h)
    x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
    x1, y1 = min(w, x1 + 2), min(h, y1 + 2)
    if x1 <= x0 or y1 <= y0:
        return out, instances

    # 构建缺陷 mask(在裁剪区域内)
    crop = out[y0:y1, x0:x1].copy()
    poly = co.copy()
    poly[:, 0] = (poly[:, 0] * w - x0)
    poly[:, 1] = (poly[:, 1] * h - y0)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

    crop_w, crop_h = x1 - x0, y1 - y0
    # 随机选择粘贴位置(用 max(0, ...) 避免闭区间越界)
    tx = rng.randint(0, max(0, w - crop_w))
    ty = rng.randint(0, max(0, h - crop_h))

    # 粘贴(只粘贴 mask 内的像素)
    roi = out[ty:ty + crop_h, tx:tx + crop_w]
    roi[mask == 1] = crop[mask == 1]

    # 生成新的多边形标签(平移)
    new_poly = co.copy()
    new_poly[:, 0] = (new_poly[:, 0] * w + (tx - x0)) / w
    new_poly[:, 1] = (new_poly[:, 1] * h + (ty - y0)) / h
    if np.all((new_poly >= 0.0) & (new_poly <= 1.0)):
        instances = instances + [(cls, new_poly)]

    return out, instances


# ---------------- 离线增强 + 过采样 ----------------
def build_augmented_dataset(data_yaml_path, aug_cfg):
    """
    生成过采样 + 增强后的训练数据,写临时目录和临时 yaml。
    返回 (temp_yaml_path, 统计信息)。

    aug_cfg 只包含增强相关参数(oversample_map / copy_paste 等),
    数据集路径(train/val)从这里读取的原始 data.yaml 获取。
    """
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_yaml_path).parent
    train_img_dir = dataset_root / data_cfg['train']
    train_label_dir = train_img_dir.parent / 'labels'

    aug_img_dir = dataset_root / 'train_aug_improved'
    aug_label_dir = dataset_root / 'labels_aug_improved'
    aug_img_dir.mkdir(exist_ok=True)
    aug_label_dir.mkdir(exist_ok=True)
    # 清空旧内容
    for p in aug_img_dir.iterdir():
        p.unlink()
    for p in aug_label_dir.iterdir():
        p.unlink()

    all_imgs = sorted(
        list(train_img_dir.rglob('*.jpg')) + list(train_img_dir.rglob('*.png'))
    )
    rng = random.Random(42)

    oversample_map = aug_cfg.get("oversample_map", {})
    copy_paste = aug_cfg.get("copy_paste", False)
    cp_classes = aug_cfg.get("copy_paste_classes", [])
    cp_times = aug_cfg.get("copy_paste_times", 2)

    new_train_list = []
    for p in all_imgs:
        new_train_list.append(str(p))

    stat = {"original": len(all_imgs), "augmented": 0}

    for img_path in all_imgs:
        label_path = train_label_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        instances = parse_label(label_path)
        if not instances:
            continue

        # 该图包含哪些类
        classes_present = {c for c, _ in instances}
        # 需要过采样的类别
        need = [c for c in classes_present if c in oversample_map]
        if not need:
            continue

        # 取这些类别中最大的倍率作为该图的增强次数
        max_times = max(oversample_map[c] for c in need)
        for k in range(max_times):
            aug_img, aug_inst = augment_image_and_labels(img, instances, rng)
            # 可选:再对稀有类做 copy-paste
            if copy_paste:
                for _ in range(cp_times):
                    tc = rng.choice(cp_classes) if cp_classes else None
                    aug_img, aug_inst = copy_paste_instance(
                        aug_img, aug_inst, rng, target_cls=tc
                    )
            # 保存
            name = f"{img_path.stem}_aug_{k}{img_path.suffix}"
            aug_img_path = aug_img_dir / name
            cv2.imwrite(str(aug_img_path), aug_img)
            lines = instances_to_lines(aug_inst)
            with open(aug_label_dir / (name.rsplit('.', 1)[0] + '.txt'),
                      'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            new_train_list.append(str(aug_img_path))
            stat["augmented"] += 1

    # 合并原始 + 增强,打乱
    rng.shuffle(new_train_list)
    temp_train_txt = dataset_root / 'temp_train_improved.txt'
    with open(temp_train_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_train_list))

    new_cfg = dict(data_cfg)
    new_cfg['train'] = str(temp_train_txt)
    temp_yaml = dataset_root / 'temp_data_improved.yaml'
    with open(temp_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(new_cfg, f, allow_unicode=True)

    return str(temp_yaml), stat


def train():
    # 组装增强参数(数据集路径由 build_augmented_dataset 内部读取 data.yaml)
    build_cfg = {
        "oversample_map": CONFIG["oversample_map"],
        "copy_paste": CONFIG["copy_paste"],
        "copy_paste_classes": CONFIG["copy_paste_classes"],
        "copy_paste_times": CONFIG["copy_paste_times"],
    }
    print("📦 开始离线增强 + 过采样 ...")
    temp_yaml, stat = build_augmented_dataset(
        CONFIG["original_config_path"], build_cfg)
    print(f"   原始样本: {stat['original']} | 新增增强样本: {stat['augmented']}")
    print(f"   临时数据配置: {temp_yaml}")

    print(f"📌 加载模型: {CONFIG['model_name']}")
    model = YOLO(CONFIG["model_name"])

    print(f"\n🚀 开始训练 (epochs={CONFIG['epochs']}, "
          f"batch={CONFIG['batch_size']}, imgsz={CONFIG['imgsz']})")
    results = model.train(
        data=temp_yaml,
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch_size"],
        imgsz=CONFIG["imgsz"],
        device=CONFIG["device"],
        patience=CONFIG["patience"],
        save=True,
        project=CONFIG["project"],
        name=CONFIG["name"],
        exist_ok=True,
        verbose=True,
        task="segment",
        rect=True,
        mosaic=0.0,          # 细长缺陷不适用 mosaic
        mixup=0.0,
        overlap_mask=True,
        mask_ratio=4,
        workers=8,           # 多进程数据加载,提升 GPU 利用率(14核机器)
        single_cls=False,
        amp=True,
        lr0=CONFIG["lr0"],
        lrf=CONFIG["lrf"],
        warmup_epochs=CONFIG["warmup_epochs"],
        cos_lr=CONFIG["cos_lr"],
        dropout=CONFIG["dropout"],
        optimizer=CONFIG["optimizer"],
        weight_decay=CONFIG["weight_decay"],
        close_mosaic=CONFIG["close_mosaic"],
    )

    print(f"\n✅ 训练完成!最佳模型:{results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train()
