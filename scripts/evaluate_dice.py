# -*- coding: utf-8 -*-
"""
Dice 系数评估脚本 (evaluate_dice.py)
=====================================
Severstal 钢铁缺陷竞赛使用 Dice 系数(像素级)作为评分指标,而非 mAP。
本脚本在验证集上计算逐类别 / 整体 Dice,更贴近真实竞赛分数。

做法:
  1. 从 YOLO 分割标签(多边形)重建每个类别的 ground-truth mask;
  2. 用训练好的模型预测,得到每个类别的预测 mask;
  3. 对每张图、每个类别计算 Dice;某类别在 gt 与 pred 都为空时计为 1(完美)。

用法:
    python scripts/evaluate_dice.py
"""

import os
import numpy as np
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 配置区
# ============================================================
CONFIG = {
    "model_path": "runs/steel_yolov8s_seg_improved/weights/best.pt",
    "data_yaml": "/home/ubuntu/Project/steel_dataset/data.yaml",
    "imgsz": 1024,
    "conf": 0.25,          # 预测置信度阈值
    "iou": 0.6,            # NMS IoU 阈值
    "device": 0,
    # 每张图每个类别的 mask 分辨率(与原图 1600x256 一致)
    "mask_size": (256, 1600),  # (H, W)
    "max_imgs": None,      # 只评估前 N 张,None 表示全部
}
# ============================================================


def load_val_images(data_yaml_path):
    cfg, root = load_cfg(data_yaml_path)
    val_dir = root / cfg['val']
    return sorted(list(val_dir.rglob('*.jpg')) + list(val_dir.rglob('*.png')))


def load_cfg(data_yaml_path):
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg, Path(data_yaml_path).parent


def label_to_masks(label_path, nc, size):
    """把 YOLO 多边形标签转为每类一张二值 mask。返回 shape (nc, H, W) uint8。"""
    H, W = size
    masks = np.zeros((nc, H, W), dtype=np.uint8)
    if not label_path.exists():
        return masks
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            if cls >= nc:
                continue
            coords = np.array([float(p) for p in parts[1:]], dtype=np.float32)
            coords = coords.reshape(-1, 2)
            coords[:, 0] *= W
            coords[:, 1] *= H
            cv2.fillPoly(masks[cls], [coords.astype(np.int32)], 1)
    return masks


def predict_to_masks(result, nc, size):
    """把模型预测结果转为每类一张二值 mask。"""
    H, W = size
    masks = np.zeros((nc, H, W), dtype=np.uint8)
    if result.masks is None:
        return masks
    det_masks = result.masks.data.cpu().numpy()          # (N, h, w)
    det_cls = result.boxes.cls.cpu().numpy().astype(int)
    h, w = det_masks.shape[1], det_masks.shape[2]
    for m, c in zip(det_masks, det_cls):
        if c >= nc:
            continue
        m = (m > 0.5).astype(np.uint8)
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        masks[c] = np.maximum(masks[c], m)
    return masks


def dice_score(pred, gt):
    """单张二值 mask 的 Dice。两者都为空返回 1.0,一方为空返回 0.0。"""
    p = pred.astype(bool)
    g = gt.astype(bool)
    p_sum, g_sum = p.sum(), g.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0
    inter = np.logical_and(p, g).sum()
    return 2.0 * inter / (p_sum + g_sum + 1e-7)


def main():
    model = YOLO(CONFIG["model_path"])
    nc = model.model.nc
    size = CONFIG["mask_size"]
    img_paths = load_val_images(CONFIG["data_yaml"])
    if CONFIG["max_imgs"]:
        img_paths = img_paths[:CONFIG["max_imgs"]]

    label_dir = img_paths[0].parent.parent / 'labels'
    per_class = {c: [] for c in range(nc)}
    all_scores = []

    for img_path in img_paths:
        label_path = label_dir / (img_path.stem + '.txt')
        gt = label_to_masks(label_path, nc, size)

        results = model.predict(
            str(img_path), imgsz=CONFIG["imgsz"],
            conf=CONFIG["conf"], iou=CONFIG["iou"],
            device=CONFIG["device"], verbose=False,
        )
        pred = predict_to_masks(results[0], nc, size)

        for c in range(nc):
            d = dice_score(pred[c], gt[c])
            per_class[c].append(d)
            all_scores.append(d)

    print("\n" + "=" * 50)
    print("Dice 系数评估结果 (Severstal 竞赛指标)")
    print("=" * 50)
    for c in range(nc):
        vals = per_class[c]
        print(f"  defect_{c+1} (class {c}): "
              f"Dice={np.mean(vals):.4f}  (样本数={len(vals)})")
    print("-" * 50)
    print(f"  整体平均 Dice: {np.mean(all_scores):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
