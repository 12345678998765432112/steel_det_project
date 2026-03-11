from ultralytics import YOLO
import torch
import os
import yaml
from pathlib import Path
import random
import numpy as np
import cv2

# 强制单进程，避免多进程问题
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ---------------------- 1. 离线增强defect_2样本（核心） ----------------------
def augment_defect2_samples_offline(data_yaml_path, defect2_cls_id=1, augment_times=1):
    """
    离线增强defect_2样本：生成增强后的图片+标注，保存到临时目录
    augment_times: 每个defect_2样本增强1次（等价于翻倍）
    """
    # 读取数据集配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(data_yaml_path).parent
    train_img_dir = dataset_root / cfg['train']
    train_label_dir = train_img_dir.parent / 'labels'
    
    # 创建临时增强目录（不修改原始数据）
    aug_img_dir = dataset_root / 'train_aug_defect2'
    aug_label_dir = dataset_root / 'labels_aug_defect2'
    aug_img_dir.mkdir(exist_ok=True)
    aug_label_dir.mkdir(exist_ok=True)
    
    # 筛选defect_2样本
    all_imgs = list(train_img_dir.rglob('*.jpg')) + list(train_img_dir.rglob('*.png'))
    defect2_imgs = []
    for img_path in all_imgs:
        label_path = train_label_dir / img_path.name.replace(img_path.suffix, '.txt')
        if not label_path.exists():
            continue
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        has_defect2 = any(int(float(line.strip().split()[0])) == defect2_cls_id for line in lines if line.strip())
        if has_defect2:
            defect2_imgs.append((img_path, label_path))
    
    # 定义增强函数（OpenCV实现，不依赖albumentations）
    def augment_image(img):
        # 1. 随机亮度/对比度
        alpha = random.uniform(0.7, 1.4)  # 对比度
        beta = random.uniform(-30, 30)     # 亮度
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # 2. 随机缩放
        scale = random.uniform(0.7, 1.5)
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        # 3. 随机轻微旋转
        angle = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img
    
    # 批量增强defect_2样本
    for idx, (img_path, label_path) in enumerate(defect2_imgs):
        # 读取原始图片和标注
        img = cv2.imread(str(img_path))
        with open(label_path, 'r', encoding='utf-8') as f:
            label_content = f.read()
        
        # 生成增强版本
        for aug_idx in range(augment_times):
            aug_img = augment_image(img)
            # 保存增强后的图片
            aug_img_name = f"{img_path.stem}_aug_{idx}_{aug_idx}{img_path.suffix}"
            aug_img_path = aug_img_dir / aug_img_name
            cv2.imwrite(str(aug_img_path), aug_img)
            # 保存标注（增强不改变标注位置，只增强图片）
            aug_label_name = f"{label_path.stem}_aug_{idx}_{aug_idx}.txt"
            aug_label_path = aug_label_dir / aug_label_name
            with open(aug_label_path, 'w', encoding='utf-8') as f:
                f.write(label_content)
    
    # 生成合并后的训练列表（原始样本 + 增强的defect_2样本）
    all_train_imgs = []
    # 加入原始样本
    for img_path in all_imgs:
        all_train_imgs.append(str(img_path))
    # 加入增强的defect_2样本
    for aug_img_path in aug_img_dir.rglob('*.*'):
        all_train_imgs.append(str(aug_img_path))
    random.shuffle(all_train_imgs)
    
    # 生成临时data.yaml
    temp_train_txt = dataset_root / 'temp_train_aug.txt'
    with open(temp_train_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_train_imgs))
    
    cfg['train'] = str(temp_train_txt)
    temp_yaml = dataset_root / 'temp_data_aug.yaml'
    with open(temp_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    
    return str(temp_yaml)

# ---------------------- 2. 核心训练函数（纯原生YOLOv8，无任何自定义逻辑） ----------------------
def train_steel_defect_no_manual_check():
    # 配置参数（只改这里！）
    original_config_path = "/home/ubuntu/Project/steel_dataset/data.yaml"
    model_name = "/home/ubuntu/Project/runs/steel_yolov8s_seg_debug_gpu/weights/best.pt"
    epochs = 40
    batch_size = 12
    imgsz = 1024
    defect2_cls_id = 1  # defect_2的类别ID
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # 1. 离线生成增强后的数据集配置（核心优化）
    temp_yaml = augment_defect2_samples_offline(
        original_config_path,
        defect2_cls_id=defect2_cls_id,
        augment_times=1  # 每个defect_2样本增强1次
    )
    
    # 2. 加载模型（纯原生）
    model = YOLO(model_name)
    
    # 3. 开始训练（纯原生参数，无任何自定义逻辑）
    results = model.train(
        data=temp_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        patience=20,
        save=True,
        project="/home/ubuntu/Project/runs",
        name="steel_yolov8s_seg_defect2_auto_opt",
        exist_ok=True,
        verbose=True,
        task="segment",
        rect=True,
        mosaic=0.0,
        mixup=0.0,
        overlap_mask=True,
        mask_ratio=4,
        workers=0,  # 关闭多进程，避免连接错误
        single_cls=False,
        amp=True,
        lr0=0.0008,
        lrf=0.01,
        warmup_epochs=5.0
    )
    
    print(f"\n✅ 训练完成！模型路径：{results.save_dir}/weights/best.pt")

# ---------------------- 执行训练 ----------------------
if __name__ == "__main__":
    train_steel_defect_no_manual_check()