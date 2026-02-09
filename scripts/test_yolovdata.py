import os
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def visualize_yolo_seg_labels(
    images_dir,          # 原始图片文件夹
    labels_dir,          # 多边形标注txt文件夹
    output_dir=None,     # 保存可视化结果的文件夹
    num_samples=20,      # 检查的样本数
    thickness=2,
    font_scale=0.6,
    scale_show=0.5       # 图片缩放比例（方便查看）
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有jpg图片
    image_paths = list(Path(images_dir).glob("*.jpg"))
    if not image_paths:
        print("❌ 未找到任何图片！")
        return
    
    # 随机选择样本
    selected = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    # 类别配色和名称（对应你的1-4类→0-3）
    class_colors = {
        1: (0, 255, 0),    # defect_1 绿色
        2: (0, 0, 255),    # defect_2 红色
        3: (255, 0, 0),    # defect_3 蓝色
        4: (255, 255, 0)   # defect_4 黄色
    }
    class_names = ['defect_1', 'defect_2', 'defect_3', 'defect_4']
    
    for img_path in tqdm(selected, desc="可视化标注"):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  无法读取图片：{img_path.name}")
            continue
        
        # 固定图片尺寸（你的数据集是256×1600）
        h, w = 256, 1600
        # 缩放图片方便查看
        img_resized = cv2.resize(img, (int(w*scale_show), int(h*scale_show)))
        
        # 对应标注文件路径
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"⚠️  无标注文件：{img_path.name}")
            # 保存/显示无标注的图片
            if output_dir:
                save_path = Path(output_dir) / img_path.name
                cv2.imwrite(str(save_path), img_resized)
            else:
                cv2.imshow("无标注 - " + img_path.name, img_resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            continue
        
        # 读取标注文件
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            print(f"ℹ️  空标注文件（无缺陷）：{img_path.name}")
            # 保存/显示空标注的图片
            if output_dir:
                save_path = Path(output_dir) / img_path.name
                cv2.imwrite(str(save_path), img_resized)
            else:
                cv2.imshow("无缺陷 - " + img_path.name, img_resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            continue
        
        # 解析每一行多边形标注
        for line in lines:
            parts = line.strip().split()
            # 多边形标注：至少需要 1（类别） + 3×2（3个顶点）=7 个元素
            if len(parts) < 7 or len(parts) % 2 != 1:
                print(f"⚠️  标注格式错误：{img_path.name} → {line}")
                continue
            
            # 解析类别和坐标
            cls = int(parts[0])
            
            # 增加合法性检查
            if cls not in class_colors:
                print(f"⚠️  跳过无效类别ID {cls}：{img_path.name} → {line}")
                continue
            
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            
            # 反归一化：转成像素坐标 → 再缩放（匹配显示尺寸）
            coords[:, 0] = coords[:, 0] * w * scale_show
            coords[:, 1] = coords[:, 1] * h * scale_show
            coords = coords.astype(np.int32)
            
            # 获取类别颜色
            color = class_colors.get(cls, (200, 200, 200))
            
            # 绘制多边形轮廓
            cv2.polylines(img_resized, [coords], isClosed=True, color=color, thickness=thickness)
            
            # 绘制类别名称（放在第一个顶点位置）
            label_text = f"{class_names[cls]}"
            cv2.putText(img_resized, label_text, 
                        (coords[0][0], coords[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # 保存或显示结果
        if output_dir:
            save_path = Path(output_dir) / img_path.name
            cv2.imwrite(str(save_path), img_resized)
            print(f"✅ 保存可视化图：{save_path}")
        else:
            cv2.imshow("标注检查 - " + img_path.name, img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    # 配置你的实际路径
    visualize_yolo_seg_labels(
        images_dir = "/home/ubuntu/Project/data/train_images",
        labels_dir = "/home/ubuntu/Project/train_labels",
        output_dir = "/home/ubuntu/Project/train_visualized",  # 保存到文件夹，方便批量查看
        num_samples = 30,
        scale_show = 0.5  # 图片缩小到50%，方便查看
    )