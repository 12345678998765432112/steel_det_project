"""定向检查defect_2标注异常的问题"""

import os
import yaml
from pathlib import Path

def check_defect2_labels(data_yaml_path, defect2_cls_id=1):
    """
    定向检查 defect_2（cls_id=1，根据你的实际ID改）的标注异常：
    1. 框/掩码面积过小（小缺陷但标注退化）；
    2. 坐标越界；
    3. 掩码点数过少（轮廓不完整）；
    """
    # 1. 读取数据集配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(data_yaml_path).parent
    label_paths = list((dataset_root/"train/labels").rglob("*.txt")) + list((dataset_root/"val/labels").rglob("*.txt"))
    
    # 2. 定义异常阈值（工业经验值）
    MIN_BOX_AREA = 1e-5  # 标注框面积占比 < 1e-5 算过小（相对图像）
    MIN_POINT_NUM = 6     # 掩码点数 < 6 算轮廓不完整
    defect2_abnormal = []
    
    # 3. 遍历标注文件，只检查 defect_2
    for txt_path in label_paths:
        if txt_path.stat().st_size == 0:
            continue
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
                if cls_id != defect2_cls_id:  # 只检查 defect_2
                    continue
                
                # 计算标注框/掩码面积（YOLO格式是相对坐标）
                coords = [float(p) for p in parts[1:]]
                if len(coords) < 4:  # 至少有框坐标
                    defect2_abnormal.append(f"{txt_path} 行{line_num}: 坐标不足")
                    continue
                
                # 计算掩码/框的面积（简化版：用最小外接矩形）
                xs = coords[0::2]
                ys = coords[1::2]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                area = w * h
                
                # 4. 判定异常
                if area < MIN_BOX_AREA:
                    defect2_abnormal.append(f"{txt_path} 行{line_num}: defect_2 标注面积过小（{area:.6f}）")
                if len(coords) < MIN_POINT_NUM:
                    defect2_abnormal.append(f"{txt_path} 行{line_num}: defect_2 掩码点数过少（{len(coords)}）")
                if any(x < 0 or x > 1 for x in coords):
                    defect2_abnormal.append(f"{txt_path} 行{line_num}: defect_2 坐标越界")
            except Exception as e:
                defect2_abnormal.append(f"{txt_path} 行{line_num}: 解析错误 - {e}")
    
    # 5. 输出异常样本（人工核对即可）
    if defect2_abnormal:
        print(f"发现 {len(defect2_abnormal)} 条 defect_2 标注异常：")
        for err in defect2_abnormal[:20]:  # 只看前20条，避免刷屏
            print(err)
    else:
        print("defect_2 标注无自动化可检测的异常")
    return defect2_abnormal

# 执行检查（替换成你的 data.yaml 路径）
check_defect2_labels("/home/ubuntu/Project/steel_dataset/data.yaml", defect2_cls_id=1)