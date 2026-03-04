import os
from pathlib import Path
from collections import defaultdict

def count_class_id_distribution(labels_dir):
    """
    统计标注文件中类别ID的分布，并检查是否有无效ID
    :param labels_dir: 标注txt文件夹路径
    """
    # 初始化统计字典
    class_count = defaultdict(int)  # 类别ID: 出现次数
    invalid_files = []              # 包含无效ID的文件
    empty_files = 0                 # 空标注文件数（无缺陷）
    total_files = 0                 # 总标注文件数
    total_annotations = 0           # 总标注行数（缺陷数）

    # 遍历所有txt文件
    label_paths = list(Path(labels_dir).glob("*.txt"))
    for label_path in label_paths:
        total_files += 1
        # 跳过空文件
        if os.path.getsize(label_path) == 0:
            empty_files += 1
            continue
        
        # 读取标注文件
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"⚠️  读取文件失败：{label_path.name} → {str(e)}")
            continue
        
        # 解析每一行的类别ID
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
                total_annotations += 1
                # 统计类别ID
                if 0 <= cls_id <= 3:
                    class_count[cls_id] += 1
                else:
                    # 记录无效ID的文件
                    invalid_files.append({
                        "file": label_path.name,
                        "line": line,
                        "invalid_cls": cls_id
                    })
            except ValueError:
                invalid_files.append({
                    "file": label_path.name,
                    "line": line,
                    "invalid_cls": "非数字"
                })

    # 输出统计结果
    print("="*50)
    print("📊 标注文件类别ID分布统计")
    print("="*50)
    print(f"总标注文件数：{total_files}")
    print(f"空标注文件数（无缺陷）：{empty_files}")
    print(f"有缺陷标注的文件数：{total_files - empty_files}")
    print(f"总缺陷标注行数：{total_annotations}")
    print("\n🔍 类别ID分布：")
    for cls_id in sorted(class_count.keys()):
        count = class_count[cls_id]
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        print(f"  类别 {cls_id} (defect_{cls_id+1})：{count} 条 ({percentage:.2f}%)")
    
    # 输出无效ID信息
    if invalid_files:
        print("\n❌ 发现无效类别ID的文件：")
        for idx, item in enumerate(invalid_files[:10], 1):  # 只显示前10个
            print(f"  {idx}. 文件：{item['file']} → 行：{item['line']} → 无效ID：{item['invalid_cls']}")
        if len(invalid_files) > 10:
            print(f"  ... 还有 {len(invalid_files)-10} 个文件包含无效ID")
    else:
        print("\n✅ 所有标注文件的类别ID均有效（0-3）！")

# 使用示例
if __name__ == "__main__":
    # 替换为你的标注文件夹路径
    LABELS_DIR = "/home/ubuntu/Project/steel_dataset/val/labels"
    count_class_id_distribution(LABELS_DIR)