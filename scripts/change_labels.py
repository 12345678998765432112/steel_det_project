import os
from pathlib import Path

def fix_class_ids(labels_dir):
    """
    将标注文件中的类别ID从1-4修正为0-3（适配YOLO nc:4的标准配置）
    :param labels_dir: 标注txt文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(labels_dir):
        print(f"⚠️  文件夹不存在：{labels_dir}，跳过！")
        return
    
    label_paths = list(Path(labels_dir).glob("*.txt"))
    if not label_paths:
        print(f"⚠️  {labels_dir} 中未找到任何txt文件，跳过！")
        return
    
    # 统计修正的文件数/行数
    fixed_files = 0
    fixed_lines = 0
    
    for label_path in label_paths:
        # 跳过空文件
        if os.path.getsize(label_path) == 0:
            continue
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"❌ 读取文件失败：{label_path} → {str(e)}")
            continue
        
        new_lines = []
        file_updated = False
        
        for line in lines:
            parts = line.split()
            if len(parts) < 7:  # 跳过格式错误的行（至少1个ID+3个坐标点）
                new_lines.append(line + '\n')
                continue
            
            try:
                old_cls = int(parts[0])
                new_cls = old_cls - 1
                # 只修正1-4的ID为0-3，其他ID跳过（防止异常）
                if 1 <= old_cls <= 4:
                    parts[0] = str(new_cls)
                    fixed_lines += 1
                    file_updated = True
                new_lines.append(' '.join(parts) + '\n')
            except ValueError:
                # 类别ID不是数字，保留原行
                new_lines.append(line + '\n')
                print(f"⚠️  {label_path} 中发现非数字类别ID：{line}")
        
        # 只在文件有更新时重写（避免空写）
        if file_updated:
            with open(label_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            fixed_files += 1
    
    print(f"✅ {labels_dir} 修正完成 → 处理文件数：{len(label_paths)} | 修正文件数：{fixed_files} | 修正行数：{fixed_lines}")

# 运行修正（按顺序处理所有标注文件夹）
if __name__ == "__main__":
    # 替换为你的实际路径
    LABEL_DIRS = [
        "/home/ubuntu/Project/train_labels",
        "/home/ubuntu/Project/steel_dataset/train/labels",
        "/home/ubuntu/Project/steel_dataset/val/labels"
    ]
    
    for dir_path in LABEL_DIRS:
        fix_class_ids(dir_path)
    
    print("\n🎉 所有文件夹的类别ID修正完成！")