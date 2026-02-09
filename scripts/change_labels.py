import os
from pathlib import Path

labels_dir = Path("/home/ubuntu/Project/data/train/labels")  # 先改 train，再改 val
# labels_dir = Path("/home/ubuntu/Project/data/val/labels")   # 第二步跑这个

for txt_file in labels_dir.glob("*.txt"):
    if txt_file.stat().st_size == 0:
        continue  # 空文件跳过
    
    lines = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                if class_id >= 1:  # 只改 >=1 的
                    parts[0] = str(class_id - 1)
                lines.append(' '.join(parts))
    
    with open(txt_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"Processed: {txt_file}")