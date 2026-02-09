import os

def count_files_in_folder(folder_path, allowed_extensions=None):
    """
    统计文件夹内的文件数量（可指定文件类型）
    :param folder_path: 文件夹路径
    :param allowed_extensions: 允许的文件后缀（如 ['.jpg', '.png']），None 则统计所有文件
    :return: 文件数量
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"⚠️  文件夹不存在：{folder_path}")
        return 0
    
    file_count = 0
    # 遍历文件夹内所有内容
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # 只统计文件（排除子文件夹），且排除隐藏文件（以.开头）
        if os.path.isfile(item_path) and not item.startswith('.'):
            # 如果指定了文件类型，只统计符合的
            if allowed_extensions:
                file_ext = os.path.splitext(item)[1].lower()
                if file_ext in allowed_extensions:
                    file_count += 1
            else:
                file_count += 1
    return file_count

# ---------------------- 配置你的文件夹路径 ----------------------
# 请替换成你实际的文件夹路径！！！
TRAIN_IMAGES_FOLDER = "/home/ubuntu/Project/data/train_images"       # 原始图片文件夹（jpg/png）
LABELS_TXT_FOLDER = "/home/ubuntu/Project/train_labels"     # YOLO标注txt文件夹
MASKS_PNG_FOLDER = "/home/ubuntu/Project/save_folder"       # 掩码PNG文件夹

# ---------------------- 执行统计 ----------------------
print("===== 文件夹文件数量统计 =====")
# 统计原始图片（只算jpg/png）
img_count = count_files_in_folder(TRAIN_IMAGES_FOLDER, ['.jpg', '.png'])
print(f"1. 原始图片文件夹 ({TRAIN_IMAGES_FOLDER})：{img_count} 个文件（jpg/png）")

# 统计标注txt文件（只算txt）
txt_count = count_files_in_folder(LABELS_TXT_FOLDER, ['.txt'])
print(f"2. YOLO标注文件夹 ({LABELS_TXT_FOLDER})：{txt_count} 个文件（txt）")

# 统计掩码PNG文件（只算png）
png_count = count_files_in_folder(MASKS_PNG_FOLDER, ['.png'])
print(f"3. 掩码图片文件夹 ({MASKS_PNG_FOLDER})：{png_count} 个文件（png）")

# 额外校验：标注文件数应 ≤ 图片文件数（因为有些图片可能无缺陷）
print("\n===== 数据校验 =====")
if txt_count > img_count:
    print(f"❌ 警告：标注文件数({txt_count}) > 原始图片数({img_count})，可能存在重复标注！")
elif txt_count == 0:
    print(f"❌ 错误：标注文件夹为空，请检查代码是否正确执行！")
else:
    print(f"✅ 数据校验正常：标注文件数({txt_count}) ≤ 原始图片数({img_count})")