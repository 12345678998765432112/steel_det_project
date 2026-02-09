import os
import random
import shutil
from pathlib import Path

def split_dataset():
    # 你的路径
    img_folder = "/home/ubuntu/Project/data/train_images"
    txt_folder = "/home/ubuntu/Project/train_labels"
    save_root = "/home/ubuntu/Project/steel_dataset"  # 划分后保存路径
    
    # 创建目录
    for dir_path in [f"{save_root}/train/images", f"{save_root}/train/labels",
                     f"{save_root}/val/images", f"{save_root}/val/labels"]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 随机划分8:2
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
    random.shuffle(all_imgs)
    train_imgs = all_imgs[:int(len(all_imgs)*0.8)]
    val_imgs = all_imgs[int(len(all_imgs)*0.8):]
    
    # 复制文件
    def copy_files(imgs, src_img, src_txt, dst_img, dst_txt):
        for img in imgs:
            shutil.copy(os.path.join(src_img, img), os.path.join(dst_img, img))
            txt = os.path.splitext(img)[0] + ".txt"
            shutil.copy(os.path.join(src_txt, txt), os.path.join(dst_txt, txt))
    
    copy_files(train_imgs, img_folder, txt_folder,
               f"{save_root}/train/images", f"{save_root}/train/labels")
    copy_files(val_imgs, img_folder, txt_folder,
               f"{save_root}/val/images", f"{save_root}/val/labels")
    
    print(f"✅ 数据集划分完成！")
    print(f"训练集：{len(train_imgs)} 张 | 验证集：{len(val_imgs)} 张")

if __name__ == "__main__":
    split_dataset()