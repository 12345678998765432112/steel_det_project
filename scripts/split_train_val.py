import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

def split_dataset():
    data_dir = os.path.expanduser('~/Project/data')
    train_img_dir = os.path.join(data_dir, 'train_images')
    labels_dir = os.path.join(data_dir, 'labels')  # 你转好的 txt 标注目录

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

    # 读取 train.csv 获取所有 image_id
    csv_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(csv_path)
    image_ids = df['ImageId'].unique()  # 所有唯一图片名

    # 随机划分 80% train, 20% val（可改比例）
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")

    # 复制图片和标签到新目录
    for img_id in tqdm(train_ids):
        img_src = os.path.join(train_img_dir, img_id)
        label_src = os.path.join(labels_dir, img_id.replace('.jpg', '.txt'))
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(train_dir, 'images', img_id))
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(train_dir, 'labels', img_id.replace('.jpg', '.txt')))

    for img_id in tqdm(val_ids):
        img_src = os.path.join(train_img_dir, img_id)
        label_src = os.path.join(labels_dir, img_id.replace('.jpg', '.txt'))
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(val_dir, 'images', img_id))
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(val_dir, 'labels', img_id.replace('.jpg', '.txt')))

    print("划分完成！")

if __name__ == '__main__':
    split_dataset()