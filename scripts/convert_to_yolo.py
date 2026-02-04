import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def main():
    csv_path = os.path.expanduser('~/Project/data/train.csv')
    img_dir = os.path.expanduser('~/Project/data/train_images')
    label_dir = os.path.expanduser('~/Project/data/labels')
    os.makedirs(label_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    grouped = df.groupby('ImageId')

    for image_id, group in tqdm(grouped):
        img_path = os.path.join(img_dir, image_id)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        label_path = os.path.join(label_dir, image_id.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                if pd.isna(row['EncodedPixels']):
                    continue
                class_id = int(row['ClassId']) - 1  # 0-3
                mask = rle_decode(row['EncodedPixels'], (height, width))
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 10:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = (x + w / 2) / width
                    cy = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")

if __name__ == '__main__':
    main()