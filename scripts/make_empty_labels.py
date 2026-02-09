import os
import glob

def generate_empty_labels():
    img_folder = "/home/ubuntu/Project/data/train_images"
    txt_folder = "/home/ubuntu/Project/train_labels"

    all_imgs = glob.glob(os.path.join(img_folder, "*.jpg"))
    all_img_names = [os.path.basename(p) for p in all_imgs]

    existing_txts = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    existing_img_names = [os.path.splitext(txt)[0] + ".jpg" for txt in existing_txts]

    empty_count = 0
    for img_name in all_img_names:
        if img_name not in existing_img_names:
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(txt_folder, txt_name)
            open(txt_path, 'w').close()
            empty_count += 1

    print("✅ 空标签补齐完成")
    print(f"总图片：{len(all_img_names)}")
    print(f"已有标注：{len(existing_txts)}")
    print(f"新增空标签：{empty_count}")

if __name__ == "__main__":
    generate_empty_labels()