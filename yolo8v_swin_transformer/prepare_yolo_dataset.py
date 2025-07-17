import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import shutil
import yaml

# Path
DATASET_DIR = '../dataset_medis'
IMAGE_DIR = os.path.join(DATASET_DIR, 'image')
MASK_DIR = os.path.join(DATASET_DIR, 'mask')
OUTPUT_IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
OUTPUT_LABEL_DIR = os.path.join(DATASET_DIR, 'labels')

# Buat output folder
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_LABEL_DIR, split), exist_ok=True)

def get_all_image_mask_pairs(image_dir, mask_dir):
    image_paths = glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
    pairs = []
    for img_path in image_paths:
        rel_path = os.path.relpath(img_path, image_dir)
        mask_path = os.path.join(mask_dir, rel_path)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs

def mask_to_yolo_boxes(mask_path, img_shape):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Threshold: objek = putih (255), background = hitam (0)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_shape[:2]
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Konversi ke format YOLO (relatif)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw_rel = bw / w
        bh_rel = bh / h
        boxes.append([0, x_center, y_center, bw_rel, bh_rel])  # 0 = class id
    return boxes

def save_yolo_label(label_path, boxes):
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(' '.join([str(round(x, 6)) for x in box]) + '\n')

def main():
    pairs = get_all_image_mask_pairs(IMAGE_DIR, MASK_DIR)
    print(f'Total pairs found: {len(pairs)}')
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    splits = [('train', train_pairs), ('val', val_pairs)]

    for split, split_pairs in splits:
        for img_path, mask_path in split_pairs:
            img = cv2.imread(img_path)
            boxes = mask_to_yolo_boxes(mask_path, img.shape)
            # Copy image
            img_name = os.path.basename(img_path)
            out_img_path = os.path.join(OUTPUT_IMAGE_DIR, split, img_name)
            shutil.copy(img_path, out_img_path)
            # Save label
            label_name = os.path.splitext(img_name)[0] + '.txt'
            out_label_path = os.path.join(OUTPUT_LABEL_DIR, split, label_name)
            save_yolo_label(out_label_path, boxes)

    # Buat data_medis.yaml
    data_yaml = {
        'train': os.path.abspath(os.path.join(OUTPUT_IMAGE_DIR, 'train')),
        'val': os.path.abspath(os.path.join(OUTPUT_IMAGE_DIR, 'val')),
        'nc': 1,
        'names': ['medis']
    }
    with open(os.path.join(DATASET_DIR, 'data_medis.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
    print('Dataset preparation complete!')

if __name__ == '__main__':
    main() 