import cv2
import os
import random
import numpy as np
from glob import glob

IMG_SIZE = (640, 640)  # ukuran output
NUM_OBJECTS = 3        # jumlah objek per gambar synthetic

# Folder dataset
img_dir = "dataset/images/train"
label_dir = "dataset/labels/train"
save_img_dir = "dataset_aug/images/train"
save_label_dir = "dataset_aug/labels/train"

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_label_dir, exist_ok=True)

# Load semua file gambar
img_files = glob(os.path.join(img_dir, "*.jpg"))

def load_objects():
    objects = []
    for img_path in img_files:
        label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                objects.append((crop, cls))
    return objects

objects = load_objects()
print(f"Total objek tersedia: {len(objects)}")

for idx in range(50):  # buat 50 gambar synthetic
    bg = 255 * np.ones((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    label_lines = []

    for _ in range(NUM_OBJECTS):
        obj_img, cls_id = random.choice(objects)
        oh, ow = obj_img.shape[:2]

        scale = random.uniform(0.5, 1.2)
        obj_img_resized = cv2.resize(obj_img, (int(ow*scale), int(oh*scale)))

        max_x = IMG_SIZE[0] - obj_img_resized.shape[1]
        max_y = IMG_SIZE[1] - obj_img_resized.shape[0]
        if max_x <= 0 or max_y <= 0:
            continue

        x_offset = random.randint(0, max_x)
        y_offset = random.randint(0, max_y)

        bg[y_offset:y_offset+obj_img_resized.shape[0], x_offset:x_offset+obj_img_resized.shape[1]] = obj_img_resized

        # Hitung label YOLO
        cx = (x_offset + obj_img_resized.shape[1]/2) / IMG_SIZE[0]
        cy = (y_offset + obj_img_resized.shape[0]/2) / IMG_SIZE[1]
        bw = obj_img_resized.shape[1] / IMG_SIZE[0]
        bh = obj_img_resized.shape[0] / IMG_SIZE[1]

        label_lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    cv2.imwrite(os.path.join(save_img_dir, f"synthetic_{idx}.jpg"), bg)
    with open(os.path.join(save_label_dir, f"synthetic_{idx}.txt"), "w") as f:
        f.write("\n".join(label_lines))

print("✅ Augmentasi selesai!")
