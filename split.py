import os
import random
import shutil

# Path asal dan tujuan
BASE_PATH = "dataset"
IMAGE_PATH = os.path.join(BASE_PATH, "images", "train")
LABEL_PATH = os.path.join(BASE_PATH, "labels", "train")

VAL_IMAGE_PATH = os.path.join(BASE_PATH, "images", "val")
VAL_LABEL_PATH = os.path.join(BASE_PATH, "labels", "val")

# Buat folder val kalau belum ada
os.makedirs(VAL_IMAGE_PATH, exist_ok=True)
os.makedirs(VAL_LABEL_PATH, exist_ok=True)

# Ambil semua file gambar
image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith((".jpg", ".png", ".jpeg"))]

# Shuffle dan ambil 20% untuk val
val_size = int(len(image_files) * 0.2)
val_files = random.sample(image_files, val_size)

print(f"Total gambar: {len(image_files)}")
print(f"Jumlah data val: {len(val_files)}")

for file in val_files:
    # Copy gambar ke images/val
    src_img = os.path.join(IMAGE_PATH, file)
    dst_img = os.path.join(VAL_IMAGE_PATH, file)
    shutil.move(src_img, dst_img)

    # Copy label ke labels/val (ganti ekstensi jadi .txt)
    label_name = os.path.splitext(file)[0] + ".txt"
    src_label = os.path.join(LABEL_PATH, label_name)
    dst_label = os.path.join(VAL_LABEL_PATH, label_name)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    else:
        print(f"[⚠️] Label tidak ditemukan: {label_name}")

print("✅ Dataset berhasil di-split ke train/val!")
