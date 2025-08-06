import cv2
import os
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime

# Konfigurasi folder
IMAGE_FOLDER = "dataset/images/train"
LABEL_FOLDER = "dataset/labels/train"
LABEL_CLASSES = ["lengan_kanan", "lengan_kiri", "kerah", "kantong", "badan"]
IMG_SIZE = 640  # simpan ukuran gambar tetap
NUM_COPIES = 5  # jumlah foto disalin

# Buat folder jika belum ada
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(LABEL_FOLDER, exist_ok=True)

# Fungsi bantu simpan file label YOLO
def save_label(img_name, bbox, label_index, width, height):
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    w = abs(x2 - x1) / width
    h = abs(y2 - y1) / height

    label_path = os.path.join(LABEL_FOLDER, f"{img_name}.txt")
    with open(label_path, "w") as f:
        f.write(f"{label_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Input label dari pengguna via Tkinter
def ask_label():
    root = tk.Tk()
    root.withdraw()
    label = simpledialog.askstring("Label", f"Masukkan label (pilih salah satu):\n{LABEL_CLASSES}")
    root.destroy()
    if label in LABEL_CLASSES:
        return LABEL_CLASSES.index(label)
    return None

# Buka kamera
CAMERA_IP = "http://10.94.239.254:8080/video/mjpeg"  # Ganti sesuai IP kamu
cap = cv2.VideoCapture(CAMERA_IP)
cv2.namedWindow("Labeling Kamera")

drawing = False
ix, iy = -1, -1
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = current_frame.copy()
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Labeling Kamera", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)

        img_copy = current_frame.copy()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Labeling Kamera", img_copy)

        label_index = ask_label()
        if label_index is not None:
            height, width = current_frame.shape[:2]
            for i in range(NUM_COPIES):
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
                img_filename = f"img_{timestamp}_{i}"
                img_path = os.path.join(IMAGE_FOLDER, f"{img_filename}.jpg")
                cv2.imwrite(img_path, current_frame)
                save_label(img_filename, (x1, y1, x2, y2), label_index, width, height)
            print(f"[✓] Disimpan {NUM_COPIES} salinan + label: {LABEL_CLASSES[label_index]}")
        else:
            print("[!] Label tidak dikenali atau dibatalkan.")

# Set mouse callback
cv2.setMouseCallback("Labeling Kamera", mouse_callback)

print("[INFO] Tekan ESC untuk keluar, drag mouse untuk crop dan labeling...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Tidak bisa mengambil frame dari kamera.")
        break

    current_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    cv2.imshow("Labeling Kamera", current_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
