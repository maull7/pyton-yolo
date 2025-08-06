from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train12/weights/best.pt")  # Ganti sesuai path model kamu
img = cv2.imread("kerah.jpg")  # Ganti sesuai path gambar
results = model(img)

# Pastikan folder output
output_dir = "crop_output"
os.makedirs(output_dir, exist_ok=True)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.3:
            continue  # lewati kalau terlalu rendah

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        label = model.names[cls_id]

        filename = f"{label}_{conf:.2f}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, crop)
        print(f"[✔] Disimpan: {path}")
