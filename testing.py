from ultralytics import YOLO
import cv2

# Load model
model = YOLO("runs/detect/train19/weights/best.pt")

# Baca gambar
img = cv2.imread("kemeja.jpeg")

# Jalankan deteksi
results = model(img, conf=0.1)
result = results[0]

# Rasio pixel ke cm (HARUS dikalibrasi dulu)
PIXEL_PER_CM = 37  # contoh: 37 pixel = 1 cm

# Annotasi manual
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    label = result.names[int(box.cls[0])] if result.names else f"class_{int(box.cls[0])}"

    # Ukuran dalam pixel
    width_px = x2 - x1
    height_px = y2 - y1

    # Konversi ke cm
    width_cm = round(width_px / PIXEL_PER_CM, 2)
    height_cm = round(height_px / PIXEL_PER_CM, 2)

    # Gambar bounding box dan label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}: {width_cm} x {height_cm} cm"
    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Tampilkan hasil
cv2.imshow("Deteksi Gambar + Ukuran", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
