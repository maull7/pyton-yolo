from ultralytics import YOLO
import cv2
import os

def load_all_models(base_path="runs/detect", max_train=12):
    models = []
    for i in range(1, max_train + 1):
        folder = os.path.join(base_path, f"train{i}")
        best_model = os.path.join(folder, "weights", "best.pt")

        if os.path.exists(best_model):
            print(f"[INFO] Model ditemukan: {best_model}")
            models.append((f"train{i}", YOLO(best_model)))
        else:
            weights_folder = os.path.join(folder, "weights")
            if os.path.exists(weights_folder):
                for file in os.listdir(weights_folder):
                    if file.endswith(".pt"):
                        model_path = os.path.join(weights_folder, file)
                        print(f"[INFO] Model alternatif ditemukan: {model_path}")
                        models.append((f"train{i}", YOLO(model_path)))
                        break
    if not models:
        raise FileNotFoundError("Tidak ada model YOLO (.pt) yang ditemukan.")
    return models

# GANTI: Sesuaikan skala ini berdasarkan kamera kamu
skala_cm_per_px = 0.05  # 1 piksel = 0.05 cm

models = load_all_models()
ip_camera_url = 'http://192.168.25.192:8080/video/mjpeg'
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Gagal mengakses IP Camera.")
    exit()

print("[INFO] Memulai deteksi dari IP Camera...")

os.makedirs("crop_output", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame dari IP Camera.")
        break

    combined_frame = frame.copy()

    for model_name, model in models:
        results = model(frame, imgsz=640, conf=0.3)
        result = results[0]
        annotated = result.plot()
        cv2.putText(annotated, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])] if result.names else f"class_{int(box.cls[0])}"

            # Hitung panjang dalam cm
            width_px = x2 - x1
            height_px = y2 - y1
            width_cm = width_px * skala_cm_per_px
            height_cm = height_px * skala_cm_per_px

            # Gambar kotak dan tulis label ukuran
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label}: {width_cm:.1f}cm x {height_cm:.1f}cm",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Simpan crop
            crop = frame[y1:y2, x1:x2]
            crop_path = f"crop_output/{model_name}_{label}_{i}.jpg"
            cv2.imwrite(crop_path, crop)

        combined_frame = annotated

    cv2.imshow("YOLOv8 - Multi Model IP Camera", combined_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
