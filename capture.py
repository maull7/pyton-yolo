import cv2
import os

# === KONFIGURASI ===
# Gunakan 0 untuk webcam lokal, atau URL IP kamera
CAMERA_SOURCE = 'http://192.168.25.192:8080/video/mjpeg'  # atau 0 untuk webcam biasa
DATASET_DIR = 'dataset'  # Folder utama dataset

# Mapping tombol ke nama label
LABEL_KEYS = {
    'k': 'kerah',
    'l': 'lengan_kanan',
    'j': 'lengan_kiri',
    'b': 'badan',
    't': 'kantong'
}

# ====================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_next_filename(label_dir):
    existing = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]
    return os.path.join(label_dir, f"{len(existing) + 1}.jpg")

cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("[ERROR] Gagal membuka kamera.")
    exit()

print("=== INSTRUKSI ===")
for key, label in LABEL_KEYS.items():
    print(f"Tekan [{key.upper()}] → Simpan ke label '{label}'")
print("Tekan [P] pause/resume, [ESC] keluar")
print("=================\n")

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame.")
            break

        # Tampilkan frame
        cv2.imshow("Ambil Dataset - Tekan tombol untuk label", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC untuk keluar
        print("[INFO] Keluar...")
        break
    elif key == ord('p'):
        paused = not paused
        print("[INFO] Paused." if paused else "[INFO] Resume.")
    elif key != -1:
        chr_key = chr(key).lower()
        if chr_key in LABEL_KEYS:
            label = LABEL_KEYS[chr_key]
            label_dir = os.path.join(DATASET_DIR, label)
            ensure_dir(label_dir)

            filename = get_next_filename(label_dir)
            cv2.imwrite(filename, frame)
            print(f"[SAVED] {filename}")

cap.release()
cv2.destroyAllWindows()
