from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

class PendeteksiBagianPakaian:
    def __init__(self, jalur_dasar="runs/detect", maks_pelatihan=19, skala_cm_per_px=0.05):
        self.model_model = self.muat_semua_model(jalur_dasar, maks_pelatihan)
        self.skala_cm_per_px = skala_cm_per_px
        self.riwayat_deteksi = defaultdict(list)

       # Definisi bagian-bagian pakaian yang akan dideteksi (sesuai data.yaml)
        self.bagian_pakaian = {
            'kerah': ['collar', 'kerah', 'neck'],
            'lengan_kanan': ['sleeve_right', 'lengan_kanan', 'right_sleeve'],
            'lengan_kiri': ['sleeve_left', 'lengan_kiri', 'left_sleeve'],
            'kantong': ['pocket', 'kantong', 'chest_pocket'],
            'badan': ['badan', 'body', 'torso', 'bagian_depan']  # tambahkan sinonim umum
        }

        # Warna untuk setiap bagian pakaian (disesuaikan)
        self.warna_bagian = {
            'kerah': (255, 0, 0),         # Merah
            'lengan_kanan': (0, 255, 255), # Cyan
            'lengan_kiri': (0, 255, 0),   # Hijau
            'kantong': (255, 255, 0),     # Kuning
            'badan': (0, 128, 255)        # Biru (dari warna 'dada' sebelumnya)
        }


    def muat_semua_model(self, jalur_dasar, maks_pelatihan):
        """Muat semua model YOLO yang tersedia"""
        model_model = []
        for i in range(1, maks_pelatihan + 1):
            folder = os.path.join(jalur_dasar, f"train{i}")
            model_terbaik = os.path.join(folder, "weights", "best.pt")

            if os.path.exists(model_terbaik):
                print(f"[INFO] Model ditemukan: {model_terbaik}")
                model_model.append((f"train{i}", YOLO(model_terbaik)))
            else:
                folder_bobot = os.path.join(folder, "weights")
                if os.path.exists(folder_bobot):
                    for berkas in os.listdir(folder_bobot):
                        if berkas.endswith(".pt"):
                            jalur_model = os.path.join(folder_bobot, berkas)
                            print(f"[INFO] Model alternatif ditemukan: {jalur_model}")
                            model_model.append((f"train{i}", YOLO(jalur_model)))
                            break

        if not model_model:
            raise FileNotFoundError("Tidak ada model YOLO (.pt) yang ditemukan.")
        return model_model

    def klasifikasi_bagian_pakaian(self, label):
        """Klasifikasi label ke bagian pakaian yang sesuai"""
        label_kecil = label.lower()
        for bagian, kata_kunci in self.bagian_pakaian.items():
            for kata in kata_kunci:
                if kata in label_kecil:
                    return bagian
        return 'tidak_dikenal'

    def hitung_skor_kepercayaan(self, deteksi_list):
        """Hitung skor kepercayaan gabungan dari beberapa model"""
        if not deteksi_list:
            return 0.0
        return np.mean([det['kepercayaan'] for det in deteksi_list])

    def supresi_maksimal_non_custom(self, deteksi_list, ambang_iou=0.4):
        """NMS kustom untuk menggabungkan deteksi dari beberapa model"""
        if len(deteksi_list) <= 1:
            return deteksi_list

        # Urutkan berdasarkan kepercayaan
        deteksi_list = sorted(deteksi_list, key=lambda x: x['kepercayaan'], reverse=True)

        deteksi_tersaring = []
        for deteksi in deteksi_list:
            simpan = True
            for tersaring in deteksi_tersaring:
                if self.hitung_iou(deteksi['kotak_pembatas'], tersaring['kotak_pembatas']) > ambang_iou:
                    simpan = False
                    break
            if simpan:
                deteksi_tersaring.append(deteksi)

        return deteksi_tersaring

    def hitung_iou(self, kotak1, kotak2):
        """Hitung Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = kotak1
        x1_2, y1_2, x2_2, y2_2 = kotak2

        # Hitung area intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area_kotak1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_kotak2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        area_union = area_kotak1 + area_kotak2 - area_inter

        return area_inter / area_union if area_union > 0 else 0.0

    def deteksi_bagian_pakaian(self, frame):
        """Deteksi bagian-bagian pakaian dengan ensemble models"""
        semua_deteksi = defaultdict(list)

        # Jalankan semua model
        for nama_model, model in self.model_model:
            hasil = model(frame, imgsz=640, conf=0.2, iou=0.4)
            result = hasil[0]

            for kotak in result.boxes:
                if kotak.conf[0] > 0.3:  # Ambang kepercayaan minimum
                    x1, y1, x2, y2 = map(int, kotak.xyxy[0])
                    label = result.names[int(kotak.cls[0])] if result.names else f"kelas_{int(kotak.cls[0])}"
                    kepercayaan = float(kotak.conf[0])

                    # Klasifikasi bagian pakaian
                    jenis_bagian = self.klasifikasi_bagian_pakaian(label)

                    deteksi = {
                        'kotak_pembatas': (x1, y1, x2, y2),
                        'label': label,
                        'kepercayaan': kepercayaan,
                        'model': nama_model,
                        'jenis_bagian': jenis_bagian
                    }

                    semua_deteksi[jenis_bagian].append(deteksi)

        # Terapkan NMS untuk setiap jenis bagian
        deteksi_akhir = {}
        for jenis_bagian, deteksi_list in semua_deteksi.items():
            if deteksi_list:
                tersaring = self.supresi_maksimal_non_custom(deteksi_list, ambang_iou=0.4)
                deteksi_akhir[jenis_bagian] = tersaring

        return deteksi_akhir

    def tingkatkan_akurasi_deteksi(self, deteksi_dict, frame):
        """Tingkatkan akurasi dengan post-processing"""
        deteksi_ditingkatkan = {}

        for jenis_bagian, deteksi_bagian in deteksi_dict.items():
            deteksi_bagian_ditingkatkan = []

            for deteksi in deteksi_bagian:
                x1, y1, x2, y2 = deteksi['kotak_pembatas']

                # Validasi ukuran deteksi (filter yang terlalu kecil/besar)
                lebar = x2 - x1
                tinggi = y2 - y1
                area = lebar * tinggi
                area_frame = frame.shape[0] * frame.shape[1]

                # Lewati deteksi yang terlalu kecil (<0.1% frame) atau terlalu besar (>50% frame)
                if area < area_frame * 0.001 or area > area_frame * 0.5:
                    continue

                # Validasi rasio aspek berdasarkan jenis bagian
                rasio_aspek = lebar / tinggi if tinggi > 0 else 0

                deteksi_valid = True
                if jenis_bagian == 'kerah' and (rasio_aspek < 0.3 or rasio_aspek > 5.0):
                    deteksi_valid = False
                elif jenis_bagian in ['lengan_kiri', 'lengan_kanan'] and (rasio_aspek < 0.2 or rasio_aspek > 3.0):
                    deteksi_valid = False
                elif jenis_bagian == 'kancing' and (rasio_aspek < 0.5 or rasio_aspek > 2.0):
                    deteksi_valid = False

                if deteksi_valid:
                    # Tambahkan informasi tambahan
                    deteksi['area_px'] = area
                    deteksi['rasio_aspek'] = rasio_aspek
                    deteksi['lebar_cm'] = lebar * self.skala_cm_per_px
                    deteksi['tinggi_cm'] = tinggi * self.skala_cm_per_px

                    deteksi_bagian_ditingkatkan.append(deteksi)

            if deteksi_bagian_ditingkatkan:
                deteksi_ditingkatkan[jenis_bagian] = deteksi_bagian_ditingkatkan

        return deteksi_ditingkatkan

    def gambar_deteksi(self, frame, deteksi_dict):
        """Gambar hasil deteksi pada frame"""
        frame_beranotasi = frame.copy()
        info_deteksi = []

        for jenis_bagian, deteksi_bagian in deteksi_dict.items():
            warna = self.warna_bagian.get(jenis_bagian, (255, 255, 255))

            for i, deteksi in enumerate(deteksi_bagian):
                x1, y1, x2, y2 = deteksi['kotak_pembatas']

                # Gambar kotak pembatas
                cv2.rectangle(frame_beranotasi, (x1, y1), (x2, y2), warna, 2)

                # Label dengan informasi detail
                teks_label = f"{jenis_bagian.upper()}: {deteksi['lebar_cm']:.1f}x{deteksi['tinggi_cm']:.1f}cm"
                teks_kepercayaan = f"Yakin: {deteksi['kepercayaan']:.2f}"

                # Background untuk teks
                (w1, h1), _ = cv2.getTextSize(teks_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                (w2, h2), _ = cv2.getTextSize(teks_kepercayaan, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(frame_beranotasi, (x1, y1-h1-h2-10), (x1+max(w1,w2)+10, y1), warna, -1)

                # Teks
                cv2.putText(frame_beranotasi, teks_label, (x1+5, y1-h2-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_beranotasi, teks_kepercayaan, (x1+5, y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Simpan info untuk crop
                info_deteksi.append({
                    'jenis_bagian': jenis_bagian,
                    'kotak_pembatas': (x1, y1, x2, y2),
                    'indeks': i,
                    'kepercayaan': deteksi['kepercayaan'],
                    'dimensi': (deteksi['lebar_cm'], deteksi['tinggi_cm'])
                })

        return frame_beranotasi, info_deteksi

    def simpan_potongan(self, frame, info_deteksi, waktu):
        """Simpan potongan dari setiap deteksi"""
        for info in info_deteksi:
            x1, y1, x2, y2 = info['kotak_pembatas']
            potongan = frame[y1:y2, x1:x2]

            if potongan.size > 0:  # Pastikan potongan tidak kosong
                nama_berkas = f"output_potongan/{waktu}_{info['jenis_bagian']}_{info['indeks']}_yakin{info['kepercayaan']:.2f}.jpg"
                cv2.imwrite(nama_berkas, potongan)

    def simpan_log_deteksi(self, deteksi_dict, waktu):
        """Simpan log deteksi ke file JSON"""
        data_log = {
            'waktu': waktu,
            'deteksi': {}
        }

        for jenis_bagian, deteksi_bagian in deteksi_dict.items():
            data_log['deteksi'][jenis_bagian] = []
            for deteksi in deteksi_bagian:
                data_log['deteksi'][jenis_bagian].append({
                    'kotak_pembatas': deteksi['kotak_pembatas'],
                    'kepercayaan': deteksi['kepercayaan'],
                    'dimensi_cm': (deteksi['lebar_cm'], deteksi['tinggi_cm']),
                    'area_px': deteksi['area_px'],
                    'rasio_aspek': deteksi['rasio_aspek']
                })

        with open(f"log_deteksi/{waktu}_deteksi.json", 'w') as f:
            json.dump(data_log, f, indent=2)

def main():
    # Inisialisasi detektor
    detektor = PendeteksiBagianPakaian(skala_cm_per_px=0.05)

    # Setup IP Camera
    url_ip_camera = 'http://192.168.25.192:8080/video/mjpeg'
    cap = cv2.VideoCapture(url_ip_camera)

    if not cap.isOpened():
        print("[ERROR] Gagal mengakses IP Camera.")
        return

    print("[INFO] Memulai deteksi bagian-bagian pakaian dari IP Camera...")

    # Buat folder output
    os.makedirs("output_potongan", exist_ok=True)
    os.makedirs("log_deteksi", exist_ok=True)

    jumlah_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari IP Camera.")
            break

        jumlah_frame += 1
        waktu = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Deteksi bagian-bagian pakaian
        deteksi = detektor.deteksi_bagian_pakaian(frame)

        # Tingkatkan akurasi
        deteksi_ditingkatkan = detektor.tingkatkan_akurasi_deteksi(deteksi, frame)

        # Gambar hasil deteksi
        frame_beranotasi, info_deteksi = detektor.gambar_deteksi(frame, deteksi_ditingkatkan)

        # Tambahkan info frame
        teks_info = f"Frame: {jumlah_frame} | Bagian terdeteksi: {len(info_deteksi)}"
        cv2.putText(frame_beranotasi, teks_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Tampilkan frame
        cv2.imshow("Deteksi Bagian Pakaian yang Ditingkatkan", frame_beranotasi)

        # Simpan potongan dan log setiap 30 frame (atau sesuai kebutuhan)
        if jumlah_frame % 30 == 0 and info_deteksi:
            detektor.simpan_potongan(frame, info_deteksi, waktu)
            detektor.simpan_log_deteksi(deteksi_ditingkatkan, waktu)
            print(f"[INFO] Menyimpan potongan dan log untuk frame {jumlah_frame}")

        # Keluar dengan ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Deteksi selesai.")

if __name__ == "__main__":
    main()
