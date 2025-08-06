from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import math
from scipy.spatial.distance import cdist
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SistemDeteksiBajuCanggih:
    def __init__(self, base_path="runs/detect", max_train=15, skala_cm_per_px=0.05):
        self.models = self.muat_semua_model(base_path, max_train)
        self.skala_cm_per_px = skala_cm_per_px
        self.riwayat_deteksi = defaultdict(list)
        self.frame_buffer = []  # Buffer untuk temporal consistency
        self.deteksi_stabil = defaultdict(list)  # Tracking deteksi stabil

        # Definisi bagian baju sesuai dataset Indonesia
        self.bagian_baju = {
            'kerah': {
                'keywords': ['kerah', 'collar', 'neck', 'leher'],
                'posisi_relatif': 'atas',
                'ukuran_min': 0.02,  # 2% dari frame
                'ukuran_max': 0.15,  # 15% dari frame
                'aspect_ratio_min': 0.5,
                'aspect_ratio_max': 4.0,
                'prioritas': 1
            },
            'lengan_kanan': {
                'keywords': ['lengan_kanan', 'right_sleeve', 'sleeve_right', 'lengan kanan'],
                'posisi_relatif': 'kanan',
                'ukuran_min': 0.05,
                'ukuran_max': 0.25,
                'aspect_ratio_min': 0.3,
                'aspect_ratio_max': 2.5,
                'prioritas': 2
            },
            'lengan_kiri': {
                'keywords': ['lengan_kiri', 'left_sleeve', 'sleeve_left', 'lengan kiri'],
                'posisi_relatif': 'kiri',
                'ukuran_min': 0.05,
                'ukuran_max': 0.25,
                'aspect_ratio_min': 0.3,
                'aspect_ratio_max': 2.5,
                'prioritas': 2
            },
            'badan': {
                'keywords': ['badan', 'body', 'chest', 'torso', 'dada', 'tubuh'],
                'posisi_relatif': 'tengah',
                'ukuran_min': 0.1,
                'ukuran_max': 0.6,
                'aspect_ratio_min': 0.4,
                'aspect_ratio_max': 2.0,
                'prioritas': 3
            },
            'kantong': {
                'keywords': ['kantong', 'pocket', 'saku'],
                'posisi_relatif': 'depan',
                'ukuran_min': 0.005,
                'ukuran_max': 0.08,
                'aspect_ratio_min': 0.6,
                'aspect_ratio_max': 2.5,
                'prioritas': 4
            }
        }

        # Warna yang lebih menarik untuk setiap bagian
        self.warna_bagian = {
            'kerah': (0, 100, 255),      # Merah terang
            'lengan_kanan': (0, 255, 100),  # Hijau terang
            'lengan_kiri': (255, 100, 0),   # Biru terang
            'badan': (100, 255, 255),       # Kuning terang
            'kantong': (255, 0, 255),       # Magenta
            'unknown': (128, 128, 128)      # Abu-abu
        }

        # Confidence threshold yang adaptif
        self.confidence_threshold = {
            'kerah': 0.4,
            'lengan_kanan': 0.35,
            'lengan_kiri': 0.35,
            'badan': 0.3,
            'kantong': 0.45  # Kantong lebih sulit dideteksi
        }

        logging.info(f"Sistem deteksi baju canggih telah diinisialisasi dengan {len(self.models)} model")

    def muat_semua_model(self, base_path, max_train):
        """Muat semua model YOLO dari folder runs/detect/train, train2, dst"""
        models = []

        print(f"🔍 Mencari model di: {base_path}")

        # Pola pencarian: train, train2, train3, ..., train12
        for i in range(1, max_train + 1):
            if i == 1:
                folder_name = "train"  # Folder pertama adalah "train"
            else:
                folder_name = f"train{i}"  # Folder selanjutnya train2, train3, dst

            folder_path = os.path.join(base_path, folder_name)
            best_model_path = os.path.join(folder_path, "weights", "best.pt")

            # Cek apakah best.pt ada
            if os.path.exists(best_model_path):
                try:
                    model = YOLO(best_model_path)
                    models.append((folder_name, model))
                    logging.info(f"✅ Model {folder_name} berhasil dimuat: {best_model_path}")
                    print(f"✅ Model {folder_name} berhasil dimuat")
                except Exception as e:
                    logging.warning(f"❌ Gagal memuat model {folder_name}: {e}")
                    print(f"❌ Gagal memuat model {folder_name}: {e}")
            else:
                # Cari file .pt alternatif di folder weights
                weights_folder = os.path.join(folder_path, "weights")
                if os.path.exists(weights_folder):
                    print(f"🔍 Mencari model alternatif di {weights_folder}")
                    for file in os.listdir(weights_folder):
                        if file.endswith(".pt"):
                            model_path = os.path.join(weights_folder, file)
                            try:
                                model = YOLO(model_path)
                                models.append((f"{folder_name}_{file[:-3]}", model))
                                logging.info(f"✅ Model alternatif {folder_name}/{file} berhasil dimuat")
                                print(f"✅ Model alternatif {folder_name}/{file} berhasil dimuat")
                                break
                            except Exception as e:
                                logging.warning(f"❌ Gagal memuat model alternatif {folder_name}/{file}: {e}")
                else:
                    print(f"⚠️  Folder {folder_path} tidak ditemukan")

        if not models:
            error_msg = f"❌ Tidak ada model YOLO (.pt) yang dapat dimuat dari {base_path}"
            print(error_msg)
            print("📁 Pastikan struktur folder seperti ini:")
            print("   runs/detect/train/weights/best.pt")
            print("   runs/detect/train2/weights/best.pt")
            print("   runs/detect/train3/weights/best.pt")
            print("   ...")
            raise FileNotFoundError(error_msg)

        print(f"🎯 Total {len(models)} model berhasil dimuat")
        return models

    def klasifikasi_bagian_baju(self, label, model_name=None):
        """Klasifikasi label ke bagian baju berdasarkan nama label dari dataset"""
        if not label:
            return 'unknown'

        label_lower = label.lower().strip()

        # Keyword matching langsung tanpa mapping model
        for bagian, info in self.bagian_baju.items():
            if label_lower in info['keywords']:
                return bagian

        # Fuzzy matching menggunakan substring
        best_match = 'unknown'
        best_score = 0

        for bagian, info in self.bagian_baju.items():
            for keyword in info['keywords']:
                # Hitung similarity score
                if keyword in label_lower or label_lower in keyword:
                    score = len(keyword) / max(len(keyword), len(label_lower))
                    if score > best_score:
                        best_score = score
                        best_match = bagian

        # Jika tidak ada match yang bagus, coba klasifikasi berdasarkan class number
        if best_score < 0.3 and label_lower.startswith('class_'):
            try:
                class_num = int(label_lower.split('_')[1])
                class_mapping = {
                    0: 'kerah',
                    1: 'lengan_kanan',
                    2: 'lengan_kiri',
                    3: 'badan',
                    4: 'kantong'
                }
                if class_num in class_mapping:
                    return class_mapping[class_num]
            except:
                pass

        return best_match if best_score > 0.3 else 'unknown'

    def validasi_posisi_relatif(self, bbox, bagian, frame_shape):
        """Validasi posisi relatif bagian baju dalam frame"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        frame_width, frame_height = frame_shape[1], frame_shape[0]

        # Normalisasi koordinat (0-1)
        norm_x = center_x / frame_width
        norm_y = center_y / frame_height

        info_bagian = self.bagian_baju.get(bagian, {})
        posisi_relatif = info_bagian.get('posisi_relatif', 'tengah')

        valid = True

        if posisi_relatif == 'atas' and norm_y > 0.6:  # Kerah seharusnya di atas
            valid = False
        elif posisi_relatif == 'kanan' and norm_x < 0.3:  # Lengan kanan seharusnya di kanan
            valid = False
        elif posisi_relatif == 'kiri' and norm_x > 0.7:  # Lengan kiri seharusnya di kiri
            valid = False
        elif posisi_relatif == 'tengah' and (norm_x < 0.2 or norm_x > 0.8):  # Badan di tengah
            valid = False

        return valid

    def hitung_iou_canggih(self, box1, box2):
        """Hitung IoU dengan penanganan edge case yang lebih baik"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Validasi input
        if x2_1 <= x1_1 or y2_1 <= y1_1 or x2_2 <= x1_2 or y2_2 <= y1_2:
            return 0.0

        # Hitung intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def nms_canggih(self, detections, iou_threshold=0.4):
        """Non-Maximum Suppression dengan weighted averaging"""
        if len(detections) <= 1:
            return detections

        # Sort berdasarkan confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        filtered_detections = []

        for detection in detections:
            keep = True
            merged = False

            for i, filtered in enumerate(filtered_detections):
                iou = self.hitung_iou_canggih(detection['bbox'], filtered['bbox'])

                if iou > iou_threshold:
                    # Merge deteksi dengan weighted average
                    if detection['confidence'] > filtered['confidence'] * 0.8:
                        # Weighted average of bounding boxes
                        w1 = detection['confidence']
                        w2 = filtered['confidence']
                        total_weight = w1 + w2

                        x1_1, y1_1, x2_1, y2_1 = detection['bbox']
                        x1_2, y1_2, x2_2, y2_2 = filtered['bbox']

                        new_bbox = (
                            int((x1_1 * w1 + x1_2 * w2) / total_weight),
                            int((y1_1 * w1 + y1_2 * w2) / total_weight),
                            int((x2_1 * w1 + x2_2 * w2) / total_weight),
                            int((y2_1 * w1 + y2_2 * w2) / total_weight)
                        )

                        # Update filtered detection
                        filtered_detections[i]['bbox'] = new_bbox
                        filtered_detections[i]['confidence'] = max(detection['confidence'], filtered['confidence'])
                        filtered_detections[i]['merged_count'] = filtered_detections[i].get('merged_count', 1) + 1

                        merged = True
                    keep = False
                    break

            if keep and not merged:
                detection['merged_count'] = 1
                filtered_detections.append(detection)

        return filtered_detections

    def deteksi_temporal_consistency(self, detections_current):
        """Tambahkan temporal consistency untuk mengurangi flicker"""
        self.frame_buffer.append(detections_current)

        # Pertahankan buffer maksimal 5 frame
        if len(self.frame_buffer) > 5:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) < 3:
            return detections_current

        # Analisis konsistensi deteksi
        consistent_detections = {}

        for bagian in self.bagian_baju.keys():
            bagian_detections = []

            # Kumpulkan deteksi dari beberapa frame terakhir
            for frame_detections in self.frame_buffer[-3:]:
                if bagian in frame_detections:
                    bagian_detections.extend(frame_detections[bagian])

            if bagian_detections:
                # Cluster deteksi yang mirip
                clustered = self.cluster_detections(bagian_detections)
                if clustered:
                    consistent_detections[bagian] = clustered

        return consistent_detections

    def cluster_detections(self, detections):
        """Cluster deteksi yang mirip untuk konsistensi temporal"""
        if not detections:
            return []

        # Hitung centroid dari setiap deteksi
        centroids = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
            centroids.append(centroid)

        # Simple clustering berdasarkan jarak
        clusters = []
        used = set()

        for i, det in enumerate(detections):
            if i in used:
                continue

            cluster = [det]
            used.add(i)

            for j, other_det in enumerate(detections):
                if j in used:
                    continue

                # Hitung jarak centroid
                dist = math.sqrt((centroids[i][0] - centroids[j][0])**2 +
                               (centroids[i][1] - centroids[j][1])**2)

                if dist < 50:  # Threshold jarak clustering
                    cluster.append(other_det)
                    used.add(j)

            if cluster:
                # Ambil deteksi dengan confidence tertinggi dari cluster
                best_detection = max(cluster, key=lambda x: x['confidence'])
                clusters.append(best_detection)

        return clusters

    def deteksi_bagian_baju_canggih(self, frame):
        """Deteksi bagian baju dengan algoritma yang lebih canggih"""
        semua_deteksi = defaultdict(list)
        frame_area = frame.shape[0] * frame.shape[1]

        # Preprocessing frame untuk meningkatkan akurasi
        frame_enhanced = self.enhance_frame(frame)

        # Jalankan semua model dengan berbagai ukuran input
        for model_name, model in self.models:
            # Multi-scale detection
            for img_size in [640, 800]:  # Coba berbagai ukuran
                try:
                    results = model(frame_enhanced, imgsz=img_size, conf=0.15, iou=0.3, verbose=False)
                    result = results[0]

                    if result.boxes is not None:
                        for box in result.boxes:
                            confidence = float(box.conf[0])

                            # Skip deteksi dengan confidence terlalu rendah
                            if confidence < 0.2:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Validasi bounding box
                            if x2 <= x1 or y2 <= y1:
                                continue

                            label = result.names[int(box.cls[0])] if result.names else f"class_{int(box.cls[0])}"

                            # Klasifikasi bagian baju (dengan info model untuk mapping yang lebih baik)
                            bagian = self.klasifikasi_bagian_baju(label, model_name)

                            # Skip jika tidak dikenali
                            if bagian == 'unknown':
                                continue

                            # Validasi ukuran dan aspect ratio
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            aspect_ratio = width / height if height > 0 else 0

                            info_bagian = self.bagian_baju[bagian]
                            area_ratio = area / frame_area

                            # Filter berdasarkan kriteria bagian
                            if (area_ratio < info_bagian['ukuran_min'] or
                                area_ratio > info_bagian['ukuran_max'] or
                                aspect_ratio < info_bagian['aspect_ratio_min'] or
                                aspect_ratio > info_bagian['aspect_ratio_max']):
                                continue

                            # Validasi posisi relatif
                            if not self.validasi_posisi_relatif((x1, y1, x2, y2), bagian, frame.shape):
                                confidence *= 0.7  # Kurangi confidence jika posisi tidak sesuai

                            # Filter berdasarkan confidence threshold adaptif
                            if confidence < self.confidence_threshold.get(bagian, 0.3):
                                continue

                            deteksi = {
                                'bbox': (x1, y1, x2, y2),
                                'label': label,
                                'confidence': confidence,
                                'model': model_name,
                                'bagian': bagian,
                                'area_px': area,
                                'aspect_ratio': aspect_ratio,
                                'img_size': img_size
                            }

                            semua_deteksi[bagian].append(deteksi)

                except Exception as e:
                    logging.warning(f"Error dalam deteksi dengan model {model_name}: {e}")
                    continue

        # Apply NMS untuk setiap bagian
        deteksi_final = {}
        for bagian, detections in semua_deteksi.items():
            if detections:
                filtered = self.nms_canggih(detections, iou_threshold=0.3)
                if filtered:
                    deteksi_final[bagian] = filtered

        # Temporal consistency
        deteksi_konsisten = self.deteksi_temporal_consistency(deteksi_final)

        return deteksi_konsisten

    def enhance_frame(self, frame):
        """Enhance frame untuk meningkatkan akurasi deteksi"""
        # Normalisasi exposure
        enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

        # Slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    def gambar_deteksi_canggih(self, frame, detections):
        """Gambar hasil deteksi dengan visualisasi yang lebih canggih"""
        annotated_frame = frame.copy()
        info_deteksi = []

        # Background overlay untuk informasi
        overlay = annotated_frame.copy()

        for bagian, deteksi_bagian in detections.items():
            warna = self.warna_bagian.get(bagian, (255, 255, 255))

            for i, deteksi in enumerate(deteksi_bagian):
                x1, y1, x2, y2 = deteksi['bbox']
                confidence = deteksi['confidence']

                # Hitung ukuran dalam cm
                width_cm = (x2 - x1) * self.skala_cm_per_px
                height_cm = (y2 - y1) * self.skala_cm_per_px

                # Gambar bounding box dengan efek glow
                thickness = 3 if confidence > 0.7 else 2

                # Outer glow effect
                cv2.rectangle(annotated_frame, (x1-2, y1-2), (x2+2, y2+2),
                             tuple(int(c*0.5) for c in warna), thickness+1)

                # Main bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), warna, thickness)

                # Corner markers untuk efek modern
                corner_size = 15
                corner_thickness = 3

                # Top-left corner
                cv2.line(annotated_frame, (x1, y1), (x1+corner_size, y1), warna, corner_thickness)
                cv2.line(annotated_frame, (x1, y1), (x1, y1+corner_size), warna, corner_thickness)

                # Top-right corner
                cv2.line(annotated_frame, (x2, y1), (x2-corner_size, y1), warna, corner_thickness)
                cv2.line(annotated_frame, (x2, y1), (x2, y1+corner_size), warna, corner_thickness)

                # Bottom-left corner
                cv2.line(annotated_frame, (x1, y2), (x1+corner_size, y2), warna, corner_thickness)
                cv2.line(annotated_frame, (x1, y2), (x1, y2-corner_size), warna, corner_thickness)

                # Bottom-right corner
                cv2.line(annotated_frame, (x2, y2), (x2-corner_size, y2), warna, corner_thickness)
                cv2.line(annotated_frame, (x2, y2), (x2, y2-corner_size), warna, corner_thickness)

                # Label dengan background yang stylish
                label_utama = f"{bagian.upper().replace('_', ' ')}"
                label_ukuran = f"{width_cm:.1f} x {height_cm:.1f} cm"
                label_confidence = f"Akurasi: {confidence:.1%}"

                # Hitung ukuran text
                font = cv2.FONT_HERSHEY_DUPLEX
                (w1, h1), _ = cv2.getTextSize(label_utama, font, 0.7, 2)
                (w2, h2), _ = cv2.getTextSize(label_ukuran, font, 0.5, 1)
                (w3, h3), _ = cv2.getTextSize(label_confidence, font, 0.5, 1)

                max_width = max(w1, w2, w3) + 20
                total_height = h1 + h2 + h3 + 20

                # Background dengan transparansi
                label_bg = (x1, y1 - total_height - 5, x1 + max_width, y1)
                cv2.rectangle(overlay, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]),
                             warna, -1)

                # Blend dengan transparansi
                alpha = 0.8
                cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

                # Text labels
                y_offset = y1 - total_height + h1
                cv2.putText(annotated_frame, label_utama, (x1+10, y_offset),
                           font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                y_offset += h2 + 5
                cv2.putText(annotated_frame, label_ukuran, (x1+10, y_offset),
                           font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                y_offset += h3 + 5
                cv2.putText(annotated_frame, label_confidence, (x1+10, y_offset),
                           font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Tambahkan titik tengah
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated_frame, (center_x, center_y), 4, warna, -1)
                cv2.circle(annotated_frame, (center_x, center_y), 6, (255, 255, 255), 2)

                # Simpan info untuk crop
                info_deteksi.append({
                    'bagian': bagian,
                    'bbox': (x1, y1, x2, y2),
                    'index': i,
                    'confidence': confidence,
                    'dimensions': (width_cm, height_cm),
                    'merged_count': deteksi.get('merged_count', 1)
                })

        return annotated_frame, info_deteksi

    def simpan_crop_canggih(self, frame, info_deteksi, timestamp):
        """Simpan crop dengan metadata yang lebih lengkap"""
        for info in info_deteksi:
            x1, y1, x2, y2 = info['bbox']

            # Padding untuk crop yang lebih baik
            padding = 10
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(frame.shape[1], x2 + padding)
            y2_pad = min(frame.shape[0], y2 + padding)

            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

            if crop.size > 0:
                # Nama file yang lebih deskriptif
                filename = (f"crop_output/{timestamp}_{info['bagian']}_"
                           f"idx{info['index']}_conf{info['confidence']:.2f}_"
                           f"merged{info['merged_count']}.jpg")

                # Enhance crop quality
                if crop.shape[0] > 50 and crop.shape[1] > 50:
                    # Upscale jika terlalu kecil
                    if crop.shape[0] < 200 or crop.shape[1] < 200:
                        scale_factor = 2
                        crop = cv2.resize(crop, None, fx=scale_factor, fy=scale_factor,
                                        interpolation=cv2.INTER_CUBIC)

                    cv2.imwrite(filename, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def simpan_log_deteksi_canggih(self, detections, timestamp, frame_info):
        """Simpan log deteksi dengan analisis yang lebih detail"""
        log_data = {
            'timestamp': timestamp,
            'frame_info': frame_info,
            'summary': {
                'total_bagian_terdeteksi': len(detections),
                'bagian_terdeteksi': list(detections.keys()),
                'total_objek': sum(len(dets) for dets in detections.values())
            },
            'detections': {}
        }

        for bagian, deteksi_bagian in detections.items():
            log_data['detections'][bagian] = []

            for deteksi in deteksi_bagian:
                detail_deteksi = {
                    'bbox': deteksi['bbox'],
                    'confidence': deteksi['confidence'],
                    'dimensions_cm': (deteksi['bbox'][2] - deteksi['bbox'][0]) * self.skala_cm_per_px,
                    'area_px': deteksi['area_px'],
                    'aspect_ratio': deteksi['aspect_ratio'],
                    'model_source': deteksi['model'],
                    'img_size_used': deteksi.get('img_size', 640),
                    'merged_count': deteksi.get('merged_count', 1),
                    'label_original': deteksi['label']
                }

                log_data['detections'][bagian].append(detail_deteksi)

        # Analisis tambahan
        log_data['analysis'] = self.analisis_komposisi_baju(detections)

        with open(f"detection_logs/{timestamp}_deteksi_canggih.json", 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def analisis_komposisi_baju(self, detections):
        """Analisis komposisi dan kelengkapan deteksi baju"""
        analisis = {
            'kelengkapan_baju': {},
            'kualitas_deteksi': {},
            'rekomendasi': []
        }

        # Cek kelengkapan bagian baju
        bagian_penting = ['kerah', 'badan', 'lengan_kanan', 'lengan_kiri']
        bagian_terdeteksi = list(detections.keys())

        for bagian in bagian_penting:
            analisis['kelengkapan_baju'][bagian] = bagian in bagian_terdeteksi

        # Hitung persentase kelengkapan
        kelengkapan_persen = (sum(analisis['kelengkapan_baju'].values()) / len(bagian_penting)) * 100
        analisis['persentase_kelengkapan'] = kelengkapan_persen

        # Analisis kualitas deteksi per bagian
        for bagian, deteksi_bagian in detections.items():
            if deteksi_bagian:
                avg_confidence = np.mean([d['confidence'] for d in deteksi_bagian])
                max_confidence = max([d['confidence'] for d in deteksi_bagian])
                jumlah_deteksi = len(deteksi_bagian)

                analisis['kualitas_deteksi'][bagian] = {
                    'rata_rata_confidence': avg_confidence,
                    'confidence_tertinggi': max_confidence,
                    'jumlah_deteksi': jumlah_deteksi,
                    'kualitas': 'Baik' if avg_confidence > 0.7 else 'Sedang' if avg_confidence > 0.5 else 'Rendah'
                }

        # Generate rekomendasi
        if kelengkapan_persen < 50:
            analisis['rekomendasi'].append("Posisi baju kurang optimal untuk deteksi lengkap")

        if 'kerah' not in bagian_terdeteksi:
            analisis['rekomendasi'].append("Pastikan kerah terlihat jelas dalam frame")

        if not ('lengan_kanan' in bagian_terdeteksi and 'lengan_kiri' in bagian_terdeteksi):
            analisis['rekomendasi'].append("Posisikan baju agar kedua lengan terlihat")

        return analisis

def main():
    """Fungsi utama untuk menjalankan sistem deteksi canggih"""
    print("=" * 60)
    print("🔍 SISTEM DETEKSI BAGIAN BAJU INDONESIA - ADVANCED")
    print("=" * 60)

    try:
        # Inisialisasi detector dengan path yang benar
        detector = SistemDeteksiBajuCanggih(
            base_path="runs/detect",  # Path ke folder runs/detect
            skala_cm_per_px=0.05
        )

        # Setup kamera (gunakan webcam default atau IP camera)
        # Uncomment baris yang sesuai dengan setup Anda

        # Untuk webcam default:
        # cap = cv2.VideoCapture(0)

        # Untuk IP Camera (uncomment jika menggunakan IP camera):
        ip_camera_url = 'http://10.94.239.254:8080//video/mjpeg'
        cap = cv2.VideoCapture(0)

        # Set resolusi untuk kualitas yang lebih baik
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ [ERROR] Gagal mengakses kamera.")
            return

        print("✅ [INFO] Kamera berhasil diakses")
        print("📱 [INFO] Memulai deteksi bagian-bagian baju...")
        print("⌨️  [INFO] Tekan 'ESC' untuk keluar, 'S' untuk screenshot, 'R' untuk reset")

        # Buat folder output
        os.makedirs("crop_output", exist_ok=True)
        os.makedirs("detection_logs", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)

        # Variabel tracking
        frame_count = 0
        total_deteksi = 0
        start_time = datetime.now()

        # FPS calculation
        fps_counter = 0
        fps_start_time = datetime.now()
        current_fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ [ERROR] Gagal membaca frame dari kamera.")
                break

            frame_count += 1
            fps_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Hitung FPS setiap 30 frame
            if fps_counter >= 30:
                current_time = datetime.now()
                time_diff = (current_time - fps_start_time).total_seconds()
                current_fps = fps_counter / time_diff
                fps_counter = 0
                fps_start_time = current_time

            # Deteksi bagian-bagian baju
            try:
                detections = detector.deteksi_bagian_baju_canggih(frame)

                # Gambar hasil deteksi
                annotated_frame, info_deteksi = detector.gambar_deteksi_canggih(frame, detections)

                # Update total deteksi
                jumlah_deteksi_frame = len(info_deteksi)
                total_deteksi += jumlah_deteksi_frame

                # Informasi status sistem
                status_info = [
                    f"Frame: {frame_count}",
                    f"FPS: {current_fps:.1f}",
                    f"Terdeteksi: {jumlah_deteksi_frame} bagian",
                    f"Total: {total_deteksi} deteksi"
                ]

                # Gambar panel informasi
                panel_height = 120
                panel = np.zeros((panel_height, annotated_frame.shape[1], 3), dtype=np.uint8)
                panel[:] = (30, 30, 30)  # Background gelap

                # Border panel
                cv2.rectangle(panel, (0, 0), (panel.shape[1]-1, panel.shape[0]-1),
                             (100, 100, 100), 2)

                # Judul sistem
                cv2.putText(panel, "SISTEM DETEKSI BAJU INDONESIA - ADVANCED",
                           (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

                # Status info
                y_pos = 45
                for i, info in enumerate(status_info):
                    color = (255, 255, 255) if i < 2 else (0, 255, 0)
                    cv2.putText(panel, info, (10 + (i % 2) * 300, y_pos + (i // 2) * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Instruksi kontrol
                cv2.putText(panel, "ESC: Keluar | S: Screenshot | R: Reset Counter",
                           (10, panel_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           (200, 200, 200), 1)

                # Gabungkan panel dengan frame utama
                display_frame = np.vstack([panel, annotated_frame])

                # Tampilkan frame
                cv2.imshow("🔍 Sistem Deteksi Baju Indonesia - Advanced", display_frame)

                # Simpan crops dan logs setiap 60 frame atau ketika ada deteksi bagus
                if (frame_count % 60 == 0 or jumlah_deteksi_frame >= 3) and info_deteksi:
                    frame_info = {
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'fps': current_fps,
                        'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
                    }

                    detector.simpan_crop_canggih(frame, info_deteksi, timestamp)
                    detector.simpan_log_deteksi_canggih(detections, timestamp, frame_info)

                    print(f"💾 [INFO] Frame {frame_count}: Disimpan {jumlah_deteksi_frame} deteksi")

            except Exception as e:
                logging.error(f"Error dalam proses deteksi frame {frame_count}: {e}")
                continue

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                break
            elif key == ord('s') or key == ord('S'):  # Screenshot
                screenshot_path = f"screenshots/screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"📸 [INFO] Screenshot disimpan: {screenshot_path}")
            elif key == ord('r') or key == ord('R'):  # Reset counter
                frame_count = 0
                total_deteksi = 0
                start_time = datetime.now()
                print("🔄 [INFO] Counter direset")

        # Cleanup dan statistik final
        runtime = (datetime.now() - start_time).total_seconds()
        avg_fps = frame_count / runtime if runtime > 0 else 0

        print("\n" + "=" * 60)
        print("📊 STATISTIK AKHIR:")
        print(f"⏱️  Total waktu berjalan: {runtime:.1f} detik")
        print(f"🎞️  Total frame diproses: {frame_count}")
        print(f"📈 FPS rata-rata: {avg_fps:.1f}")
        print(f"🎯 Total deteksi: {total_deteksi}")
        print(f"📁 File tersimpan di: crop_output/, detection_logs/, screenshots/")
        print("=" * 60)

    except Exception as e:
        logging.error(f"Error dalam sistem utama: {e}")
        print(f"❌ [ERROR] Terjadi kesalahan: {e}")

    finally:
        # Pastikan resource dibersihkan
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("✅ [INFO] Sistem deteksi telah dihentikan dengan aman.")

if __name__ == "__main__":
    main()
