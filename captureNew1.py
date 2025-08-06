import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter import font as tkFont
from datetime import datetime
import json
import numpy as np
from PIL import Image, ImageTk
import threading
import time

class SistemPenangkapanDataset:
    def __init__(self):
        # Konfigurasi dasar
        self.BASE_FOLDER = "dataset_pakaian"
        self.IMAGE_FOLDER = os.path.join(self.BASE_FOLDER, "images")
        self.LABEL_FOLDER = os.path.join(self.BASE_FOLDER, "labels")
        self.BACKUP_FOLDER = os.path.join(self.BASE_FOLDER, "backup")
        self.STATS_FILE = os.path.join(self.BASE_FOLDER, "dataset_stats.json")

        # Kelas label pakaian
        self.LABEL_CLASSES = [
            "lengan_kanan", "lengan_kiri", "kerah", "kantong",
            "badan", "kancing", "manset", "kelim", "bahu"
        ]

        # Warna untuk setiap kelas
        self.CLASS_COLORS = {
            "lengan_kanan": (0, 255, 0),    # Hijau
            "lengan_kiri": (255, 0, 0),     # Biru
            "kerah": (0, 0, 255),           # Merah
            "kantong": (255, 255, 0),       # Cyan
            "badan": (255, 0, 255),         # Magenta
            "kancing": (0, 255, 255),       # Kuning
            "manset": (128, 255, 0),        # Lime
            "kelim": (255, 128, 0),         # Orange
            "bahu": (128, 0, 255)           # Violet
        }

        self.IMG_SIZE = 640
        self.MIN_BOX_SIZE = 20
        self.current_frame = None
        self.original_frame = None
        self.drawing = False
        self.boxes = []
        self.current_box = None
        self.selected_class = 0
        self.dataset_stats = self.load_stats()

        # Setup folders
        self.setup_folders()

        # Camera setup
        self.cap = None
        self.camera_active = False

        # Setup GUI
        self.setup_gui()

    def setup_folders(self):
        """Buat struktur folder yang diperlukan"""
        folders = [
            self.IMAGE_FOLDER,
            self.LABEL_FOLDER,
            self.BACKUP_FOLDER,
            os.path.join(self.IMAGE_FOLDER, "train"),
            os.path.join(self.IMAGE_FOLDER, "val"),
            os.path.join(self.IMAGE_FOLDER, "test"),
            os.path.join(self.LABEL_FOLDER, "train"),
            os.path.join(self.LABEL_FOLDER, "val"),
            os.path.join(self.LABEL_FOLDER, "test")
        ]

        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def load_stats(self):
        """Muat statistik dataset"""
        if os.path.exists(self.STATS_FILE):
            with open(self.STATS_FILE, 'r') as f:
                return json.load(f)
        else:
            return {
                "total_images": 0,
                "total_annotations": 0,
                "class_distribution": {cls: 0 for cls in self.LABEL_CLASSES},
                "last_session": datetime.now().isoformat(),
                "split_counts": {"train": 0, "val": 0, "test": 0}
            }

    def save_stats(self):
        """Simpan statistik dataset"""
        with open(self.STATS_FILE, 'w') as f:
            json.dump(self.dataset_stats, f, indent=2)

    def setup_gui(self):
        """Setup GUI utama"""
        self.root = tk.Tk()
        self.root.title("Sistem Penangkapan Dataset Pakaian - Advanced")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')

        # Font styling
        self.title_font = tkFont.Font(family="Arial", size=12, weight="bold")
        self.label_font = tkFont.Font(family="Arial", size=10)

        self.setup_main_frame()
        self.setup_control_panel()
        self.setup_stats_panel()
        self.setup_class_panel()

    def setup_main_frame(self):
        """Setup frame utama untuk video"""
        self.main_frame = tk.Frame(self.root, bg='#34495e', bd=2, relief='raised')
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(self.main_frame, text="🎥 Live Camera Feed",
                              font=self.title_font, bg='#34495e', fg='white')
        title_label.pack(pady=10)

        # Canvas untuk video
        self.canvas = tk.Canvas(self.main_frame, width=640, height=640, bg='black')
        self.canvas.pack(padx=10, pady=10)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Siap untuk menangkap dataset")
        status_bar = tk.Label(self.main_frame, textvariable=self.status_var,
                             bg='#2c3e50', fg='#ecf0f1', font=self.label_font)
        status_bar.pack(fill=tk.X, pady=5)

    def setup_control_panel(self):
        """Setup panel kontrol"""
        control_frame = tk.Frame(self.root, bg='#34495e', bd=2, relief='raised')
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

        # Title
        title_label = tk.Label(control_frame, text="🎛️ Panel Kontrol",
                              font=self.title_font, bg='#34495e', fg='white')
        title_label.pack(pady=10)

        # Camera controls
        cam_frame = tk.LabelFrame(control_frame, text="Kontrol Kamera",
                                 bg='#34495e', fg='white', font=self.label_font)
        cam_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_btn = tk.Button(cam_frame, text="▶️ Mulai Kamera",
                                  command=self.start_camera, bg='#27ae60', fg='white',
                                  font=self.label_font, width=15)
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(cam_frame, text="⏹️ Stop Kamera",
                                 command=self.stop_camera, bg='#e74c3c', fg='white',
                                 font=self.label_font, width=15, state='disabled')
        self.stop_btn.pack(pady=5)

        # Dataset controls
        dataset_frame = tk.LabelFrame(control_frame, text="Kontrol Dataset",
                                     bg='#34495e', fg='white', font=self.label_font)
        dataset_frame.pack(fill=tk.X, padx=10, pady=5)

        # Split selection
        tk.Label(dataset_frame, text="Split Dataset:", bg='#34495e', fg='white').pack()
        self.split_var = tk.StringVar(value="train")
        split_combo = ttk.Combobox(dataset_frame, textvariable=self.split_var,
                                  values=["train", "val", "test"], state="readonly", width=12)
        split_combo.pack(pady=5)

        # Auto capture
        self.auto_capture_var = tk.BooleanVar()
        auto_check = tk.Checkbutton(dataset_frame, text="Auto Capture (5s)",
                                   variable=self.auto_capture_var, bg='#34495e', fg='white',
                                   selectcolor='#34495e', font=self.label_font)
        auto_check.pack(pady=5)

        # Buttons
        tk.Button(dataset_frame, text="💾 Simpan Frame", command=self.save_current_frame,
                 bg='#3498db', fg='white', font=self.label_font, width=15).pack(pady=2)

        tk.Button(dataset_frame, text="🗑️ Hapus Boxes", command=self.clear_boxes,
                 bg='#e67e22', fg='white', font=self.label_font, width=15).pack(pady=2)

        tk.Button(dataset_frame, text="📊 Export Dataset", command=self.export_dataset,
                 bg='#9b59b6', fg='white', font=self.label_font, width=15).pack(pady=2)

        # Augmentation controls
        aug_frame = tk.LabelFrame(control_frame, text="Augmentasi Data",
                                 bg='#34495e', fg='white', font=self.label_font)
        aug_frame.pack(fill=tk.X, padx=10, pady=5)

        self.brightness_var = tk.BooleanVar()
        self.rotation_var = tk.BooleanVar()
        self.flip_var = tk.BooleanVar()

        tk.Checkbutton(aug_frame, text="Brightness", variable=self.brightness_var,
                      bg='#34495e', fg='white', selectcolor='#34495e').pack(anchor='w')
        tk.Checkbutton(aug_frame, text="Rotation", variable=self.rotation_var,
                      bg='#34495e', fg='white', selectcolor='#34495e').pack(anchor='w')
        tk.Checkbutton(aug_frame, text="Flip", variable=self.flip_var,
                      bg='#34495e', fg='white', selectcolor='#34495e').pack(anchor='w')

    def setup_stats_panel(self):
        """Setup panel statistik"""
        stats_frame = tk.LabelFrame(self.root.winfo_toplevel(), text="📈 Statistik Dataset",
                                   bg='#34495e', fg='white', font=self.label_font)
        stats_frame.place(x=10, y=650, width=300, height=140)

        self.stats_text = tk.Text(stats_frame, height=6, width=35, bg='#2c3e50', fg='#ecf0f1',
                                 font=('Courier', 9), state='disabled')
        self.stats_text.pack(padx=5, pady=5)

        self.update_stats_display()

    def setup_class_panel(self):
        """Setup panel pemilihan kelas"""
        class_frame = tk.LabelFrame(self.root.winfo_toplevel(), text="🏷️ Pilih Kelas",
                                   bg='#34495e', fg='white', font=self.label_font)
        class_frame.place(x=320, y=650, width=300, height=140)

        # Class buttons
        for i, class_name in enumerate(self.LABEL_CLASSES):
            row = i // 3
            col = i % 3
            color = self.CLASS_COLORS[class_name]
            color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"  # BGR to RGB hex

            btn = tk.Button(class_frame, text=class_name,
                           command=lambda idx=i: self.select_class(idx),
                           bg=color_hex, fg='white', font=('Arial', 8, 'bold'),
                           width=8, height=1)



        # Selected class display
        self.selected_class_var = tk.StringVar()
        self.selected_class_var.set(f"Terpilih: {self.LABEL_CLASSES[0]}")
        selected_label = tk.Label(class_frame, textvariable=self.selected_class_var,
                                 bg='#34495e', fg='#e74c3c', font=self.title_font)
        selected_label.grid(row=3, column=0, columnspan=3, pady=5)

    def select_class(self, class_idx):
        """Pilih kelas untuk labeling"""
        self.selected_class = class_idx
        self.selected_class_var.set(f"Terpilih: {self.LABEL_CLASSES[class_idx]}")
        self.status_var.set(f"Kelas aktif: {self.LABEL_CLASSES[class_idx]}")

    def start_camera(self):
        """Mulai kamera"""
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.camera_active = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_var.set("Kamera aktif - Drag mouse untuk membuat bounding box")
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            # Auto capture thread
            if self.auto_capture_var.get():
                self.auto_capture_thread = threading.Thread(target=self.auto_capture_loop, daemon=True)
                self.auto_capture_thread.start()
        else:
            messagebox.showerror("Error", "Tidak dapat mengakses kamera!")

    def stop_camera(self):
        """Stop kamera"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Kamera dimatikan")

    def camera_loop(self):
        """Loop utama kamera"""
        while self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                self.original_frame = frame.copy()
                self.current_frame = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))
                self.update_canvas()
            time.sleep(0.03)  # ~30 FPS

    def auto_capture_loop(self):
        """Loop auto capture"""
        while self.camera_active and self.auto_capture_var.get():
            time.sleep(5)  # Capture every 5 seconds
            if self.current_frame is not None and len(self.boxes) > 0:
                self.save_current_frame()

    def update_canvas(self):
        """Update canvas dengan frame terbaru"""
        if self.current_frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

            # Draw existing boxes
            frame_with_boxes = frame_rgb.copy()
            for box in self.boxes:
                x1, y1, x2, y2, class_idx = box
                color = self.CLASS_COLORS[self.LABEL_CLASSES[class_idx]]
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)

                # Label
                label_text = self.LABEL_CLASSES[class_idx]
                cv2.putText(frame_with_boxes, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert to PIL and display
            image = Image.fromarray(frame_with_boxes)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.delete("all")
            self.canvas.create_image(320, 320, image=self.photo)

    def on_mouse_down(self, event):
        """Mouse down event"""
        self.drawing = True
        self.start_x, self.start_y = event.x - 320, event.y - 320  # Center offset
        self.start_x = max(0, min(self.IMG_SIZE, self.start_x + 320)) - 320
        self.start_y = max(0, min(self.IMG_SIZE, self.start_y + 320)) - 320

    def on_mouse_drag(self, event):
        """Mouse drag event"""
        if self.drawing:
            current_x = max(0, min(self.IMG_SIZE, event.x - 320 + 320)) - 320
            current_y = max(0, min(self.IMG_SIZE, event.y - 320 + 320)) - 320

            # Update canvas with current box
            self.update_canvas()

            # Draw current box
            x1 = min(self.start_x + 320, current_x + 320)
            y1 = min(self.start_y + 320, current_y + 320)
            x2 = max(self.start_x + 320, current_x + 320)
            y2 = max(self.start_y + 320, current_y + 320)

            color = self.CLASS_COLORS[self.LABEL_CLASSES[self.selected_class]]
            color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color_hex, width=2)

    def on_mouse_up(self, event):
        """Mouse up event"""
        if self.drawing:
            self.drawing = False

            end_x = max(0, min(self.IMG_SIZE, event.x - 320 + 320)) - 320
            end_y = max(0, min(self.IMG_SIZE, event.y - 320 + 320)) - 320

            # Calculate box coordinates
            x1 = min(self.start_x, end_x) + 320
            y1 = min(self.start_y, end_y) + 320
            x2 = max(self.start_x, end_x) + 320
            y2 = max(self.start_y, end_y) + 320

            # Validate box size
            if abs(x2 - x1) > self.MIN_BOX_SIZE and abs(y2 - y1) > self.MIN_BOX_SIZE:
                self.boxes.append((x1, y1, x2, y2, self.selected_class))
                self.status_var.set(f"Box ditambahkan: {self.LABEL_CLASSES[self.selected_class]} (Total: {len(self.boxes)})")
            else:
                self.status_var.set("Box terlalu kecil - diabaikan")

    def on_mouse_move(self, event):
        """Mouse move event untuk menampilkan koordinat"""
        if not self.drawing:
            x = event.x - 320 + 320
            y = event.y - 320 + 320
            if 0 <= x < self.IMG_SIZE and 0 <= y < self.IMG_SIZE:
                self.status_var.set(f"Posisi: ({x}, {y}) | Kelas: {self.LABEL_CLASSES[self.selected_class]}")

    def clear_boxes(self):
        """Hapus semua bounding boxes"""
        self.boxes = []
        self.status_var.set("Semua boxes dihapus")

    def save_current_frame(self):
        """Simpan frame saat ini dengan anotasi"""
        if self.current_frame is None or len(self.boxes) == 0:
            messagebox.showwarning("Peringatan", "Tidak ada frame atau boxes untuk disimpan!")
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_filename = f"pakaian_{timestamp}"

        split = self.split_var.get()
        img_path = os.path.join(self.IMAGE_FOLDER, split, f"{img_filename}.jpg")
        label_path = os.path.join(self.LABEL_FOLDER, split, f"{img_filename}.txt")

        # Save image with augmentations
        images_to_save = [self.current_frame]
        labels_to_save = [self.boxes]
        filenames = [img_filename]

        # Apply augmentations
        if self.brightness_var.get():
            bright_img = cv2.convertScaleAbs(self.current_frame, alpha=1.2, beta=30)
            images_to_save.append(bright_img)
            labels_to_save.append(self.boxes)
            filenames.append(f"{img_filename}_bright")

        if self.rotation_var.get():
            # Simple 90 degree rotation
            rotated_img = cv2.rotate(self.current_frame, cv2.ROTATE_90_CLOCKWISE)
            rotated_boxes = self.rotate_boxes(self.boxes, 90)
            images_to_save.append(rotated_img)
            labels_to_save.append(rotated_boxes)
            filenames.append(f"{img_filename}_rot90")

        if self.flip_var.get():
            flipped_img = cv2.flip(self.current_frame, 1)
            flipped_boxes = self.flip_boxes(self.boxes)
            images_to_save.append(flipped_img)
            labels_to_save.append(flipped_boxes)
            filenames.append(f"{img_filename}_flip")

        # Save all versions
        saved_count = 0
        for img, boxes, filename in zip(images_to_save, labels_to_save, filenames):
            img_file = os.path.join(self.IMAGE_FOLDER, split, f"{filename}.jpg")
            label_file = os.path.join(self.LABEL_FOLDER, split, f"{filename}.txt")

            # Save image
            cv2.imwrite(img_file, img)

            # Save labels
            self.save_yolo_labels(label_file, boxes, self.IMG_SIZE, self.IMG_SIZE)
            saved_count += 1

        # Update statistics
        self.dataset_stats["total_images"] += saved_count
        self.dataset_stats["total_annotations"] += len(self.boxes) * saved_count
        self.dataset_stats["split_counts"][split] += saved_count

        for box in self.boxes:
            class_name = self.LABEL_CLASSES[box[4]]
            self.dataset_stats["class_distribution"][class_name] += saved_count

        self.save_stats()
        self.update_stats_display()

        # Clear boxes after saving
        self.boxes = []

        self.status_var.set(f"✅ Disimpan {saved_count} gambar ke {split}!")

    def save_yolo_labels(self, label_path, boxes, img_width, img_height):
        """Simpan label dalam format YOLO"""
        with open(label_path, 'w') as f:
            for box in boxes:
                x1, y1, x2, y2, class_idx = box

                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height

                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def rotate_boxes(self, boxes, angle):
        """Rotasi bounding boxes"""
        # Simplified rotation for 90 degrees
        rotated_boxes = []
        for box in boxes:
            x1, y1, x2, y2, class_idx = box
            if angle == 90:
                new_x1 = y1
                new_y1 = self.IMG_SIZE - x2
                new_x2 = y2
                new_y2 = self.IMG_SIZE - x1
                rotated_boxes.append((new_x1, new_y1, new_x2, new_y2, class_idx))
        return rotated_boxes

    def flip_boxes(self, boxes):
        """Flip bounding boxes horizontal"""
        flipped_boxes = []
        for box in boxes:
            x1, y1, x2, y2, class_idx = box
            new_x1 = self.IMG_SIZE - x2
            new_x2 = self.IMG_SIZE - x1
            flipped_boxes.append((new_x1, y1, new_x2, y2, class_idx))
        return flipped_boxes

    def update_stats_display(self):
        """Update tampilan statistik"""
        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)

        stats_text = f"""📊 STATISTIK DATASET 📊
Total Gambar: {self.dataset_stats['total_images']}
Total Anotasi: {self.dataset_stats['total_annotations']}

Split Distribution:
• Train: {self.dataset_stats['split_counts']['train']}
• Val: {self.dataset_stats['split_counts']['val']}
• Test: {self.dataset_stats['split_counts']['test']}

Top 3 Kelas:"""

        # Sort classes by count
        sorted_classes = sorted(self.dataset_stats['class_distribution'].items(),
                               key=lambda x: x[1], reverse=True)[:3]

        for class_name, count in sorted_classes:
            stats_text += f"\n• {class_name}: {count}"

        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state='disabled')

    def export_dataset(self):
        """Export dataset dalam format yang siap untuk training"""
        try:
            # Create dataset.yaml
            yaml_content = f"""path: {os.path.abspath(self.BASE_FOLDER)}
            train: images/train
            val: images/val
            test: images/test

            nc: {len(self.LABEL_CLASSES)}
            names: {self.LABEL_CLASSES}
            """

            yaml_path = os.path.join(self.BASE_FOLDER, "dataset.yaml")
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)

            # Create data split info
            split_info = {
                "dataset_info": {
                    "name": "Pakaian Dataset",
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "classes": self.LABEL_CLASSES,
                    "total_images": self.dataset_stats['total_images'],
                    "splits": self.dataset_stats['split_counts']
                }
            }

            info_path = os.path.join(self.BASE_FOLDER, "dataset_info.json")
            with open(info_path, 'w') as f:
                json.dump(split_info, f, indent=2)

            messagebox.showinfo("Sukses", f"Dataset berhasil diekspor!\n\nFiles created:\n• {yaml_path}\n• {info_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengekspor dataset: {str(e)}")

    def run(self):
        """Jalankan aplikasi"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle saat aplikasi ditutup"""
        if self.camera_active:
            self.stop_camera()
        self.save_stats()
        self.root.destroy()

if __name__ == "__main__":
    print("🚀 Memulai Sistem Penangkapan Dataset Pakaian - Advanced")
    app = SistemPenangkapanDataset()
    app.run()
