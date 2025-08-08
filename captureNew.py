import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime

# Konfigurasi folder
IMAGE_FOLDER = "dataset/images/train"
LABEL_FOLDER = "dataset/labels/train"
LABEL_CLASSES = ["kerah", "lengan_kanan", "lengan_kiri", "badan", "kantong"]
IMG_SIZE = 640
NUM_COPIES = 2
SNAP_THRESHOLD = 15  # Jarak snap ke garis dalam pixels

# Buat folder jika belum ada
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(LABEL_FOLDER, exist_ok=True)

# Kelas untuk menyimpan informasi bounding box
class BoundingBox:
    def __init__(self, x1, y1, x2, y2, label_index, label_name):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.label_index = label_index
        self.label_name = label_name
        self.color = self._get_color()

    def _get_color(self):
        # Warna berbeda untuk setiap kelas
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        return colors[self.label_index % len(colors)]

    def draw(self, img):
        # Gambar bounding box
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), self.color, 2)

        # Label background
        label_text = f"{self.label_name}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (self.x1, self.y1 - text_h - 5),
                     (self.x1 + text_w + 5, self.y1), self.color, -1)

        # Label text
        cv2.putText(img, label_text, (self.x1 + 2, self.y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def to_yolo_format(self, img_width, img_height):
        x_center = ((self.x1 + self.x2) / 2) / img_width
        y_center = ((self.y1 + self.y2) / 2) / img_height
        w = abs(self.x2 - self.x1) / img_width
        h = abs(self.y2 - self.y1) / img_height
        return f"{self.label_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

# Fungsi bantu simpan multiple labels YOLO
def save_multiple_labels(img_name, bounding_boxes, width, height):
    label_path = os.path.join(LABEL_FOLDER, f"{img_name}.txt")
    with open(label_path, "w") as f:
        for bbox in bounding_boxes:
            f.write(bbox.to_yolo_format(width, height) + "\n")

# Input label dari pengguna via Tkinter
def ask_label():
    root = tk.Tk()
    root.withdraw()

    # Buat dialog yang lebih user-friendly
    label_options = "\n".join([f"{i}: {label}" for i, label in enumerate(LABEL_CLASSES)])
    label = simpledialog.askstring(
        "Pilih Label",
        f"Masukkan nama label atau nomor:\n\n{label_options}\n\nKetik nama atau nomor (0-{len(LABEL_CLASSES)-1}):"
    )
    root.destroy()

    if label is None:
        return None

    # Coba parse sebagai nomor dulu
    try:
        label_idx = int(label)
        if 0 <= label_idx < len(LABEL_CLASSES):
            return label_idx
    except ValueError:
        pass

    # Coba cari berdasarkan nama
    if label in LABEL_CLASSES:
        return LABEL_CLASSES.index(label)

    return None

# Deteksi edge menggunakan Canny
def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Fungsi snap ke garis terdekat
def snap_to_edge(x, y, edges, threshold=SNAP_THRESHOLD):
    height, width = edges.shape
    x_start = max(0, x - threshold)
    x_end = min(width, x + threshold)
    y_start = max(0, y - threshold)
    y_end = min(height, y + threshold)

    min_dist = float('inf')
    snap_x, snap_y = x, y

    for py in range(y_start, y_end):
        for px in range(x_start, x_end):
            if edges[py, px] > 0:
                dist = np.sqrt((px - x)**2 + (py - y)**2)
                if dist < min_dist:
                    min_dist = dist
                    snap_x, snap_y = px, py

    return snap_x, snap_y

# Fungsi menggambar garis bantu
def draw_guide_lines(img, x, y):
    height, width = img.shape[:2]
    cv2.line(img, (0, y), (width, y), (0, 255, 255), 1)
    cv2.line(img, (x, 0), (x, height), (0, 255, 255), 1)

# Fungsi menggambar grid
def draw_grid(img, grid_size=25):
    height, width = img.shape[:2]
    for x in range(0, width, grid_size):
        cv2.line(img, (x, 0), (x, height), (80, 80, 80), 1)
    for y in range(0, height, grid_size):
        cv2.line(img, (0, y), (width, y), (80, 80, 80), 1)

# Fungsi snap ke grid
def snap_to_grid(x, y, grid_size=25):
    snap_x = round(x / grid_size) * grid_size
    snap_y = round(y / grid_size) * grid_size
    return snap_x, snap_y

# Fungsi menampilkan info
def draw_info(img, x1, y1, x2, y2, snap_mode):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    info_text = f"Size: {width}x{height}"
    coord_text = f"({x1},{y1}) -> ({x2},{y2})"
    mode_text = f"Snap: {snap_mode}"

    # Background untuk teks
    cv2.rectangle(img, (x1, y1-55), (x1+220, y1), (0, 0, 0), -1)
    cv2.putText(img, mode_text, (x1+2, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(img, coord_text, (x1+2, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, info_text, (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Fungsi untuk menggambar garis pensil
def draw_pencil_lines(img, lines, color=(0, 255, 0), thickness=2):
    for line in lines:
        if len(line) > 1:
            for i in range(len(line) - 1):
                cv2.line(img, line[i], line[i + 1], color, thickness)

# Fungsi untuk menggambar garis lurus
def draw_straight_line(img, start, end, color=(255, 0, 0), thickness=2):
    cv2.line(img, start, end, color, thickness)

# Fungsi untuk menghitung bounding box dari garis pensil
def get_pencil_bbox(lines):
    if not lines or all(len(line) == 0 for line in lines):
        return None

    all_points = []
    for line in lines:
        all_points.extend(line)

    if not all_points:
        return None

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    margin = 10
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(IMG_SIZE, x2 + margin)
    y2 = min(IMG_SIZE, y2 + margin)

    return (x1, y1, x2, y2)

# Fungsi untuk menampilkan summary bounding boxes
def show_bbox_summary(bounding_boxes):
    if not bounding_boxes:
        return "No objects labeled"

    summary = []
    label_counts = {}

    for bbox in bounding_boxes:
        label_counts[bbox.label_name] = label_counts.get(bbox.label_name, 0) + 1

    for label, count in label_counts.items():
        summary.append(f"{label}: {count}")

    return f"Objects: {len(bounding_boxes)} | " + " | ".join(summary)

# Buka kamera
CAMERA_IP = "http://10.94.239.254:8080/video/mjpeg"
cap = cv2.VideoCapture(0)
cv2.namedWindow("Multi-Label Object Detection")

# Global variables
drawing = False
show_grid = True
show_edges = True
snap_mode = "edge"  # "edge", "grid", "off"
draw_mode = "box"  # "box", "pencil", "line"
pencil_drawing = False
ix, iy = -1, -1
current_frame = None
current_edges = None
mouse_x, mouse_y = 0, 0
pencil_points = []
drawn_lines = []

# NEW: Multiple bounding boxes storage
current_bounding_boxes = []  # List of BoundingBox objects
temp_bbox = None  # Temporary bbox being drawn

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, pencil_drawing, current_frame, current_edges
    global mouse_x, mouse_y, pencil_points, drawn_lines, temp_bbox

    # Terapkan snapping berdasarkan mode (kecuali untuk pencil mode)
    if draw_mode != "pencil":
        if snap_mode == "edge" and current_edges is not None:
            x, y = snap_to_edge(x, y, current_edges)
        elif snap_mode == "grid":
            x, y = snap_to_grid(x, y)

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        if draw_mode == "box":
            drawing = True
            ix, iy = x, y
            print(f"[INFO] Start box at ({ix}, {iy})")
        elif draw_mode == "pencil":
            pencil_drawing = True
            pencil_points = [(x, y)]
            print("[INFO] Start pencil drawing")
        elif draw_mode == "line":
            drawing = True
            ix, iy = x, y
            print(f"[INFO] Start line at ({ix}, {iy})")

    elif event == cv2.EVENT_MOUSEMOVE:
        img_copy = current_frame.copy()

        # Tampilkan edges jika diaktifkan
        if show_edges and current_edges is not None:
            edge_overlay = cv2.cvtColor(current_edges, cv2.COLOR_GRAY2BGR)
            edge_overlay[:, :, 1:] = 0
            img_copy = cv2.addWeighted(img_copy, 0.8, edge_overlay, 0.3, 0)

        # Tampilkan grid jika diaktifkan
        if show_grid:
            draw_grid(img_copy)

        # Tampilkan semua bounding boxes yang sudah disimpan
        for bbox in current_bounding_boxes:
            bbox.draw(img_copy)

        # Tampilkan garis bantu (kecuali untuk pencil mode)
        if draw_mode != "pencil":
            draw_guide_lines(img_copy, x, y)

        # Tampilkan garis yang sudah digambar sebelumnya
        draw_pencil_lines(img_copy, drawn_lines, (0, 255, 0), 2)

        if draw_mode == "box" and drawing:
            # Mode bounding box
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            draw_info(img_copy, ix, iy, x, y, snap_mode)

            if snap_mode == "edge":
                cv2.circle(img_copy, (x, y), 5, (0, 0, 255), 2)
            elif snap_mode == "grid":
                cv2.circle(img_copy, (x, y), 3, (255, 0, 255), 2)

        elif draw_mode == "pencil" and pencil_drawing:
            pencil_points.append((x, y))
            if len(pencil_points) > 1:
                for i in range(len(pencil_points) - 1):
                    cv2.line(img_copy, pencil_points[i], pencil_points[i + 1], (255, 0, 255), 3)
            cv2.circle(img_copy, (x, y), 5, (255, 0, 255), 2)

        elif draw_mode == "line" and drawing:
            draw_straight_line(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
            length = int(np.sqrt((x - ix)**2 + (y - iy)**2))
            cv2.putText(img_copy, f"Length: {length}px", (ix, iy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        else:
            # Mode idle
            coord_text = f"({x},{y}) [{draw_mode.upper()}]"
            if draw_mode != "pencil" and snap_mode != "off":
                coord_text += f" [SNAP: {snap_mode.upper()}]"

            cv2.putText(img_copy, coord_text, (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if draw_mode != "pencil":
                if snap_mode == "edge":
                    cv2.circle(img_copy, (x, y), 3, (0, 0, 255), 1)
                elif snap_mode == "grid":
                    cv2.circle(img_copy, (x, y), 2, (255, 0, 255), 1)

            if draw_mode == "pencil":
                cv2.circle(img_copy, (x, y), 3, (255, 0, 255), 2)

        cv2.imshow("Multi-Label Object Detection", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        if draw_mode == "box" and drawing:
            drawing = False
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)

            print(f"[INFO] Box completed: ({x1},{y1}) to ({x2},{y2})")

            if abs(x2-x1) < 5 or abs(y2-y1) < 5:
                print("[!] Box terlalu kecil! Minimal 5x5 pixels.")
                return

            _add_bounding_box(x1, y1, x2, y2)

        elif draw_mode == "pencil" and pencil_drawing:
            pencil_drawing = False
            if len(pencil_points) > 1:
                drawn_lines.append(pencil_points[:])
                print(f"[INFO] Pencil line added with {len(pencil_points)} points")
            pencil_points = []

        elif draw_mode == "line" and drawing:
            drawing = False
            line_points = [(ix, iy), (x, y)]
            drawn_lines.append(line_points)
            print(f"[INFO] Straight line added: ({ix},{iy}) to ({x},{y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if draw_mode == "pencil" and drawn_lines:
            bbox = get_pencil_bbox(drawn_lines)
            if bbox:
                x1, y1, x2, y2 = bbox
                print(f"[INFO] Generated bbox from pencil: ({x1},{y1}) to ({x2},{y2})")
                _add_bounding_box(x1, y1, x2, y2)
        elif draw_mode == "line" and drawn_lines:
            bbox = get_pencil_bbox(drawn_lines)
            if bbox:
                x1, y1, x2, y2 = bbox
                print(f"[INFO] Generated bbox from lines: ({x1},{y1}) to ({x2},{y2})")
                _add_bounding_box(x1, y1, x2, y2)

def _add_bounding_box(x1, y1, x2, y2):
    """Helper function untuk menambah bounding box ke list"""
    label_index = ask_label()
    if label_index is not None:
        bbox = BoundingBox(x1, y1, x2, y2, label_index, LABEL_CLASSES[label_index])
        current_bounding_boxes.append(bbox)
        print(f"[✓] Added bounding box: {LABEL_CLASSES[label_index]} ({len(current_bounding_boxes)} total)")

        # Clear drawn lines after adding bbox
        drawn_lines.clear()
    else:
        print("[!] Label tidak dikenali atau dibatalkan.")

def save_current_image():
    """Save current image with all bounding boxes"""
    if not current_bounding_boxes:
        print("[!] Tidak ada bounding box untuk disimpan!")
        return False

    # Konfirmasi save
    root = tk.Tk()
    root.withdraw()

    summary = show_bbox_summary(current_bounding_boxes)
    result = messagebox.askyesno(
        "Konfirmasi Simpan",
        f"Save gambar dengan {len(current_bounding_boxes)} objek?\n\n{summary}\n\n"
        f"Akan membuat {NUM_COPIES} salinan gambar + label."
    )
    root.destroy()

    if not result:
        return False

    height, width = current_frame.shape[:2]

    # Save multiple copies
    for i in range(NUM_COPIES):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        img_filename = f"img_{timestamp}_{i}"
        img_path = os.path.join(IMAGE_FOLDER, f"{img_filename}.jpg")

        # Save image
        cv2.imwrite(img_path, current_frame)

        # Save labels
        save_multiple_labels(img_filename, current_bounding_boxes, width, height)

    print(f"[✓] Disimpan {NUM_COPIES} salinan dengan {len(current_bounding_boxes)} objek!")

    # Clear current bounding boxes after saving
    current_bounding_boxes.clear()
    return True

def delete_last_bbox():
    """Hapus bounding box terakhir"""
    if current_bounding_boxes:
        removed = current_bounding_boxes.pop()
        print(f"[✓] Removed last bbox: {removed.label_name} ({len(current_bounding_boxes)} remaining)")
    else:
        print("[!] No bounding boxes to remove")

# Set mouse callback
cv2.setMouseCallback("Multi-Label Object Detection", mouse_callback)

print("[INFO] === MULTI-LABEL OBJECT DETECTION TOOL ===")
print("[INFO] Kontrol:")
print("  === DRAWING MODES ===")
print("  - 'q': Box Mode (bounding box)")
print("  - 'w': Pencil Mode (drawing bebas)")
print("  - 'r': Line Mode (garis lurus)")
print("  === SNAP MODES ===")
print("  - '1': Snap to Edges (deteksi tepi objek)")
print("  - '2': Snap to Grid (25x25 pixels)")
print("  - '3': No Snap (free drawing)")
print("  === DISPLAY ===")
print("  - 'g': Toggle grid display on/off")
print("  - 'e': Toggle edge display on/off")
print("  - 'c': Clear all drawn lines")
print("  === MULTI-LABEL FEATURES ===")
print("  - 's': SAVE current image with ALL bounding boxes")
print("  - 'd': DELETE last bounding box")
print("  - 'z': Clear ALL bounding boxes (start over)")
print("  === LABEL WORKFLOW ===")
print("  1. Draw multiple bounding boxes (each will ask for label)")
print("  2. Press 's' to save image with all labels")
print("  3. Continue with new image or add more boxes")
print("  === SPECIAL ===")
print("  - ESC: Keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Tidak bisa mengambil frame dari kamera.")
        break

    current_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Update edge detection
    if show_edges:
        current_edges = detect_edges(current_frame)

    # Tampilkan frame dengan overlay jika tidak sedang drawing
    if not drawing:
        display_frame = current_frame.copy()

        # Overlay edges
        if show_edges and current_edges is not None:
            edge_overlay = cv2.cvtColor(current_edges, cv2.COLOR_GRAY2BGR)
            edge_overlay[:, :, 1:] = 0
            display_frame = cv2.addWeighted(display_frame, 0.8, edge_overlay, 0.3, 0)

        # Grid
        if show_grid:
            draw_grid(display_frame)

        # Tampilkan semua bounding boxes
        for bbox in current_bounding_boxes:
            bbox.draw(display_frame)

        # Tampilkan garis yang sudah digambar
        draw_pencil_lines(display_frame, drawn_lines, (0, 255, 0), 2)

        # Guide lines dan info mouse
        if mouse_x > 0 and mouse_y > 0:
            if draw_mode != "pencil":
                draw_guide_lines(display_frame, mouse_x, mouse_y)

            coord_text = f"({mouse_x},{mouse_y}) [{draw_mode.upper()}]"
            if draw_mode != "pencil" and snap_mode != "off":
                coord_text += f" [{snap_mode.upper()}]"
            cv2.putText(display_frame, coord_text, (mouse_x+10, mouse_y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if draw_mode == "pencil":
                cv2.circle(display_frame, (mouse_x, mouse_y), 3, (255, 0, 255), 2)

        # Status bar (enhanced for multi-label)
        status = f"Mode: {draw_mode.upper()} | Snap: {snap_mode.upper()} | Grid: {'ON' if show_grid else 'OFF'} | Edges: {'ON' if show_edges else 'OFF'}"
        if drawn_lines:
            status += f" | Lines: {len(drawn_lines)}"

        # Bounding box summary
        bbox_summary = show_bbox_summary(current_bounding_boxes)

        # Status bar background
        status_height = 45 if current_bounding_boxes else 25
        cv2.rectangle(display_frame, (5, 5), (len(max(status, bbox_summary, key=len))*7 + 10, status_height), (0, 0, 0), -1)

        # Status text
        cv2.putText(display_frame, status, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        if current_bounding_boxes:
            cv2.putText(display_frame, bbox_summary, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.imshow("Multi-Label Object Detection", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('q'):  # Box mode
        draw_mode = "box"
        print("[INFO] Draw mode: BOUNDING BOX")
    elif key == ord('w'):  # Pencil mode
        draw_mode = "pencil"
        print("[INFO] Draw mode: PENCIL (free drawing)")
    elif key == ord('r'):  # Line mode
        draw_mode = "line"
        print("[INFO] Draw mode: STRAIGHT LINE")
    elif key == ord('1'):  # Snap to edges
        snap_mode = "edge"
        print("[INFO] Snap mode: EDGES")
    elif key == ord('2'):  # Snap to grid
        snap_mode = "grid"
        print("[INFO] Snap mode: GRID")
    elif key == ord('3'):  # No snap
        snap_mode = "off"
        print("[INFO] Snap mode: OFF")
    elif key == ord('g'):  # Toggle grid
        show_grid = not show_grid
        print(f"[INFO] Grid display: {'ON' if show_grid else 'OFF'}")
    elif key == ord('e'):  # Toggle edges
        show_edges = not show_edges
        print(f"[INFO] Edge display: {'ON' if show_edges else 'OFF'}")
    elif key == ord('c'):  # Clear drawn lines
        drawn_lines.clear()
        pencil_points.clear()
        print("[INFO] All drawn lines cleared")
    elif key == ord('s'):  # SAVE current image with all bounding boxes
        save_current_image()
    elif key == ord('d'):  # DELETE last bounding box
        delete_last_bbox()
    elif key == ord('z'):  # Clear ALL bounding boxes
        if current_bounding_boxes:
            current_bounding_boxes.clear()
            print("[INFO] All bounding boxes cleared")
        else:
            print("[INFO] No bounding boxes to clear")
    elif key == ord(' '):  # SPACE - Save last line as bbox (Line mode only)
        if draw_mode == "line" and drawn_lines:
            last_line = [drawn_lines[-1]]
            bbox = get_pencil_bbox(last_line)
            if bbox:
                x1, y1, x2, y2 = bbox
                print(f"[INFO] Converting last line to bbox: ({x1},{y1}) to ({x2},{y2})")
                _add_bounding_box(x1, y1, x2, y2)
                # Remove the line that was converted
                drawn_lines.pop()

cap.release()
cv2.destroyAllWindows()
