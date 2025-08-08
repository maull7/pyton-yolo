import os
from collections import Counter, defaultdict

# Path ke folder label
label_folder = "dataset/labels/train"

# Simpan jumlah objek per gambar
combo_counter = Counter()
class_names = {0: 'kerah', 1: 'lengan_kanan', 2: 'lengan_kiri', 3: 'badan', 4: 'kantong'}

for file in os.listdir(label_folder):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(label_folder, file)
    with open(path, "r") as f:
        classes_in_img = set()
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            classes_in_img.add(cls)

        if classes_in_img:
            combo_counter[tuple(sorted(classes_in_img))] += 1

# Tampilkan hasil
print("Kombinasi objek yang muncul di 1 gambar:")
for combo, count in combo_counter.most_common():
    labels = [class_names[c] for c in combo]
    print(f"{labels}: {count} gambar")

print("\nTotal gambar:", sum(combo_counter.values()))

