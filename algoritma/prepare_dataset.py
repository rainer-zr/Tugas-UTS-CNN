import os
import glob
import numpy as np
from PIL import Image

RAW_BASE = r"C:\Users\Raine\Downloads\coffee_raw"  # Ubah sesuai lokasi ekstrak .zip
REPO_DATASET = os.path.join("..", "dataset")
CLASSES = ["Green", "Light", "Medium", "Dark"]

# 1. Konversi seluruh dataset TRAIN ke 32x32 BMP di dalam repo (dataset/train/)
for cls_name in CLASSES:
    out_dir = os.path.join(REPO_DATASET, "train", cls_name.lower())
    os.makedirs(out_dir, exist_ok=True)
    for i, fpath in enumerate(sorted(glob.glob(os.path.join(RAW_BASE, "train", cls_name, "*.*")))):
        img = Image.open(fpath).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        img.save(os.path.join(out_dir, f"{cls_name.lower()}_{i:03d}.bmp"), format="BMP")

# 2. Konversi seluruh dataset TEST ke 32x32 BMP di dalam repo (dataset/test/) + file .mem Verilog
out_test = os.path.join(REPO_DATASET, "test")
os.makedirs(out_test, exist_ok=True)
hex_pixels, hex_labels = [], []

for label_idx, cls_name in enumerate(CLASSES):
    for i, fpath in enumerate(sorted(glob.glob(os.path.join(RAW_BASE, "test", cls_name, "*.*")))):
        img = Image.open(fpath).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        img.save(os.path.join(out_test, f"{cls_name.lower()}_{i:03d}.bmp"), format="BMP")

        for px in np.array(img, dtype=np.uint8).flatten():
            hex_pixels.append(f"{px:02x}")
        hex_labels.append(f"{label_idx:01x}")

with open(os.path.join(out_test, "test_vectors.mem"), "w") as f:
    f.write("\n".join(hex_pixels) + "\n")

with open(os.path.join(out_test, "test_labels.mem"), "w") as f:
    f.write("\n".join(hex_labels) + "\n")

print(f"Selesai! Total {len(hex_labels)} citra uji 32x32 .bmp tersimpan di dataset/test/")