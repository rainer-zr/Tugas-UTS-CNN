import os
import glob
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BMP_DIR = os.path.join(BASE_DIR, "dataset", "test_32x32_bmp")
WEIGHTS_PATH = os.path.join(BASE_DIR, "algoritma", "weights_int8.json")
OUT_IMG_DIR = os.path.join(BASE_DIR, "output", "images")
os.makedirs(OUT_IMG_DIR, exist_ok=True)

CLASSES = ["green", "light", "medium", "dark"]

with open(WEIGHTS_PATH, "r") as f:
    w = json.load(f)

c1_w = np.array(w["c1_w"], dtype=np.int64)
c1_b = np.array(w["c1_b"], dtype=np.int64)
c2_w = np.array(w["c2_w"], dtype=np.int64)
c2_b = np.array(w["c2_b"], dtype=np.int64)
fc_w = np.array(w["fc_w"], dtype=np.int64)
fc_b = np.array(w["fc_b"], dtype=np.int64)

def infer_bit_exact(img_32x32x3):
    img = img_32x32x3.astype(np.int64)

    # Layer 1: Conv2D 3x3 + ReLU + Bit-Shift (>>> 8)
    conv1_out = np.zeros((30, 30, 8), dtype=np.int64)
    for f in range(8):
        for r in range(30):
            for c in range(30):
                acc = int(c1_b[f])
                for ch in range(3):
                    acc += int(np.sum(img[r:r+3, c:c+3, ch] * c1_w[f, ch]))
                conv1_out[r, c, f] = (acc >> 8) if acc > 0 else 0

    # Layer 2: MaxPooling 2x2
    pool1_out = np.zeros((15, 15, 8), dtype=np.int64)
    for f in range(8):
        for r in range(15):
            for c in range(15):
                pool1_out[r, c, f] = np.max(conv1_out[r*2:r*2+2, c*2:c*2+2, f])

    # Layer 3: Conv2D 3x3 + ReLU + Global Average Pooling (>>> 10)
    gap_sum = np.zeros(8, dtype=np.int64)
    for f in range(8):
        s = 0
        for r in range(13):
            for c in range(13):
                acc = int(c2_b[f])
                for ch in range(8):
                    acc += int(np.sum(pool1_out[r:r+3, c:c+3, ch] * c2_w[f, ch]))
                if acc > 0:
                    s += (acc >> 8)
        gap_sum[f] = s >> 10

    # Layer 4: Fully Connected (Dense)
    logits = np.zeros(4, dtype=np.int64)
    for cls in range(4):
        logits[cls] = int(np.sum(gap_sum * fc_w[cls])) + int(fc_b[cls])

    return int(np.argmax(logits)), conv1_out, pool1_out

# Evaluasi seluruh 400 citra .bmp & simpan visualisasi Feature Map (Conv1 + ReLU)
correct = 0
total = 0
sample_img = None
sample_conv1 = None

for cls_idx, cls_name in enumerate(CLASSES):
    for fpath in sorted(glob.glob(os.path.join(BMP_DIR, f"{cls_name}_*.bmp"))):
        arr = np.array(Image.open(fpath).convert("RGB"), dtype=np.uint8)
        pred_idx, conv1_map, _ = infer_bit_exact(arr)
        total += 1
        if pred_idx == cls_idx:
            correct += 1
        if cls_name == "medium" and fpath.endswith("_000.bmp"):
            sample_img = arr
            sample_conv1 = conv1_map

print(f"Total Citra Uji (.bmp 32x32): {total}")
print(f"Prediksi Benar (Python INT8): {correct}/{total} ({(correct/total)*100:.2f}%)")

# Plot Input BMP 32x32 + 8 Channel Feature Map hasil Conv1 + ReLU
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes[0, 0].imshow(sample_img)
axes[0, 0].set_title("Input BMP 32x32\n(Medium Roast)")
axes[0, 0].axis("off")
axes[1, 0].axis("off")

for f in range(8):
    row = f // 4
    col = (f % 4) + 1
    axes[row, col].imshow(sample_conv1[:, :, f], cmap="viridis")
    axes[row, col].set_title(f"Conv1+ReLU F{f}\n(30x30)")
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_IMG_DIR, "step1b_feature_maps_relu.png"), dpi=150)
plt.show()