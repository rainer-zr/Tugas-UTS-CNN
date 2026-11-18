import os
import glob
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

TEST_DIR = os.path.join("..", "dataset", "test")
CLASSES = ["green", "light", "medium", "dark"]

with open("weights_int8.json", "r") as jf:
    params = json.load(jf)

conv_w = np.array(params["conv_w"], dtype=np.int32)
conv_b = np.array(params["conv_b"], dtype=np.int32)
dense_w = np.array(params["dense_w"], dtype=np.int32)
dense_b = np.array(params["dense_b"], dtype=np.int32)

def predict_int8_bit_exact(img_32x32):
    img = img_32x32.astype(np.int32)
    pooled = np.zeros(4, dtype=np.int32)
    for f in range(4):
        max_val = 0
        for r in range(30):
            for c in range(30):
                acc = int(np.sum(img[r:r+3, c:c+3] * conv_w[f])) + int(conv_b[f])
                if acc < 0:
                    acc = 0
                if acc > max_val:
                    max_val = acc
        pooled[f] = max_val >> 6

    logits = np.zeros(4, dtype=np.int32)
    for cls in range(4):
        logits[cls] = int(np.sum(pooled * dense_w[cls])) + int(dense_b[cls])
    return int(np.argmax(logits))

bmp_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.bmp")))
correct = 0
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
shown = 0

for fpath in bmp_files:
    fname = os.path.basename(fpath)
    gt_label = fname.split("_")[0]
    gt_idx = CLASSES.index(gt_label)

    img = Image.open(fpath).convert("L")
    arr = np.array(img, dtype=np.uint8)
    pred_idx = predict_int8_bit_exact(arr)

    if pred_idx == gt_idx:
        correct += 1

    if shown < 4 and fname.endswith("_000.bmp"):
        axes[shown].imshow(arr, cmap="gray", vmin=0, vmax=255)
        axes[shown].set_title(f"Asli: {gt_label}\nPred: {CLASSES[pred_idx]}")
        axes[shown].axis("off")
        shown += 1

acc = (correct / len(bmp_files)) * 100.0
print("========================================")
print(f"Total Gambar Uji (.bmp 32x32) : {len(bmp_files)}")
print(f"Prediksi Benar (Python INT8)  : {correct}/{len(bmp_files)}")
print(f"Akurasi Algoritma Python      : {acc:.2f}%")
print("========================================")

plt.tight_layout()
plt.savefig(os.path.join("..", "output", "images", "step1_python_bmp_preview.png"), dpi=150)
plt.show()