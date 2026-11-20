import os
import glob
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Konfigurasi Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "dataset", "train")
OUTPUT_WEIGHTS_JSON = os.path.join(BASE_DIR, "weights_int8.json")
OUTPUT_VH = os.path.join(BASE_DIR, "..", "implementasi", "rtl", "cnn_weights.vh")

CLASSES = ["green", "light", "medium", "dark"]

print("1. Memuat Dataset Pelatihan (RGB 32x32)...")
X_train, y_train = [], []
for idx, cls_name in enumerate(CLASSES):
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    for fpath in glob.glob(os.path.join(cls_dir, "*.bmp")):
        # PERBAIKAN: Gunakan "RGB" agar sesuai dengan Verilog dan run_cnn_coffee.py
        img = Image.open(fpath).convert("RGB")
        X_train.append(np.array(img, dtype=np.float32) / 255.0)
        y_train.append(idx)

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=np.int32)
print(f"   Total data latih: {X_train.shape[0]} citra, Shape: {X_train.shape}")

print("2. Membangun & Melatih Model 2-Layer CNN (RGB)...")
# PERBAIKAN: Arsitektur disesuaikan dengan run_cnn_coffee.py dan Verilog
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    # Layer 1: Conv2D 3x3, 8 filter, ReLU
    tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding="valid", activation="relu", name="conv1"),
    # Pooling 2x2
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
    # Layer 2: Conv2D 3x3, 8 filter, ReLU
    tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding="valid", activation="relu", name="conv2"),
    # Global Average Pooling
    tf.keras.layers.GlobalAveragePooling2D(name="gap"),
    # Dense Layer (4 kelas)
    tf.keras.layers.Dense(4, name="fc")
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

print("3. Melakukan Kuantisasi Post-Training ke INT8...")
# Ekstrak bobot dari model yang sudah dilatih
c1_w_f, c1_b_f = model.get_layer("conv1").get_weights() # Shape: (3, 3, 3, 8), (8,)
c2_w_f, c2_b_f = model.get_layer("conv2").get_weights() # Shape: (3, 3, 8, 8), (8,)
fc_w_f, fc_b_f = model.get_layer("fc").get_weights()    # Shape: (8, 4), (4,)

# Fungsi bantu kuantisasi
def quantize_weights(w_f, b_f, w_scale_factor=127.0, b_scale_factor=127.0):
    w_max = np.max(np.abs(w_f))
    scale_w = w_scale_factor / w_max if w_max > 0 else 1.0
    w_q = np.clip(np.round(w_f * scale_w), -127, 127).astype(int)
    
    # Bias diskalakan sesuai dengan akumulasi fitur map
    b_q = np.round(b_f * scale_w * b_scale_factor).astype(int)
    return w_q, b_q, scale_w

c1_w_q, c1_b_q, _ = quantize_weights(c1_w_f, c1_b_f, b_scale_factor=255.0)
c2_w_q, c2_b_q, _ = quantize_weights(c2_w_f, c2_b_f, b_scale_factor=255.0)
fc_w_q, fc_b_q, _ = quantize_weights(fc_w_f, fc_b_f, b_scale_factor=64.0)

# Konversi ke format List agar bisa di-JSON serialize
# c1_w: [filter][channel][row][col] -> 8 x 3 x 3 x 3
c1_w_list = [[[[int(c1_w_q[r, c, ch, f]) for c in range(3)] for r in range(3)] for ch in range(3)] for f in range(8)]
c1_b_list = [int(c1_b_q[f]) for f in range(8)]

# c2_w: [filter][channel][row][col] -> 8 x 8 x 3 x 3
c2_w_list = [[[[int(c2_w_q[r, c, ch, f]) for c in range(3)] for r in range(3)] for ch in range(8)] for f in range(8)]
c2_b_list = [int(c2_b_q[f]) for f in range(8)]

# fc_w: [class][filter] -> 4 x 8
fc_w_list = [[int(fc_w_q[f, cls]) for f in range(8)] for cls in range(4)]
fc_b_list = [int(fc_b_q[cls]) for cls in range(4)]

print("4. Menyimpan Bobot ke JSON dan File Header Verilog (.vh)...")
# Simpan ke JSON (PERBAIKAN: Nama key disesuaikan dengan run_cnn_coffee.py)
weights_dict = {
    "c1_w": c1_w_list,
    "c1_b": c1_b_list,
    "c2_w": c2_w_list,
    "c2_b": c2_b_list,
    "fc_w": fc_w_list,
    "fc_b": fc_b_list
}

with open(OUTPUT_WEIGHTS_JSON, "w") as jf:
    json.dump(weights_dict, jf, indent=2)

# Generate file .vh untuk Verilog
with open(OUTPUT_VH, "w") as vf:
    vf.write("// Auto-generated INT8 CNN Weights from Python Golden Model\n")
    vf.write("`ifndef CNN_WEIGHTS_VH\n`define CNN_WEIGHTS_VH\n\n")
    
    # Conv 1 Weights & Bias
    for f in range(8):
        for ch in range(3):
            for r in range(3):
                for c in range(3):
                    idx = f * 27 + ch * 9 + r * 3 + c
                    vf.write(f"localparam signed [7:0] C1_W_{idx} = 8'sd{c1_w_list[f][ch][r][c]};\n")
        vf.write(f"localparam signed [31:0] C1_B_{f} = 32'sd{c1_b_list[f]};\n")
    
    # Conv 2 Weights & Bias
    for f in range(8):
        for ch in range(8):
            for r in range(3):
                for c in range(3):
                    idx = f * 72 + ch * 9 + r * 3 + c
                    vf.write(f"localparam signed [7:0] C2_W_{idx} = 8'sd{c2_w_list[f][ch][r][c]};\n")
        vf.write(f"localparam signed [31:0] C2_B_{f} = 32'sd{c2_b_list[f]};\n")

    # FC Weights & Bias
    for cls in range(4):
        for f in range(8):
            vf.write(f"localparam signed [7:0] FC_W_{cls}_{f} = 8'sd{fc_w_list[cls][f]};\n")
        vf.write(f"localparam signed [31:0] FC_B_{cls} = 32'sd{fc_b_list[cls]};\n")
        
    vf.write("\n`endif // CNN_WEIGHTS_VH\n")

print("✅ Selesai! weights_int8.json dan implementasi/rtl/cnn_weights.vh berhasil dibuat.")
print("   Silakan jalankan 'python run_cnn_coffee.py' untuk memvalidasi akurasi Python INT8.")
