import os
import glob
import json
import numpy as np
from PIL import Image
import tensorflow as tf

TRAIN_DIR = os.path.join("..", "dataset", "train")
CLASSES = ["green", "light", "medium", "dark"]

X_train, y_train = [], []
for idx, cls_name in enumerate(CLASSES):
    for fpath in glob.glob(os.path.join(TRAIN_DIR, cls_name, "*.bmp")):
        img = Image.open(fpath).convert("L")
        X_train.append(np.array(img, dtype=np.float32) / 255.0)
        y_train.append(idx)

X_train = np.expand_dims(np.array(X_train), axis=-1)
y_train = np.array(y_train, dtype=np.int32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(4, kernel_size=3, strides=1, padding="valid", activation="relu", name="conv1"),
    tf.keras.layers.GlobalMaxPooling2D(name="gmp"),
    tf.keras.layers.Dense(4, name="fc1")
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

# Kuantisasi bobot & bias ke INT8 (-127..127)
conv_w_f, conv_b_f = model.get_layer("conv1").get_weights()
dense_w_f, dense_b_f = model.get_layer("fc1").get_weights()

scale_c = 127.0 / np.max(np.abs(conv_w_f))
conv_w_q = np.clip(np.round(conv_w_f * scale_c), -127, 127).astype(int)
conv_b_q = np.round(conv_b_f * scale_c * 255.0).astype(int)

scale_d = 127.0 / np.max(np.abs(dense_w_f))
dense_w_q = np.clip(np.round(dense_w_f * scale_d), -127, 127).astype(int)
dense_b_q = np.round(dense_b_f * scale_d * 64.0).astype(int)

conv_w_list = [[[int(conv_w_q[r, c, 0, f]) for c in range(3)] for r in range(3)] for f in range(4)]
conv_b_list = [int(conv_b_q[f]) for f in range(4)]
dense_w_list = [[int(dense_w_q[f, cls]) for f in range(4)] for cls in range(4)]
dense_b_list = [int(dense_b_q[cls]) for cls in range(4)]

with open("weights_int8.json", "w") as jf:
    json.dump({
        "conv_w": conv_w_list,
        "conv_b": conv_b_list,
        "dense_w": dense_w_list,
        "dense_b": dense_b_list
    }, jf, indent=2)

vh_path = os.path.join("..", "implementasi", "rtl", "cnn_weights.vh")
with open(vh_path, "w") as vf:
    vf.write("// Auto-generated INT8 CNN Weights from Python Golden Model\n")
    for f in range(4):
        for r in range(3):
            for c in range(3):
                vf.write(f"localparam signed [7:0] CONV_W_{f}_{r}_{c} = 8'sd{conv_w_list[f][r][c]};\n")
        vf.write(f"localparam signed [31:0] CONV_B_{f} = 32'sd{conv_b_list[f]};\n")
    for cls in range(4):
        for f in range(4):
            vf.write(f"localparam signed [7:0] DENSE_W_{cls}_{f} = 8'sd{dense_w_list[cls][f]};\n")
        vf.write(f"localparam signed [31:0] DENSE_B_{cls} = 32'sd{dense_b_list[cls]};\n")

print("Selesai! weights_int8.json dan implementasi/rtl/cnn_weights.vh berhasil dibuat.")