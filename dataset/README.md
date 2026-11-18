# Dataset: Coffee Bean Roast Level (32×32 Grayscale)

- **Sumber Asli:** Kaggle — *Coffee Bean Dataset (Resized 224×224)* (`gpiosenka/coffee-bean-dataset-resized-224-x-224`).
- **Preprocessing Target:** Dikonversi menjadi citra **32×32 piksel grayscale (8-bit, 0–255)** dalam format `.bmp` untuk pengujian algoritma Python dan simulasi Verilog.
- **Label Kelas:**
  - `0` (`green`) — Biji kopi mentah (*Green Bean*)
  - `1` (`light`) — Sangrai terang (*Light Roast*)
  - `2` (`medium`) — Sangrai sedang (*Medium Roast*)
  - `3` (`dark`) — Sangrai gelap (*Dark Roast*)
- **Isi Folder `dataset/test/`:** 40 file uji `.bmp` (10 sampel per kelas) serta `test_vectors.mem` dan `test_labels.mem` untuk dibaca oleh *testbench* Verilog (`$readmemh`).
